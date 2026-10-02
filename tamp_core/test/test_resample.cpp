// Unit tests for the uniform-Δt resampler.
//
// This is the piece the offset identity (b - a = k - l) rests on: it only holds
// if one slot means the same duration for every robot and every trajectory. So
// these tests are not box-ticking -- if the resampler is wrong, every collision
// matrix downstream is wrong, and the schedule is unsound in a way nothing else
// would catch.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "multi_robot_cell_tamp/resample.hpp"

using multi_robot_cell_tamp::makeDwell;
using multi_robot_cell_tamp::ObjectState;
using multi_robot_cell_tamp::Phase;
using multi_robot_cell_tamp::resampleUniform;
using multi_robot_cell_tamp::Segment;
using multi_robot_cell_tamp::TimedWaypoint;

namespace
{

/// A straight ramp of one joint from `p0` to `p1` over `duration`, with the
/// matching constant velocity, as a time-parameterised segment.
Segment ramp(double p0, double p1, double duration, Phase phase)
{
  const double v = (p1 - p0) / duration;
  Segment s;
  s.phase = phase;
  s.waypoints.push_back(TimedWaypoint{0.0, {p0}, {v}});
  s.waypoints.push_back(TimedWaypoint{duration, {p1}, {v}});
  return s;
}

}  // namespace

// --------------------------------------------------------------------------- #
// The clock
// --------------------------------------------------------------------------- #
TEST(Resample, SamplesLandOnAUniformGrid)
{
  // 1 second at Δt=0.1 => slots at t=0.0 .. 1.0 => 11 samples.
  const auto out = resampleUniform({ramp(0.0, 1.0, 1.0, Phase::ToPick)}, 0.1);

  ASSERT_EQ(out.num_samples, 11u);
  ASSERT_EQ(out.num_joints, 1u);
  EXPECT_TRUE(out.used_velocities);

  // A constant-velocity ramp is reproduced exactly by cubic Hermite, so each
  // sample must be its exact linear value -- any drift here is an arithmetic bug,
  // not interpolation error.
  for (std::size_t k = 0; k < out.num_samples; ++k) {
    EXPECT_NEAR(out.sample(k)[0], 0.1 * static_cast<double>(k), 1e-12) << "at slot " << k;
  }
}

TEST(Resample, EndpointsAreExactNotInterpolated)
{
  // 0.95 s at Δt=0.1: ceil(9.5)+1 = 11 samples, so the last slot lands PAST the
  // end of the path. The final sample must still be the final waypoint exactly.
  // ADR-0004 parks an idle robot AT home, and the home-clearance check runs
  // against that exact configuration -- "very close to home" is not home.
  const auto out = resampleUniform({ramp(0.0, 1.0, 0.95, Phase::ToPick)}, 0.1);

  EXPECT_NEAR(out.sample(0)[0], 0.0, 1e-12);
  EXPECT_NEAR(out.sample(out.num_samples - 1)[0], 1.0, 1e-12);
}

TEST(Resample, DifferentDurationsShareTheSameClock)
{
  // Two trajectories of unequal length must agree on what a slot MEANS: slot k of
  // one is simultaneous with slot k of the other. Without this, b - a = k - l is
  // simply false and the whole method collapses.
  const auto fast = resampleUniform({ramp(0.0, 1.0, 0.5, Phase::ToPick)}, 0.1);
  const auto slow = resampleUniform({ramp(0.0, 1.0, 2.0, Phase::ToPick)}, 0.1);

  EXPECT_EQ(fast.num_samples, 6u);    // 0.5 s / 0.1 + 1
  EXPECT_EQ(slow.num_samples, 21u);   // 2.0 s / 0.1 + 1
  EXPECT_DOUBLE_EQ(fast.delta_t, slow.delta_t);

  // Slot 3 is t=0.3 s for both, regardless of their different total durations.
  EXPECT_NEAR(fast.sample(3)[0], 0.3 / 0.5, 1e-12);
  EXPECT_NEAR(slow.sample(3)[0], 0.3 / 2.0, 1e-12);
}

// --------------------------------------------------------------------------- #
// Gripper dwell as slots
// --------------------------------------------------------------------------- #
TEST(Resample, DwellFreezesTheArmAndConsumesSlots)
{
  // home -> grasp (1.0 s), close gripper (10 slots = 1.0 s), carry (1.0 s).
  // The dwell must consume exactly 10 slots with the arm perfectly still.
  std::vector<Segment> segs;
  segs.push_back(ramp(0.0, 1.0, 1.0, Phase::ToPick));
  segs.push_back(makeDwell({1.0}, Phase::GripClose, 10, 0.1));
  segs.push_back(ramp(1.0, 2.0, 1.0, Phase::Carrying));

  const auto out = resampleUniform(segs, 0.1);

  // 1.0 + 1.0 + 1.0 = 3.0 s => 31 samples.
  ASSERT_EQ(out.num_samples, 31u);

  std::size_t frozen = 0;
  for (std::size_t k = 0; k < out.num_samples; ++k) {
    if (out.phase[k] == Phase::GripClose) {
      ++frozen;
      // Frozen means frozen: the arm sits exactly at the grasp configuration.
      EXPECT_NEAR(out.sample(k)[0], 1.0, 1e-12) << "arm moved during the dwell, at slot " << k;
    }
  }
  EXPECT_EQ(frozen, 10u) << "the gripper dwell must consume exactly its slots";
}

// --------------------------------------------------------------------------- #
// Where the object is
// --------------------------------------------------------------------------- #
TEST(Resample, ObjectStateTracksThePhase)
{
  // The object does not stop existing while the robot flies out to fetch it: it
  // is at its SPAWN pose on the way there, on the gripper while carried, and at
  // its PLACE pose on the way home. Getting this wrong lets the collision stage
  // route the other robot straight through an object that is really there.
  std::vector<Segment> segs;
  segs.push_back(ramp(0.0, 1.0, 1.0, Phase::ToPick));
  segs.push_back(makeDwell({1.0}, Phase::GripClose, 2, 0.1));
  segs.push_back(ramp(1.0, 2.0, 1.0, Phase::Carrying));
  segs.push_back(makeDwell({2.0}, Phase::GripOpen, 2, 0.1));
  segs.push_back(ramp(2.0, 3.0, 1.0, Phase::ToHome));

  const auto out = resampleUniform(segs, 0.1);
  ASSERT_EQ(out.num_samples, out.phase.size());
  ASSERT_EQ(out.num_samples, out.object_state.size());

  for (std::size_t k = 0; k < out.num_samples; ++k) {
    switch (out.phase[k]) {
      case Phase::ToPick:
        EXPECT_EQ(out.object_state[k], ObjectState::AtSpawn) << "slot " << k;
        break;
      case Phase::GripClose:
      case Phase::Carrying:
      case Phase::GripOpen:
        EXPECT_EQ(out.object_state[k], ObjectState::Attached) << "slot " << k;
        break;
      case Phase::ToHome:
        EXPECT_EQ(out.object_state[k], ObjectState::AtPlace) << "slot " << k;
        break;
      case Phase::ProcessOn:
      case Phase::Processing:
      case Phase::ProcessOff:
        ADD_FAILURE() << "a pick-and-place cycle produced a process phase at slot " << k;
        break;
    }
  }

  // Every phase must actually occur -- otherwise this test passes vacuously.
  for (const auto p : {Phase::ToPick, Phase::GripClose, Phase::Carrying, Phase::GripOpen,
                       Phase::ToHome}) {
    EXPECT_NE(std::find(out.phase.begin(), out.phase.end(), p), out.phase.end())
      << "phase " << multi_robot_cell_tamp::phaseName(p) << " never appears";
  }
}

// --------------------------------------------------------------------------- #
// A process (weld) cycle
// --------------------------------------------------------------------------- #
TEST(Resample, AProcessCycleCarriesNoObject)
{
  // The commutes of a process task deliberately reuse ToPick/ToHome so the
  // ADR-0008 yield refinement, which finds a parking pose by scanning for exactly
  // those two phases, keeps working. The price is that `objectStateFor` would
  // report AtSpawn on the way out and AtPlace on the way back for a task that has
  // no object at all. The resampler is not the place that fixes it -- the
  // generator flattens `object_state` afterwards -- but the three process phases
  // must at least be neutral, and they must survive the round trip through the
  // sampler intact.
  std::vector<Segment> segs;
  segs.push_back(ramp(0.0, 1.0, 1.0, Phase::ToPick));
  segs.push_back(makeDwell({1.0}, Phase::ProcessOn, 2, 0.1));
  segs.push_back(ramp(1.0, 1.2, 0.5, Phase::Processing));
  segs.push_back(makeDwell({1.2}, Phase::ProcessOff, 2, 0.1));
  segs.push_back(ramp(1.2, 0.0, 1.0, Phase::ToHome));

  const auto out = resampleUniform(segs, 0.1);
  for (const auto p : {Phase::ProcessOn, Phase::Processing, Phase::ProcessOff}) {
    EXPECT_NE(std::find(out.phase.begin(), out.phase.end(), p), out.phase.end())
      << "phase " << multi_robot_cell_tamp::phaseName(p) << " never appears";
  }
  for (std::size_t k = 0; k < out.num_samples; ++k) {
    if (out.phase[k] == Phase::ProcessOn || out.phase[k] == Phase::Processing ||
      out.phase[k] == Phase::ProcessOff)
    {
      EXPECT_EQ(out.object_state[k], ObjectState::AtSpawn) << "slot " << k;
    }
  }

  // The arm is frozen for the whole of both dwells -- that is what makes the arc
  // strike and the arc-out cost real slots without moving the tool off the seam.
  for (std::size_t k = 1; k < out.num_samples; ++k) {
    if (out.phase[k] == Phase::ProcessOn && out.phase[k - 1] == Phase::ProcessOn) {
      EXPECT_NEAR(out.sample(k)[0], 1.0, 1e-12) << "slot " << k;
    }
    if (out.phase[k] == Phase::ProcessOff && out.phase[k - 1] == Phase::ProcessOff) {
      EXPECT_NEAR(out.sample(k)[0], 1.2, 1e-12) << "slot " << k;
    }
  }
}

// --------------------------------------------------------------------------- #
// The support-surface flag follows its segment exactly
// --------------------------------------------------------------------------- #
TEST(Resample, SupportContactFollowsItsSegment)
{
  // The flag scopes a collision allowance, so it must be neither wider nor
  // narrower than the segments that set it -- in particular a free-space carry
  // between two support-contact segments must come out with the flag CLEAR.
  auto flagged = [](Segment s) {s.support_contact = true; return s;};
  std::vector<Segment> segs;
  segs.push_back(ramp(0.0, 1.0, 1.0, Phase::ToPick));                     // fly out
  segs.push_back(flagged(ramp(1.0, 0.9, 0.5, Phase::ToPick)));            // descent
  segs.push_back(flagged(makeDwell({0.9}, Phase::GripClose, 3, 0.1)));
  segs.push_back(flagged(ramp(0.9, 1.0, 0.5, Phase::Carrying)));          // lift
  segs.push_back(ramp(1.0, 2.0, 1.0, Phase::Carrying));                   // transfer
  segs.push_back(flagged(ramp(2.0, 1.9, 0.5, Phase::Carrying)));          // place descent

  const auto out = resampleUniform(segs, 0.1);
  ASSERT_EQ(out.support_contact.size(), out.num_samples);
  // Segment starts: 0, 1.0, 1.5, 1.8, 2.3, 3.3; a sample on a boundary belongs to
  // the segment that STARTS there.
  for (std::size_t k = 0; k < out.num_samples; ++k) {
    const double t = k * 0.1;
    const bool transfer = t >= 2.3 - 1e-9 && t < 3.3 - 1e-9;
    const bool fly_out = t < 1.0 - 1e-9;
    EXPECT_EQ(out.support_contact[k] != 0, !(transfer || fly_out)) << "slot " << k;
  }
}

// --------------------------------------------------------------------------- #
// The phase encoding is part of the artifact format
// --------------------------------------------------------------------------- #
TEST(Resample, PhaseValuesAreFrozen)
{
  // These integers are written into `tamp_trajectories.json` and read back by the
  // collision stages, the refinement and both executors. Renumbering them would
  // silently reinterpret every archived artifact under `artifacts/runs/*`.
  EXPECT_EQ(static_cast<int>(Phase::ToPick), 0);
  EXPECT_EQ(static_cast<int>(Phase::GripClose), 1);
  EXPECT_EQ(static_cast<int>(Phase::Carrying), 2);
  EXPECT_EQ(static_cast<int>(Phase::GripOpen), 3);
  EXPECT_EQ(static_cast<int>(Phase::ToHome), 4);
  EXPECT_EQ(static_cast<int>(Phase::ProcessOn), 5);
  EXPECT_EQ(static_cast<int>(Phase::Processing), 6);
  EXPECT_EQ(static_cast<int>(Phase::ProcessOff), 7);
}

// --------------------------------------------------------------------------- #
// The Δt soundness metric
// --------------------------------------------------------------------------- #
TEST(Resample, ReportsTheLargestJointStep)
{
  // A joint sweeping 2 rad in 1 s, sampled at 0.1 s, moves ~0.2 rad per slot. If
  // that is larger than the smallest feature in the scene, the collision check can
  // step over a collision. The number has to be reported, not assumed.
  const auto out = resampleUniform({ramp(0.0, 2.0, 1.0, Phase::ToPick)}, 0.1);
  EXPECT_NEAR(out.max_joint_step, 0.2, 1e-9);
  EXPECT_EQ(out.max_step_joint, 0u);

  // Halving Δt must halve the step -- that is the lever you pull when it is too big.
  const auto finer = resampleUniform({ramp(0.0, 2.0, 1.0, Phase::ToPick)}, 0.05);
  EXPECT_NEAR(finer.max_joint_step, 0.1, 1e-9);
}

TEST(Resample, FallsBackToLinearAndSaysSo)
{
  // No velocities => linear interpolation. It must be reported, not silently
  // substituted: a linear interpolant cuts the corner of a curved path, and that
  // is exactly the error that puts a resampled sample somewhere MoveIt never
  // checked.
  Segment s;
  s.phase = Phase::ToPick;
  s.waypoints.push_back(TimedWaypoint{0.0, {0.0}, {}});
  s.waypoints.push_back(TimedWaypoint{1.0, {1.0}, {}});

  const auto out = resampleUniform({s}, 0.1);
  EXPECT_FALSE(out.used_velocities);
  EXPECT_NEAR(out.sample(5)[0], 0.5, 1e-12);
}

// --------------------------------------------------------------------------- #
// Refusals
// --------------------------------------------------------------------------- #
TEST(Resample, RejectsGarbage)
{
  EXPECT_THROW(resampleUniform({}, 0.1), std::invalid_argument);
  EXPECT_THROW(resampleUniform({ramp(0.0, 1.0, 1.0, Phase::ToPick)}, 0.0),
               std::invalid_argument);
  EXPECT_THROW(makeDwell({1.0}, Phase::GripClose, 0, 0.1), std::invalid_argument);

  Segment mismatched;
  mismatched.phase = Phase::ToPick;
  mismatched.waypoints.push_back(TimedWaypoint{0.0, {0.0, 0.0}, {0.0, 0.0}});
  mismatched.waypoints.push_back(TimedWaypoint{1.0, {1.0}, {1.0}});
  EXPECT_THROW(resampleUniform({mismatched}, 0.1), std::invalid_argument);
}
