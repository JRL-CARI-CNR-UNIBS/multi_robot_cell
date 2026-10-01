// Uniform-Δt resampling of a concatenated, time-parameterised trajectory.
//
// WHY THIS EXISTS
//
// MoveIt returns a trajectory whose waypoints sit at NON-uniform
// `time_from_start` -- the time parameteriser (TOTG) places them wherever the
// velocity/acceleration limits demand. The scheduling model needs the opposite:
// samples at a UNIFORM step Δt, on a clock SHARED by both robots. That is not a
// cosmetic difference. The whole method rests on the identity
//
//     sample k of (r,i) and sample l of (s,j) are simultaneous  <=>  b - a = k - l
//
// which is only true if one slot means the same duration for every robot and
// every trajectory. Resampling to a common Δt is what buys that identity.
//
// Gripper actuation becomes a DWELL: the arm is frozen at a configuration for a
// whole number of slots. The slots are still consumed, so K_ri accounts for the
// time the gripper takes -- and a robot is, at every slot, either executing one
// trajectory or parked at home (ADR-0004).
//
// THE TRAP
//
// A resampled configuration lies BETWEEN waypoints MoveIt validated, so it is a
// configuration MoveIt never checked. On a 7-DOF arm skirting an obstacle, the
// interpolant can bulge into collision between two collision-free waypoints.
// Every resampled sample must therefore be re-validated against the scene by the
// caller (see `TrajectoryGenerator::validateSamples`). This header only produces
// the samples; it cannot know about geometry.

#ifndef MULTI_ROBOT_CELL_TAMP__RESAMPLE_HPP_
#define MULTI_ROBOT_CELL_TAMP__RESAMPLE_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace multi_robot_cell_tamp
{

/// One waypoint on a time-parameterised path: joint positions, velocities, and
/// the time at which it is reached. `velocities` may be empty (then the
/// resampler falls back to linear interpolation, and says so).
struct TimedWaypoint
{
  double time_from_start{0.0};
  std::vector<double> positions;
  std::vector<double> velocities;
};

/// What the robot is doing during a stretch of the trajectory.
///
/// The numeric values are part of the artifact format -- they are written into
/// `tamp_trajectories.json` as integers and read back by the collision stages,
/// the refinement and the executors. Values 0-4 are frozen: archived artifacts
/// under `artifacts/runs/*` must keep loading. New phases append.
///
/// 5-7 describe a PROCESS task (a weld pass): the tool is taken to the seam
/// start, switched on, run along the seam at the process speed, switched off,
/// and retreated. Nothing is grasped, so there is no gripper actuation and no
/// object -- the commutes deliberately reuse `ToPick`/`ToHome` so that the
/// ADR-0008 yield refinement, which locates a parking pose by scanning for those
/// two phases, works on a process task without knowing what one is.
enum class Phase : std::uint8_t
{
  ToPick = 0,      ///< home -> pre-grasp -> grasp, hands empty
  GripClose = 1,   ///< dwell at the grasp pose, arm frozen
  Carrying = 2,    ///< grasp -> retreat -> pre-place -> place, object attached
  GripOpen = 3,    ///< dwell at the place pose, arm frozen
  ToHome = 4,      ///< place -> retreat -> home, hands empty
  ProcessOn = 5,   ///< dwell at the seam start, arm frozen, the arc is struck
  Processing = 6,  ///< seam start -> seam end, at the process speed
  ProcessOff = 7,  ///< dwell at the seam end, arm frozen, the arc is out
};

/// Where the task's own object is, at a given sample.
///
/// This matters more than it looks. ADR-0003 excludes object `i` from its own
/// environment `E_i` and attaches it to the gripper instead -- but the object does
/// not stop existing while the robot flies out to fetch it. It sits at its SPAWN
/// pose during the approach, rides the gripper while carried, and sits at its
/// PLACE pose while the robot returns home. A plain "carrying: yes/no" flag makes
/// the object disappear for the approach and the return, and the collision stage
/// would then be free to route the OTHER robot straight through it.
///
/// The two dwells are `Attached`: the object is co-located with the gripper at the
/// grasp (resp. place) pose, so attaching it there is both correct and the same
/// geometry as leaving it static would give.
enum class ObjectState : std::uint8_t
{
  AtSpawn = 0,   ///< static, at the object's spawn pose  (phase ToPick)
  Attached = 1,  ///< rigidly attached to the gripper      (GripClose/Carrying/GripOpen)
  AtPlace = 2,   ///< static, at the task's place pose     (phase ToHome)
};

/// The object state implied by a phase. One rule, one place.
constexpr ObjectState objectStateFor(Phase p)
{
  switch (p) {
    case Phase::ToPick:
      return ObjectState::AtSpawn;
    case Phase::GripClose:
    case Phase::Carrying:
    case Phase::GripOpen:
      return ObjectState::Attached;
    case Phase::ToHome:
      return ObjectState::AtPlace;
    // A process task carries nothing. `AtSpawn` is the neutral answer here, and
    // the generator additionally flattens `object_state` over a whole process
    // trajectory so that the commutes (which reuse ToPick/ToHome) do not fake an
    // object moving from a spawn pose to a place pose that do not exist.
    case Phase::ProcessOn:
    case Phase::Processing:
    case Phase::ProcessOff:
      return ObjectState::AtSpawn;
  }
  return ObjectState::AtSpawn;
}

/// A contiguous, time-parameterised piece of a task, with its phase.
/// `waypoints[0].time_from_start` must be 0: segments are stitched by the
/// concatenator, which supplies the global time offset.
struct Segment
{
  Phase phase{Phase::ToPick};
  /// Whether the task's object may touch its SUPPORT SURFACE during this segment.
  /// True only for the stretches where the object is physically resting on it:
  /// the pick descent, the GripClose dwell and the lift, and the place descent,
  /// the GripOpen dwell and the retreat -- the scope MoveIt's own pick/place gives
  /// `support_surface_name`. Never for a free-space transfer, and it concerns
  /// that one surface only; every other obstacle is checked in full.
  ///
  /// A weld uses the same flag for its PROCESS allowance (the torch's process links
  /// against the parts the weld touches): the descent onto the seam, the strike, the
  /// traverse, the cut and the retreat -- and only when the scene grants one. What the
  /// flag allows is the generator's business (`contactAcm`); here it is one bit.
  bool support_contact{false};
  std::vector<TimedWaypoint> waypoints;
};

/// The uniform-Δt result: `num_samples` rows of `num_joints` positions.
struct ResampledTrajectory
{
  double delta_t{0.0};
  std::size_t num_joints{0};
  std::size_t num_samples{0};                 ///< K_ri, in slots
  std::vector<double> positions;              ///< row-major, num_samples x num_joints
  std::vector<Phase> phase;                   ///< per sample
  std::vector<ObjectState> object_state;      ///< per sample: where the task's object is
  /// per sample: taken from a `support_contact` segment (1) or not (0). Same
  /// knot rule as `phase`, so the two always agree on where a segment starts.
  std::vector<std::uint8_t> support_contact;

  /// Largest per-joint change between consecutive samples. THE number that says
  /// whether Δt is fine enough: if a joint can swing further than the smallest
  /// feature in the scene between two samples, the collision check can step
  /// straight over a collision and the schedule is unsound.
  double max_joint_step{0.0};
  std::size_t max_step_joint{0};              ///< which joint hit that maximum
  bool used_velocities{false};                ///< cubic Hermite (true) or linear (false)

  const double * sample(std::size_t k) const { return &positions[k * num_joints]; }
};

/// Splice a dwell (arm frozen) onto the end of a segment list.
///
/// `slots` slots at the segment's final configuration, with zero velocity. This
/// is how gripper actuation enters the model: real time passes, the arm does not
/// move, and the slots are counted in K_ri.
Segment makeDwell(const std::vector<double> & configuration, Phase phase, int slots,
                  double delta_t);

/// Concatenate segments onto one timeline and resample at uniform `delta_t`.
///
/// Interpolation is cubic Hermite when velocities are present (exact at the
/// knots, C1 between them, and it honours the velocity profile TOTG computed);
/// linear otherwise, which is reported via `used_velocities` rather than hidden.
///
/// The first sample is the first waypoint and the last sample is the LAST
/// waypoint exactly -- not an interpolation near it. The endpoints are the home
/// configuration, and ADR-0004 requires a robot to be exactly at home when idle,
/// so an endpoint that is merely close is not good enough.
///
/// Throws std::invalid_argument on an empty segment list, inconsistent joint
/// counts, or non-monotonic times.
ResampledTrajectory resampleUniform(const std::vector<Segment> & segments, double delta_t);

/// Human-readable name, for logs and the artifact manifest.
std::string phaseName(Phase p);

}  // namespace multi_robot_cell_tamp

#endif  // MULTI_ROBOT_CELL_TAMP__RESAMPLE_HPP_
