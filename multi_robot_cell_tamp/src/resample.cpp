#include "multi_robot_cell_tamp/resample.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace multi_robot_cell_tamp
{
namespace
{

/// Cubic Hermite on one joint. Exact at both knots, and it reproduces the
/// velocity profile TOTG computed rather than cutting the corner a straight line
/// would.
double hermite(double p0, double v0, double p1, double v1, double dt, double u)
{
  const double u2 = u * u;
  const double u3 = u2 * u;
  const double h00 = 2 * u3 - 3 * u2 + 1;
  const double h10 = u3 - 2 * u2 + u;
  const double h01 = -2 * u3 + 3 * u2;
  const double h11 = u3 - u2;
  return h00 * p0 + h10 * dt * v0 + h01 * p1 + h11 * dt * v1;
}

}  // namespace

std::string phaseName(Phase p)
{
  switch (p) {
    case Phase::ToPick:    return "to_pick";
    case Phase::GripClose: return "grip_close";
    case Phase::Carrying:  return "carrying";
    case Phase::GripOpen:  return "grip_open";
    case Phase::ToHome:    return "to_home";
  }
  return "unknown";
}

Segment makeDwell(const std::vector<double> & configuration, Phase phase, int slots,
                  double delta_t)
{
  if (slots <= 0) {
    throw std::invalid_argument("dwell must last at least one slot");
  }

  // Two waypoints, zero velocity, spanning exactly `slots * delta_t`. Any sample
  // taken inside this span interpolates between two identical configurations, so
  // it reproduces that configuration exactly -- the arm is frozen, by
  // construction rather than by rounding.
  Segment seg;
  seg.phase = phase;
  seg.waypoints.push_back(
    TimedWaypoint{0.0, configuration, std::vector<double>(configuration.size(), 0.0)});
  seg.waypoints.push_back(
    TimedWaypoint{slots * delta_t, configuration,
                  std::vector<double>(configuration.size(), 0.0)});
  return seg;
}

ResampledTrajectory resampleUniform(const std::vector<Segment> & segments, double delta_t)
{
  if (delta_t <= 0.0) {
    throw std::invalid_argument("delta_t must be positive");
  }
  if (segments.empty()) {
    throw std::invalid_argument("no segments to resample");
  }

  // ---- Flatten the segments onto one global timeline. ---------------------- #
  // Each segment's times are relative to its own start; the offset is the total
  // duration of everything before it.
  //
  // A segment boundary produces TWO knots at the same instant, and both are kept.
  // It is tempting to drop the duplicate -- the positions agree, after all -- but
  // the VELOCITIES do not: a ramp arrives at the grasp pose with a non-zero
  // velocity, and the dwell that follows starts at rest. That velocity
  // discontinuity is real, and it is the whole content of "the arm stops here".
  //
  // Dropping the incoming knot throws the discontinuity away, and the Hermite
  // interpolant then carries the ramp's velocity straight through the dwell:
  // the arm bulges ~0.15 rad past the grasp pose while the gripper is supposedly
  // closed on the object. (A linear interpolant would have hidden this; the
  // cubic exposed it, which is one reason to prefer it.) So: keep both knots,
  // and let the resampling loop below skip the zero-span interval between them.
  std::vector<TimedWaypoint> path;
  std::vector<Phase> knot_phase;
  double offset = 0.0;
  std::size_t num_joints = 0;
  bool have_velocities = true;

  for (const auto & seg : segments) {
    if (seg.waypoints.empty()) {
      throw std::invalid_argument("segment has no waypoints");
    }
    if (num_joints == 0) {
      num_joints = seg.waypoints.front().positions.size();
      if (num_joints == 0) {
        throw std::invalid_argument("waypoint has no joints");
      }
    }

    for (const auto & wp : seg.waypoints) {
      TimedWaypoint pt = wp;
      if (pt.positions.size() != num_joints) {
        throw std::invalid_argument("inconsistent joint count across waypoints");
      }
      if (pt.velocities.empty()) {
        have_velocities = false;
        pt.velocities.assign(num_joints, 0.0);
      } else if (pt.velocities.size() != num_joints) {
        throw std::invalid_argument("velocity/position size mismatch");
      }

      pt.time_from_start += offset;
      if (!path.empty() && pt.time_from_start + 1e-9 < path.back().time_from_start) {
        throw std::invalid_argument("segment times are not monotonic");
      }
      path.push_back(pt);
      knot_phase.push_back(seg.phase);
    }
    offset += seg.waypoints.back().time_from_start;
  }

  if (path.size() < 2 || path.back().time_from_start <= 0.0) {
    throw std::invalid_argument("need at least two waypoints spanning positive time");
  }

  // ---- Resample at uniform Δt. -------------------------------------------- #
  const double total = path.back().time_from_start;
  const std::size_t num_samples = static_cast<std::size_t>(std::ceil(total / delta_t)) + 1;

  ResampledTrajectory out;
  out.delta_t = delta_t;
  out.num_joints = num_joints;
  out.num_samples = num_samples;
  out.used_velocities = have_velocities;
  out.positions.resize(num_samples * num_joints);
  out.phase.resize(num_samples);
  out.object_state.resize(num_samples);

  std::size_t seg_idx = 0;
  for (std::size_t k = 0; k < num_samples; ++k) {
    const double t = std::min(k * delta_t, total);

    // The final sample is the final waypoint EXACTLY. `ceil` means the last slot
    // usually lands past the end of the path, and clamping t would interpolate to
    // it anyway -- but only up to floating-point error. The home configuration
    // has to be exact: ADR-0004 has the robot parked AT home when idle, and the
    // home-clearance check is run against that exact configuration.
    if (k == num_samples - 1) {
      std::copy(path.back().positions.begin(), path.back().positions.end(),
                out.positions.begin() + k * num_joints);
      out.phase[k] = knot_phase.back();
      out.object_state[k] = objectStateFor(out.phase[k]);
      break;
    }

    // `<=`, not `<`. At a segment boundary two knots share an instant; advancing
    // past the earlier one means a sample landing exactly on the boundary is
    // interpolated forward from the STARTING segment -- with the starting
    // segment's velocity. That is what pins the arm still for the first slot of a
    // dwell instead of letting the previous ramp's velocity carry it onward. It
    // also steps cleanly over the zero-span interval between the twin knots.
    while (seg_idx + 2 < path.size() && path[seg_idx + 1].time_from_start <= t) {
      ++seg_idx;
    }
    const TimedWaypoint & a = path[seg_idx];
    const TimedWaypoint & b = path[seg_idx + 1];
    const double span = b.time_from_start - a.time_from_start;
    const double u = span > 0.0 ? std::clamp((t - a.time_from_start) / span, 0.0, 1.0) : 0.0;

    for (std::size_t j = 0; j < num_joints; ++j) {
      out.positions[k * num_joints + j] =
        have_velocities
          ? hermite(a.positions[j], a.velocities[j], b.positions[j], b.velocities[j], span, u)
          : (1.0 - u) * a.positions[j] + u * b.positions[j];
    }

    // A sample belongs to the phase of the interval it falls in. Take it from the
    // knot that STARTS the interval, so the instant the dwell begins is already
    // labelled as the dwell.
    out.phase[k] = knot_phase[seg_idx];
    out.object_state[k] = objectStateFor(out.phase[k]);
  }

  // ---- The Δt soundness metric. ------------------------------------------- #
  for (std::size_t k = 1; k < num_samples; ++k) {
    for (std::size_t j = 0; j < num_joints; ++j) {
      const double step =
        std::abs(out.positions[k * num_joints + j] - out.positions[(k - 1) * num_joints + j]);
      if (step > out.max_joint_step) {
        out.max_joint_step = step;
        out.max_step_joint = j;
      }
    }
  }

  return out;
}

}  // namespace multi_robot_cell_tamp
