"""Offline inter-robot collision stage — trajectory artifact -> SchedulingProblem.

Reads the trajectories from the previous stage, computes the inter-robot collision
matrices via FCL on the loaded robot model, reduces them to forbidden start-offset
sets, and writes the geometry-free problem the Python scheduler solves.

    ros2 launch multi_robot_cell_tamp generate_collisions.launch.py

Defaults read/write the persistent ``artifacts/`` dir (``artifacts/tamp_trajectories.json``
-> ``artifacts/fcl/tamp_problem.json``); override ``traj_file``/``out_file`` to change them.

Needs the robot model (for FK + collision geometry) but not the running cell.
"""

import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from cell_moveit import cell_of_task, description_and_semantic, scene_path  # noqa: E402

# Persistent (non-/tmp) output dir under the package SOURCE tree; realpath() resolves
# the --symlink-install symlink back to source. The FCL seam lives in artifacts/fcl/.
ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "artifacts"
)


def launch_setup(context):
    task_file = LaunchConfiguration("task_file").perform(context) or scene_path("nominal")

    # The collision node only needs the model's geometry — no controllers, no
    # planning pipeline — so the URDF/SRDF are loaded directly (of the scene's cell,
    # launch/cell_moveit.py). MoveItConfigsBuilder is avoided on purpose: its
    # to_moveit_configs() auto-loads the package's moveit_controllers.yaml, which is
    # fully commented out and makes it throw.
    robot_description, robot_description_semantic = description_and_semantic(
        cell_of_task(task_file), context)

    os.makedirs(os.path.dirname(LaunchConfiguration("out_file").perform(context)), exist_ok=True)

    return [
        Node(
            package="multi_robot_cell_tamp",
            executable="collision_generator",
            output="screen",
            parameters=[
                {"robot_description": robot_description},
                {"robot_description_semantic": robot_description_semantic},
                {
                    "traj_file": LaunchConfiguration("traj_file").perform(context),
                    "task_file": task_file,
                    "out_file": LaunchConfiguration("out_file").perform(context),
                },
            ],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "traj_file",
                default_value=os.path.join(ARTIFACTS_DIR, "tamp_trajectories.json"),
            ),
            DeclareLaunchArgument("task_file", default_value=""),
            DeclareLaunchArgument(
                "out_file",
                default_value=os.path.join(ARTIFACTS_DIR, "fcl", "tamp_problem.json"),
                description="FCL geometry-free seam (default artifacts/fcl/).",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
