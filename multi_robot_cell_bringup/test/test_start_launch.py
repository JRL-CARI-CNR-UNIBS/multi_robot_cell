import time
import pytest
import unittest

import launch
import launch_testing
import launch_testing.actions

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


@pytest.mark.launch_test
def generate_test_description():
    start_launch = PathJoinSubstitution([
        FindPackageShare("multi_robot_cell_bringup"),
        "launch",
        "start.launch.py",
    ])

    launch_start = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(start_launch),
        launch_arguments={
            "fake": "true",
            "rviz": "false",
        }.items(),
    )

    return launch.LaunchDescription([
        launch_start,
        launch_testing.actions.ReadyToTest(),
    ])


class TestStartLaunch(unittest.TestCase):

    def test_does_not_crash_immediately(self):
        time.sleep(15)


# @launch_testing.post_shutdown_test()
# class TestStartLaunchExitCodes(unittest.TestCase):

#     def test_exit_codes(self, proc_info):
#         launch_testing.asserts.assertExitCodes(proc_info)