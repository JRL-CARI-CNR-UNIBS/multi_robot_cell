#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String


class RobotDescriptionPublisher(Node):
    def __init__(self):
        super().__init__("robot_description_publisher")

        self.declare_parameter("robot_description", "")
        # By default, publish once (latched via TRANSIENT_LOCAL) to avoid
        # repeatedly reconfiguring joint_state_publisher_gui.
        self.declare_parameter("publish_period_s", 0.0)

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._pub = self.create_publisher(String, "/robot_description", qos)

        period = float(self.get_parameter("publish_period_s").value)
        self._publish()
        self._timer = None
        if period and period > 0.0:
            self._timer = self.create_timer(max(period, 0.1), self._publish)

    def _publish(self):
        urdf = str(self.get_parameter("robot_description").value)
        if not urdf:
            self.get_logger().warn("Parameter 'robot_description' is empty; not publishing.")
            return
        msg = String()
        msg.data = urdf
        self._pub.publish(msg)


def main():
    rclpy.init()
    node = RobotDescriptionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
