#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2024 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

__author__ = "Pedro Roque, Jaeyoung Lim"
__contact__ = "padr@kth.se, jalim@ethz.ch"

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Transform, Twist
from nav_msgs.msg import Odometry, Path
from px4_msgs.msg import (
    ActuatorMotors,
    OffboardControlMode,
    VehicleAngularVelocity,
    VehicleAttitude,
    VehicleLocalPosition,
    VehicleRatesSetpoint,
    VehicleStatus,
    VehicleThrustSetpoint,
    VehicleTorqueSetpoint,
)
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from visualization_msgs.msg import Marker

from mpc_msgs.srv import SetPose

DATA_VALIDITY_STREAM = 0.5  # seconds, threshold for (pos,att,vel) messages
DATA_VALIDITY_STATUS = 2.0  # seconds, threshold for status message


class SpacecraftMPC(Node):

    def __init__(self):
        super().__init__("spacecraft_mpc")

        # Get mode; rate, wrench, direct_allocation
        self.mode = self.declare_parameter(
            "mode",
            "wrench",
        ).value
        self.sitl = True

        # Get setpoint from rviz (true/false)
        self.setpoint_from_rviz = self.declare_parameter(
            "setpoint_from_rviz",
            False,
        ).value

        # Select target mode for navigation (setpoint or trajectory)
        # NOTE: setpoint not tested
        self.target_mode = self.declare_parameter(
            "target_mode",
            "trajectory",
        ).value
        if self.target_mode not in ["setpoint", "trajectory"]:
            raise ValueError(
                f"Invalid target_mode: {self.target_mode}. Must be "
                "'setpoint' or 'trajectory'."
            )
        if self.setpoint_from_rviz and self.target_mode == "trajectory":
            self.get_logger().warn(
                (
                    "Trajectory target mode is not compatible with "
                    "setpoint_from_rviz. Switching to setpoint target mode."
                )
            )
            self.target_mode = "setpoint"

        # QoS profiles
        qos_profile_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0,
        )

        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0,
        )

        # Setup publishers and subscribers
        self.set_publishers_subscribers(qos_profile_pub, qos_profile_sub)

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arm_state = 0

        # Create Spacecraft and controller objects
        if self.mode == "rate":
            from px4_mpc.controllers.spacecraft_rate_mpc import SpacecraftRateMPC
            from px4_mpc.models.spacecraft_rate_model import SpacecraftRateModel

            self.model = SpacecraftRateModel()
            self.mpc = SpacecraftRateMPC(self.model)
        elif self.mode == "wrench":
            from px4_mpc.controllers.spacecraft_wrench_mpc import SpacecraftWrenchMPC
            from px4_mpc.models.spacecraft_wrench_model import SpacecraftWrenchModel

            self.model = SpacecraftWrenchModel()
            self.mpc = SpacecraftWrenchMPC(self.model)
        elif self.mode == "direct_allocation":
            from px4_mpc.controllers.spacecraft_direct_allocation_mpc import (
                SpacecraftDirectAllocationMPC,
            )
            from px4_mpc.models.spacecraft_direct_allocation_model import (
                SpacecraftDirectAllocationModel,
            )

            self.model = SpacecraftDirectAllocationModel()
            self.mpc = SpacecraftDirectAllocationMPC(self.model)

        # State variables
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_angular_velocity = np.array([0.0, 0.0, 0.0])

        # Setpoint variables
        # NOTE: setpoint_velocity is currently ignored (see upstream repo)
        self.setpoint_position = np.array([0.0, 0.0, 0.0])
        self.setpoint_velocity = np.array([0.0, 0.0, 0.0])
        self.setpoint_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.setpoint_omega = np.array([0.0, 0.0, 0.0])
        self.setpoint_ok = False

        # Trajectory variables
        # rows: data, columns: time steps (same convention as in mpc)
        self.trajectory_position = np.zeros((3, self.mpc.N + 1))
        self.trajectory_velocity = np.zeros((3, self.mpc.N + 1))
        self.trajectory_attitude = np.zeros((4, self.mpc.N + 1))
        self.trajectory_attitude[0, :] = 1.0  # initialize with identity quaternions
        self.trajectory_omega = np.zeros((3, self.mpc.N + 1))
        self.trajectory_ok = False

        # Set initial timestamps
        self.vehicle_local_position_timestamp = -np.inf
        self.vehicle_local_velocity_timestamp = -np.inf
        self.vehicle_attitude_timestamp = -np.inf
        self.vehicle_angular_velocity_timestamp = -np.inf
        self.vehicle_status_timestamp = -np.inf

    def set_publishers_subscribers(self, qos_profile_pub, qos_profile_sub):
        """
        Create subscribers to PX4 outputs, subscribers to setpoint or trajectory inputs,
        and publisher for output commands.
        """
        # Subscribe to both multiple status topics using the same callback. Depending on
        # the PX4 version, the right one will be used (but not multiple)
        self.status_sub = self.create_subscription(
            VehicleStatus,
            "fmu/out/vehicle_status",
            self.vehicle_status_callback,
            qos_profile_sub,
        )
        self.status_sub_v1 = self.create_subscription(
            VehicleStatus,
            "fmu/out/vehicle_status_v1",
            self.vehicle_status_callback,
            qos_profile_sub,
        )
        self.status_sub_v2 = self.create_subscription(
            VehicleStatus,
            "fmu/out/vehicle_status_v2",
            self.vehicle_status_callback,
            qos_profile_sub,
        )
        self.status_sub_v4 = self.create_subscription(
            VehicleStatus,
            "fmu/out/vehicle_status_v4",
            self.vehicle_status_callback,
            qos_profile_sub,
        )

        # Subscribe to PX4 state topics
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,  # NOTE: also includes velocity
            "fmu/out/vehicle_local_position",
            self.vehicle_local_position_callback,
            qos_profile_sub,
        )
        self.local_position_sub_v1 = self.create_subscription(
            VehicleLocalPosition,
            "fmu/out/vehicle_local_position_v1",
            self.vehicle_local_position_callback,
            qos_profile_sub,
        )
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            "fmu/out/vehicle_attitude",
            self.vehicle_attitude_callback,
            qos_profile_sub,
        )
        self.angular_vel_sub = self.create_subscription(
            VehicleAngularVelocity,
            "fmu/out/vehicle_angular_velocity",
            self.vehicle_angular_velocity_callback,
            qos_profile_sub,
        )

        # Subscribe to setpoint or trajectory topics
        if self.setpoint_from_rviz:
            self.set_pose_srv = self.create_service(
                SetPose,
                "set_pose",
                self.add_set_pos_callback,
            )
        else:
            if self.target_mode == "setpoint":
                self.setpoint_pose_sub = self.create_subscription(
                    Odometry,
                    "setpoint_pose",
                    self.get_setpoint_pose_callback,
                    0,
                )
                # NOTE [PRAS]: I changed this because the test_setpoint.py doesnt seem
                # to receive the namespace args and publish into topic without the
                # namespace [# CHECKME: is this still an issue?]
                # self.setpoint_pose_sub = self.create_subscription(
                #     Odometry,
                #     "/setpoint_pose",
                #     self.get_setpoint_pose_callback,
                #     0,
                # )

            elif self.target_mode == "trajectory":
                self.trajectory_sub = self.create_subscription(
                    MultiDOFJointTrajectory,
                    "reference_trajectory",
                    self.get_reference_trajectory_callback,
                    0,
                )

        # Create publishers
        self.publisher_offboard_mode = self.create_publisher(
            OffboardControlMode,
            "fmu/in/offboard_control_mode",
            qos_profile_pub,
        )
        self.publisher_direct_actuator = self.create_publisher(
            ActuatorMotors,
            "fmu/in/actuator_motors",
            qos_profile_pub,
        )
        self.publisher_rates_setpoint = self.create_publisher(
            VehicleRatesSetpoint,
            "fmu/in/vehicle_rates_setpoint",
            qos_profile_pub,
        )
        self.publisher_thrust_setpoint = self.create_publisher(
            VehicleThrustSetpoint,
            "fmu/in/vehicle_thrust_setpoint",
            qos_profile_pub,
        )
        self.publisher_torque_setpoint = self.create_publisher(
            VehicleTorqueSetpoint,
            "fmu/in/vehicle_torque_setpoint",
            qos_profile_pub,
        )

        # Create publisher for rviz visualization
        self.predicted_path_pub = self.create_publisher(
            Path,
            "px4_mpc/predicted_path",
            10,
        )
        self.reference_pub = self.create_publisher(
            Marker,
            "px4_mpc/reference_setpoint",  # TODO: rename to px4_mpc/reference_setpoint
            10,
        )

        # Create odometry publisher for SITL visualization in rivz
        if self.sitl:
            self.odom_pub = self.create_publisher(
                Odometry,
                "odom",  # TODO: rename to px4_mpc/odometry
                qos_profile_pub,
            )
        return

    def vehicle_attitude_callback(self, msg):
        # NED-> ENU transformation
        # Receives quaternion in NED frame as (qw, qx, qy, qz)
        self.vehicle_attitude_timestamp = self.get_clock().now().nanoseconds / 1e9
        q_enu = (
            1
            / np.sqrt(2)
            * np.array(
                [
                    msg.q[0] + msg.q[3],
                    msg.q[1] + msg.q[2],
                    msg.q[1] - msg.q[2],
                    msg.q[0] - msg.q[3],
                ]
            )
        )
        q_enu /= np.linalg.norm(q_enu)
        self.vehicle_attitude = q_enu.astype(float)

    def vehicle_local_position_callback(self, msg):
        # NED-> ENU transformation
        self.vehicle_local_position_timestamp = self.get_clock().now().nanoseconds / 1e9
        self.vehicle_local_position[0] = msg.y
        self.vehicle_local_position[1] = msg.x
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vy
        self.vehicle_local_velocity[1] = msg.vx
        self.vehicle_local_velocity[2] = -msg.vz

    def vehicle_angular_velocity_callback(self, msg):
        # NED-> ENU transformation
        self.vehicle_angular_velocity_timestamp = (
            self.get_clock().now().nanoseconds / 1e9
        )
        self.vehicle_angular_velocity[0] = msg.xyz[0]
        self.vehicle_angular_velocity[1] = -msg.xyz[1]
        self.vehicle_angular_velocity[2] = -msg.xyz[2]

    def vehicle_status_callback(self, msg):
        # self.get_logger().info("Vehicle status received!")
        # print("NAV_STATUS: ", msg.nav_state)
        # print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.vehicle_status_timestamp = self.get_clock().now().nanoseconds / 1e9
        self.nav_state = msg.nav_state
        self.arm_state = msg.arming_state

    def publish_reference(self, pub, reference):
        msg = Marker()
        msg.action = Marker.ADD
        msg.header.frame_id = "map"
        # msg.header.stamp = self.get_clock().now().nanoseconds / 1000
        msg.ns = "arrow"
        msg.id = 1
        msg.type = Marker.SPHERE
        msg.scale.x = 0.5
        msg.scale.y = 0.5
        msg.scale.z = 0.5
        msg.color.r = 1.0
        msg.color.g = 0.0
        msg.color.b = 0.0
        msg.color.a = 1.0
        msg.pose.position.x = reference[0]
        msg.pose.position.y = reference[1]
        msg.pose.position.z = reference[2]
        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0

        pub.publish(msg)

    def publish_rate_setpoint(self, u_pred):
        F_cmd = u_pred[0, 0:3]
        w_cmd = u_pred[0, 3:6]

        # The PX4 uses normalized force input. Scaling with respect to maximum force.
        F_scaling = 1 / (2 * 1.5)
        F_cmd *= F_scaling

        rates_setpoint_msg = VehicleRatesSetpoint()
        rates_setpoint_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        rates_setpoint_msg.roll = float(w_cmd[0])
        rates_setpoint_msg.pitch = -float(w_cmd[1])
        rates_setpoint_msg.yaw = -float(w_cmd[2])
        rates_setpoint_msg.thrust_body[0] = float(F_cmd[0])
        rates_setpoint_msg.thrust_body[1] = -float(F_cmd[1])
        rates_setpoint_msg.thrust_body[2] = -float(F_cmd[2])
        self.publisher_rates_setpoint.publish(rates_setpoint_msg)

    def publish_wrench_setpoint(self, u_pred):
        # u_pred is [Fx, Fy, Tz]] in FLU frame

        # The PX4 uses normalized wrench input. Scaling w.r.t. maximum force and torque.
        F_scaling = 1 / (2 * 1.5)
        T_scaling = 1 / (4 * 0.12 * 1.5)
        u_pred[0, 0] *= F_scaling
        u_pred[0, 1] *= F_scaling
        u_pred[0, 2] *= T_scaling

        thrust_outputs_msg = VehicleThrustSetpoint()
        thrust_outputs_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        torque_outputs_msg = VehicleTorqueSetpoint()
        torque_outputs_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        thrust_outputs_msg.xyz = [u_pred[0, 0], -u_pred[0, 1], -0.0]
        torque_outputs_msg.xyz = [0.0, -0.0, -u_pred[0, 2]]

        self.publisher_thrust_setpoint.publish(thrust_outputs_msg)
        self.publisher_torque_setpoint.publish(torque_outputs_msg)

    def publish_direct_actuator_setpoint(self, u_pred):
        actuator_outputs_msg = ActuatorMotors()
        actuator_outputs_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        # Normalize thrust values w.r.t. max thrust
        thrust = u_pred[0, :] / self.model.max_thrust

        # Generate actuator outputs dynamically
        thrust_command = []
        for t in thrust:
            thrust_command.extend([max(t, 0.0), max(-t, 0.0)])
        thrust_command = np.clip(np.array(thrust_command, dtype=np.float32), 0.0, 1.0)

        actuator_outputs_msg.control[: len(thrust_command)] = thrust_command
        self.publisher_direct_actuator.publish(actuator_outputs_msg)

    def publish_sitl_odometry(self):
        msg = Odometry()
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = self.vehicle_local_position[0]
        msg.pose.pose.position.y = self.vehicle_local_position[1]
        msg.pose.pose.position.z = self.vehicle_local_position[2]
        msg.pose.pose.orientation.w = self.vehicle_attitude[0]
        msg.pose.pose.orientation.x = self.vehicle_attitude[1]
        msg.pose.pose.orientation.y = self.vehicle_attitude[2]
        msg.pose.pose.orientation.z = self.vehicle_attitude[3]
        msg.twist.twist.linear.x = self.vehicle_local_velocity[0]
        msg.twist.twist.linear.y = self.vehicle_local_velocity[1]
        msg.twist.twist.linear.z = self.vehicle_local_velocity[2]
        msg.twist.twist.angular.x = self.vehicle_angular_velocity[0]
        msg.twist.twist.angular.y = self.vehicle_angular_velocity[1]
        msg.twist.twist.angular.z = self.vehicle_angular_velocity[2]
        self.odom_pub.publish(msg)

    def check_data_validity(self):
        current_time = self.get_clock().now().nanoseconds / 1e9

        ret_val = True

        # Check if the data is valid based on the timestamps
        if current_time - self.vehicle_attitude_timestamp > DATA_VALIDITY_STREAM:
            self.get_logger().warn(
                "Vehicle attitude data is too old. Skipping offboard control..."
            )
            self.get_logger().warn(
                (
                    f"Current time: {current_time}, attitude "
                    f"timestamp: {self.vehicle_attitude_timestamp}"
                )
            )
            ret_val = False

        if current_time - self.vehicle_local_position_timestamp > DATA_VALIDITY_STREAM:
            self.get_logger().warn(
                "Vehicle position data is too old. Skipping offboard control..."
            )
            ret_val = False

        if (
            current_time - self.vehicle_angular_velocity_timestamp
            > DATA_VALIDITY_STREAM
        ):
            self.get_logger().warn(
                "Vehicle angular velocity data is too old. Skipping offboard control..."
            )
            ret_val = False

        if current_time - self.vehicle_status_timestamp > DATA_VALIDITY_STATUS:
            self.get_logger().warn(
                "Vehicle status data is too old. Skipping offboard control..."
            )
            ret_val = False

        return ret_val

    def cmdloop_callback(self):

        # Publish odometry for SITL
        if self.sitl:
            self.publish_sitl_odometry()

        # Check data validity
        if not self.check_data_validity():
            return

        # Publish offboard control modes
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        offboard_msg.direct_actuator = False
        if self.mode == "rate":
            offboard_msg.body_rate = True
        elif self.mode == "direct_allocation":
            offboard_msg.direct_actuator = True
        elif self.mode == "wrench":
            offboard_msg.thrust_and_torque = True
        self.publisher_offboard_mode.publish(offboard_msg)

        # Set state and references for each MPC mode
        if self.mode == "rate":
            x0 = np.array(
                [
                    self.vehicle_local_position[0],
                    self.vehicle_local_position[1],
                    self.vehicle_local_position[2],
                    self.vehicle_local_velocity[0],
                    self.vehicle_local_velocity[1],
                    self.vehicle_local_velocity[2],
                    self.vehicle_attitude[0],
                    self.vehicle_attitude[1],
                    self.vehicle_attitude[2],
                    self.vehicle_attitude[3],
                ]
            ).reshape(10, 1)
            ref = np.concatenate(
                (
                    self.setpoint_position,  # position
                    np.zeros(3),  # velocity
                    self.setpoint_attitude,  # attitude
                    np.zeros(6),
                ),
                axis=0,
            )  # inputs reference (F, w)
            ref = np.repeat(ref.reshape((-1, 1)), self.mpc.N + 1, axis=1)
        elif self.mode == "wrench":
            x0 = np.array(
                [
                    self.vehicle_local_position[0],
                    self.vehicle_local_position[1],
                    self.vehicle_local_position[2],
                    self.vehicle_local_velocity[0],
                    self.vehicle_local_velocity[1],
                    self.vehicle_local_velocity[2],
                    self.vehicle_attitude[0],
                    self.vehicle_attitude[1],
                    self.vehicle_attitude[2],
                    self.vehicle_attitude[3],
                    self.vehicle_angular_velocity[0],
                    self.vehicle_angular_velocity[1],
                    self.vehicle_angular_velocity[2],
                ]
            ).reshape(13, 1)

            # Build reference depending on target_mode (setpoint vs trajectory)
            if self.target_mode == "setpoint" and self.setpoint_ok:
                ref = np.concatenate(
                    (
                        self.setpoint_position,  # position
                        self.setpoint_velocity,  # velocity
                        self.setpoint_attitude,  # attitude
                        self.setpoint_omega,  # angular velocity
                        np.zeros(3),  # control input
                    ),
                    axis=0,
                )
                ref = np.repeat(ref.reshape((-1, 1)), self.mpc.N + 1, axis=1)

            elif self.target_mode == "trajectory" and self.trajectory_ok:
                ref = np.zeros((16, self.mpc.N + 1))  # initialize reference array
                ref[0:3, :] = self.trajectory_position
                ref[3:6, :] = self.trajectory_velocity
                ref[6:10, :] = self.trajectory_attitude
                ref[10:13, :] = self.trajectory_omega
                # input reference (rows 13:16) are left set to 0

            else:
                # If the reference is not ok, MPC solution is skipped
                self.get_logger().warn(
                    "No valid reference available yet "
                    f"(target_mode={self.target_mode}, setpoint_ok={self.setpoint_ok}, "
                    f"trajectory_ok={self.trajectory_ok}). Skipping offboard control.",
                    throttle_duration_sec=1.0,
                )
                return

        elif self.mode == "direct_allocation":
            x0 = np.array(
                [
                    self.vehicle_local_position[0],
                    self.vehicle_local_position[1],
                    self.vehicle_local_position[2],
                    self.vehicle_local_velocity[0],
                    self.vehicle_local_velocity[1],
                    self.vehicle_local_velocity[2],
                    self.vehicle_attitude[0],
                    self.vehicle_attitude[1],
                    self.vehicle_attitude[2],
                    self.vehicle_attitude[3],
                    self.vehicle_angular_velocity[0],
                    self.vehicle_angular_velocity[1],
                    self.vehicle_angular_velocity[2],
                ]
            ).reshape(13, 1)
            ref = np.concatenate(
                (
                    self.setpoint_position,  # position
                    np.zeros(3),  # velocity
                    self.setpoint_attitude,  # attitude
                    np.zeros(3),  # angular velocity
                    np.zeros(4),  # inputs reference (u1, ..., u4) for 2D platform
                ),
                axis=0,
            )
            ref = np.repeat(ref.reshape((-1, 1)), self.mpc.N + 1, axis=1)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # Solve MPC
        u_pred, x_pred = self.mpc.solve(x0, ref=ref)

        # Colect data
        idx = 0
        predicted_path_msg = Path()
        for predicted_state in x_pred:
            idx = idx + u_pred
            # Publish time history of the vehicle path
            predicted_pose_msg = self.vector2PoseMsg(
                "map", predicted_state[0:3], self.setpoint_attitude
            )
            predicted_path_msg.header = predicted_pose_msg.header
            predicted_path_msg.poses.append(predicted_pose_msg)
        self.predicted_path_pub.publish(predicted_path_msg)
        self.publish_reference(self.reference_pub, self.setpoint_position)

        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if self.mode == "rate":
                self.publish_rate_setpoint(u_pred)
            elif (
                self.mode == "direct_allocation"
                or self.mode == "direct_allocation_trajectory"
            ):
                self.publish_direct_actuator_setpoint(u_pred)
            elif (
                self.mode == "wrench"
            ):  # and self.arm_state == VehicleStatus.ARMING_STATE_ARMED:
                # if self.setpoint_ok:
                self.publish_wrench_setpoint(u_pred)
                # self.get_logger().info("Publishing Wrench Setpoint")

    def add_set_pos_callback(self, request, response):
        self.setpoint_position[0] = request.pose.position.x
        self.setpoint_position[1] = request.pose.position.y
        self.setpoint_position[2] = request.pose.position.z
        self.setpoint_attitude[0] = request.pose.orientation.w
        self.setpoint_attitude[1] = request.pose.orientation.x
        self.setpoint_attitude[2] = request.pose.orientation.y
        self.setpoint_attitude[3] = request.pose.orientation.z
        return response

    def get_setpoint_pose_callback(self, msg):
        self.setpoint_position[0] = msg.pose.pose.position.x
        self.setpoint_position[1] = msg.pose.pose.position.y
        self.setpoint_position[2] = msg.pose.pose.position.z
        self.setpoint_attitude[0] = msg.pose.pose.orientation.w
        self.setpoint_attitude[1] = msg.pose.pose.orientation.x
        self.setpoint_attitude[2] = msg.pose.pose.orientation.y
        self.setpoint_attitude[3] = msg.pose.pose.orientation.z

        self.setpoint_omega[0] = msg.twist.twist.angular.x
        self.setpoint_omega[1] = msg.twist.twist.angular.y
        self.setpoint_omega[2] = msg.twist.twist.angular.z

        self.setpoint_ok = True

    def get_reference_trajectory_callback(self, msg: MultiDOFJointTrajectory) -> None:
        """
        Extract reference trajectory from the received message. The trajectory is
        composed by a sequence of N+1 MultiDOFJointTrajectoryPoint objects.

        Args:
            msg(MultiDOFJointTrajectory): message containing the reference trajectory.
                Each point in the trajectory is a MultiDOFJointTrajectoryPoint.

        Returns:
            None

        Raises:
            ValueError: if the number of points in the trajectory does not match
                the expected number (N+1)
            ValueError: if any point is missing transforms or velocities.
        """
        # Debug setting (for development)
        DEBUG = True

        # Validate input
        n_points = len(msg.points)
        if n_points != self.mpc.N + 1:
            self.get_logger().error(
                f"Received trajectory with {n_points} points, expected "
                f"{self.mpc.N + 1}. Ignoring message."
            )
            return

        # Iterate over trajectory points
        for i in range(n_points):

            # Validate point
            point: MultiDOFJointTrajectoryPoint = msg.points[i]
            if not point.transforms or not point.velocities:
                self.get_logger().error(
                    f"Trajectory point {i} is missing transforms or velocities."
                )
                return

            # Extract data from point
            pose: Transform = point.transforms[0]
            twist: Twist = point.velocities[0]

            self.trajectory_position[0, i] = pose.translation.x
            self.trajectory_position[1, i] = pose.translation.y
            self.trajectory_position[2, i] = pose.translation.z

            self.trajectory_attitude[0, i] = pose.rotation.w
            self.trajectory_attitude[1, i] = pose.rotation.x
            self.trajectory_attitude[2, i] = pose.rotation.y
            self.trajectory_attitude[3, i] = pose.rotation.z

            self.trajectory_velocity[0, i] = twist.linear.x
            self.trajectory_velocity[1, i] = twist.linear.y
            self.trajectory_velocity[2, i] = twist.linear.z

            self.trajectory_omega[0, i] = twist.angular.x
            self.trajectory_omega[1, i] = twist.angular.y
            self.trajectory_omega[2, i] = twist.angular.z

        self.trajectory_ok = True

        # Print debug info
        if DEBUG is True:
            self.get_logger().info(
                f"Trajectory received: pos[:, 0]={self.trajectory_position[:, 0]}, "
                f"att[:, 0]={self.trajectory_attitude[:, 0]}",
                throttle_duration_sec=1.0,
            )

    def vector2PoseMsg(self, frame_id, position, attitude):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = frame_id
        pose_msg.pose.orientation.w = attitude[0]
        pose_msg.pose.orientation.x = attitude[1]
        pose_msg.pose.orientation.y = attitude[2]
        pose_msg.pose.orientation.z = attitude[3]
        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])
        return pose_msg

    def forward_propagate_attitude(self, q0, omega_z, dt, N):
        """
        Propagate yaw-only attitude over N steps.

        Returns (N+1) x 4 array of quaternions and (N+1) x 3 array of angular
        velocities. Angular rate is constant (no torque prediction in the planner
        horizon).
        """
        quats = np.zeros((N + 1, 4))
        ang_vels = np.zeros((N + 1, 3))
        quats[0] = q0
        ang_vels[0] = [0.0, 0.0, omega_z]

        for i in range(1, N + 1):
            dtheta = omega_z * dt
            w0, x0, y0, z0 = quats[i - 1]
            c, s = np.cos(dtheta / 2), np.sin(dtheta / 2)
            quats[i] = np.array(
                [
                    w0 * c - z0 * s,
                    x0 * c + y0 * s,
                    -x0 * s + y0 * c,
                    w0 * s + z0 * c,
                ]
            )
            quats[i] /= np.linalg.norm(quats[i])
            ang_vels[i] = [0.0, 0.0, omega_z]
            # quats[i] = [1.0, 0.0, 0.0, 0.0]
            # ang_vels[i] = [0.0, 0.0, 0.0]

        return quats, ang_vels


def main(args=None):
    rclpy.init(args=args)

    spacecraft_mpc = SpacecraftMPC()

    rclpy.spin(spacecraft_mpc)

    spacecraft_mpc.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
