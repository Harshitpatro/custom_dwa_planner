#!/usr/bin/env python3
"""
DWA Local Planner for ROS 2

This node implements a Dynamic Window Approach (DWA) local planner for navigation.
It subscribes to laser scan, odometry, and goal topics, and publishes velocity commands
based on cost-optimized trajectory predictions. Trajectories are visualized in RViz.
"""


import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import math
from typing import List, Tuple
from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import TransformException

class DWAConfig:
    """
    Configuration parameters for the DWA planner.
    """
    def __init__(self):
        # TurtleBot3 Burger specifications
        self.max_linear_vel = 0.22
        self.min_linear_vel = 0.0
        self.max_angular_vel = 2.84
        self.min_angular_vel = -2.84
        self.max_linear_accel = 1.0
        self.max_angular_accel = 2.0
        self.velocity_resolution = 0.01
        self.angular_resolution = 0.1
        self.dt = 0.1
        self.predict_time = 3.0
        self.goal_cost_gain = 1.0
        self.obstacle_cost_gain = 2.0
        self.velocity_cost_gain = 0.1
        self.robot_radius = 0.105
        self.goal_tolerance = 0.1  # Goal tolerance in meters
        self.min_obstacle_distance = 1.0  # Distance beyond which obstacle cost is zero

class DWALocalPlanner(Node):
    """
    DWA Local Planner ROS 2 Node.
    """
    def __init__(self):
        super().__init__('dwa_local_planner')
        self.config = DWAConfig()
        self.current_pose = None
        self.current_twist = None
        self.laser_data = None
        self.goal_pose = None
        self.goal_queue = []

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, qos)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, qos)
        self.rviz_goal_sub = self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_callback, qos)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', qos)
        self.trajectory_vis_pub = self.create_publisher(MarkerArray, '/dwa_trajectories', qos)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("DWA Local Planner initialized")

    def odom_callback(self, msg):
        """Callback for odometry updates."""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def laser_callback(self, msg):
        """Callback for laser scan data."""
        self.laser_data = msg

    def goal_callback(self, msg):
        """Callback for goal input. Queues multiple waypoints."""
        self.get_logger().info(f"Received goal: x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}")
        self.goal_queue.append(msg.pose)
        if self.goal_pose is None:
            self.goal_pose = self.goal_queue.pop(0)

    def control_loop(self):
        """
        Main control loop called by ROS 2 timer.
        Selects and executes the best trajectory using DWA.
        """
        if not self.is_ready():
            return

        if self.is_goal_reached():
            self.stop_robot()
            self.get_logger().info("Goal reached!")

            if self.goal_queue:
                self.goal_pose = self.goal_queue.pop(0)
                self.get_logger().info(f"Fetching next goal from queue: x={self.goal_pose.position.x:.2f}, y={self.goal_pose.position.y:.2f}")
            else:
                self.goal_pose = None  # Only clear if no goals left

            return


        best_cmd = self.dwa_planning()
        if best_cmd is not None:
            cmd_msg = Twist()
            cmd_msg.linear.x = best_cmd[0]
            cmd_msg.angular.z = best_cmd[1]
            self.cmd_vel_pub.publish(cmd_msg)
        else:
            self.stop_robot()
            self.get_logger().warn("No safe trajectory found - stopping robot")

    def is_ready(self):
        """Check if all required sensor inputs and goals are available."""
        return all([
            self.current_pose is not None,
            self.current_twist is not None,
            self.goal_pose is not None,
            self.laser_data is not None
        ])

    def is_goal_reached(self):
        """Return True if current pose is within goal tolerance."""
        if self.goal_pose is None or self.current_pose is None:
            return False
        dx = self.current_pose.position.x - self.goal_pose.position.x
        dy = self.current_pose.position.y - self.goal_pose.position.y
        return math.sqrt(dx ** 2 + dy ** 2) < self.config.goal_tolerance

    def stop_robot(self):
        """Publishes a zero velocity command."""
        self.cmd_vel_pub.publish(Twist())

    def dwa_planning(self):
        """Main DWA planning routine"""
        # Get dynamic window
        v_window = self.generate_dynamic_window()
        
        trajectories = []  # Collect all for visualization
        min_cost = float('inf')
        best_cmd = None

        # Generate and evaluate trajectories
        for v in np.arange(v_window[0], v_window[1], self.config.velocity_resolution):
            for w in np.arange(v_window[2], v_window[3], self.config.angular_resolution):
                trajectory = self.predict_trajectory(v, w)
                
                if trajectory:
                    # Calculate costs
                    goal_cost = self.calculate_goal_cost(trajectory[-1])
                    obstacle_cost = self.calculate_obstacle_cost(trajectory)
                    velocity_cost = self.calculate_velocity_cost(v, w)
                    
                    # Total cost
                    total_cost = (self.config.goal_cost_gain * goal_cost +
                                  self.config.obstacle_cost_gain * obstacle_cost +
                                  self.config.velocity_cost_gain * velocity_cost)
                    
                    vis_cost = total_cost if obstacle_cost != float('inf') else float('inf')
                    trajectories.append((trajectory, vis_cost))
                    
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_cmd = (v, w)
        
        # Visualize trajectories
        self.visualize_trajectories(trajectories)
        return best_cmd

    def generate_dynamic_window(self):
        """Calculate dynamic window based on current state"""
        # Dynamic window from robot specification
        vs = [self.config.min_linear_vel, self.config.max_linear_vel,
              self.config.min_angular_vel, self.config.max_angular_vel]
        
        # Dynamic window from motion model
        vd = [self.current_twist.linear.x - self.config.max_linear_accel * self.config.dt,
              self.current_twist.linear.x + self.config.max_linear_accel * self.config.dt,
              self.current_twist.angular.z - self.config.max_angular_accel * self.config.dt,
              self.current_twist.angular.z + self.config.max_angular_accel * self.config.dt]
        
        # Final dynamic window
        return [max(vs[0], vd[0]), min(vs[1], vd[1]),
                max(vs[2], vd[2]), min(vs[3], vd[3])]

    def predict_trajectory(self, lv, av):
        """
        Predicts the robot's trajectory over a short horizon using given velocities.
        """
        traj = [(0.0, 0.0)]  # Include initial position
        x, y, theta = 0.0, 0.0, 0.0
        steps = int(self.config.predict_time / self.config.dt)
        for _ in range(steps):
            x += lv * math.cos(theta) * self.config.dt
            y += lv * math.sin(theta) * self.config.dt
            theta += av * self.config.dt
            traj.append((x, y))
        return traj

    def calculate_goal_cost(self, endpoint):
        """Returns Euclidean distance from predicted endpoint to goal."""
        if self.goal_pose is None or self.current_pose is None:
            return float('inf')
        cp = self.current_pose.position
        orientation = self.current_pose.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y ** 2 + orientation.z ** 2))
        ex = cp.x + endpoint[0] * math.cos(yaw) - endpoint[1] * math.sin(yaw)
        ey = cp.y + endpoint[0] * math.sin(yaw) + endpoint[1] * math.cos(yaw)
        dx = self.goal_pose.position.x - ex
        dy = self.goal_pose.position.y - ey
        return math.sqrt(dx ** 2 + dy ** 2)

    def calculate_obstacle_cost(self, traj):
        """
        Checks if the trajectory is safe with respect to LaserScan data.
        Returns inverse distance to closest obstacle.
        """
        if self.laser_data is None:
            return 0.0
        min_dist = float('inf')
        for x, y in traj:
            d = math.sqrt(x ** 2 + y ** 2)
            a = math.atan2(y, x)
            idx = self.angle_to_laser_index(a)
            if 0 <= idx < len(self.laser_data.ranges):
                l_dist = self.laser_data.ranges[idx]
                if l_dist < float('inf') and d + self.config.robot_radius > l_dist:
                    return float('inf')
                if l_dist < float('inf'):
                    buffer = l_dist - d - self.config.robot_radius
                    if buffer >= 0:
                        min_dist = min(min_dist, buffer)
        return 1.0 / max(min_dist, 0.01) if min_dist < self.config.min_obstacle_distance else 0.0

    def angle_to_laser_index(self, angle):
        """Converts a given angle to a corresponding index in the LaserScan array."""
        if self.laser_data is None:
            return -1
        angle = angle - self.laser_data.angle_min
        index = int(angle / self.laser_data.angle_increment)
        return max(0, min(index, len(self.laser_data.ranges) - 1))

    def calculate_velocity_cost(self, lv, av):
        """Returns normalized cost for velocity magnitude."""
        return abs(lv) / self.config.max_linear_vel + abs(av) / self.config.max_angular_vel

    def visualize_trajectories(self, trajectories):
        """
        Publishes all sampled trajectories to RViz for visualization.
        Colors reflect cost: green for best, red for collision, blue-magenta gradient for others.
        """
        if not trajectories:
            return
        min_cost = min(cost for _, cost in trajectories if cost != float('inf')) if any(cost != float('inf') for _, cost in trajectories) else float('inf')
        marker_array = MarkerArray()
        for i, (traj, cost) in enumerate(trajectories):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.color.a = 0.5
            marker.scale.x = 0.02
            if cost == min_cost:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            elif cost == float('inf'):
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                norm = min((cost - min_cost) / (10.0 - min_cost), 1.0) if min_cost < 10.0 else 1.0  # Normalize for gradient
                marker.color.r = norm
                marker.color.g = 0.0
                marker.color.b = 1.0 - norm
            for x, y in traj:
                p = Point()
                p.x, p.y = x, y
                p.z = 0.0
                marker.points.append(p)
            marker_array.markers.append(marker)
        self.trajectory_vis_pub.publish(marker_array)

def main(args=None):
    """ROS 2 node entry point."""
    rclpy.init(args=args)
    planner = DWALocalPlanner()
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()