import unittest
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

class DWAConfig:
    def __init__(self):
        self.goal_tolerance = 0.2  # or your desired value in meters

class TestDWAPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Node('test_dwa_planner')

    def test_goal_reaching(self):
        # Test if planner can reach a simple goal
        pass

    def test_obstacle_avoidance(self):
        # Test if planner avoids obstacles
        pass

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    unittest.main()