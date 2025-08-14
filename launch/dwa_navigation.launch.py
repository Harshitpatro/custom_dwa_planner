#!/usr/bin/env python3
"""
Launch file for DWA Local Planner with TurtleBot3 simulation
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_name = LaunchConfiguration('world', default='turtlebot3_world')
    launch_rviz = LaunchConfiguration('rviz', default='true')
    
    # Package directories
    pkg_turtlebot3_gazebo = FindPackageShare('turtlebot3_gazebo')
    pkg_turtlebot3_bringup = FindPackageShare('turtlebot3_bringup')
    
    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    declare_world_cmd = DeclareLaunchArgument(
        'world',
        default_value='turtlebot3_world',
        description='World model name'
    )
    
    declare_rviz_cmd = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Launch RViz if true'
    )
    
    # Launch Gazebo world
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_turtlebot3_gazebo, 'launch', 'turtlebot3_world.launch.py'])
        ]),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )
    
    # Launch RViz
    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_turtlebot3_bringup, 'launch', 'rviz2.launch.py'])
        ]),
        condition=IfCondition(launch_rviz),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )
    
    # DWA Local Planner node
    dwa_planner_node = Node(
        package='dwa_local_planner',
        executable='dwa_planner',
        name='dwa_local_planner',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('/goal_pose', '/goal_pose'),
            ('/cmd_vel', '/cmd_vel'),
            ('/odom', '/odom'),
            ('/scan', '/scan'),
        ]
    )
    
    # Goal pose publisher helper node (optional - for programmatic goal setting)
    goal_publisher_node = Node(
        package='dwa_local_planner', 
        executable='goal_publisher',
        name='goal_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
        }]
    )
    
    # Create launch description
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_world_cmd)
    ld.add_action(declare_rviz_cmd)
    
    # Add nodes
    ld.add_action(gazebo_launch)
    ld.add_action(rviz_launch)
    ld.add_action(dwa_planner_node)
    
    return ld