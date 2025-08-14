# DWA Local Planner for TurtleBot3 - ROS2 Humble

This project implements a custom Dynamic Window Approach (DWA) local planner for TurtleBot3 navigation in ROS2 Humble with Gazebo simulation.

#Demo
![alt text](<Screencast from 08-14-2025 02_58_35 PM-VEED.gif>)

## Prerequisites

- Ubuntu 22.04 LTS
- ROS2 Humble
- Python 3.10+
- Gazebo 11

## Installation and Setup

### 1. Install ROS2 Humble

```bash
# Add ROS2 repository
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Humble
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-cartographer
sudo apt install ros-humble-cartographer-ros
sudo apt install ros-humble-navigation2
sudo apt install ros-humble-nav2-bringup
```

### 2. Install TurtleBot3 Packages

```bash
sudo apt install ros-humble-turtlebot3*
```

### 3. Set Environment Variables

Add to your `~/.bashrc`:

```bash
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/humble/share/turtlebot3_gazebo/models
```

Then reload:
```bash
source ~/.bashrc
```

### 4. Clone the Planner Repository

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/Harshitpatro/dwa_local_planner.git

```

### 5. Build the Package

```bash
cd ~/ros2_ws
colcon build --packages-select dwa_local_planner
source install/setup.bash
```

## Usage

### 1. Launch TurtleBot3 Simulation

Terminal 1 - Launch Gazebo world:
```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

Terminal 2 - Launch RViz for visualization:
```bash
ros2 launch turtlebot3_bringup rviz2.launch.py
```

Terminal 3 - Launch the DWA planner:
```bash
ros2 run dwa_local_planner dwa_planner
```

### 2. Set Navigation Goal

In RViz:
1. Add MarkerArray display with topic `/dwa_trajectories`
2. Use the "2D Goal Pose" tool to set a target destination
3. Watch the robot navigate while avoiding obstacles

### 3. Monitor Performance

View debug messages:
```bash
ros2 topic echo /rosout
```

Check current velocity commands:
```bash
ros2 topic echo /cmd_vel
```

## Algorithm Overview

### Dynamic Window Approach (DWA)

The DWA algorithm implemented here follows these steps:

1. **Dynamic Window Generation**: Sample velocity commands (linear and angular) within the robot's dynamic constraints
2. **Trajectory Prediction**: For each velocity sample, predict the robot's trajectory over a time horizon
3. **Cost Evaluation**: Evaluate each trajectory using a multi-objective cost function:
   - **Goal Distance**: Distance from trajectory endpoint to goal
   - **Obstacle Avoidance**: Minimum distance to obstacles along trajectory
   - **Path Smoothness**: Velocity magnitude and angular velocity penalties
4. **Best Command Selection**: Choose the velocity command with the lowest total cost

### Cost Function

```
Total Cost = α * goal_cost + β * obstacle_cost + γ * velocity_cost
```

Where:
- `α = 1.0` (goal weight)
- `β = 2.0` (obstacle weight) 
- `γ = 0.1` (velocity weight)

### Parameters

Key configurable parameters:
- `max_linear_vel`: 0.5 m/s
- `max_angular_vel`: 1.5 rad/s
- `linear_acceleration`: 2.0 m/s²
- `angular_acceleration`: 3.0 rad/s²
- `prediction_time`: 3.0 seconds
- `dt`: 0.1 seconds (simulation step)
- `goal_tolerance`: 0.2 meters

## Troubleshooting

### Common Issues

1. **Robot not moving**: Check if goal is set and within reasonable distance
2. **Oscillating behavior**: Reduce angular velocity samples or adjust cost weights
3. **Collision with obstacles**: Increase obstacle cost weight or reduce prediction time
4. **RViz not showing trajectories**: Ensure MarkerArray topic `/dwa_trajectories` is added

### Performance Tuning

- **Increase responsiveness**: Reduce `dt` or `prediction_time`
- **Smoother paths**: Increase velocity cost weight
- **Better obstacle avoidance**: Increase obstacle cost weight
- **Faster goal reaching**: Increase goal cost weight

## File Structure

```
dwa_local_planner/
├── dwa_local_planner/
│   ├── __init__.py
│   └── dwa_planner.py
├── launch/
│   └── dwa_navigation.launch.py
├── package.xml
├── setup.py
└── README.md
```

## Testing Scenarios

1. **Simple Navigation**: Place goal in open space
2. **Obstacle Avoidance**: Navigate around static obstacles in Gazebo world
3. **Narrow Passages**: Test navigation through doorways

