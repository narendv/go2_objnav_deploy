from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare the launch argument
    declare_run_to_json = DeclareLaunchArgument(
        'save_to_json',
        default_value='false',
        description='Flag to run the to_json node'
    )

    return LaunchDescription([
        declare_run_to_json,

        Node(
            package='mask_to_bbox_cpp',
            executable='mask_to_bbox',
            name='mask_to_bbox',
            output='screen'
        ),
        Node(
            package='mask_to_bbox_cpp',
            executable='bb_to_global_pose_graph.py',
            name='bb_to_global',
            output='screen'
        ),
        Node(
            package='mask_to_bbox_cpp',
            executable='segment_ros2.py',
            name='segment_ros2',
            output='screen'
        ),
        Node(
            package='mask_to_bbox_cpp',
            executable='to_json.py',
            name='to_json',
            # output='screen',
            condition=IfCondition(LaunchConfiguration('save_to_json'))
        ),
    ])
