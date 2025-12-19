import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='scene_graph',
            executable='room_detection',
            name='room_clusters',
            output='screen',
            parameters=[
                {'visualize': False},
                {'use_descriptions': False},
                {'dbscan_eps': 1.5},
                {'min_cluster_size': 2}
            ]
        ),
        Node(
            package='scene_graph',
            executable='scene_tree_generation',
            name='scene_tree',
            output='screen',
            parameters=[
                {'max_dist_duplicate': 0.0},
                {'cutoff_to_nearest_room': 3.5}
            ]
        )
    ])

if __name__ == '__main__':
    generate_launch_description()
