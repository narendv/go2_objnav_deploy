#!/usr/bin/env python3

from scene_graph.voting_metrics import *
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import cv2
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from scipy.optimize import minimize
from cv_bridge import CvBridge, CvBridgeError
from ollama import Client
# from sentence_transformers import SentenceTransformer
from scipy.spatial import ConvexHull
from hsg_msgs.msg import ObjectsWRoom, ObjectsWRoomList
from builtin_interfaces.msg import Duration
import random
import os
import re

class HeirarchyNode(Node):
    def __init__(self):
        super().__init__('room_clusters')
        self.declare_parameter('visualize', True)
        self.declare_parameter('use_descriptions', False)
        self.declare_parameter('dbscan_eps', 0.5)
        self.declare_parameter('min_cluster_size', 2)
        self.visualize = self.get_parameter('visualize').get_parameter_value().bool_value
        self.use_descriptions = self.get_parameter('use_descriptions').get_parameter_value().bool_value
        self.dbscan_eps = self.get_parameter('dbscan_eps').get_parameter_value().double_value
        self.min_cluster_size = self.get_parameter('min_cluster_size').get_parameter_value().integer_value
        if self.use_descriptions:
            self.prompt_code = 'room_labels_sentence.txt'
        else:
            self.prompt_code = 'room_labels.txt'
        prompts_path = os.path.join('/workspace/quad_ws/src/scene_graph/scene_graph/prompts', self.prompt_code)

        self.subscription_objects = self.create_subscription(MarkerArray,'bbox_centers_global', self.listener_callback,2)
        self.visualizer_pub = self.create_publisher(MarkerArray, 'room_markers', 1)
        self.scene_graph_pub = self.create_publisher(ObjectsWRoomList, 'scene_areas', 1)
        self.subscription_objects  # prevent unused variable warning
        if self.use_descriptions:
            with open(prompts_path, 'r') as f:
                self.room_description = [line.split(':')[0] for line in f.readlines()]
            with open(prompts_path, 'r') as f:
                self.room_labels = [line.split(':')[1].strip() for line in f.readlines()]
        else:
            with open(prompts_path, 'r') as f:
                self.room_description = [line.strip() for line in f.readlines()]
            with open(prompts_path, 'r') as f:
                self.room_labels = [line.split('/')[0] for line in f.readlines()]

        # model_name = 'llama3'
        # ollama.pull(model_name)
        self._client = Client(host='http://localhost:11434')
        llama_system_prompt_path = os.path.join('/workspace/quad_ws/src/scene_graph/scene_graph/prompts', 'llama_system_prompt.txt')
        llama_user_prompt_path = os.path.join('/workspace/quad_ws/src/scene_graph/scene_graph/prompts', 'llama_user_prompt.txt')
        with open(llama_system_prompt_path, 'r') as f:
            self.system_prompt = f.read()
        with open(llama_user_prompt_path, 'r') as f:
            self.user_prompt = f.read()
        self.rgb_values = self.generate_random_rgb(len(self.room_labels))
        # self.clip = SentenceTransformer('all-mpnet-base-v2') # 'multi-qa-mpnet-base-dot-v1'
        # self.room_embeddings = self.clip.encode(self.room_description)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.revise_tree_count = 0

        self.current_pose = None
        self.create_timer(1.0, self.lookup)

    @staticmethod
    def generate_random_rgb(n):
        random.seed(42)
        # blue:corridor, green:research laboratory, red:conference room, yellow:pantry, purple:library, turquoise:office
        color_list = [(random.random(), random.random(), random.random()) for _ in range(n)]
        return color_list

    def predict_room(self, object_labels):
        room = topk_plus_majority(self.clip, object_labels, self.room_embeddings, self.room_labels, k=20)
        # room = sentence_comparision(self.clip, object_labels, self.room_embeddings, self.room_labels)
        return room

    def predict_room_llama(self, object_labels):
        objects = ', '.join(object_labels)
        # rooms = ', '.join(self.room_labels)
        user_prompt = self.user_prompt.format(objects=objects)
        model_name = 'llama3'
        response = self._client.chat(model=model_name, messages=[
        {
            'role': 'system',
            'content': self.system_prompt,
        },
        {
            'role': 'user',
            'content': user_prompt,
        },
        ])
        pattern = r"Room label:\s*(\w+)"
        self.get_logger().info(f"Response: {response['message']['content']}")
        # Search for the pattern in the response
        match = re.search(pattern, response['message']['content'])
        if match:
            room_label = match.group(1)
        else:
            # If no match is found, assign room label as 'room'
            room_label = 'room'
        return room_label.lower()

    def lookup(self):
        try:
            # Lookup the transform from 'map' to 'base_link'
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0))
            # Extract x, y coordinates from the transform
            self.current_pose = (transform.transform.translation.x, transform.transform.translation.y)
        except TransformException as ex:
            self.get_logger().warn(f'Could not transform map to base_link: {ex}')

    def check_cluster_in_range(self,cluster_points):
        self.get_logger().info(f"Current pose: {self.current_pose}")
        cluster_center = np.mean(cluster_points, axis=0)
        dist = np.linalg.norm(cluster_center - self.current_pose)
        self.get_logger().info(f"Distance to cluster center: {dist}")
        if dist > 30.0: # max distance to a cluster
            return False
        return True

    def reset_counter(self):
        return {'office': 0, 'break_room': 0}

    def listener_callback(self, msg):
        points = []
        object_labels = []
        for marker in msg.markers:
            points.append([marker.pose.position.x, marker.pose.position.y])
            object_labels.append(marker.text)
        if points == []:
            self.get_logger().info('No points received')
            return
        # pose = self.lookup()
        # points, object_labels = self.find_points_within_radius_kdtree(points, pose[0], pose[1], 2.0)
        self.get_logger().info(f'Received {len(points)} points')
        object_labels = np.array(object_labels)
        points = np.array(points)
        self.revise_tree_count = (self.revise_tree_count + 1) % 1
        self.get_logger().info(f"Revise tree count: {self.revise_tree_count}")
        self.process_points(points,object_labels)

    def process_points(self, points, object_labels, plot_type='convex_hull'):
        counter = self.reset_counter()
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.min_cluster_size)  # adjust parameters as needed
        # dbscan = HDBSCAN(min_cluster_size=4, min_samples=None, cluster_selection_epsilon=0.8, max_cluster_size=None)
        labels = dbscan.fit_predict(points)
        scene_graph_array = ObjectsWRoomList()
        scene_graph_array.header.stamp = self.get_clock().now().to_msg()
        scene_graph_array.refresh_tree.data = True if self.revise_tree_count == 0 else False
        unique_labels = set(labels)
        if self.visualize:
            marker_array = MarkerArray()
        for k in unique_labels:
            class_member_mask = (labels == k)
            cluster_points = points[class_member_mask]
            # print(f" Refresh: {scene_graph_array.refresh_tree.data}")
            # print(f" Distance check: {self.check_cluster_in_range(cluster_points)}")
            if not scene_graph_array.refresh_tree.data and not self.check_cluster_in_range(cluster_points):
                continue
            cluster_object_labels = object_labels[class_member_mask]
            mask = cluster_object_labels != 'light'
            cluster_object_labels = cluster_object_labels[mask]
            cluster_points = cluster_points[mask]
            if k == -1:
                room_name = 'corridor'
            else:
                # room_name = self.predict_room(cluster_object_labels)
                room_name = self.predict_room_llama(cluster_object_labels)
            if room_name != 'corridor':
                counter[room_name.lower()] += 1
                instance_number = counter[room_name.lower()]
                room_name = room_name + '-' + str(instance_number)
            self.get_logger().info(f"Predicted room for cluster {k}: {room_name}")
            try:
                # room_index = self.room_labels.index(room_name)
                self.scene_graph_collate(scene_graph_array, cluster_points, cluster_object_labels, room_name)
            except ValueError:
                print(f"{room_name} not found in the list of room labels")
                return
            if self.visualize and len(cluster_points) > 4 and k != -1:
                self.visualize_cluster(marker_array, cluster_points, room_index, k, room_name, plot_type)
        # Publish the scene graph data
        self.scene_graph_pub.publish(scene_graph_array)
        # Publish the scene graph data
        if self.visualize:
            self.visualizer_pub.publish(marker_array)

    def scene_graph_collate(self, scene_graph_list, cluster_points, cluster_object_labels, room_name):
        scene_graph_array = ObjectsWRoom()
        # scene_graph_array.header.stamp = self.get_clock().now().to_msg()
        # scene_graph_array.header.frame_id = 'map'
        scene_graph_array.room_label.data = room_name
        for label in cluster_object_labels:
            object_msg = String()
            object_msg.data = label
            scene_graph_array.object_labels.append(object_msg)
        for point in cluster_points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0.0
            scene_graph_array.points.append(p)
        return scene_graph_list.clusters.append(scene_graph_array)

    def visualize_cluster(self, marker_array, cluster_points, room_index, cluster_number, text, plot_type ='convex_hull'):
        if not isinstance(cluster_number, int):
            cluster_number = int(cluster_number)
        if plot_type == 'circle':
            center, radius = self.find_minimum_enclosing_circle(cluster_points)
            marker = self.create_circle_marker(center, radius, self.rgb_values[room_index], cluster_number)
            marker_array.markers.append(marker)
        elif plot_type == 'lines':
            marker = self.create_room_lines(cluster_points, self.rgb_values[room_index], cluster_number, height=0.0)
            marker_array.markers.append(marker)
        elif plot_type == 'convex_hull':
            hull_points = self.find_convex_hull(cluster_points)
            marker = self.create_room_convex(hull_points, self.rgb_values[room_index], cluster_number, height=0.0)
            marker_array.markers.append(marker)
        self.visualizer_pub.publish(marker_array)


    @staticmethod
    def find_minimum_enclosing_circle(points):
        # Function to minimize: distance from a center point to the farthest point in the cluster
        def objective(center):
            distances = np.linalg.norm(points - center, axis=1)
            return np.max(distances)

        # Initial guess for the center: mean of the points
        initial_center = np.mean(points, axis=0)
        result = minimize(objective, initial_center, method='Nelder-Mead')
        center = result.x
        radius = objective(center)
        return center, radius+0.05 # add a small margin to the radius

    @staticmethod
    def find_convex_hull(points):
        hull = ConvexHull(points)
        return points[hull.vertices]

    def create_circle_marker(self, center, radius, color, marker_id):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "circles"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = 0.0  # Adjust if necessary
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = radius * 2  # diameter
        marker.scale.y = radius * 2  # diameter
        marker.scale.z = 0.1  # flat circle
        marker.color.a = 0.5  # Reduce opacity
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]

        return marker

    def create_room_lines(self,cluster_points, color, marker_id, height=2.0):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rooms"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        for point in cluster_points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = height
            marker.points.append(p)

        return marker

    def create_room_convex(self,hull_points, color, marker_id, height=2.0):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "convex_hull"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        for point in hull_points:
            p = Point()
            p.x, p.y, p.z = float(point[0]), float(point[1]), 0.0
            marker.points.append(p)
        # Close the loop
        p = Point()
        p.x, p.y, p.z = float(hull_points[0][0]), float(hull_points[0][1]), height
        marker.points.append(p)
        # marker_viz_time = Duration()
        # marker_viz_time.nanosec = 500000000
        # marker.lifetime = marker_viz_time
        return marker


def main(args=None):
    rclpy.init(args=args)
    marker_array_subscriber = HeirarchyNode()
    rclpy.spin(marker_array_subscriber)
    marker_array_subscriber.destroy_node()
    rclpy.shutdown()
