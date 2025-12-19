#!/usr/bin/env python3

import numpy as np
import time
import os
import json
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from hsg_msgs.msg import ObjectsWRoomList


class SceneNode:
    def __init__(self, name, level, parent=None, location=None, instance=1):
        self.name = name
        self.level = level
        self.parent = parent
        self.location = location if location is not None else [0, 0]
        self.instance = instance
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self
        self.update_location()

    def remove_child(self, child_node):
        if child_node in self.children:
            self.children.remove(child_node)
            self.update_location()

    def find_node(self, name, level, instance=None):
        if self.name == name and self.level == level and (instance is None or self.instance == instance):
            return self
        for child in self.children:
            result = child.find_node(name, level, instance)
            if result:
                return result
        return None

    def find_duplicate(self, name, level, location, threshold):
        if self.name == name and self.level == level and np.linalg.norm(np.array(self.location) - np.array(location)) < threshold:
            return self
        for child in self.children:
            result = child.find_duplicate(name, level, location, threshold)
            if result:
                return result
        return None

    def update_location(self):
        if self.children:
            locations = [child.location for child in self.children]
            self.location = list(np.mean(locations, axis=0))

    def get_hierarchy(self):
        node = self
        hierarchy = []
        while node:
            hierarchy.append((node.level, node.name, node.instance))
            node = node.parent
        return hierarchy[::-1]  # Reverse to get from root to leaf


class SceneGraph(Node):
    def __init__(self):
        super().__init__('scene_tree')
        self.declare_parameter('max_dist_duplicate', 0.0)
        self.declare_parameter('cutoff_to_nearest_room', 2.0)
        self.max_dist_duplicate = self.get_parameter('max_dist_duplicate').get_parameter_value().double_value
        self.cutoff_to_nearest_room = self.get_parameter('cutoff_to_nearest_room').get_parameter_value().double_value
        self.package_dir = '/workspace/quad_ws/src/scene_graph/scene_graph'
        # if maps directory exists, remove it and create a empty one
        if os.path.exists(os.path.join(self.package_dir, 'maps')):
            os.system(f'rm -r {os.path.join(self.package_dir, "maps")}')
        os.makedirs(os.path.join(self.package_dir, 'maps'))
        self.tree_root = SceneNode("Scene Graph", "root")
        # subscribe to a MarkerArray topic
        self.subscription = self.create_subscription(ObjectsWRoomList, 'scene_areas', self.marker_callback, 1)
        self.subscription  # prevent unused variable warning
        self.start_time = time.time()

        # self.create_timer(5.0, self.reset_tree)

    def reset_tree(self):
        self.tree_root = SceneNode("Scene Graph", "root")

    def add_to_tree(self, building_name, floor_name, room_name, object_name, location):
        # Find or create building node
        building_node = self.tree_root.find_node(building_name, "building")
        if not building_node:
            building_node = SceneNode(building_name, "building", self.tree_root)
            self.tree_root.add_child(building_node)

        # Find or create floor node
        floor_node = building_node.find_node(floor_name, "floor")
        if not floor_node:
            floor_node = SceneNode(floor_name, "floor", building_node)
            building_node.add_child(floor_node)

        # Check for existing room nodes and their distances
        room_node = None
        if room_name == 'corridor':
            existing_room_nodes = [child for child in floor_node.children if child.name == room_name]
            min_dist = float('inf')
            for existing_room_node in existing_room_nodes:
                dist = np.linalg.norm(np.array(existing_room_node.location) - np.array(location))
                if dist < min(self.cutoff_to_nearest_room, min_dist):
                    min_dist = dist
                    room_node = existing_room_node
                    instance_number_room = room_node.instance

        # If no suitable room node is found, create a new one with a new instance
        if not room_node:
            if 'corridor' in room_name:
                instance_number_room = sum(1 for child in floor_node.children if child.name == room_name) + 1
                room_node = SceneNode(room_name, "room", floor_node, instance=instance_number_room)
                floor_node.add_child(room_node)
            else:
                instance_number_room = int(room_name.split('-')[-1])
                room_name = room_name.split('-')[0]
                existing_room_nodes = [child for child in floor_node.children if child.name == room_name and  child.instance == instance_number_room]
                if len(existing_room_nodes) > 0:
                    room_node = existing_room_nodes[0]
                else:
                    room_node = SceneNode(room_name, "room", floor_node, instance=instance_number_room)
                    floor_node.add_child(room_node)

        # Check for duplicate object by comparing locations
        # duplicate_node = self.tree_root.find_duplicate(object_name, "object", location, self.max_dist_duplicate)
        # if duplicate_node:
        #     current_room_node = duplicate_node.parent
        #     if current_room_node.name != room_name:
        #         # print(f"Moving {object_name} from {current_room_node.name} to {room_name}")
        #         current_room_node.remove_child(duplicate_node)
        #         room_node.add_child(duplicate_node)
        #         duplicate_node.parent = room_node
        #     # else:
        #     #     print(f"Object {object_name} with location {location} already exists as {duplicate_node.name} in room {room_node.name}")
        #     return

        # Handle object instance counting by looking at the number of objects of the same type in room and incrementing
        instance_number_object = sum(1 for child in room_node.children if child.name == object_name) + 1

        # Add object node with instance number and location
        self.get_logger().info(f"Adding {object_name} {instance_number_object} to {room_name} {instance_number_room}")
        object_node = SceneNode(object_name, "object", room_node, location, instance_number_object)
        room_node.add_child(object_node)

    def marker_callback(self, msg):
        # print(self.tree_root.parent,self.tree_root.children)
        # if not self.tree_root:
        #     return
        if msg.refresh_tree.data:
            self.reset_tree()
        for cluster in msg.clusters:
            for i in range(len(cluster.object_labels)):
                self.add_to_tree("metrotech", "basement", cluster.room_label.data, cluster.object_labels[i].data, [cluster.points[i].x,cluster.points[i].y])
        # save the dictionary to a JSON file
        save_path = os.path.join(self.package_dir, os.path.join('maps',str(round(time.time()))+'.json'))
        self.get_logger().info(f"Saving scene tree {str(round(time.time()))}")
        with open(save_path, "w") as file:
            json.dump(scene_tree_to_dict(self.tree_root), file, indent=4)

        current_time = time.time()
        time_diff = current_time - self.start_time
        freq = 1/time_diff
        self.start_time = current_time
        self.get_logger().info(f"Time difference is {time_diff} s. The frequency of HSG generation is {freq}")


# Print the updated structure
def print_scene_tree(node, level=0):
    print("    " * level + f"{node.level}: {node.name} (Instance: {node.instance}, Location: {node.location})")
    for child in node.children:
        print_scene_tree(child, level + 1)

def prune_empty_rooms(node):
    for child in node.children:
        if child.level == "room" and not child.children:
            node.remove_child(child)
        else:
            prune_empty_rooms(child)

def save_scene_tree_to_file(node, filename):
    # prune_empty_rooms(node)
    with open(filename, 'w') as file:
        def write_node_to_file(node, level=0):
            file.write("    " * level + f"{node.level}: {node.name} (Instance: {node.instance}, Location: {node.location})\n")
            for child in node.children:
                write_node_to_file(child, level + 1)
        write_node_to_file(node)

def scene_tree_to_dict(node):
    # def node_to_dict(node):
    #     node_dict = {
    #         "location": node.location,
    #         "children": {}
    #     }
    #     for child in node.children:
    #         key = f"{child.name}_{child.instance}"
    #         node_dict["children"][key] = node_to_dict(child)
    #     return node_dict

    tree_dict = {
        "buildings": {}
    }

    for building in node.children:
        building_key = f"{building.name}-{building.instance}"
        tree_dict["buildings"][building_key] = {"floors": {}}
        for floor in building.children:
            floor_key = f"{floor.name}-{floor.instance}"
            tree_dict["buildings"][building_key]["floors"][floor_key] = {"rooms": {}}
            for room in floor.children:
                room_key = f"{room.name}-{room.instance}"
                tree_dict["buildings"][building_key]["floors"][floor_key]["rooms"][room_key] = {"location": room.location, "objects": {}}
                for obj in room.children:
                    obj_key = f"{obj.name}-{obj.instance}"
                    tree_dict["buildings"][building_key]["floors"][floor_key]["rooms"][room_key]["objects"][obj_key] = {"location": obj.location}

    return tree_dict

def main(args=None):
    rclpy.init(args=args)
    hsg_subscriber = SceneGraph()
    rclpy.spin(hsg_subscriber)
    hsg_subscriber.destroy_node()
    rclpy.shutdown()
