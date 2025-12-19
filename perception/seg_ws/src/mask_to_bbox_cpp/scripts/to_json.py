#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
import json
from datetime import datetime
import numpy as np
from rclpy.time import Time
import shutil
import os


# object_colors = np.loadtxt("/workspace/ws/src/classes/colors.txt")
object_colors = np.loadtxt("/workspace/ws/src/classes/colors_new.txt")

# object_names_filename = "/workspace/ws/src/classes/model_classes.txt"
object_names_filename = "/workspace/ws/src/classes/model_classes_new.txt"
object_names = []

with open(object_names_filename, "r") as file:
    for line in file:
        # Strip any leading/trailing whitespace (including newline characters)
        line = line.strip()

        # Find the first comma
        comma_index = line.find(",")

        # If there's a comma, use the substring up to the comma
        if comma_index != -1:
            object_names.append(line[:comma_index])
        else:
            # If there's no comma, use the entire line
            object_names.append(line)


def clear_folder(folder_path):
    # List all files and directories in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # Remove the file or link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory and its contents
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


class MarkerArraySubscriber(Node):

    def __init__(self):
        super().__init__("marker_array_subscriber")
        self.subscription = self.create_subscription(
            MarkerArray,
            "/bbox_centers_global",  # replace with your topic name
            self.listener_callback,
            10,
        )
        self.subscription  # prevent unused variable warning
        clear_folder("/workspace/jsons")

    def listener_callback(self, msg):
        object_instances = np.zeros(len(object_names))
        map_object_bboxes = []
        for marker in msg.markers:
            object_name = marker.text
            object_idx = object_names.index(object_name)
            object_instances[object_idx] += 1
            map_object_bboxes.append(
                {
                    "center": {
                        "x": marker.pose.position.x,
                        "y": marker.pose.position.y,
                        "z": marker.pose.position.z,
                    },
                    "instance": int(object_instances[object_idx]),
                    "label": marker.text,
                }
            )
        if len(msg.markers) == 0:
            return
        time2 = Time.from_msg(msg.markers[0].header.stamp)
        time_in_nanosecs = time2.nanoseconds
        data = {
            "metrotech": {
                "basement": {
                    "map_object_bboxes": map_object_bboxes,
                    "timestamp": time_in_nanosecs,
                }
            }
        }

        with open(
            "/workspace/jsons/"
            + str(time_in_nanosecs)
            + ".json",
            "w",
        ) as f:
            json.dump(data, f, indent=4)

        self.get_logger().info("Data saved to marker_array_data.json")


def main(args=None):
    rclpy.init(args=args)
    marker_array_subscriber = MarkerArraySubscriber()
    rclpy.spin(marker_array_subscriber)
    marker_array_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
