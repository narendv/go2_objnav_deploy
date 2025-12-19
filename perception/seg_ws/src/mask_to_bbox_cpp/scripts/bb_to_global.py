#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
import tf2_ros
import tf2_geometry_msgs
import transforms3d.quaternions as quat

object_colors = np.loadtxt("/workspace/ws/src/classes/colors.txt")

object_names_filename = "/workspace/ws/src/classes/model_classes.txt"
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


def euclidean_distance(marker1, marker2):
    return np.sqrt(
        (marker1.pose.position.x - marker2.pose.position.x) ** 2
        + (marker1.pose.position.y - marker2.pose.position.y) ** 2
    )


class BoundingBox3D(Node):
    def __init__(self):
        super().__init__("bounding_box_3d")

        # Subscribers
        self.image_sub = message_filters.Subscriber(
            self, Image, "/go2/cam/depth/image_rect_raw"
        )
        self.info_sub = message_filters.Subscriber(
            self, CameraInfo, "/go2/cam/depth/camera_info"
        )
        self.bbox_sub = message_filters.Subscriber(
            self, Detection2DArray, "/bounding_boxes"
        )

        # Synchronize messages
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.info_sub, self.bbox_sub], 500, 0.1
        )
        self.ts.registerCallback(self.callback)

        # Publisher
        self.marker_pub = self.create_publisher(MarkerArray, "/bbox_centers_global", 10)

        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()
        self.marker_array = MarkerArray()
        self.camera_in_bl = np.array(
            [
                [0.01273343, -0.07259746, 0.99728004, 0.10951693],
                [-0.99986312, 0.00961243, 0.01346616, 0.02625135],
                [-0.01056389, -0.997315, -0.07246512, -0.11930069],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.marker_points = []
        self.k = 0
        self.points_2d = []
        self.points_3d = []
        self.detection_id = []
        self.trans = np.eye(4)

    def callback(self, depth_image, camera_info, bbox_array):
        try:
            depth_img = self.bridge.imgmsg_to_cv2(
                depth_image, desired_encoding="passthrough"
            )
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridgeError: {e}")
            return

        # Camera intrinsic parameters
        K = np.array(camera_info.k).reshape(3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        trans = self.get_transformation_matrix(depth_image.header.stamp)
        if trans is not None:
            self.trans = trans
        for detection in bbox_array.detections:
            bbox = detection.bbox
            u, v = int(bbox.center.position.x), int(bbox.center.position.y)
            # print(u,v,depth_img.shape)
            z = depth_img[v, u] / 1000.0  # Convert from mm to meters
            if z > 5 or z < 1:
                continue

            if z == 0:
                continue  # Skip invalid depth

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            point_3d = np.array([x, y, z, 1.0])  # Homogeneous coordinates

            point = self.camera_in_bl @ point_3d[..., None]
            # print(point, self.trans)
            transformed_point = self.trans @ point
            transformed_point = transformed_point[:, 0]

            self.points_2d.append(transformed_point[:2])
            self.points_3d.append(transformed_point)
            self.detection_id.append(detection.id)
        # print(len(self.points_2d), len(self.points_3d), len(self.detection_id))

        self.create_marker_array()

        self.marker_pub.publish(self.marker_array)

    def create_marker_array(self):
        self.marker_array = MarkerArray()

        points_2d = np.array(self.points_2d)
        points_3d = np.array(self.points_3d)
        detection_id = np.array(self.detection_id)
        indices = []
        for object_id in np.unique(detection_id):
            object_indices = np.where(detection_id == object_id)[0]
            # object_point = points_3d[object_indices]
            object_point = points_2d[object_indices]
            _, select_object_indices = np.unique(
                (np.round(object_point * 1)).astype(np.int32), axis=0, return_index=True
            )
            object_indices = object_indices[select_object_indices].tolist()
            indices.extend(object_indices)

        # _, indices = np.unique((np.round(points_2d,1)*1).astype(np.int32), axis=0, return_index=True)
        # print(points_2d.shape, indices.shape)
        # print(self.detection_id[indices])
        self.points_2d = points_2d[indices].tolist()
        self.points_3d = points_3d[indices].tolist()
        self.detection_id = detection_id[indices].tolist()

        for point, object_idx in zip(self.points_3d, self.detection_id):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rclpy.clock.Clock().now().to_msg()
            # marker.ns = 'bbox_centers'
            marker.id = self.k
            self.k += 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = 0.0
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.a = 1.0
            if object_idx in object_names:
                index = object_names.index(object_idx)
                color = object_colors[index]
                marker.color.r = color[0] / 255.0
                marker.color.g = color[1] / 255.0
                marker.color.b = color[2] / 255.0
            else:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 1.0
            marker.text = object_idx
            self.marker_array.markers.append(marker)

    def get_transformation_matrix(self, stamp):
        try:
            # trans = self.tf_buffer.lookup_transform('map', 'base_link', stamp, rclpy.time.Duration(seconds=1.0))
            # Duration to subtract
            duration_sec = 1.0
            duration_nsec = 0

            # Calculate the new time
            new_sec = stamp.sec - duration_sec
            new_nsec = stamp.nanosec - duration_nsec

            # # Adjust if the nanoseconds are negative
            # if new_nsec < 0:
            #     new_sec -= 1
            #     new_nsec += 1_000_000_000

            # Create the new Time object
            adjusted_stamp = rclpy.time.Time(seconds=new_sec, nanoseconds=new_nsec)

            # trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            trans = self.tf_buffer.lookup_transform("map", "base_link", stamp)
            translation = trans.transform.translation
            rotation = trans.transform.rotation

            # Extract the translation
            t = np.array([translation.x, translation.y, translation.z])

            # Convert quaternion to transformation matrix
            q = [
                rotation.w,
                rotation.x,
                rotation.y,
                rotation.z,
            ]  # transforms3d uses scalar-first quaternion
            R = quat.quat2mat(q)

            # Construct the transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[0:3, 0:3] = R
            transformation_matrix[0:3, 3] = t

            return transformation_matrix

        except Exception as e:
            self.get_logger().error(f"Failed to get transformation: {str(e)}")
            return None


def main(args=None):
    rclpy.init(args=args)
    node = BoundingBox3D()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
