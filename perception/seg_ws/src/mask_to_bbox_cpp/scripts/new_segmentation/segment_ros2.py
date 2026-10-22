#!/usr/bin/env python3
import sys
import os

# Add the current directory to the Python path
sys.path.append("/workspace/ws/src/mask_to_bbox_cpp/scripts/new_segmentation")

import rclpy
from rclpy.node import Node
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from infer import TensorrtEngine

# from scipy.io import loadmat
import copy
import os
import torch


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind="mergesort" if return_index else "quicksort")
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode="BGR"):
    labelmap = labelmap.astype("int")
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3), dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * np.tile(
            colors[label], (labelmap.shape[0], labelmap.shape[1], 1)
        )

    if mode == "BGR":
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


class ImageConverter(Node):

    def __init__(self):
        super().__init__("image_converter")
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, "/go2/cam/color/image_raw", self.callback1, 10
        )
        self.publisher_ = self.create_publisher(Image, "sem_image", 10)

        self.get_logger().info("Loading model")

        # class_list = "/workspace/ws/src/classes/model_classes.txt"
        # model_path = "/workspace/ws/src/mask_to_bbox_cpp/scripts/new_segmentation/optimized_fcclip_384_512.trt"
        # self.img_resize_shape = (16 * 32, 12 * 32)
        class_list = "/workspace/ws/src/classes/model_classes_new.txt"
        model_path = "/workspace/ws/src/mask_to_bbox_cpp/scripts/new_segmentation/optimized_fcclip_480_640_bf16.trt"
        self.img_resize_shape = (640, 480)


        stuff_classes = []
        with open(class_list, "r") as f:
            stuff_classes = f.read().splitlines()
        self.seg_engine = TensorrtEngine(model_path, stuff_classes)
        self.seg_engine.activate()

        self.get_logger().info("Model loaded")
        # self.warmup_model()
        # self.get_logger().info('Warmup done')
        # self.colors = loadmat('data/color150.mat')['colors']
        self.sem_image = None

    def warmup_model(self):
        for i in range(10):
            cv_image_RGB = np.random.rand(480, 640, 3)
            rgb_image = np.array(cv_image_RGB)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

            rgb_image = (rgb_image - mean) / std
            rgb_image = (
                rgb_image[np.newaxis, :, :, :].transpose(0, 3, 1, 2).astype(np.float16)
            )
            pred = self.sem_engine.infer(rgb_image)[0][0]

    def callback1(self, imgRGB):
        img = self.bridge.imgmsg_to_cv2(imgRGB, "bgr8")
        img = cv2.resize(img, self.img_resize_shape)
        preprocessed_input = (
            torch.from_numpy(np.asarray(img, dtype=np.float32))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to("cuda")
        ) / 255.0
        self.seg_engine.infer({"img_and_crops": preprocessed_input})
        pred_class_mask, pred_mask, mask_id_list, class_id_list, label_id_list = (
            self.seg_engine.postprocess_seginfo()
        )
        pred_class_mask = pred_class_mask.astype(np.uint8)

        pred_color_msg = self.bridge.cv2_to_imgmsg(pred_class_mask, "bgr8")
        pred_color_msg.header = imgRGB.header
        self.publisher_.publish(pred_color_msg)


def main(args=None):
    rclpy.init(args=args)
    ic = ImageConverter()
    try:
        rclpy.spin(ic)
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
    ic.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
