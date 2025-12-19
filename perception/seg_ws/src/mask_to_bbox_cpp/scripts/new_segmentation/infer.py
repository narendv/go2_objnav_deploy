import argparse
import glob
import os
from collections import OrderedDict

import cv2
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F
import tqdm
import zmq
import copy
from cuda import cudart
from flatbuffers import Builder

from frame import Frame, Label

numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
    np.bool_: torch.bool,
}


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--class-list",
        default="models/model_classes.txt",
        metavar="FILE",
        help="path to class list file",
    )
    parser.add_argument(
        "--model-path",
        default="/workspace/ws/src/mask_to_bbox_cpp/scripts/new_segmentation/optimized_fcclip_384_512.trt",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--image-dir",
        default="data/images",
        metavar="DIR",
        help="directory of .png images to run inference on",
    )
    return parser


def _cuda_assert(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None


class TensorrtEngine:
    def __init__(
        self,
        engine_path,
        stuff_classes,
    ):
        self.stream = _cuda_assert(cudart.cudaStreamCreate())
        self.stuff_classes = stuff_classes
        self.shared_device_memory = None
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None
        self.test_topk_per_image = 100
        self.device = "cuda"
        self.overlap_threshold = 0.4
        # self.colors = np.loadtxt("/workspace/ws/src/classes/colors.txt")
        self.colors = np.loadtxt("/workspace/ws/src/classes/colors_new.txt")

    def __del__(self):
        del self.engine
        del self.context
        del self.tensors
        if self.shared_device_memory:
            cudart.cudaFree(self.shared_device_memory)
        cudart.cudaStreamDestroy(self.stream)
        del self.stream

    def activate(self):
        trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.INFO), "")
        with open(self.engine_path, "rb") as f, trt.Runtime(
            trt.Logger(trt.Logger.WARNING)
        ) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.context.set_optimization_profile_async(0, self.stream)
        self.allocate_buffers()

    def allocate_buffers(self, device="cuda"):
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            shape = self.engine.get_tensor_shape(binding)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(binding, shape)
            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
            ).to(device=device)
            self.tensors[binding] = tensor

    def instance_inference(self, segment_indices, mask_final, encoded_segments_info):
        segment_indices = torch.from_numpy(segment_indices).to(
            self.device
        )  # (250, 1787)
        mask_final = torch.from_numpy(mask_final).to(self.device)  # (250, 72, 96)

        image_size = mask_final.shape[-2:]
        panoptic_seg = torch.zeros(image_size, dtype=torch.int32, device=self.device)

        mask_indices = torch.nonzero(mask_final, as_tuple=False)
        panoptic_seg[mask_indices[:, 1], mask_indices[:, 2]] = segment_indices[
            mask_indices[:, 0]
        ]
        panoptic_seg = (
            (
                F.interpolate(
                    panoptic_seg.unsqueeze(0).unsqueeze(0).float(),
                    # scale_factor=4,
                    size=(480, 640),
                    mode="nearest-exact",
                )
            )
            .squeeze(0)
            .squeeze(0)
        ).int()

        panoptic_seg = panoptic_seg.cpu().numpy()
        return panoptic_seg, encoded_segments_info

    def infer(self, feed_dict):
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        noerror = self.context.execute_async_v3(self.stream)
        if not noerror:
            raise ValueError("ERROR: inference failed.")

    def postprocess_seginfo(self):
        label_set = set()
        instance_count = dict()
        label_id_list = []
        mask_id_list = []
        class_id_list = []
        # print(self.tensors.keys())
        # segment_indices = self.tensors["segment_indices"].cpu().numpy()
        mask_final = self.tensors["mask_final"].cpu().numpy()
        encoded_segments_info = self.tensors["encoded_segments_info"].cpu().numpy()
        segments_info = encoded_segments_info
        # print(mask_final.shape)
        # print(encoded_segments_info.shape)
        # pred_mask, segments_info = self.instance_inference(
        #     segment_indices, mask_final, encoded_segments_info
        # )
        pred_mask = (
            (
                F.interpolate(
                    torch.Tensor(mask_final).to(self.device).unsqueeze(0).unsqueeze(0).float(),
                    # scale_factor=4,
                    size=(480, 640),
                    mode="nearest-exact",
                )
            )
            .squeeze(0)
            .squeeze(0)
        ).int().cpu().numpy()
        pred_class_mask = np.zeros([pred_mask.shape[0], pred_mask.shape[1], 3])

        # print('aaa')
        # print(pred_mask)
        # print(segments_info)
        for indx in range(segments_info.shape[0]):
            mask_id = int(segments_info[indx, 0])
            # print(mask_id)
            if segments_info[indx, 0] == 0:
                break
            mask_id_list.append(mask_id)

            category_idx = int(segments_info[indx, 1])
            pred_class_mask[np.where(pred_mask == mask_id)] = self.colors[category_idx]
            if category_idx in label_set:
                instance_count[category_idx] += 1
            else:
                instance_count[category_idx] = 1
            label_set.add(category_idx)
            label_id_list.append(
                self.stuff_classes[category_idx].split(",")[0]
                + "-"
                + str(instance_count[category_idx])
            )
            class_id_list.append(category_idx)
        return pred_class_mask, pred_mask, mask_id_list, class_id_list, label_id_list


if __name__ == "__main__":
    class_list = "classes/model_classes.txt"
    model_path = "optimized_fcclip_384_512.trt"
    args = get_parser().parse_args()
    img_resize_shape = (16 * 32, 12 * 32)
    # initialize plugin registry

    b = Builder(1)
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://0.0.0.0:5555")

    stuff_classes = []
    with open(class_list, "r") as f:
        stuff_classes = f.read().splitlines()

    seg_engine = TensorrtEngine(model_path, stuff_classes)
    seg_engine.activate()
    i = 0
    for path in tqdm.tqdm(
        sorted(
            glob.glob(
                os.path.expanduser(
                    args.image_dir + "/*.png"
                )
            )
        )
    ):
        print(path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        print(img.shape)
        img = cv2.resize(img, img_resize_shape)
        preprocessed_input = (
            torch.from_numpy(np.asarray(img, dtype=np.float32))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to("cuda")
        ) / 255.0

        print(preprocessed_input.shape)
        seg_engine.infer({"img_and_crops": preprocessed_input})
        pred_class_mask, pred_mask, mask_id_list, class_id_list, label_id_list = (
            seg_engine.postprocess_seginfo()
        )
        print(pred_mask.shape)
        print(mask_id_list)
        print(class_id_list)
        print(label_id_list)
        print(pred_class_mask)
        input()
        timestamp_ms = int(i + 1)
        _, jpeg_img = cv2.imencode(".jpg", img)
        jpeg_bytes = jpeg_img.tobytes()

        Frame.FrameStartImageVector(b, len(jpeg_bytes))
        for byte in reversed(jpeg_bytes):
            b.PrependByte(byte)
        image_offset = b.EndVector()

        Frame.FrameStartSegMaskVector(b, pred_mask.size)
        for px in reversed(pred_mask.flatten()):
            b.PrependByte(px)
        seg_mask_offset = b.EndVector()

        name_offsets = [b.CreateString(name) for name in label_id_list]

        # Step 2: Construct Label objects and collect their offsets
        label_offsets = []
        for label_id, name_offset, class_id in zip(
            mask_id_list, name_offsets, class_id_list
        ):
            Label.Start(b)
            Label.AddId(b, label_id)
            Label.AddCategoryId(b, class_id)
            Label.AddName(b, name_offset)
            label_offset = Label.End(b)
            label_offsets.append(label_offset)

        # Step 3: Start the labels vector and add the collected label offsets
        Frame.FrameStartLabelsVector(b, len(label_offsets))
        for label_offset in reversed(label_offsets):
            b.PrependUOffsetTRelative(label_offset)
        labels_offset = b.EndVector()

        Frame.FrameStart(b)
        Frame.FrameAddImage(b, image_offset)
        Frame.FrameAddSegMask(b, seg_mask_offset)
        Frame.FrameAddTimestampMs(b, timestamp_ms)
        Frame.FrameAddLabels(b, labels_offset)  # Add the labels vector to the Frame
        frame = Frame.FrameEnd(b)

        # Finish the creation of the FlatBuffer and get the result
        b.Finish(frame)
        buf = b.Output()
        zmq_message = bytearray(buf)
        socket.send(zmq_message, zmq.NOBLOCK)
        i = i + 33
