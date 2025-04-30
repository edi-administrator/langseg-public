import rclpy
import rclpy.executors
import argparse
from rclpy.node import Node
from typing import Tuple, Callable
import numpy as np
from sensor_msgs.msg import Image
from mvdb_interface.msg import VectorImage
import cv_bridge
import torch
from model_load import load_model
import time

def stamp(s, t1, t2):
    return f"{s} {t2 - t1:.4f}"

def normalize_c(x: torch.Tensor) -> torch.Tensor:
    return x / (torch.linalg.norm(x, dim=1, keepdims=True) + torch.finfo(x.dtype).eps)

def image_preprocess(img: Image, transform: Callable, device: str = "cuda:0") -> Tuple[torch.Tensor | None, bool]:
    bridge =  cv_bridge.CvBridge()
    valid = False
    batch = None
    encodings = {
        "bayer_rggb8": "bgr8",
    }
    try:
        if img.encoding in encodings:
            encoding = encodings[img.encoding]
        else:
            encoding = "passthrough"
        img_mat = bridge.imgmsg_to_cv2(img, desired_encoding=encoding)
        torch_img = transform(img_mat).to(device)
        batch = torch.unsqueeze(torch_img, 0)
        valid = True
    except Exception as e:
        print(str(e))
    return batch, valid


class SegmentedImagePublisher(Node):

    def __init__(self, topic_in: str, topic_out: str, size: Tuple[int,int] = (256, 399)):
        
        super().__init__("segmentation_node")

        self.lseg_model, self.transform = load_model(inference=True, size=size)
        
        self.sub = self.create_subscription(Image, topic_in, self.image_cb, 1)
        self.pub = self.create_publisher(VectorImage, topic_out, 1)
        
        self.get_logger().info("segmentation republisher node up!")
        self.get_logger().info(f"images in: {self.sub.topic}")
        self.get_logger().info(f"seg out  : {self.pub.topic}")

    def image_cb(self, msg: Image):

        t1 = time.time()

        batch, valid = image_preprocess(msg, self.transform, device="cuda:0")

        t2 = time.time()

        if valid:

            vimg = VectorImage()

            with torch.no_grad():
                output: torch.Tensor = self.lseg_model.net(batch)

            t3 = time.time()

            vimg.header = msg.header
            vimg.c = output.shape[-3]
            vimg.h = output.shape[-2]
            vimg.w = output.shape[-1]
            vimg.dtype = str(output.dtype)

            self.get_logger().info(f"output shape: {output.shape}")
            output = output.permute(0, 2, 3, 1)
            vimg._data = output.detach().cpu().numpy().tobytes()

            t4 = time.time()

            self.get_logger().info(stamp("preprocess", t1, t2))
            self.get_logger().info(stamp("infer", t2, t3))

            self.pub.publish(vimg)
            self.get_logger().info(stamp("message", t3, t4))

        else:

            self.get_logger().error("did not convert image to tensor!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "repub",
        usage="--topic_in IMAGES_IN --topic_out IMAGES_OUT"
    )

    parser.add_argument(
        "--topic_in",
        default="/camera_left/image_raw"
    )

    parser.add_argument(
        "--topic_out",
        default="/segmented_images"
    )

    args = parser.parse_args()

    rclpy.init()
    ex = rclpy.executors.SingleThreadedExecutor()
    s = SegmentedImagePublisher(args.topic_in, args.topic_out)
    ex.add_node(s)
    
    while rclpy.ok():
        ex.spin()

    s.destroy_node()