import rclpy
import rclpy.executors
from rclpy.node import Node
from typing import Tuple, Callable, Type
from functools import partial
import numpy as np
from sensor_msgs.msg import Image
from mvdb_interface.srv import Segmentation, Embedding
from source.models.lseg_net import LSegOptimizedInference
import clip
import cv_bridge
import torch
from model_load import load_model
import time

def stamp(s, t1, t2):
    return print(f"{s} {t2 - t1:.4f}")

def normalize_c(x: torch.Tensor) -> torch.Tensor:
    return x / (torch.linalg.norm(x, dim=1, keepdims=True) + torch.finfo(x.dtype).eps)
        
class AbstractServiceNode(Node):

    def __init__(self, srv_t: Type[Segmentation | Embedding], name: str, path: str, cb: Callable = lambda _, rs: rs):
        super().__init__(name)
        self._cb = cb
        self.srv = self.create_service(srv_t, path, self.cb)
        self.get_logger().info(f"{name} up!")

    def cb(self, *args):
        t1 = self.get_clock().now()
        self.get_logger().info(f"{self.get_name()} called!")
        t1 = self.get_clock().now()
        res = self._cb(*args)
        t2 = self.get_clock().now()
        dt = (t2.nanoseconds - t1.nanoseconds) / 1e9
        self.get_logger().info(f"{self.get_name()} returning! dt = {dt:.4f}")
        return res

def image_preprocess(img: Image, transform: Callable, device: str = "cuda:0") -> Tuple[torch.Tensor | None, bool]:
    bridge =  cv_bridge.CvBridge()
    valid = False
    batch = None
    imgs = [img]
    try:
        if len(imgs) > 0:
            batch = torch.stack([transform(bridge.imgmsg_to_cv2(img)) for img in imgs]).to(device)
            valid = True
    except Exception as e:
        print(str(e))
    return batch, valid

def make_services(seg_path: str = "/segmentation", emb_path: str = "/embedding", size: Tuple[int,int] = [256, 399]) -> Tuple[AbstractServiceNode, AbstractServiceNode]:

    lseg_model, transform = load_model(inference=True, size=size)
    lseg_model: LSegOptimizedInference

    text_embedder = lseg_model.net.clip_pretrained
    image_embedder = lseg_model.net

    def embed_cb(req: Embedding.Request, resp: Embedding.Response, text_embedder = None) -> Embedding.Response:

        print(f"Query = '{req.query}'")
        tokenized_text = clip.tokenize(req.query).to(torch.device("cuda:0"))
        with torch.no_grad():
            text_features: np.ndarray = text_embedder.encode_text(tokenized_text).detach().cpu().numpy()

        resp.dim = text_features.shape[-1]
        resp.embedding.extend(text_features.astype(np.float32)[0])

        return resp

    def segment_cb(req: Segmentation.Request, resp: Segmentation.Response, image_embedder = None, transform = None ) -> Segmentation.Response:

        t1 = time.time()

        batch, valid = image_preprocess(req.img, transform, device="cuda:0")

        t2 = time.time()

        if valid:

            with torch.no_grad():
                output: torch.Tensor = image_embedder(batch)

            t3 = time.time()

            resp.vimg.c = output.shape[-3]
            resp.vimg.h = output.shape[-2]
            resp.vimg.w = output.shape[-1]
            resp.vimg.dtype = str(output.dtype)

            print(f"output shape: {output.shape}")
            output = output.permute(0, 2, 3, 1)
            resp.vimg._data = output.detach().cpu().numpy().tobytes()

            t4 = time.time()

            stamp("preprocess", t1, t2)
            stamp("infer", t2, t3)
            stamp("message", t3, t4)

        return resp

    embed_cb = partial(embed_cb, text_embedder=text_embedder)
    segment_cb = partial(segment_cb, image_embedder=image_embedder, transform=transform)

    embedder = AbstractServiceNode(Embedding, "embedding_node", emb_path, embed_cb)
    segmenter = AbstractServiceNode(Segmentation, "segmentation_node", seg_path, segment_cb)

    return segmenter, embedder

if __name__ == "__main__":
    rclpy.init()
    ex = rclpy.executors.MultiThreadedExecutor()
    s, e = make_services()
    ex.add_node(s)
    ex.add_node(e)
    ex.spin()
    e.destroy_node()
    s.destroy_node()