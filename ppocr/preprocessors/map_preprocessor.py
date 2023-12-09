# Modified from https://github.com/clovaai/deep-text-recognition-benchmark
#
# Licensed under the Apache License, Version 2.0 (the "License");s
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from .base_preprocessor import BasePreprocessor

from ppocr.utils.self_segmentation.kmeans import clusterpixels
from ppocr.utils.skeleton_tracing import trace_skeleton 
import paddle

def cluster_skeleton_detector(img, csize, maxIter):
    """
        Args:
            img (numpy): Images to be rectified with size
                :math:`(C, H, W)`.

        Returns:
            Tensor: Skeleton map with size :math:`(1, H, W)`.
    """

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)
    img_bg = np.zeros(gray_img.shape, dtype="uint8")

    #0, 1 array
    mask = clusterpixels(gray_img, 2).astype(np.uint8)
    polys = trace_skeleton.from_numpy(mask, csize, maxIter)

    # mask = thinning(mask)

    # rects = []
    # polys = traceSkeleton(mask, 0, 0, mask.shape[1], mask.shape[0], csize, maxIter, rects)
    try:
        for poly in polys:
            poly = np.int0(poly)
            for p in poly:
                x,y = p.ravel()
                # print(x,y)
                img_bg[y,x] = 1
    except TypeError:
        print('No poly detected!')
    # print("-------------------")

    return img_bg

def corner_detector(img, maxCorners, qualityLevel, minDistance):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)
    img_bg = np.zeros(gray_img.shape, dtype="uint8")

    corners = cv2.goodFeaturesToTrack(gray_img, maxCorners, qualityLevel, minDistance)
    try:
        corners = np.int0(corners)
        for corner in corners:
            x,y = corner.ravel()
            # print(x,y)
            img_bg[y,x] = 1
    except TypeError:
        print('No corner detected!')
    # print("-------------------")
    return img_bg


class MapPreprocessor(BasePreprocessor):

    def __init__(self, map_type = "corner"):
        """
            map_type: "corner", "cluster_skeleton"
        """
        super().__init__()
        self.map_type = map_type
        if map_type == "corner":
            self.maxCorners = 200
            self.qualityLevel = 0.01
            self.minDistance = 3
        elif map_type == "cluster_skeleton":
            self.csize = 10
            self.maxIter = 999

    def forward(self, batch_img):
        """
        Args:
            batch_img (paddle.Tensor): Images to be rectified with size
                :math:`(N, C, H, W)`.

        Returns:
            paddle.Tensor: Corner map with size :math:`(N, 1, H, W)`.
        """
        device = batch_img.place
        img_np = batch_img.numpy()
        batch_map = []
        for i in range(img_np.shape[0]):
            
            sin_img = img_np[i].transpose(1,2,0) * 255
            img_bg = np.zeros(sin_img.shape[:2], dtype="uint8")
            if self.map_type == "corner":
                img_bg = corner_detector(sin_img, self.maxCorners, self.qualityLevel, self.minDistance)
            
            elif self.map_type == "cluster_skeleton":
                img_bg = cluster_skeleton_detector(sin_img, self.csize, self.maxIter)
           
            map = np.expand_dims(img_bg, axis=0).astype(np.float32)
            batch_map.append(map)

        batch_map = np.concatenate(batch_map, axis=0)
        batch_map = paddle.to_tensor(batch_map, place=device)

        # device = batch_img.device
        # img_np = batch_img.cpu().numpy()
        # batch_map = torch.Tensor()
        # for i in range(img_np.shape[0]):
            
        #     sin_img = img_np[i].transpose(1,2,0) * 255
        #     img_bg = np.zeros(sin_img.shape[:2], dtype="uint8")
        #     if self.map_type == "corner":
        #         img_bg = corner_detector(sin_img, self.maxCorners, self.qualityLevel, self.minDistance)
            
        #     elif self.map_type == "cluster_skeleton":
        #         img_bg = cluster_skeleton_detector(sin_img, self.csize, self.maxIter)
           
        #     mask = torch.tensor(img_bg).unsqueeze(0).unsqueeze(0)
        #     mask = mask.to(torch.float32)
        #     batch_map = torch.cat([batch_map, mask], dim=0)

        # batch_map = batch_map.to(device)

        return batch_map