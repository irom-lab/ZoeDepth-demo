# Test monocular depth estimation
# Converts image from an iPhone to a Kinect's intrinsic matrix and then estimates depth
# Kinect is the camera used for the NYU dataset

import torch
import time

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.geometry as geometry
import numpy as np
import cv2

# import IPython as ipy

##################################
# Local file
from PIL import Image
image = Image.open("./test_img2.jpg").convert("RGB")  # load


##################################
# Convert image to Kinect's intrinsic matrix

# Kinect intrinsic matrix
p = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
M_prime = p.intrinsic_matrix # Kinect intrinsic matrix
# M_prime = np.array([[5.1885790117450188e+02, 0.00000000e+00, 3.2558244941119034e+02],[0.00000000e+00, 5.1946961112127485e+02, 2.5373616633400465e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# iPhone X intrinsic matrix (I believe this is if you're holding the phone upright)
# Source: https://stackoverflow.com/questions/50402026/what-is-the-intrinsicmatrix-for-an-iphone-x-rear-camera
M = np.array([[3.20512987e+03, 0.00000000e+00, 1.99443897e+03],[0.00000000e+00, 3.17391061e+03, 1.41309060e+03],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# Affine transformation matrix
M_trans = M_prime @ np.linalg.inv(M)

# Transform image
image = cv2.warpAffine(np.asarray(image), M_trans[0:2,:], (640,480)) # 640 x 480 is Kinect's resolution
# plt.imshow(image); plt.show()


##################################
# # Model: ZoeD_NK (fine-tuned on KITTI and NYU)
# conf = get_config("zoedepth_nk", "infer")
# model_zoe = build_model(conf)

# # ZoeD_N (fine-tuned on NYU)
conf = get_config("zoedepth", "infer")
model_zoe = build_model(conf)


##################################
# Prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe.to(DEVICE)

t_start = time.time()
depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor
t_end = time.time()
print("Time taken (tensor): ", t_end - t_start)

# Convert to numpy
depth_numpy = depth_tensor.numpy()

plt.imshow(depth_numpy)
plt.show()

##################################
# Camera params
width = depth_tensor.shape[1]
height = depth_tensor.shape[0]

# Kinect
params = o3d.camera.PinholeCameraIntrinsic(width, height, M_prime[0][0], M_prime[1][1], M_prime[0][2], M_prime[1][2])

##################################
# Convert to open3d
depth_o3d = geometry.Image(depth_numpy)
color_o3d = geometry.Image(np.asarray(image))
rgbd = geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d)

##################################
# Point cloud
# pc = geometry.PointCloud.create_from_depth_image(depth_o3d, params, depth_scale=1.0, depth_trunc=1000.0, stride=1, project_valid_depth_only=True)
pc = geometry.PointCloud.create_from_rgbd_image(rgbd, params)
o3d.visualization.draw_geometries([pc])

