import pyexr
import cv2
import numpy as np

ao = pyexr.open('../../../cleargrasp-dataset-train/cup-with-waves-train/depth-imgs-rectified/000000000-depth-rectified.exr').get()[:, :, 0]
ao = ao*255.

ao = np.clip(ao, 0, 255)
ao = ao.astype(np.uint8)
print(ao.shape)
cv2.imwrite('../../../cleargrasp-dataset-train/cup-with-waves-train/utils_exr_2_png/00001.png', ao)