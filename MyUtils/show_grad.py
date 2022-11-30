import cv2
import matplotlib.pyplot as plt
import numpy as np
depth = cv2.imread('../dataset1/00001-depth.png', -1).astype(np.float32)
# color = cv2.imread('E:\Projects\Datasets\\NyuV2Dataset\\0001-color.jpg')[:, :, (2,1,0)]
condition = 1400
depth = np.where(depth>condition, condition, depth)#np.where(condition,x,y) 当where内有三个参数时

# mask = depth > 0

# valid = depth[mask]
# c, d = load_h5(path)
grad_x = depth[1:, :] - depth[:-1,:]
grad_y = depth[:,1:] - depth[:,:-1]
grad_x = np.where(grad_x!=0, 255, 0)
grad_y = np.where(grad_y!=0, 255, 0)
plt.subplot(131), plt.imshow(depth, cmap='gray')
plt.subplot(132), plt.imshow(1-grad_x, cmap='gray')
plt.subplot(133), plt.imshow(grad_y, cmap='gray')
plt.show()