import matplotlib
import matplotlib.cm
import numpy as np
import torch
import cv2
import struct
import open3d as o3d
def DepthNorm_plus(depth):
    maxDepth = []
    depth = depth.cpu()
    depth = np.array(depth)

    for idx in range(0,4):
        maxDepth.append( np.max(depth[idx][0]) )
        depth[idx][0] = depth[idx][0] / maxDepth[idx]

    depth = torch.from_numpy(depth)
    depth = torch.autograd.Variable(depth.cuda(non_blocking=True))
    return  depth

def DepthNorm_grad(depth, maxDepth = 1000):
    maxDepth = []
    depth = depth.cpu()
    depth = np.array(depth.detach())

    for idx in range(0,4):
        maxDepth.append( np.max(depth[idx][0]) )
        depth[idx][0] = depth[idx][0] / maxDepth[idx]

    depth = torch.from_numpy(depth)
    depth = torch.autograd.Variable(depth.cuda(non_blocking=True))
    return  depth

def DepthNorm(depth, maxDepth=1000):
    return  maxDepth / depth

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def colorize(value, vmin=500, vmax=40000, cmap='jet'):
#     value = value.cpu().numpy()[0, :, :]
#
#     vmax = None
#     vmin = None
#     # normalize
#     vmin = value.min() if vmin is None else vmin
#     vmax = value.max() if vmax is None else vmax
#     if vmin != vmax:
#         value = (value - vmin) / (vmax - vmin)  # vmin..vmax
#     else:
#         # Avoid 0-division
#         value = value*0.
#     # squeeze last dim if it exists
#     #value = value.squeeze(axis=0)
#
#     cmapper = matplotlib.cm.get_cmap(cmap)
#     value = cmapper(value,bytes=True)  # (nxmx4)
#
#     img = value[:, :, :3]
#
#     return img.transpose((2, 0, 1))

import matplotlib.pyplot as plt
def colorize(value, vmin=0, vmax=0.65, cmap='jet',real = False): # 10 1000
    #plasma - > jet
    value = value.cpu().numpy()[0, :, :]
    if real:
        value[0,0] = 0
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    #value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True)  # (nxmx4)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))

def colorize_old(input, ddc, ours, cmap='jet'):
    input = input.cpu().numpy()
    ddc = ddc.cpu().numpy().squeeze()
    ours = ours.cpu().numpy().squeeze()

    rgb = input[:, :3]*255.
    rgb = rgb.astype(np.uint8)
    raw = input[:, 3]
    out = []
    for i in range(rgb.shape[0]):
        c = np.transpose(rgb[i], (1, 2, 0))
        r = raw[i]
        d = ddc[i]
        o = ours[i]
        res = np.concatenate([r, d, o], axis=1)
        min_ = np.min(res)
        max_ = np.max(res)
        res = (res-min_)/(max_-min_)
        # plt.imshow(res.squeeze(), cmap='jet')
        # plt.show()
        cmapper = matplotlib.cm.get_cmap(cmap)
        res = cmapper(res,bytes=True)  # (nxmx4)
        res = np.concatenate([c, res[:, :, :3]], axis=1)
        out.append(res)
        # plt.imshow(res.squeeze())
        # plt.show()
    out = np.concatenate(out, axis=0)
    # plt.imshow(out)
    # plt.show()

    # value = value.cpu().numpy()[0, :, :]
    #
    # vmax = None
    # vmin = None
    # # normalize
    # vmin = value.min() if vmin is None else vmin
    # vmax = value.max() if vmax is None else vmax
    # if vmin != vmax:
    #     value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    # else:
    #     # Avoid 0-division
    #     value = value*0.
    # # squeeze last dim if it exists
    # #value = value.squeeze(axis=0)
    #
    # cmapper = matplotlib.cm.get_cmap(cmap)
    # value = cmapper(value,bytes=True)  # (nxmx4)
    #
    # img = value[:, :, :3]

    return out

def _normalize_depth_img(depth_img, dtype=np.uint8, min_depth=0.0, max_depth=1.0):
    '''Converts a floating point depth image to uint8 or uint16 image.
    The depth image is first scaled to (0.0, max_depth) and then scaled and converted to given datatype.

    Args:
        depth_img (numpy.float32): Depth image, value is depth in meters
        dtype (numpy.dtype, optional): Defaults to np.uint16. Output data type. Must be np.uint8 or np.uint16
        max_depth (float, optional): The max depth to be considered in the input depth image. The min depth is
            considered to be 0.0.
    Raises:
        ValueError: If wrong dtype is given

    Returns:
        numpy.ndarray: Depth image scaled to given dtype
    '''

    if dtype != np.uint16 and dtype != np.uint8:
        raise ValueError('Unsupported dtype {}. Must be one of ("np.uint8", "np.uint16")'.format(dtype))

    # Clip depth image to given range
    depth_img = np.ma.masked_array(depth_img, mask=(depth_img == 0.0))
    depth_img = np.ma.clip(depth_img, min_depth, max_depth)

    # Get min/max value of given datatype
    type_info = np.iinfo(dtype)
    min_val = type_info.min
    max_val = type_info.max

    # Scale the depth image to given datatype range
    depth_img = ((depth_img - min_depth) / (max_depth - min_depth)) * max_val
    depth_img = depth_img.astype(dtype)

    depth_img = np.ma.filled(depth_img, fill_value=0)  # Convert back to normal numpy array from masked numpy array

    return depth_img

def depth2rgb(depth_img, min_depth=0.0, max_depth=1.5, color_mode=cv2.COLORMAP_JET, reverse_scale=False,
              dynamic_scaling=False):
    '''Generates RGB representation of a depth image.
    To do so, the depth image has to be normalized by specifying a min and max depth to be considered.

    Holes in the depth image (0.0) appear black in color.

    Args:
        depth_img (numpy.ndarray): Depth image, values in meters. Shape=(H, W), dtype=np.float32
        min_depth (float): Min depth to be considered
        max_depth (float): Max depth to be considered
        color_mode (int): Integer or cv2 object representing Which coloring scheme to use.
                          Please consult https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html

                          Each mode is mapped to an int. Eg: cv2.COLORMAP_AUTUMN = 0.
                          This mapping changes from version to version.
        reverse_scale (bool): Whether to make the largest values the smallest to reverse the color mapping
        dynamic_scaling (bool): If true, the depth image will be colored according to the min/max depth value within the
                                image, rather that the passed arguments.
    Returns:
        numpy.ndarray: RGB representation of depth image. Shape=(H,W,3)
    '''
    # Map depth image to Color Map
    if dynamic_scaling:
        depth_img_scaled = _normalize_depth_img(depth_img, dtype=np.uint8,
                                                min_depth=max(depth_img[depth_img > 0].min(), min_depth),    # Add a small epsilon so that min depth does not show up as black (invalid pixels)
                                                max_depth=min(depth_img.max(), max_depth))
    else:
        depth_img_scaled = _normalize_depth_img(depth_img, dtype=np.uint8, min_depth=min_depth, max_depth=max_depth)

    if reverse_scale is True:
        depth_img_scaled = np.ma.masked_array(depth_img_scaled, mask=(depth_img_scaled == 0.0))
        depth_img_scaled = 255 - depth_img_scaled
        depth_img_scaled = np.ma.filled(depth_img_scaled, fill_value=0)

    depth_img_mapped = cv2.applyColorMap(depth_img_scaled, color_mode)
    depth_img_mapped = cv2.cvtColor(depth_img_mapped, cv2.COLOR_BGR2RGB)

    # Make holes in input depth black:
    depth_img_mapped[depth_img_scaled == 0, :] = 0

    return depth_img_mapped


# def write_point_cloud(filename, color_image, depth_image, fx=370, fy=370, cx=256, cy=144):
#     """Creates and Writes a .ply point cloud file using RGB and Depth images.
#
#     Args:
#         filename (str): The path to the file which should be written. It should end with extension '.ply'
#         color image (numpy.ndarray): Shape=[H, W, C], dtype=np.uint8
#         depth image (numpy.ndarray): Shape=[H, W], dtype=np.float32. Each pixel contains depth in meters.
#         fx (int): The focal len along x-axis in pixels of camera used to capture image.
#         fy (int): The focal len along y-axis in pixels of camera used to capture image.
#         cx (int): The center of the image (along x-axis, pixels) as per camera used to capture image.
#         cy (int): The center of the image (along y-axis, pixels) as per camera used to capture image.
#     """
#     xyz_points, rgb_points = _get_point_cloud(color_image, depth_image, fx, fy, cx, cy)
#
#     # Write header of .ply file
#     with open(filename, 'wb') as fid:
#         fid.write(bytes('ply\n', 'utf-8'))
#         fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
#         fid.write(bytes('element vertex %d\n' % xyz_points.shape[0], 'utf-8'))
#         fid.write(bytes('property float x\n', 'utf-8'))
#         fid.write(bytes('property float y\n', 'utf-8'))
#         fid.write(bytes('property float z\n', 'utf-8'))
#         fid.write(bytes('property uchar red\n', 'utf-8'))
#         fid.write(bytes('property uchar green\n', 'utf-8'))
#         fid.write(bytes('property uchar blue\n', 'utf-8'))
#         fid.write(bytes('end_header\n', 'utf-8'))
#
#         # Write 3D points to .ply file
#         for i in range(xyz_points.shape[0]):
#             fid.write(
#                 bytearray(
#                     struct.pack("fffccc", xyz_points[i, 0], xyz_points[i, 1], xyz_points[i, 2],
#                                 rgb_points[i, 0].tostring(), rgb_points[i, 1].tostring(), rgb_points[i, 2].tostring())))
#
#
# write_point_cloud("ply/test1.ply", color_image, depth_image, fx=370, fy=370, cx=256, cy=144)