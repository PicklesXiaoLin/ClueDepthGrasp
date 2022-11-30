import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageStat
from io import BytesIO
import random
import matplotlib.pyplot as plt
import cv2

import PIL.ImageEnhance
def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, depth,mask,normal, outline = sample['image'], sample['depth'],sample['mask'], sample['normal'], sample['outline']

        if not _is_pil_image(image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if not _is_pil_image(mask):
            raise TypeError('img should be PIL Image. Got {}'.format(type(mask)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            normal = normal.transpose(Image.FLIP_LEFT_RIGHT)
            outline = outline.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth, 'mask':mask, 'normal':normal, 'outline':outline}

class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth, mask, normal, outline = sample['image'], sample['depth'], sample['mask'], sample['normal'], sample['outline']
        if not _is_pil_image(image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if not _is_pil_image(mask):
            raise TypeError('img should be PIL Image. Got {}'.format(type(mask)))
        if not _is_pil_image(normal):
            raise TypeError('img should be PIL Image. Got {}'.format(type(normal)))
        if not _is_pil_image(outline):
            raise TypeError('img should be PIL Image. Got {}'.format(type(outline)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[..., list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth, 'mask':mask, 'normal':normal, 'outline':outline}


def get_image_light_mean(img):
    im = img.convert('L')
    stat = ImageStat.Stat(im)
    return stat.mean[0]


def random_gamma_transform(img, gamma):
    if get_image_light_mean(img) <= 30:
        return img
    res = PIL.ImageEnhance.Brightness(img).enhance(gamma)
    if get_image_light_mean(res) <= 30:
        return img
    return res


class RandomGammaTransform(object):
    def __init__(self, bright_low=0.5, bright_high=1.5):
        self.bright_low = bright_low
        self.bright_high = bright_high

    def __call__(self, sample):
        image, depth, mask, normal, outline = sample['image'], sample['depth'], sample['mask'], sample['normal'], \
                                              sample['outline']
        if not _is_pil_image(image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if not _is_pil_image(mask):
            raise TypeError('img should be PIL Image. Got {}'.format(type(mask)))
        if not _is_pil_image(normal):
            raise TypeError('img should be PIL Image. Got {}'.format(type(normal)))
        if not _is_pil_image(outline):
            raise TypeError('img should be PIL Image. Got {}'.format(type(outline)))

        bright = np.random.uniform(self.bright_low, self.bright_high)
        image = random_gamma_transform(image, bright)
        return {'image': image, 'depth': depth, 'mask': mask, 'normal':normal, 'outline':outline}


class ToTensor(object):
    def __init__(self, is_test=False, is_real=False):
        self.is_test = is_test
        self.is_real = is_real
        #left = np.ones(shape=(480, 320), dtype=np.float32)
        #right = np.zeros(shape=(480, 320), dtype=np.float32)
        #self.mask = np.concatenate([left, right], axis=1) + 1e-10

        # self.eps = np.concatenate([left, right], axis=1)

    def __call__(self, sample):

        image, depth, mask, normal, outline = sample['image'], sample['depth'], sample['mask'], sample['normal'], sample['outline']
        if self.is_real :
            gt_depth = sample['gt_depth']
            land = sample['land']
            gt_depth = np.asarray(gt_depth)
            land = np.asarray(land)

        image = np.asarray(image)
        depth = np.asarray(depth) ## add
        mask  = np.asarray(mask)
        normal = np.asarray(normal)
        outline = np.asarray(outline)
        another_mask = np.asarray(1-mask)  #1110011
        
        # image = image[16:-16, 16:-16, :]
        # image = self.to_tensor(image)
        #plt.imshow(mask[1])
        #plt.imshow(mask[2])
        # depth = depth.resize((320, 240))
        #mask = np.where(mask==0,1,0)
        # plt.subplot(221),plt.imshow(image)
        # plt.subplot(222),plt.imshow(depth)
        # plt.subplot(223),plt.imshow(mask)

        # input_depth = np.asarray(depth.copy())*(1-mask)

        # plt.subplot(224),plt.imshow(input_depth)
        #plt.show()
        # input = np.concatenate([image, np.expand_dims(input_depth, axis=-1)], axis=-1)
        input = np.concatenate([image, np.expand_dims(depth, axis=-1)], axis=-1)
        input = np.concatenate([input, np.expand_dims(normal, axis=-1)], axis=-1)
        input = np.concatenate([input, np.expand_dims(outline, axis=-1)], axis=-1)
        input = np.concatenate([input, np.expand_dims(mask, axis=-1)], axis=-1)
        input = np.concatenate([input, np.expand_dims(another_mask, axis=-1)], axis=-1)
    
        if self.is_real:
            input = np.concatenate([input, np.expand_dims(gt_depth, axis=-1)], axis=-1)
            input = np.concatenate([input, np.expand_dims(land, axis=-1)], axis=-1)

            
        #input = input[16:-16, 16:-16, :]
        input = input[8:-8,:,:]
        input = self.to_tensor(input.copy())
        # input = self.to_tensor(input)
        
        if self.is_real:
            depth = gt_depth[8:-8,:]
            # depth_single = depth[:,:]
            
        else:
            depth = depth[8:-8,:]

        #plt.imshow(input_depth)
        #plt.show()
        depth = np.expand_dims(depth, axis=-1)

        #depth = self.to_tensor(depth).float()
        ###############
        if self.is_test:
            # depth = self.to_tensor(depth.copy()).float() / 400
            depth = self.to_tensor(depth.copy()).float() 

        else:
            # depth = torch.from_numpy(depth.transpose((2, 0, 1))).float().div(255).float()*400
            # depth = self.to_tensor(depth.copy()).float() * 400
            depth = self.to_tensor(depth.copy()).float() 
            
        # depth_single_2 = depth[0,:,:]
        # plt.subplot(212),plt.imshow(depth_single_2)
        # plt.show()
        #######
        #depth = self.to_tensor(depth).float() * 100

        # print(np.max(depth.cpu()))
        # print(np.min(depth.cpu()))
        # put in expected range
        # depth = torch.clamp(depth, 10, 1000)
        depth = torch.clamp(depth, 0., 1000/255)

        #print(np.max(np.array(depth[0])))
        #print(np.max(depth[0]))
        if self.is_real:
            return {'image': input, 'depth': depth, 'num': 1}
        return {'image': input, 'depth': depth}


    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose(2, 0, 1))
            # img = torch.Tensor(pic.transpose(2, 0, 1))
            img = img.float().div(255)
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            print("2")
            return img.float().div(255)
        else:
            print("3")
            return img


def loadZipToMem(zip_file):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}

    # nyu2_train = list((row.split(',') for row in (data['csv/clear_test2.csv']).decode("utf-8").split('\r\n') if len(row) > 0))
    nyu2_train = list((row.split(',') for row in (data['csv/clear_train2.csv']).decode("utf-8").split('\r\n') if len(row) > 0))
    nyu2_test  = list((row.split(',') for row in (data['csv/clear_test2.csv']).decode("utf-8").split('\r\n')  if len(row) > 0)) # test sys
    # real_known 0-59 60-119 120-172 ==> test-val-real_val
    nyu2_val  = list((row.split(',') for row in (data['csv/clear_val2.csv']).decode("utf-8").split('\r\n')  if len(row) > 0)) # val sys
    nyu2_real_test  = list((row.split(',') for row in (data['csv/real_test_415.csv']).decode("utf-8").split('\r\n')  if len(row) > 0))
    nyu2_real_val  = list((row.split(',') for row in (data['csv/real_val.csv']).decode("utf-8").split('\r\n')  if len(row) > 0))
    # real_novel 0-59 60-112 ==> real_test\test2

    # nyu2_real_test2  = list((row.split(',') for row in (data['csv/real_test_415_tree.csv']).decode("utf-8").split('\r\n')  if len(row) > 0))
    

    from sklearn.utils import shuffle
    nyu2_train = shuffle(nyu2_train, random_state=0)
#    nyu2_test  = shuffle(nyu2_test,  random_state=0)
#    nyu2_val  = shuffle(nyu2_val,  random_state=0)
#    nyu2_real_val  = shuffle(nyu2_real_val,  random_state=0)
#    nyu2_real_test  = shuffle(nyu2_real_test,  random_state=0)
    
    #if True: nyu2_train = nyu2_train[:40]
    print()
    print('Loaded train({0}).'.format(len(nyu2_train)))
    print('Loaded val({0}).'.format(len(nyu2_test)))
    print('Loaded test({0}).'.format(len(nyu2_val)))
    print('Loaded real_val({0}).'.format(len(nyu2_real_val)))
    print('Loaded real_test({0}).'.format(len(nyu2_real_test)))
    # print('Loaded real_test2({0}).'.format(len(nyu2_real_test2)))

    return data, nyu2_train, nyu2_test, nyu2_val, nyu2_real_val, nyu2_real_test
    # return data, nyu2_train, nyu2_test, nyu2_val


class depthDatasetMemory(Dataset):
    def __init__(self, data, nyu2_train, transform=None, is_real=False):
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform
        self.is_real = is_real

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        image = Image.open(BytesIO(self.data[sample[0]]))
        depth = Image.open(BytesIO(self.data[sample[1]]))
        mask  = Image.open(BytesIO(self.data[sample[2]]))
        normal = Image.open(BytesIO(self.data[sample[3]]))
        outline = Image.open(BytesIO(self.data[sample[4]]))
        if self.is_real:
            gt_depth = Image.open(BytesIO(self.data[sample[5]]))
            land = Image.open(BytesIO(self.data[sample[6]]))
            sample = {'image': image, 'depth': depth, 'mask':mask, 'normal':normal, 'outline':outline, 'gt_depth':gt_depth, 'land':land}
        else:
            sample = {'image': image, 'depth': depth, 'mask':mask, 'normal':normal, 'outline':outline}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.nyu_dataset)


def getNoTransform(is_test=False, is_real=False):
    return transforms.Compose([ToTensor(is_test=is_test, is_real=is_real)])


def getDefaultTrainTransform():
    return transforms.Compose([RandomHorizontalFlip(), RandomChannelSwap(0.5), ToTensor()])


def getTrainingTestingData(batch_size):
    #data, nyu2_train, nyu2_test = loadZipToMem('datazip/Big_data_clear_val.zip')
    # data, nyu2_train, nyu2_test, nyu2_val, nyu2_real_val, nyu2_real_test = loadZipToMem('datazip/Big_clear_real_ttr_all_fourth_only4.zip')
    data, nyu2_train, nyu2_test, nyu2_val, nyu2_real_val, nyu2_real_test = loadZipToMem('datazip/Big_data_land_four.zip')
    # data, nyu2_train, nyu2_test, nyu2_val = loadZipToMem('datazip/Big_data_clear_train+test_p.zip')

    transformed_training = depthDatasetMemory(data, nyu2_train, transform=getDefaultTrainTransform())
    transformed_testing = depthDatasetMemory(data, nyu2_test, transform=getNoTransform())
    transformed_valing = depthDatasetMemory(data, nyu2_val, transform=getNoTransform())
    transformed_real_valing = depthDatasetMemory(data, nyu2_real_val, transform=getNoTransform(is_real=True),is_real=True)
    transformed_real_testing = depthDatasetMemory(data, nyu2_real_test, transform=getNoTransform(is_real=True),is_real=True)
    # transformed_real_testing2 = depthDatasetMemory(data, nyu2_real_test, transform=getNoTransform(is_real=True),is_real=True)

    #transformed_testing2 = depthDatasetMemory(data, nyu2_test, transform=getNoTransform())
    #, num_workers = 2  , pin_memory=True
    return DataLoader(transformed_training, batch_size, shuffle=True,num_workers = 2, pin_memory=True), \
           DataLoader(transformed_testing, batch_size, shuffle=False,num_workers = 2 ),\
           DataLoader(transformed_valing, batch_size,  shuffle=False,num_workers = 2), \
           DataLoader(transformed_real_valing, batch_size,shuffle=False), \
           DataLoader(transformed_real_testing, batch_size,shuffle=False)
           # DataLoader(transformed_real_testing2, batch_size)


def getTestingData(batch_size):
    data, nyu2_train = loadZipToMem('datazip/Big_data_clear_train+test_p_big.zip')
   # data, nyu2_train = loadZipToMem('datazip/Big_data_clear_val.zip')

    transformed_testing = depthDatasetMemory(data, nyu2_train, transform=getNoTransform())

    return DataLoader(DataLoader(transformed_testing, batch_size, shuffle=True))