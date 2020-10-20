"""
Expected formats:
Image: 1-channel, 8-bit (0->255), TIF
Mask: 4-channel, data channel -> 0th, PNG
Ice_Conc: 1-channel, 0->100 % ice concentration, numpy.ndarray, NPY

Output:
sample{'image': channel 0 (image patch, 0->1), channel 1 (ice_conc_patch, 0->1)
       'mask' : mask_patch [0,1]
"""

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

import os
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2


def get_patch(img, mask, patch_size):
    """
    Returns a patch such that patch does not contain more than 50% black pixels. 
    A black pixel means that the area is 
    """
    img_x, img_y = img.shape
    
    patch_x = int((img_x - patch_size) * random.random())
    patch_y = int((img_y - patch_size) * random.random())
        
    img_patch = img[patch_x: (patch_x+patch_size) ,patch_y:(patch_y+patch_size)]
    mask_patch = mask[patch_x: (patch_x+patch_size) ,patch_y:(patch_y+patch_size)]
    
    valid_patch = np.sum(img_patch==0) < (0.5 * patch_size**2)
#     print(valid_patch)
        
    if(valid_patch == False):
        img_patch, mask_patch = get_patch(img, mask, patch_size)
        
    return img_patch, mask_patch  


class DatasetFloe(Dataset):
    """
    Data loader for RESUNET CRF, 
    OUPTUT: Image: 0->255, Mask: [0,1]
    """
    
    def __init__(self, patch_size: int, batch_size: int, mode = 'train'):
        self.path_images = '../data/images/' + mode
        self.path_masks = '../data/annotation_masks/'
        self.batch_size = batch_size
        self.patchsize = patch_size
        self.img_names = os.listdir(self.path_images)
        
    def transform(self, img, mask):
        
        image = TF.to_pil_image(img)
        mask = TF.to_pil_image(mask)

        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask= TF.hflip(mask)
            
        # Random Vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            
        # To Tensor
        image = TF.to_tensor(image) * 255
        mask_fb = TF.to_tensor(mask).float()
        mask_bg = (TF.to_tensor(mask)!=1).float()
        
        # Stack image and ice so that network accepts them as two channels.
        image = torch.stack((image.squeeze(), image.squeeze(), image.squeeze()), dim = 0)
        mask = torch.stack((mask_bg.squeeze(), mask_fb.squeeze()), dim = 0)
        
        return image, mask
            
        
    def __len__(self):
        return self.batch_size # We are generating images on the go.
    
    def __getitem__(self, index):
        file_name = random.choice(self.img_names)[:-4]

        img = plt.imread(os.path.join(self.path_images,file_name + '.tif'))
        mask = cv2.imread(os.path.join(self.path_masks,file_name + '.png'))
        
        mask = (mask[:,:,2] / 128).astype(np.int32)
        
#         patch_size = self.patchsize
#         img_x, img_y = img.shape
        
#         patch_x = int((img_x - self.patchsize) * random.random())
#         patch_y = int((img_y - self.patchsize) * random.random())
        
#         img_patch = img[patch_x: (patch_x+self.patchsize) ,patch_y:(patch_y+self.patchsize)]
#         mask_patch = mask[patch_x: (patch_x+self.patchsize) ,patch_y:(patch_y+self.patchsize)]
# #         ice_conc_patch = ice_conc[patch_x: (patch_x+self.patchsize) ,patch_y:(patch_y+self.patchsize)]
        
        img_patch, mask_patch = get_patch(img, mask, self.patchsize)
        
        img_patch, mask_patch = self.transform(img_patch, mask_patch)
        sample = {'image': img_patch, 'mask': mask_patch}

        return sample
    
    
class DatasetFloeVal(Dataset):
    """
    Data loader for RESUNET CRF, 
    OUPTUT: Image: 0->255, Mask: [0,1]
    """
    
    def __init__(self, patch_size: int):
        self.path_images = '../data/val/' + str(patch_size) + '/img_patches/'
        self.path_masks = '../data/val/' + str(patch_size) + '/mask_patches/'
        self.patchsize = patch_size
        self.img_names = os.listdir(self.path_images)
        self.img_names.sort()
        
    def transform(self, img, mask):
        
        image = TF.to_pil_image(img)
        mask = TF.to_pil_image(mask)
            
        # To Tensor
        image = TF.to_tensor(image) * 255
        mask_fb = TF.to_tensor(mask).float()
        mask_bg = (TF.to_tensor(mask)!=1).float()
        
        mask = torch.stack((mask_bg.squeeze(), mask_fb.squeeze()), dim = 0)
        
        return image, mask
            
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        img = plt.imread(os.path.join(self.path_images,self.img_names[index]))
        
        mask = plt.imread(os.path.join(self.path_masks,self.img_names[index]))
        
        mask = (mask[:,:,0]).astype(np.int32)
                
        img_patch, mask_patch = self.transform(img, mask)
        sample = {'image': img_patch, 'mask': mask_patch}

        return sample


##############           DEPRECATED, May work             USE IN CASE YOU WANT TO TRAIN WITH ICE CONCENTRATION MASKS AS WELL       ###########################
class DatasetFloe_Ice_Mask(Dataset):
    def __init__(self, patch_size, mode):
        self.path_images = '../data/images/' + mode
        self.path_masks = '../data/annotation_masks/'
        self.path_ice_conc = '../data/ice_conc/'
        self.patchsize = patch_size
        self.img_names = os.listdir(self.path_images)
        
    def transform(self, img, mask, ice):
        
        image = TF.to_pil_image(img)
        mask = TF.to_pil_image(mask)
        ice = TF.to_pil_image(ice.astype(np.float32)/100)
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask= TF.hflip(mask)
            ice = TF.hflip(ice)
            
        # Random Vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            ice = TF.vflip(ice)
            
        # To Tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        ice = TF.to_tensor(ice)
        
        # Stack image and ice so that network accepts them as two channels.
        image_stacked = torch.stack((image.squeeze(), ice.squeeze()), dim = 0)
        
        return image_stacked, mask
            
        
    def __len__(self):
        return 100 # We are generating images on the go.
    
    def __getitem__(self, index):
        file_name = random.choice(self.img_names)[:-4]

        img = plt.imread(os.path.join(self.path_images,file_name + '.tif'))
        mask = cv2.imread(os.path.join(self.path_masks,file_name + '.png'))
        ice_conc = np.load(os.path.join(self.path_ice_conc,file_name + '.npy'))
        
        mask = (mask[:,:,2] / 128).astype(np.int32)
        
        img_x, img_y = img.shape
        
        patch_x = int((img_x - self.patchsize) * random.random())
        patch_y = int((img_y - self.patchsize) * random.random())
        
        img_patch = img[patch_x: (patch_x+self.patchsize) ,patch_y:(patch_y+self.patchsize)]
        mask_patch = mask[patch_x: (patch_x+self.patchsize) ,patch_y:(patch_y+self.patchsize)]
        ice_conc_patch = ice_conc[patch_x: (patch_x+self.patchsize) ,patch_y:(patch_y+self.patchsize)]
        
        img_patch, mask_patch = self.transform(img_patch, mask_patch, ice_conc_patch)
        
        sample = {'image': img_patch, 'mask': mask_patch}
        
        return sample

###############                    USELESS                     #####################3
# class DatasetValidateFloe(Dataset):
#     """
#     Expecting: image_patches be 4-channel TIF, 8-bit
#              : mask_patches be 4-channel TIF, 8-bit
#              : ice_con_patches be 1-channel TIF, 0->120, 120 means land, 0->100 is the ice concentration
#     """
    
#     def __init__(self):
#         self.path_images = './data/valid_premade_patches_multi/image_patches/'
#         self.path_masks = './data/valid_premade_patches_multi/mask_patches/'
#         self.path_ice_conc = './data/valid_premade_patches_multi/con_patches/'
#         self.file_names = os.listdir(self.path_images)
        
#     def transform(self, img, mask, ice):
#         # To Tensor
#         img = TF.to_tensor(img)
#         mask = TF.to_tensor(mask)
#         ice = TF.to_tensor((np.array(ice) != 120).astype(int) *ice) 
#         # Ice concentration patches: land_mass = 120
#         #                          : ice_conc = 0 -> 100

#         img = img.float()/255   # 0->1
#         mask = torch.round(mask) # [0,1]
#         ice = ice.float()/100 # 0->1

#         img = torch.stack((img.squeeze(), ice.squeeze()), dim = 0) # all img, mask, ice as_type(float32)
        
#         return img, mask
    
#     def __len__(self):
#         return(len(self.file_names))
    
#     def __getitem__(self, index):
#         img = Image.open(self.path_images + self.file_names[index])
#         mask = Image.open(self.path_masks + self.file_names[index])
#         ice = Image.open(self.path_ice_conc + self.file_names[index])
        
        
#         img, mask = self.transform(img, mask, ice)
#         sample = {'image': img, 'mask': mask}
#         return sample
        
#         img, mask = self.transform(img, mask, ice)
#         sample = {'image': img, 'mask': mask}
#         return sample

class DL_UNET_1C(Dataset):
    
    """
    Dataloader: UNET Single Channel
    """
    def __init__(self, patch_size, mode):
        self.path_images = '../data/images/' + mode
        self.path_masks = '../data/annotation_masks/'

        self.patchsize = patch_size
        self.img_names = os.listdir(self.path_images)
        
    def transform(self, img, mask):
        
        image = TF.to_pil_image(img)
        mask = TF.to_pil_image(mask)
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask= TF.hflip(mask)
            
        # Random Vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            
        # To Tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask).float()
        
        # Stack image and ice so that network accepts them as two channels.
        image = torch.stack((image.squeeze(), image.squeeze(), image.squeeze()), dim = 0)
        
        return image, mask
            
        
    def __len__(self):
        return 8 # We are generating images on the go.
    
    def __getitem__(self, index):
        file_name = random.choice(self.img_names)[:-4]

        img = plt.imread(os.path.join(self.path_images,file_name + '.tif'))
        mask = cv2.imread(os.path.join(self.path_masks,file_name + '.png'))
        
        mask = (mask[:,:,2] / 128).astype(np.int32)
        
        img_x, img_y = img.shape
        
        patch_x = int((img_x - self.patchsize) * random.random())
        patch_y = int((img_y - self.patchsize) * random.random())
        
        img_patch = img[patch_x: (patch_x+self.patchsize) ,patch_y:(patch_y+self.patchsize)]
        mask_patch = mask[patch_x: (patch_x+self.patchsize) ,patch_y:(patch_y+self.patchsize)]
        
        img_patch, mask_patch = self.transform(img_patch, mask_patch)
        
        sample = {'image': img_patch, 'mask': mask_patch}
        
        return sample
    
#####################      DEEPLAB AND FCN DATALOADER       ###############################

"""
DONT USE. Modify the original DatasetFloe according to your utilisation
DATA LOADER FOR DEEPLABV3 AND FCN. required output is RGB image
"""
class OLD_DatasetFloe(Dataset):
    """
    DEPRECATED
    """
    def __init__(self, patch_size: int, mode : str):
        self.path_images = '../data/images/' + mode
        self.path_masks = '../data/annotation_masks/'

        self.patchsize = patch_size
        self.img_names = os.listdir(self.path_images)
        
    def transform(self, img, mask):
        
        image = TF.to_pil_image(img)
        mask = TF.to_pil_image(mask)
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask= TF.hflip(mask)
            
        # Random Vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            
        # To Tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask).float()
        
        # Stack image and ice so that network accepts them as two channels.
        image = torch.stack((image.squeeze(), image.squeeze(), image.squeeze()), dim = 0)
        
        return image, mask
            
        
    def __len__(self):
        return 8 # We are generating images on the go.
    
    def __getitem__(self, index):
        file_name = random.choice(self.img_names)[:-4]

        img = plt.imread(os.path.join(self.path_images,file_name + '.tif'))
        mask = cv2.imread(os.path.join(self.path_masks,file_name + '.png'))
        
        mask = (mask[:,:,2] / 128).astype(np.int32)
        
        img_x, img_y = img.shape
        
        patch_x = int((img_x - self.patchsize) * random.random())
        patch_y = int((img_y - self.patchsize) * random.random())
        
        img_patch = img[patch_x: (patch_x+self.patchsize) ,patch_y:(patch_y+self.patchsize)]
        mask_patch = mask[patch_x: (patch_x+self.patchsize) ,patch_y:(patch_y+self.patchsize)]
        
        img_patch, mask_patch = self.transform(img_patch, mask_patch)
        
        sample = {'image': img_patch, 'mask': mask_patch}
        
        return sample