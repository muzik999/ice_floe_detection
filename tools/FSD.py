# import os,sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir+'/model') 

from .metrics import get_centres
from model.architecture import UNet
import os
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.image as mpimg

# Functions to calculate floe soze distribution

def make_bins(areas, bin_size):
    sizes, counts = [], []
    for i, a in enumerate(range(int(np.floor(np.min(areas))), int(np.ceil(np.max(areas))), bin_size)):
        if i == 0:
            prev = a
            continue
        else:
            sizes.append(np.mean([prev, a]))
            count = 0
            for area in areas:
                if prev <= area < a:
                    count += 1
            prev = a
            counts.append(count)
    # for last bin
    sizes
    count = 0
    for area in areas:
        if a <= area < np.max(areas) + 1:
            count += 1
    sizes.append(np.max(areas))
    counts.append(count)

    return sizes, counts


def plot_FSD(size, count, ax):
    empty_inds = []
    for i, c in enumerate(count):
        if c == 0:
            empty_inds.append(i)
    for index in sorted(empty_inds, reverse=True):
        del size[index]
        del count[index]
    ax.scatter(size, count, marker='x')
    ax.set_xlabel('Floe Area (pixels)')
    ax.set_ylabel('Frequency')
    ax.set_ylim([0, None])


def get_areas_validation(image_path, mask_path, model_path):
    areas_mask_final, areas_pred_final = [], []

    file = torch.load(model_path)
    model = UNet(1)
    model.load_state_dict(file['model_state_dict'])
    model.cuda()
    model.eval()

    images = os.listdir(image_path)
    masks = os.listdir(mask_path)
    for i in range(len(images)):
        print('Processing Image: {}'.format(images[i]))
        image = os.path.join(image_path + images[i])
        mask = os.path.join(mask_path + masks[i])
        m = cv2.imread(mask)
        img = mpimg.imread(image)
        img = img * 255
        if '.png' in mask:
            msk = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

        X = TF.to_tensor(img).unsqueeze(1).cuda()
        pred = model(X)
        pred_thresh = (pred[0, 0, :, :].cpu().detach().numpy() > 0).astype(np.uint8)

        _, areas_mask = get_centres(msk, get_areas=True)
        _, areas_pred = get_centres(pred_thresh, get_areas=True)

        areas_mask_final += list(areas_mask)
        areas_pred_final += list(areas_pred)

    return areas_mask_final, areas_pred_final