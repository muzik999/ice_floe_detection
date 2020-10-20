import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
# metrics for floe validation

def get_centres(img, get_areas=False):
    img = np.uint8(img)
    ret, thresh = cv2.threshold(img, 0.001, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centres = []
    if get_areas:
        areas = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centres.append([cX, cY])
            if get_areas:
                area = cv2.contourArea(c)
                areas.append(area)

    centres = np.array(centres)
    if get_areas:
        return centres, np.array(areas)
    else:
        return centres


def precision_recall(y_true, y_pred):
    smooth = 1e-6

    centres = get_centres(y_true)
    centres_pred = get_centres(y_pred)

    tp = 0
    fn = 0
    fp = 0

    if len(centres) > 0:
        for x, y in zip(centres[:, 0], centres[:, 1]):
            if y_pred[y, x] > 0:
                tp += 1
            else:
                fn += 1

    if len(centres_pred) > 0:
        for x, y in zip(centres_pred[:, 0], centres_pred[:, 1]):
            if y_true[y, x] == 0:
                fp += 1

    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)

    return precision, recall


def mAcc(y_true, y_pred):
    intersection_pos = (y_pred.astype(bool) & y_true.astype(bool)).astype(float).sum()
    intersection_neg = (np.invert(y_pred.astype(bool)) & np.invert(y_true.astype(bool))).astype(float).sum()
    num_pixels = y_true.shape[0] * y_true.shape[1]

    mAcc = (intersection_pos + intersection_neg) / num_pixels
    return mAcc


# def mIoU(y_true, y_pred):
#     smooth = 1e-6

#     intersection_pos = (y_pred.astype(bool) & y_true.astype(bool)).astype(float).sum()
#     union_pos = (y_pred.astype(bool) | y_true.astype(bool)).astype(float).sum()

#     intersection_neg = (np.invert(y_pred.astype(bool)) & np.invert(y_true.astype(bool))).astype(float).sum()
#     union_neg = (np.invert(y_pred.astype(bool)) | np.invert(y_true.astype(bool))).astype(float).sum()

#     iou_pos = (intersection_pos.sum() + smooth) / (union_pos.sum() + smooth)
#     iou_neg = (intersection_neg.sum() + smooth) / (union_neg.sum() + smooth)

#     mIoU = (iou_pos + iou_neg) / 2
#     return mIoU

def compute_miou(y_pred, y_true):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    
    return np.mean(IoU)

def count_parameters(model):
    count = 0
    for p in model.parameters():
        count += p.data.nelement()
        
    return count

def plot_ALL():
    plt.figure(figsize=(50,50))
    plt.title(img_name + '             Pink = "TP", Yellow = "FP", Blue = "FN"', fontdict={'fontsize': 50})
    plt.imshow(image_plt, cmap='gray')
    plt.imshow(mask_GT, cmap='brg', alpha=0.5)
    plt.imshow(mask_PRED, cmap='autumn_r', alpha=0.7)
    plt.imshow(mask_OL, cmap='cool_r', alpha=0.5)

    plt.savefig('/home/muzik999/MASc/iceFloe/floes/model/weights/UNET_CRF/NEW/experiment_BACKBONE/' + 'ALL_' + img_name[:-4] + '.png')
    plt.close()
    
def plot_GT():
    plt.figure(figsize=(50,50))
    plt.title(img_name + '             Blue = "Ground Truth"', fontdict={'fontsize': 50})
    plt.imshow(image_plt, cmap='gray')
    plt.imshow(mask_GT, cmap='brg', alpha=0.5)
    # plt.imshow(mask_PRED, cmap='autumn_r', alpha=0.7)
    # plt.imshow(mask_OL, cmap='cool_r', alpha=0.5)

    plt.savefig('/home/muzik999/MASc/iceFloe/floes/model/weights/UNET_CRF/NEW/experiment_BACKBONE/' + 'GT_' + img_name[:-4] + '.png')
    plt.close()

def plot_PRED():
    plt.figure(figsize=(50,50))
    plt.title(img_name + '             Yellow = "Prediction"', fontdict={'fontsize': 50})
    plt.imshow(image_plt, cmap='gray')
    # plt.imshow(mask_GT, cmap='brg', alpha=0.5)
    plt.imshow(mask_PRED, cmap='autumn_r', alpha=0.7)
    # plt.imshow(mask_OL, cmap='cool_r', alpha=0.5)

    plt.savefig('/home/muzik999/MASc/iceFloe/floes/model/weights/UNET_CRF/NEW/experiment_BACKBONE/' + 'PRED_' + img_name[:-4] + '.png')
    plt.close()
    
def plot_OL():
    plt.figure(figsize=(50,50))
    plt.title(img_name + '             Pink = "Overlap"', fontdict={'fontsize': 50})
    plt.imshow(image_plt, cmap='gray')
    # plt.imshow(mask_GT, cmap='brg', alpha=0.5)
    # plt.imshow(mask_PRED, cmap='autumn_r', alpha=0.7)
    plt.imshow(mask_OL, cmap='cool_r', alpha=0.5)

    plt.savefig('/home/muzik999/MASc/iceFloe/floes/model/weights/UNET_CRF/NEW/experiment_BACKBONE/' + 'OL_' + img_name[:-4] + '.png')
    plt.close()