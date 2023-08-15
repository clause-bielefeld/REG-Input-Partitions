import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
from os.path import join, dirname, abspath
import numpy as np
import cv2
from skimage.filters import gaussian
import sys

sys.path.append(join(dirname(abspath(__file__)), os.pardir, 'model'))
from data_utils.utils import pad_img_to_max, crop_image_to_bb, filename_from_id


def plot_att(att):
    plt.matshow(att)


def interpolate_2d(tensor, shape):
    return (
        torch.nn.functional.interpolate(tensor.unsqueeze(0).unsqueeze(0), shape)
        .squeeze(0)
        .squeeze(0)
    )


def normalize_to_255(image):
    return image * (255 / image.max())


def make_heatmap(
    entry, att, img_base, mode="context", w=0.5, sigma=15, display_result=True, mask_out_target=False
):
    # Retrieve & display the image
    image_file = filename_from_id(entry.image_id, prefix="COCO_train2014_")
    image_filepath = join(img_base, "train2014", image_file)

    img = cv2.imread(image_filepath)

    if mode == "context":
        if mask_out_target:
            img = Image.fromarray(img)
            _, _, img, _ = crop_image_to_bb(img, entry.bbox, return_context=True)
            img = np.array(img)         
    elif mode == "target":
        img = Image.fromarray(img)
        img, _ = crop_image_to_bb(img, entry.bbox, return_context=False)
        img = np.array(img)
    else:
        raise NotImplementedError("mode has to be 'target' or 'context'")

    padded_img = np.array(pad_img_to_max(Image.fromarray(img)))

    image_shape = padded_img.shape[:2]
    att = interpolate_2d(att, image_shape).numpy()
    att = gaussian(att, sigma=sigma)

    normalized_att = normalize_to_255(att).astype("uint8")
    heatmap_img = cv2.applyColorMap(normalized_att, cv2.COLORMAP_JET)

    super_imposed_img = cv2.addWeighted(heatmap_img, w, padded_img, 1 - w, 0)
    super_imposed_img = cv2.cvtColor(super_imposed_img, cv2.COLOR_BGR2RGB)

    if display_result:
        display(Image.fromarray(super_imposed_img))

    else:
        return super_imposed_img