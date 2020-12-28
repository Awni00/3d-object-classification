import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def load_images(im_base_path):
    bgr_image = cv.imread(f'{im_base_path}_crop.png', cv.IMREAD_UNCHANGED)
    depth_image = cv.imread(f'{im_base_path}_depthcrop.png', cv.IMREAD_UNCHANGED)
    mask_image = cv.imread(f'{im_base_path}_maskcrop.png', cv.IMREAD_UNCHANGED)

    return bgr_image, depth_image, mask_image

def plot_rgb_depth(rgb_im, depth_im, mask_image=None):
    '''plots rgb, depth, and mask images next to each other'''

    if mask_image is None:
        fig, (rgb_ax, depth_ax) = plt.subplots(nrows=1, ncols=2)

        rgb_ax.imshow(rgb_im);
        rgb_ax.set_title('RGB Image')

        depth_ax.imshow(depth_im, cmap='binary');
        depth_ax.set_title('Depth Image')
    else:
        fig, (rgb_ax, depth_ax, mask_ax) = plt.subplots(nrows=1, ncols=3)

        rgb_ax.imshow(rgb_im);
        rgb_ax.set_title('RGB Image')

        depth_ax.imshow(depth_im, cmap='binary');
        depth_ax.set_title('Depth Image')

        mask_ax.imshow(mask_image, cmap='binary');
        mask_ax.set_title('Binary mask')


    return fig
