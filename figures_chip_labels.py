"""
Script reads in a drill cutting image, the train and test chips previously identified,
and creates three figures, one identifying train and test chips, another two identyfing chips' classes
The easiest thing is to essentially repeat a lot of what chip_identification.py does,
but select the colors based on the train_dir and test_dir.
Rafael Pires de Lima, July 2021
"""

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import measure, morphology
from skimage.io import imread, imsave
from skimage.transform import downscale_local_mean
from skimage.util import img_as_ubyte
from tqdm import tqdm

img_names = [
    # '485140_01_500nm_BSED',
    # '485140_03_500nm_BSED',
    # '485140_04_500nm_BSED',
    # '485140_05_500nm_BSED',
    # '485140_06_500nm_BSED',
    # '485140_07_500nm_BSED',
    # '485140_08_500nm_BSED',
    # '485140_09_500nm_BSED',
    # '485140_10_500nm_BSED',
    # '485140_11_500nm_BSED_trimmed',
    # '485140_12_500nm_BSED',
    # '485140_13_500nm_BSED',
    # '485140_14_500nm_BSED',
    '485140_15_500nm_BSED_trimmed'
]

for img_name in img_names:
    fname_in = f'C:/Projects/drill-cuttings/data/SEM_samples/{img_name}.tif'
    train_dir = 'C:/Projects/drill-cuttings/data/splits/train'
    test_dir = 'C:/Projects/drill-cuttings/data/splits/test'
    data_out_dir = 'C:/Projects/drill-cuttings/data/labels/'

    downscale_factor = 2
    min_intensity = 10.0
    max_intensity = 30.0
    # 500 nm = 500 mm-6 per pixel originally:
    scale_nm_per_pixel = 500 * downscale_factor
    # remove chips smaller than W x H mm:
    min_hole = (2e5*2e5) / (scale_nm_per_pixel**2)
    min_chip = min_hole / 2
    print(f"smallest chip: {min_chip}")
    print(f"smallest hole: {min_hole}")

    #####################################################################################

    image = imread(fname_in)
    print(f'input shape: {image.shape}')
    image_downscaled = downscale_local_mean(image,
                                            (downscale_factor, downscale_factor))
    print(f'downscaled shape: {image_downscaled.shape}')
    del image
    img = image_downscaled.astype(int)

    #######################################################
    # Make segmentation using simple amplitude thresholding.
    labels = np.zeros_like(img).astype(bool)
    foreground, background = True, False
    labels[img < min_intensity] = background
    labels[img > max_intensity] = foreground

    labels = morphology.remove_small_holes(labels, min_hole)
    labels = morphology.remove_small_objects(labels, min_chip)

    labels = measure.label(labels)
    nchips = labels.max()
    print(f"chips detected: {nchips}")
    sample_dset = np.zeros_like(labels)

    #######
    props = measure.regionprops(labels, img)
    properties = ['label', 'area', 'min_intensity',
                  'mean_intensity', 'max_intensity']
    del img
    #######################################################
    # produce colormap for sample_dset
    colors = ["white", "blue", "orange"]
    cmap_dset = ListedColormap(colors)

    # produce colormap for sample_dset
    colors = ["white", "#a10c02", "#1562bf", "#FF9999", "#65b842", "#FFC000"]
    cmap_labels = ListedColormap(colors)
    cmap_labels_scale_dict = {'bit_met': 1,
                              'cemented': 2,
                              'heterolithic': 3,
                              'mudstone': 4,
                              'porous': 5
                              }

    # reset labels variable
    labels = np.zeros_like(labels)
    # create an array to store chip set
    sample_dset = np.zeros_like(labels)

    for index in tqdm(range(1, nchips)):
        label = props[index].label

        # contour = measure.find_contours(labels == label, 0.5)[0]
        # y, x = contour.T

        # what set and class is this chip?
        bbox_lab = "_".join([str(bl) for bl in props[index]['bbox']])
        not_found = True
        while not_found:
            #dset = None
            for dset in tqdm([train_dir, test_dir]):
                for root, dirs, files in os.walk(dset):
                    for chip in files:
                        if bbox_lab in chip:
                            if img_name in chip:
                                # found the labeled chip
                                chip_dset = dset
                                chip_class = os.path.basename(root)
                                not_found = False

                                # get coordinates
                                label_coords = chip.split(
                                    img_name)[1].split('.')[0]
                                coords = label_coords.split('_')[2:]
                                coords = [int(coord) for coord in coords]

                                # get chip label image
                                chip_img = props[index]['filled_image'].astype(
                                    np.int)

                                # for some reason (?) few images actually have
                                # two chips in a single label
                                # so we re-label the chip:
                                chip_img = measure.label(chip_img)
                                # get the label value of the center pixel
                                ci, cj = [i//2 for i in chip_img.shape]
                                label_center = chip_img[ci, cj]
                                chip_img[chip_img > label_center] = 0
                                chip_img[chip_img < label_center] = 0

                                # populate sample_dset array/image:
                                if dset == train_dir:
                                    scale = 1
                                elif dset == test_dir:
                                    scale = 2
                                sample_dset[coords[0]:coords[2],
                                            coords[1]:coords[3]] += scale*chip_img.copy()

                                # populate labels array/image:
                                scale = cmap_labels_scale_dict[chip_class]
                                labels[coords[0]:coords[2],
                                       coords[1]:coords[3]] += scale*chip_img.copy()

                                break
            if dset == test_dir:
                if not_found:
                    print('sample not found')
                    not_found = False

    # save labels:
    imsave(os.path.join(data_out_dir,
                        'labeled_ts',
                        f'{img_name}_labels.tif'),
           img_as_ubyte(labels))

    fig, ax = plt.subplots()
    img = ax.imshow(labels,
                    vmin=0, vmax=len(cmap_labels.colors),
                    cmap=cmap_labels,
                    interpolation="none")
    img.write_png(os.path.join(
        data_out_dir, 'color_images', f'{img_name}_labels.png'))

    plt.close('all')
    del labels

    fig, ax = plt.subplots()
    img = ax.imshow(sample_dset,
                    vmin=0, vmax=len(cmap_dset.colors),
                    cmap=cmap_dset,
                    interpolation="none")
    img.write_png(os.path.join(data_out_dir, 'color_images',
                               f'{img_name}_train_test.png'))
    plt.close('all')
    del sample_dset
