"""
Script reads in a drill cutting image, the train and test chips previously identified,
and creates an html file with Plotly
The easiest thing is to essentially repeat a lot of what chip_identification.py does,
but select the colors based on the train_dir and test_dir.
The most expensive step is contour = measure.find_contours(labels == label, 0.5)[0].

Rafael Pires de Lima, July 2021
"""

import os

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from skimage import measure, morphology
from skimage.io import imread
from skimage.transform import downscale_local_mean
from tqdm import tqdm

img_names = [
    '485140_01_500nm_BSED',
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
    min_chip = min_hole  # /2
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
    print(f"chips detected: {labels.max()}")

    #######
    props = measure.regionprops(labels, img)
    properties = ['label', 'area', 'min_intensity',
                  'mean_intensity', 'max_intensity']

    fig = px.imshow(img, binary_string=True)
    fig.update_traces(hoverinfo='skip')  # hover is only for label info
    del img

    #####################################################################
    # produce colormap for sample_dset
    colors = ["white", "#a10c02", "#1562bf", "#FF9999", "#65b842", "#FFC000"]

    cmap_labels_scale_dict = {'bit_met': 1,
                              'cemented': 2,
                              'heterolithic': 3,
                              'mudstone': 4,
                              'porous': 5
                              }
    chip_class_dict = {'bit_met': 'DBM',
                       'cemented': 'DC_Slt',
                       'heterolithic': 'HD_Slt',
                       'mudstone': 'OR_M',
                       'porous': 'PD_Slt'
                       }
    ######################################################################
    # For each label, add a filled scatter trace for its contour,
    # and display the properties of the label in the hover of this trace.
    for index in tqdm(range(1, labels.max())):
        label = props[index].label

        contour = measure.find_contours(labels == label, 0.5)[0]
        y, x = contour.T

        # what set and class is this chip?
        bbox_lab = "_".join([str(bl) for bl in props[index]['bbox']])
        not_found = True
        while not_found:
            dset = None
            for dset in [train_dir, test_dir]:
                for root, dirs, files in os.walk(dset):
                    for chip in files:
                        if bbox_lab in chip:
                            if img_name in chip:
                                # found the labeled chip
                                chip_dset = dset
                                chip_class = os.path.basename(root)
                                not_found = False
                                break
            if dset == test_dir:
                if not_found:
                    print('sample not found')
                    not_found = False

        hoverinfo = f'<b>Class: {chip_class_dict[chip_class]}</b><br>'
        hoverinfo += f'<b>Set: {chip_dset}</b><br>'
        for prop_name in properties:
            val = getattr(props[index], prop_name)
            hoverinfo += f'<b>{prop_name}: {val:.2f}</b><br>'

        color = colors[cmap_labels_scale_dict[chip_class]]

        fig.add_trace(go.Scatter(
            x=x, y=y, name=label,
            mode='lines',
            line=dict(dash=['solid' if chip_dset == train_dir else 'dot'][0],
                      color=color),
            # color=color,  # fill='toself',
            showlegend=False,
            hovertemplate=hoverinfo, hoveron='points+fills'))

        # Dani, remove this break
        break

    if not os.path.isdir(data_out_dir):
        os.mkdir(data_out_dir)

    fig.write_html(os.path.join(data_out_dir,
                                f"{img_name}.html"))
    del fig
