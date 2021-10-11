"""
Script reads in a drill cutting image, identifies chips by simple thresholds,
and saves the identified chips, a plotly html plot, and a CSV with basic information

Rafael Pires de Lima, March 2021
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from skimage import measure, morphology
from skimage.io import imread, imsave
from skimage.transform import downscale_local_mean
from skimage.util import img_as_ubyte
from tqdm import tqdm

# list with the name of the SEM images without extension
img_names = [
    '06_500nm_BSED'
]

# to save intermediate images
save_steps = False

for img_name in img_names:
    # location of input file
    # (a filename assebled with img_names elements)
    fname_in = os.path.normpath(f'D:\Projects\data\drill-cuttings\BS\{img_name}.tif')
    # location of output file (a folder)
    data_out_dir = os.path.normpath(f'D:\Projects\data\drill-cuttings\out')

    downscale_factor = 2
    min_intensity = 10.0
    max_intensity = 30.0
    # 500 nm = 500 mm-6 per pixel originally:
    scale_nm_per_pixel = 500 * downscale_factor
    # remove chips smaller than W x H mm:
    min_hole = (2e5*2e5) / (scale_nm_per_pixel**2)
    min_chip = min_hole/2
    print(f"smallest chip: {min_chip}")
    print(f"smallest hole: {min_hole}")

    #####################################################################################
    # check if output folders exist, create them otherwise:
    if not os.path.isdir(data_out_dir):
        os.mkdir(data_out_dir)

    for dir_out in ['contours_properties', 'cropped_chips']:
        if not os.path.isdir(os.path.join(data_out_dir,
                                dir_out)):
            os.mkdir(os.path.join(data_out_dir, dir_out))
    
    #####################################################################################
    image = imread(fname_in)
    print(f'input shape: {image.shape}')
    image_downscaled = downscale_local_mean(image,
                                            (downscale_factor, downscale_factor))
    print(f'downscaled shape: {image_downscaled.shape}')
    img = image_downscaled.astype(int)
    del image

    if save_steps:
        imsave(os.path.join(data_out_dir,
                            'contours_properties',
                            f'{img_name}_0_input.tif'),
               img_as_ubyte(img))

    #######################################################
    # Make segmentation using simple amplitude thresholding.

    labels = np.zeros_like(img).astype(bool)
    foreground, background = True, False
    labels[img < min_intensity] = background
    labels[img > max_intensity] = foreground

    if save_steps:
        imsave(os.path.join(data_out_dir,
                            'contours_properties',
                            f'{img_name}_1_thresholded.tif'),
               img_as_ubyte(labels))

    labels = morphology.remove_small_holes(labels, min_hole)

    if save_steps:
        imsave(os.path.join(data_out_dir,
                            'contours_properties',
                            f'{img_name}_2_remove_holes.tif'),
               img_as_ubyte(labels))

    labels = morphology.remove_small_objects(labels, min_chip)

    if save_steps:
        temp_labels = labels.copy()

    labels = measure.label(labels)
    print(f"chips detected: {labels.max()}")

    if save_steps:
        imsave(os.path.join(data_out_dir,
                            'contours_properties',
                            f'{img_name}_3_remove_objects.tif'),
               img_as_ubyte(temp_labels))
        del temp_labels

    #######
    if save_steps:
        fig = px.imshow(img, binary_string=True)
        fig.update_traces(hoverinfo='skip')  # hover is only for label info

    props = measure.regionprops(labels, img)
    properties = ['label', 'area', 'min_intensity',
                  'mean_intensity', 'max_intensity']

    excluded = ['area',
                'bbox',
                'bbox_area'
                'eccentricity',
                'perimeter',
                'slice',
                'convex_image',
                'coords',
                'filled_image',
                'image',
                'inertia_tensor',
                'intensity_image',
                'moments',
                'moments_central',
                'moments_hu',
                'moments_normalized',
                'weighted_local_centroid',
                'weighted_moments',
                'weighted_moments_central',
                'weighted_moments_hu',
                'weighted_moments_normalized']

    # For each label, add a filled scatter trace for its contour,
    # and display the properties of the label in the hover of this trace.
    for index in tqdm(range(1, labels.max())):
        label = props[index].label

        if save_steps:
            contour = measure.find_contours(labels == label, 0.5)[0]
            y, x = contour.T

            hoverinfo = ''
            for prop_name in properties:
                val = getattr(props[index], prop_name)
                hoverinfo += f'<b>{prop_name}: {val:.2f}</b><br>'

            fig.add_trace(go.Scatter(
                x=x, y=y, name=label,
                mode='lines',
                # fill='toself',
                showlegend=False,
                hovertemplate=hoverinfo, hoveron='points+fills'))

        # save chip as image
        bbox_lab = "_".join([str(bl) for bl in props[index]['bbox']])
        chip = img_as_ubyte(props[index]["intensity_image"])
        imsave(os.path.join(data_out_dir, 'cropped_chips',
                            f'{img_name}_{label}_{bbox_lab}.tif'),
               chip)

    if save_steps:
        fig.write_html(os.path.join(data_out_dir, 'contours_properties',
                                    f"{img_name}.html"))
        del fig

    props = measure.regionprops_table(labels, img,
                                      properties=properties)

    pd.DataFrame(props).to_csv(os.path.join(data_out_dir, 'contours_properties',
                                            f"{img_name}.csv"),
                               index=False)

    del props
    del labels
    del img
