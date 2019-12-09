import glob
import os

import numpy as np
import pandas as pd

import skimage
import skimage.feature
import skimage.filters
import skimage.filters.rank
import skimage.morphology
import skimage.segmentation
import skimage.io

import scipy.stats as st
import scipy.ndimage

import colorcet
import bebi103
import holoviews as hv
import holoviews.operation.datashader

import panel as pn
pn.extension('mathjax')

bebi103.hv.set_defaults()






def image_to_data_frame(df_rois, im, thresh, selem_size):
    '''processes TIFF stack into images of workable form,
    extracts relevant data and places areas into dataframes'''
    rois = [bebi103.image.verts_to_roi(g[['x', 'y']].values, size_i=im.shape[1], size_j=im.shape[2])
            for _, g in df_rois.groupby('roi')]
    roi, roi_bbox, roi_box = rois[0]
    im_cropped_roi = roi_box
    
    frames = np.arange(0, im.shape[0], dtype=int)
    areas = []
    
    for frame in frames:
        # sobel, then conversion to binary
        im_float = skimage.img_as_float(im[frame][roi_bbox])
        im_float = (im_float.astype(float) - im_float.min()) / (im_float.max() - im_float.min())
        im_LoG = scipy.ndimage.filters.gaussian_laplace(im_float, 5.0)
        im_bw = im_LoG > thresh

        # closes for smoother plot 
        selem = skimage.morphology.disk(selem_size)
        im_edge_closed = skimage.morphology.binary_closing(im_bw, selem) 

        # labels binary image; backward kwarg says value in im_bw to consider backgr.
        im_labeled, n_labels = skimage.measure.label(im_edge_closed, background=0, return_num=True)

        # assigns labeled images and areas 
        unique, counts = np.unique(im_labeled, return_counts=True)
        counts = counts.tolist()

        # assumption that bacteria is second-largest blob onscreen
        counts.remove(max(counts)) # removes background data point
        areas.append(max(counts)) 
    df = pd.DataFrame({"frame": np.arange(0, im.shape[0], dtype=int)})
    df["areas"] = areas
    df["diffs"] = df["areas"].diff()
    
    return df





def plucking(df, max_growth, frames_away):
    '''plucks unsatisfactory segmentation errors, where neighboring bacterium
    are too close to allow for proper segmentation
    '''
    # if bacteria growing, make sure it's < max growth % per frame
    # or else it's considered a segmentation error
    diffs, areas = df["diffs"].to_numpy(), df["areas"].to_numpy()
    drop = [True]

    for i in range(1, len(diffs)):
        if (diffs[i] > 0) & (diffs[i]/areas[i] > max_growth):
            drop.append(True)
        else:
            drop.append(False)

    df["plucked_areas"] = drop
    df_plucked = (df[df.plucked_areas != True]).copy()
    
    # drops data points that are in between
    # recalculate difference in areas
    df_plucked["diffs"] = df_plucked["areas"].diff()
    diffs = df_plucked["diffs"].to_numpy()
    areas = df_plucked["areas"].to_numpy()

    # theory is basically taking out pts/groups of pts under latency
    growth = 10 
    drop = [True] * frames_away

    for i in range(frames_away, len(diffs)-frames_away):
        # throws away lone points
        if ((abs(areas[i] - areas[i-frames_away])>frames_away*growth) & 
            (abs(areas[i] - areas[i+frames_away])>frames_away*growth)):
            drop.append(True)
        else:
            drop.append(False)
    
    drop += [True] * frames_away
    
    # creates column telling us whether or not frame should be dropped
    df_plucked["plucked_areas"] = drop
    
    # final dataframe without unwanted values
    df_plucked_more = (df_plucked[df_plucked.plucked_areas != True]).copy()
    return df_plucked_more






def coloring(df, wiggle):
    '''process of coloring different growth cycles,
    returns dataframe with integer cycles in "colored columns"'''
    # begins process of coloring different growth cycles
    df["diffs"] = df["areas"].diff()
    diffs = df["diffs"].to_numpy()
    
    # count_negatives mark times of big change
    count_negatives = 0
    colored_nums = [0]

    for i in range(1, len(diffs)):
        if ((diffs[i] < 0) & (diffs[i] > wiggle)):
            diffs[i] = -diffs[i]

        if diffs[i] < 0:
            count_negatives += 1
        colored_nums.append(count_negatives)
       
    df["colored"] = colored_nums
    return df




def pixel_converter(df, interpixel_distance):
    '''returns dataframe w/ additional column for area conversion'''
    df["areas (μm^2)"] = df["areas"] * (interpixel_distance ** 2)
    return df