import skimage
import skimage.feature
import skimage.filters
import skimage.filters.rank
import skimage.morphology
import skimage.segmentation
import skimage.io

import scipy.stats as st
import scipy.ndimage

import bokeh
import colorcet
import bebi103
import holoviews as hv
import holoviews.operation.datashader
hv.extension('bokeh')

import panel as pn
pn.extension('mathjax')

bebi103.hv.set_defaults()
bokeh.io.output_notebook()
import bokeh.io



def display(im, roi_bbox, frame=0):
    return bebi103.image.imshow(im[frame][roi_bbox],
                                #interpixel_distance=interpixel_distance,
                                length_units="μm",
                                flip=False,
                                colorbar=True)

def display_marr_hildreth(im, roi_bbox, frame=0):
    im_float = skimage.img_as_float(im[frame][roi_bbox])
    im_float = (im_float.astype(float) - im_float.min()) / (im_float.max() - im_float.min())
    im_LoG = scipy.ndimage.filters.gaussian_laplace(im_float, 5.0)

    return  bebi103.image.imshow(im_LoG,
                                cmap=colorcet.coolwarm,
                                #interpixel_distance=interpixel_distance,
                                length_units="μm",
                                flip=False,
                                colorbar=True)
thresh = 0.004
def display_thresh_marr_hildreth(frame=0):
    im_float = skimage.img_as_float(im[frame][roi_bbox])
    im_float = (im_float.astype(float) - im_float.min()) / (im_float.max() - im_float.min())
    im_LoG = scipy.ndimage.filters.gaussian_laplace(im_float, 5.0)
    im_bw = im_LoG > thresh
    return  bebi103.image.imshow(im_bw,
                                #interpixel_distance=interpixel_distance,
                                length_units="μm",
                                flip=False,
                                colorbar=True)
def display_thresh_marr_hildreth_labeled(frame=0):
    im_float = skimage.img_as_float(im[frame][roi_bbox])
    im_float = (im_float.astype(float) - im_float.min()) / (im_float.max() - im_float.min())
    im_LoG = scipy.ndimage.filters.gaussian_laplace(im_float, 5.0)
    
    im_bw = im_LoG > thresh
    
    # Label binary image; backward kwarg says value in im_bw to consider backgr.
    im_labeled, n_labels = skimage.measure.label(im_bw, background=0, return_num=True)

    return  bebi103.image.imshow(im_labeled,
                                cmap=colorcet.cwr,
                                interpixel_distance=interpixel_distance,
                                length_units="μm",
                                flip=False,
                                colorbar=True,
                                )