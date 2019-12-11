import numpy as np
import pandas as pd
import random 
import scipy.stats

import bokeh
import bokeh.palettes
from bokeh.transform import linear_cmap
import bokeh_catplot
import bokeh.models
import bebi103
import holoviews as hv
hv.extension('bokeh')
import bokeh.io
bebi103.hv.set_defaults()
bokeh.io.output_notebook()


def growth_data(df):
    '''takes in dataframe with colored cycles, 
    returns data frame of min and max areas, length of cycle, frame of reference'''
    # creates new df indexing by cycle
    df_stacked = df.set_index("colored")
    mins, maxs, times, frames, fractions = [], [], [], [], []

    # going through each cycle 
    for i in range(df["colored"].max()+1):
        df_cycle = df.loc[df["colored"] == i]
        # begin collecting size @birth and @division
        mins.append(df_cycle["areas"].min())
        maxs.append(df_cycle["areas"].max())
        # as well as times of growth 
        times.append(df_cycle["frame"].max() -
                     df_cycle["frame"].min())
        frames.append(df_cycle["frame"].min())
        
    # getting growth (division-birth size)
    growths = [(maxs[j] - mins[j]) for j in range(0, len(mins))]
    
    # and what fractional area divided off
    fractions = [(mins[j]/maxs[j-1]) for j in range(1, len(mins))]
    fractions.insert(0, .5)
    
    # store and return all this data as a dataframe
    data = pd.DataFrame({"mins": mins, "maxs": maxs, 
                         "growths":growths, "times": times, 
                         "frames": frames, "fractions": fractions})
    data = data.drop(0, axis=0)
    
    return data




# used to spline data points
def splining(df, f, column1, column2):
    '''not sure but'''
    smoothing_factor = f * (df[column2]**2).sum()
    # spline instance, cubic polynomials
    column1 = df[column1].values
    column2 = df[column2].values
    spl = scipy.interpolate.UnivariateSpline(
        column1, column2, s=smoothing_factor)

    # spl is now a callable function
    column1_spline = np.linspace(column1[0], column1[-1], 400)
    column2_spline = spl(column1_spline)
    
    plot = hv.Curve(data=(column1_spline, column2_spline)
                   ).opts(color="navy", line_width=1.75)
    return plot
