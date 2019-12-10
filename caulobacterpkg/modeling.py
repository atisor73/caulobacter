import numpy as np
import pandas as pd
import random 

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









def return_ks(df):
    '''returns dataframes filled w k-values'''
    dfs = []
    for cycle in range(np.max(df["colored"].values)):
        df_cycle = df.loc[df["colored"]==cycle]
        df_cycle = df_cycle.reset_index()
        df_cycle = df_cycle.drop(columns=["index"])

        # initial size for entire cycle
        initial_size = df_cycle.loc[0].values[1]

        # frame distance from initial cycle (unit in minutes)
        df_cycle["time_since_growth"] = df_cycle["frame"] - df_cycle.loc[0].values[0]

        # calculating k-values (linear)
        df_cycle["k-linear"] = (df_cycle["areas"] / 
                                initial_size - 1) / df_cycle["time_since_growth"]
        df_cycle["k-exponential"] = np.log(df_cycle["areas"] / 
                                initial_size) /df_cycle["time_since_growth"]
        # add to dataframe of k-values
        dfs.append(pd.DataFrame({"cycle":[cycle]*df_cycle.shape[0], 
                                 "frame":df_cycle["time_since_growth"],
                                 "k-linear":df_cycle["k-linear"],
                                 "k-exponential":df_cycle["k-exponential"]}))
    return dfs


def cycle_mean_std(df, val="k-exponential"):
    '''returns mean val of all cycles in a dataframe as a list'''
    mean_ks, std_ks = [], []
    for cycle in range(max(df.cycle.values) + 1):
        mean_ks.append(np.nanmean(df.loc[df["cycle"]==cycle][val].values))
        std_ks.append(np.nanstd(df.loc[df["cycle"]==cycle][val].values))
    return (mean_ks, std_ks)


def return_a0s(df, b_k_MLEs_lin, b_k_MLEs_exp):
    '''returns dataframes filled w a0-estimates from k*'''
    dfs = []
    for cycle in range(np.max(df["colored"].values)):
        df_cycle = df.loc[df["colored"]==cycle]
        df_cycle = df_cycle.reset_index()
        df_cycle = df_cycle.drop(columns=["index"])

        # k* for cycle
        k_lin = b_k_MLEs_lin[0][cycle]
        k_exp = b_k_MLEs_exp[0][cycle]

        # frame distance from initial cycle (unit in minutes)
        df_cycle["time_since_growth"] = df_cycle["frame"] - df_cycle.loc[0].values[0]

        # calculating a_0 estimates (linear)
        df_cycle["a_0_linear_estimate"] =  df_cycle["areas"]/(1+k_lin*df_cycle["time_since_growth"])
        df_cycle["a_0_exponential_estimate"] =  (df_cycle["areas"] * 
                                                 np.exp(k_exp * df_cycle["time_since_growth"]))
        
        # add to dataframe of a0-values
        dfs.append(pd.DataFrame({"cycle":[cycle]*df_cycle.shape[0], 
                                 "frame":df_cycle["time_since_growth"],
                                 "a_0_linear_estimate":df_cycle["a_0_linear_estimate"],
                                 "a_0_exponential_estimate":df_cycle["a_0_exponential_estimate"]}))
    return dfs



def return_a_theoretical(df, b_k_MLEs, b_a0_MLEs, func="exp"):
    '''start generating predictive areas from k* and a*
    returns dataframse of all sizes for each cycle with different k* and a* for each'''
    dfs = []
    for cycle in range(np.max(df["colored"].values)):
        times = np.linspace(0, df.loc[df["colored"]==cycle]["frame"].max()
                        -df.loc[df["colored"]==cycle]["frame"].min(), 1000)
        k_star, a_star = b_k_MLEs[0][cycle], b_a0_MLEs[0][cycle]
        if func=="exp":
            theoretical_a = a_star * np.exp(k_star*times)
        elif func=="lin":
            theoretical_a = a_star * (1+k_star*times)
        
        dfs.append(pd.DataFrame({"cycle":[cycle]*1000, 
                                 "times": times+df.loc[df["colored"]==cycle]["frame"].min(), 
                                 "theoretical_a":theoretical_a}))
    return pd.concat(dfs, ignore_index=True)


# ecdf bokeh
def plotter_bokeh(df, width, bacterium):
    '''plots frames and areas of bacteria'''
    x, y = "frame", "areas (μm^2)"
    color = df["colored"]
    
    p = bokeh.plotting.figure(width=width, height=300, 
        x_axis_label=x, y_axis_label=y,
        tooltips=[(x, "@{frame}"), (y, "@{areas (μm^2)}"), ("cycle", "@{colored}")],
        title=f"{bacterium}")
    
    # alternates colors for even / odd cycles
    p.circle(source=df.loc[color % 2 == 0], x=x, y=y, color="darkblue")
    p.circle(source=df.loc[color % 2 == 1], x=x, y=y, color="lightblue")

    p.xgrid.grid_line_color, p.ygrid.grid_line_color = None, None
    p.outline_line_color = None
    p.title.align, p.title.text_font_style = "center", "bold"
    p.toolbar.autohide = True
    return p