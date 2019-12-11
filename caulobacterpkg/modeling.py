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



# LINEAR MODEL
def lin_theor(t, a_0, k):
    '''MLE for linear approximation'''
    out = a_0 * np.ones_like(t)
    out = a_0 + t * k
    return out

def resid_lin(params, t, area):
    """Residuals. `params` here is all parameters except σ."""
    return area - lin_theor(*params, t)

def mle_lin(area, t):
    '''Perform least squares'''
    res = scipy.optimize.least_squares(
        resid_lin,
        np.array([0, 0]),
        args=(t, area),
        bounds=([-np.inf, -np.inf], [np.inf, np.inf]),
    )

    # Compute residual sum of squares from optimal params
    rss_mle = np.sum(resid_lin(res.x, t, area) ** 2)

    # Compute MLE for sigma
    sigma_mle = np.sqrt(rss_mle / len(t))
    return tuple([x for x in res.x] + [sigma_mle])


# EXPONENTIAL MODEL
def exp_theor(t, a_0, k):
    out = a_0 * np.ones_like(t)
    out = a_0 * np.exp(t * k)
    return out

def resid_exp(params, t, area):
    """Residuals. `params` here is all parameters except σ."""
    return area - exp_theor(*params, t)

def mle_exp(area, t):
    
    res = scipy.optimize.least_squares(
        resid_exp,
        np.array([0,0]),
        args=(t, area),
        bounds=([-np.inf, -np.inf], [np.inf, np.inf]),
    )

    # Compute residual sum of squares from optimal params
    rss_mle = np.sum(resid_exp(res.x, t, area) ** 2)

    # Compute MLE for sigma
    sigma_mle = np.sqrt(rss_mle / len(t))
    return tuple([x for x in res.x] + [sigma_mle])






# to look at how k changes with time
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

