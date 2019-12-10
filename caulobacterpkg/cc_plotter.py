import bokeh
import bebi103
import bokeh_catplot
import colorcet
import holoviews as hv
import holoviews.operation.datashader
from holoviews.operation.datashader import dynspread
bokeh.io.output_notebook()

hv.extension('bokeh')

bokeh.io.output_notebook()
import bokeh.io

def plotter_bokeh(df, width, bacterium):
    '''plots frames and areas of bacteria'''
    x, y = "frame", "areas (μm^2)"
    color = df["colored"]
    
    p = bokeh.plotting.figure(width=width, height=500, 
        x_axis_label=x, y_axis_label=y,
        tooltips=[(x, "@{frame}"), 
                  (y, "@{areas (μm^2)}")],
        title=f"{bacterium}"
    )
    p.circle(source=df.loc[color % 2 == 0],
        x=x, y=y, color="darkred")
    p.circle(source=df.loc[color % 2 == 1],
        x=x, y=y, color="darksalmon")

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.outline_line_color = None
    p.title.align = "center"
    p.title.text_font_style = "bold"
    p.toolbar.autohide = True
    p.output_backend = 'svg'
    return p
    
    
def plotter_datashade(df, width, bacterium):
    '''plots frames and areas of bacteria'''
    
    # wants to handle large sets of data using datashade, 
    # not a lot of overlap, no need for points to be miniscule 
    # dynspread stretches points
    dynspread.max_px=9
    dynspread.threshold=.8

    points = hv.Points(
    data=df,
    kdims=["frame", "areas (μm^2)"],
    )
    p = hv.operation.datashader.datashade(
        points,
        cmap=bokeh.palettes.Blues8,
    ).opts(
        width=width,
        height=300,
        padding=0.1,
        show_grid=True,
    ).opts(
        title=bacterium
    )
    return dynspread(p)


def ecdf_plotter(data, title, xrange=None):
    p = bokeh_catplot.ecdf(data=data, cats=None, val="times", 
                            style='formal', palette=['#8c564b'], 
                           x_range=xrange, title=title)

    return p