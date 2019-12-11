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
    return bokeh_catplot.ecdf(data=data, 
                              cats=None, 
                              val="times", 
                            style='formal',
                              #palette=['#F1D4D4']
                              #palette=['#8c564b'], 
                           x_range=xrange,
                              title=title)


def plot_biphasic(df_data, bacterium, width):   
    p = bokeh.plotting.figure(width=int(width*1.5), height=300, 
        x_axis_label="frames", y_axis_label="values",
        tooltips=[("initial area", "@{mins}"), ("cycle", "@{frames}"), 
                 ("final area", "@{maxs}"), ("time", "@{times}")],
        title=("growth cycles: "+bacterium))
    p.line(source=df_data, x="frames", y="mins", color="lightblue", line_width=1.5)
    p.line(source=df_data, x="frames", y="maxs", color="darkblue", line_width=1.5)

    p.circle(source=df_data, x="frames", y="mins", color="lightblue", line_width=1.5)
    p.circle(source=df_data, x="frames", y="maxs", color="darkblue", line_width=1.5)
    p.xgrid.minor_grid_line_color = "snow"
    p.xgrid.grid_line_color = "snow"
    p.ygrid.grid_line_color = "snow"
    p.outline_line_color = None
    p.title.align = "center"
    p.output_backend = 'svg'
    p.toolbar.autohide = True
    x = np.linspace(0, 4*np.pi, 100)
    y = np.sin(x)
    
    legend = bokeh.models.Legend(items=[
        ("initial area"   , [p.circle(x,y,color="lightblue"), p.line(x,y,color="lightblue",line_width=1.5)]),
        ("final area" , [p.circle(x,y,color="darkblue"), p.line(x,y,color="darkblue",line_width=1.5)]),
        #("times" , [p.circle(x,y,color="orange"), p.line(x,y,color="orange",line_width=1.5)]),
    ], location="center")
    p.add_layout(legend, 'right')
    return p



from bokeh.themes.theme import Theme
from bokeh.io import curdoc

theme = Theme(
    json={
    'attrs' : {
        'Figure' : {
            'background_fill_color': 'white',
            'border_fill_color': 'white',
            'outline_line_color': 'white',
        },'Grid': {
            'grid_line_dash': [6, 4],
            'grid_line_alpha': .3,},
        'Axis': {
            'major_label_text_color': 'black',
            'axis_label_text_color': 'black',
            'major_tick_line_color': 'black',
            'minor_tick_line_color': 'black',
            'axis_line_color': "white"}}})
# # how i set themes for some of the plots 
# curdoc().theme = theme