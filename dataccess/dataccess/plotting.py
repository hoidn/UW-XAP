"""
Module implementing Boke and mpld3-based plots for the Jupyter notebook
with a matplotlib-esque API.
"""

from bokeh.plotting import figure, output_file
from bokeh.plotting import show as bkshow
from bokeh.io import output_notebook
from bokeh.io import gridplot
from bokeh.models import Range1d

from dataccess import utils
import config
import pdb

class BokehPlot:
    """
    Class containing a single Bokeh plot implementing a subset of the API of
    matplotlib.figure.Figure
    """
    #from bokeh.palettes import Spectral9
    #palette = Spectral9
    palette = ['#332288',  '#44aa99',  '#999933',\
         '#cc6677',  '#aa4499', '#000000']
    #palette = ['#332288', '#88ccee', '#44aa99', '#117733', '#999933',\
    #    '#ddcc77', '#cc6677', '#882255', '#aa4499']
    counter = 0
    def __init__(self, **kwargs):
        TOOLS = 'wheel_zoom,box_zoom,pan,crosshair,hover,resize,reset'
        self.figure = figure(plot_width = 800, plot_height = 500, tools = TOOLS, **kwargs)
#        self.figure.legend.border_line_alpha = 0.5
#        self.figure.legend.background_fill_alpha = 0.5
    def plot(self, x, y, label = None, **kwargs):
        if 'color' not in kwargs:
            color = BokehPlot.palette[BokehPlot.counter % len(BokehPlot.palette)]
            BokehPlot.counter += 1
            self.figure.line(x, y, legend = label, line_width = 2, color = color, **kwargs)
        else:
            self.figure.line(x, y, legend = label, line_width = 2, **kwargs)
    def set_xlabel(self, label):
        self.figure.xaxis.axis_label = label
    def set_ylabel(self, label):
        self.figure.yaxis.axis_label = label
    def set_xlim(self, range_tuple):
        self.figure.set(x_range = Range1d(*range_tuple))
    def legend(self):
        pass
    def show(self):
        bkshow(self.figure)

class Plt:
    def subplots(self, n, kwarg_list = None):
        if kwarg_list is None:
            kwarg_list = [{} for _ in range(n)]
        self.plots = [BokehPlot(**kwargs) for kwargs in kwarg_list]
        if n == 1:
            return None, self.plots[0]
        else:
            return None, self.plots
    def show(self):
        if len(self.plots) == 1:
            p = self.plots[0]
        elif len(self.plots) == 2:
            TOOLS = 'wheel_zoom,box_zoom,pan,crosshair,hover,resize,reset'
            p = gridplot([[self.plots[0].figure], [self.plots[1].figure]])
        else:
            raise NotImplementedError
        bkshow(p)

current_plot = {}

def subplots(n, kwarg_list = None):
    plt = Plt()
    current_plot[0] = plt
    return plt.subplots(n, kwarg_list = kwarg_list)

def show():
    plt = current_plot[0]
    plt.show()


def savefig(*args):
    pass

if utils.isroot():
    output_notebook(verbose = False)
