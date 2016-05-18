"""
Module implementing plotly and mpld3-based plots for the Jupyter notebook.
"""

import config
import pdb

from dataccess import utils
import matplotlib.pyplot as mplt

from bokeh.plotting import output_notebook

output_notebook()

class Plt:
    def __init__(self):
        self.show_legend = False
    def subplots(self, *args, **kwargs):
        self.mode = 'bokeh'
        self.fig, self.axes = mplt.subplots(*args, **kwargs)
        return self.fig, self.axes

    def imshow(self, *args, **kwargs):
        self.mode = 'maptlotlib'
        mplt.imshow(*args, **kwargs)
        
    def show(self):
        if self.mode == 'plotly':
            if self.show_legend:
                # The requires a connection to the web
                # (usable from psana by configuring requests to use a SOCKS proxy)
                #from plotly.offline import iplot_mpl
                from plotly.plotly import iplot_mpl
                update = dict(
                    layout=dict(
                        showlegend=True  # show legend 
                    )
                )
                return iplot_mpl(self.fig, update = update)
            else:
                from plotly.offline import iplot_mpl
                iplot_mpl(self.fig)
        elif self.mode == 'matplotlib':
            mplt.show()
        elif self.mode == 'bokeh':
            from bokeh import mpl
            from bokeh.plotting import show
            show(mpl.to_bokeh())
        else:
            raise ValueError("Invalid mode: %" % mode)

    def legend(self, *args, **kwargs):
        self.show_legend = True
        mplt.legend(*args, **kwargs)

# TODO: a more elegant way of getting around scoping limitations in the way
# we're doing here.
plt = Plt()

def subplots(*args, **kwargs):
    return plt.subplots(*args, **kwargs)

def show():
    plt.show()

def imshow():
    plt.imshow()

def savefig(*args):
    pass

def legend(*args, **kwargs):
    plt.legend(*args, **kwargs)

