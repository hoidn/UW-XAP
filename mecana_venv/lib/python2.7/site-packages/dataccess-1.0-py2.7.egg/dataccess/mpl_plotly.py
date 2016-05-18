"""
A matplotlib-esque interface for Plotly-based interactive plots for the Jupyter
notebook.
"""

import matplotlib.pyplot as mplt
import plotly.offline as py
import plotly.graph_objs as go

py.offline.init_notebook_mode()

class Figure:
    """Class containing the equivalent of a matplotlib Axis."""
    def __init__(self):
        self.lines = []
        self.xaxis = {'exponentformat': 'power'}
        self.yaxis = {}

    def plot(self, x, y, label = None, color = None):
        scatter_kwargs = dict(x = x, y = y, name = label,
                mode = 'lines', line = dict(color = color))
        if label is None:
            scatter_kwargs['showlegend'] = False
            scatter_kwargs['hoverinfo'] = 'none'
            
        else:
            scatter_kwargs['showlegend'] = True
        self.lines.append(
            go.Scatter(**scatter_kwargs))

    def set_xlabel(self, xlabel):
        self.xaxis['title'] = xlabel

    def set_ylabel(self, ylabel):
        self.yaxis['title'] = ylabel

    def set_xscale(self, value):
        if value == 'log':
            self.xaxis['type'] = 'log'
        else:
            raise NotImplementedError

    def show(self):
        layout = go.Layout(xaxis = self.xaxis, yaxis = self.yaxis)
        data = self.lines
        fig = go.Figure(data = data, layout = layout)
        py.iplot(fig)

class Plt:
    def __init__(self):
        self.mode = None
        self.figures = []
    
    def clear(self):
        self.__init__()

    def subplots(self, *args, **kwargs):
        self.mode = 'plotly'
        if len(args) > 0:
            n_plots = args[0]
            self.figures = [Figure() for _ in range(n_plots)]
            return None, self.figures
        else:
            self.figures.append(Figure())
            return None, self.figures[0]

    def legend(self):
        pass

    def show(self):
        if self.mode == 'plotly':
            for fig in self.figures:
                fig.show()
            self.clear()
        elif self.mode == 'matplotlib':
            mplt.show()
        else:
            raise ValueError("plotting mode not recognized: %s" % self.mode)

    def imshow(self, *kargs, **kwargs):
        mplt.imshow(*args, **kwargs)
        self.mode = 'matplotlib'

    def savefig(self, *args, **kwargs):
        # TODO: implement this
        pass

plt = Plt()
