BQPlot
======

Plotting system for the Jupyter notebook based on the interactive HTML.

Installation
============

.. code-block:: bash

    pip install bqplot
    jupyter nbextension enable --py bqplot


Usage
=====

.. code-block:: python

    from bqplot import pyplot as plt
    import numpy as np

    plt.figure(1)
    n = 200
    x = np.linspace(0.0, 10.0, n)
    y = np.cumsum(np.random.randn(n))
    plt.plot(x,y, axes_options={'y': {'grid_lines': 'dashed'}})
    plt.show()


