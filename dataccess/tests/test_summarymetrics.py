from dataccess import summarymetrics
from dataccess import query
import numpy as np
import config

def test_scatter():
    config.noplot = True
    def func1(imarr, **kwargs):
        return 2 * np.sum(imarr)
    def func2(imarr, **kwargs):
        return np.sum(imarr)

    df1 = 'si', func1
    df2 = 'si', func2
    x1, y1 = summarymetrics.scatter(query.existing_dataset_by_label('614'), df1, df2)
    x2, y2 = summarymetrics.scatter('614', df2, df1)

    assert np.isclose(x1, y2).all()
    assert np.isclose(x1, 2 * y1).all()
