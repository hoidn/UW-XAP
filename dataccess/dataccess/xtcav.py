import ipdb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mpldates
import numpy as np
import os
import datetime
import config
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter as filt

from dataccess import utils

xtcav_path = utils.resource_path('data/xtcav.dat')
xtcav = pd.read_table(xtcav_path)

def epoch_time(raw):
    return float(datetime.datetime.strptime(raw,"%m/%d/%Y %H:%M:%S").strftime('%s'))

# Convert to epoch time
xtcav_times = np.array(map(epoch_time, xtcav['Timestamp']))
# Get x ray pulse durations (fs)
xtcav_lengths = np.array(xtcav['X-rays'])
pulse_length_from_epoch_time = interp1d(xtcav_times, xtcav_lengths)


def autocorrelation(x):
    x = x - np.mean(x)
    return np.array([np.sum(x * np.roll(x, i)) for i in range(len(x))])/(np.std(x)**2 * len(x))

def plot_autocorrelation():
    xdata = xtcav_times[9800:12500][::2]
    ydata = xtcav_lengths[9800:12500][::2]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xdata - np.min(xdata), autocorrelation(ydata))
    ax.set_xlabel("time (s)")
    ax.set_title("X ray pulse duration autocorrelation")

#    ax.set_xticks(xdata)
#    ax.set_xticklabels(xdata)

    plt.show()

def get_run_epoch_time(run_number):
    # This appears to be the first file associated with a run that's written to the xtc directory.
    XTC_fmt  = "/reg/d/psdm/" + config.exppath + r"/xtc/smalldata/" + config.xtc_prefix + "-r%04d-s00-c00.smd.xtc"
    fname = XTC_fmt % run_number
    #print fname
    t =  os.path.getmtime(fname)
    #print t
    return t

def plot():
    height = 50
    plt.plot_date(mpldates.epoch2num(xtcav_times), filt(xtcav_lengths, 5), 'ob',tz='US/Pacific')
    rtimes = [(r, get_run_epoch_time(r)) for r in range(876, 1137)]
    runs, times = zip(*rtimes)
    plt.plot_date(mpldates.epoch2num(times), np.repeat(np.mean(xtcav_lengths), len(rtimes)), 'og',tz='US/Pacific')
    for run, time in rtimes:
        label = '%d' % run
        plt.annotate(label, xy = (mpldates.epoch2num(time), height), xytext = (0, 5),
            textcoords = 'offset points', ha = 'center', va = 'bottom',rotation = 90)
    plt.show()

def get_run_human_time(run_number):
    t = datetime.datetime.fromtimestamp(get_run_epoch_time(run_number))
    print t
    return t

def get_run_pulse_duration(run_number):
    return pulse_length_from_epoch_time(get_run_epoch_time(run_number))
