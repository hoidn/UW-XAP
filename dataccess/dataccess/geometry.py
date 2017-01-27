"""
This module contains functions related to the calculation of powder diffraction
patterns from area detector data.
"""

import numpy.ma as ma
from scipy.interpolate import griddata
import itertools
import numpy as np
from scipy.ndimage.filters import gaussian_filter

import utils
from output import log
import config

DEFAULT_SMOOTHING = 0.
# hbar c in eV * Angstrom
HBARC = 1973. 

def get_detid_parameters(detid):
    """
    Given a detector ID, extract its detector geometry parameters from
    config.py and return them.
    """
    paramdict = config.detinfo_map[detid].geometry
    (phi, x0, y0, alpha, r) = paramdict['phi'], paramdict['x0'],\
        paramdict['y0'], paramdict['alpha'], paramdict['r']
    return (phi, x0, y0, alpha, r)


def get_x_y(imarray, phi, x0, y0, alpha, r):
    """
    Given CSPAD geometry parameters and an assembeled image data array, return
    two arrays (each of the same shape as the image data) with values replaced
    by row/column indices.
    """
    length, width = imarray.shape
    y = np.vstack(np.ones(width)*i for i in range(length))
    ecks = np.vstack([1 for i in range(length)])
    x = np.hstack(ecks*i for i in range(width))
    return x, y

def get_beta_rho(imarray, phi, x0, y0, alpha, r):
    """
    Given CSPAD geometry parameters and an assembeled image data array, return
    (1) an array (of the same shape as the image data) with 2theta scattering
    angle values and (2) an array (of the same shape as the image data) with
    rho (distance) values.
    """
    x, y = get_x_y(imarray, phi, x0, y0, alpha, r)
    try:
        x2 = -np.cos(phi) *(x-x0) + np.sin(phi) * (y-y0)
        y2 = -np.sin(phi) * (x-x0) - np.cos(phi) * (y-y0)
    except AttributeError:
        raise AttributeError("Missing geometry data in config.py")
    rho = (r**2 + x2**2 + y2**2)**0.5
    y1 = y2 * np.cos(alpha) + r * np.sin(alpha)
    z1 = - y2 * np.sin(alpha) + r * np.cos(alpha)
    
    # beta is the twotheta value for a given (x,y)
    beta = np.arctan2((y1**2 + x2**2)**0.5, z1) * 180 / np.pi
    return beta, rho

def get_phi2(imarray, detid):
    """
    Given CSPAD geometry parameters and an assembeled image data array, return
    (1) an array (of the same shape as the image data) with values of phi2,
    the azimuthal angle with respect to the x-ray beam.
    """
    (phi, x0, y0, alpha, r) = get_detid_parameters(detid)
    x, y = get_x_y(imarray, phi, x0, y0, alpha, r)
    try:
        x2 = -np.cos(phi) *(x-x0) + np.sin(phi) * (y-y0)
        y2 = -np.sin(phi) * (x-x0) - np.cos(phi) * (y-y0)
    except AttributeError:
        raise AttributeError("Missing geometry data in config.py")
    y1 = y2 * np.cos(alpha) + r * np.sin(alpha)
    z1 = - y2 * np.sin(alpha) + r * np.cos(alpha)
    phi2 = np.arctan2(y1, z1)
    return phi2

def select_phi2(imarray, phi2_0, delta_phi2, detid):
    """
    Mask out all values outside of the specified phi2 range.
    """
    result = imarray.copy()
    phi2 = get_phi2(imarray, detid)
    result = np.where(np.logical_and(phi2 > phi2_0 - delta_phi2/2, phi2 < phi2_0 + delta_phi2/2), result, 0.)
    return result

# translate(phi, x0, y0, alpha, r)
# Produces I vs theta values for imarray. For older versions, see bojangles_old.py
# Inputs:  detector configuration parameters and diffraction image
# Outputs:  lists of intensity and 2theta values (data)
def translate(phi, x0, y0, alpha, r, imarray, fiducial_ellipses = None):
    # fiducial ellipse width
    ew = .1
    # beta is the twotheta value for a given (x,y)
    beta, rho = get_beta_rho(imarray, phi, x0, y0, alpha, r)
    if fiducial_ellipses is not None:
        fiducial_value = np.max(np.nan_to_num(imarray))/5.
        for ang in fiducial_ellipses:
            imarray = np.where(np.logical_and(beta > ang - ew, beta < ang + ew), fiducial_value, imarray)
    
    newpoints = np.vstack((beta.flatten(), imarray.flatten()))
    
    return newpoints.T, imarray


def binData(mi, ma, stepsize, valenza = True):
    """
    Input:  a minimum, a maximum, and a stepsize
    Output:  a list of bins
    """
    log( "creating angle bins")
    binangles = list()
    binangles.append(mi)
    i = mi
    while i < ma-(stepsize/2):
        i += stepsize
        binangles.append(i)

    return binangles

#@utils.eager_persist_to_file("cache/xrd.process_imarray/")
def process_imarray(detid, imarray, nbins = 1000,
        fiducial_ellipses = None, bgsub = True, compound_list = [],
        pre_integration_smoothing = 0,
        **kwargs):
    """
    Given a detector ID and assembeled CSPAD image data array, compute the
    powder pattern.

    Outputs:  data in bins, intensity vs. theta, as lists (NOT numpy arrays)
    """
    if bgsub and not compound_list:
        bgsub = False
        log( "Overriding bg_sub to False due to empty compound_list")
        
    @utils.eager_persist_to_file('cache/xrd/process_imarray/expanded_mask')
    def expanded_mask(arr):
        """
        Return a boolean array that masks out zero values
        in and their neighbors in the input array.\
        """
        import numpy.ma as ma
        import maskmaker
        mask = ma.make_mask(np.ones_like(arr))
        mask = np.where(arr == 0, False, True)
        return maskmaker.makemask(mask, 2 * pre_integration_smoothing)
    # TODO: make this take dataset as an argument
    (phi, x0, y0, alpha, r) = get_detid_parameters(detid)
    if bgsub:
        imarray = subtract_background_full_frame(imarray, detid, compound_list)
    
    mask = expanded_mask(imarray)
    imarray = gaussian_filter(imarray, pre_integration_smoothing) * mask
    data, imarray = translate(phi, x0, y0, alpha, r, imarray, fiducial_ellipses = fiducial_ellipses)
    
    thetas = data[:,0]
    intens = data[:,1]

    # algorithm for binning the data
    ma = max(thetas)
    mi = min(thetas)
    stepsize = (ma - mi)/(nbins)
    binangles = binData(mi, ma, stepsize)
    numPix = [0] * (nbins+1)
    intenValue = [0] * (nbins+1)
    
    log( "putting data in bins"        )
    # find which bin each theta lies in and add it to count
    for j,theta in enumerate(thetas):
        if intens[j] != 0:
            k = int(np.floor((theta-mi)/stepsize))
            numPix[k]=numPix[k]+1
            intenValue[k]=intenValue[k]+intens[j]
    # form average by dividing total intensity by the number of pixels
    log( "adjusting intensity")
    adjInten = np.nan_to_num((np.array(intenValue)/np.array(numPix)))
    
#    if np.min(adjInten) < 0:
#        log( "WARNING: Negative values have been suppressed in final powder pattern (may indicate background subtraction with an inadequate data mask).")
#        adjInten[adjInten < 0.] = 0.
    return binangles, list(  adjInten ), imarray


# From: http://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent
def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m
def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

def trim_array(imarray):
    """
    Trim the input array if it isn't square (the above 2d polynomial fitting
    function requires an nxn matrix). Returns a view of the original array.
    """
    dimx, dimy = imarray.shape
    difference = dimy - dimx
    if difference: 
        if difference > 0:
            trimmed = imarray[:, :dimy - difference]
        elif difference < 0:
            trimmed = imarray[:dimx + difference, :]
    else:
        trimmed = imarray
    return trimmed

def pad_array(imarray):
    """
    Pad the input array if it isn't square (the above 2d polynomial fitting
    function requires an nxn matrix). Returns a new array. 
    """
    dimx, dimy = imarray.shape
    difference = dimy - dimx
    if difference: 
        if difference > 0:
            padded = np.vstack((imarray, np.zeros((difference, dimy))))
        elif difference < 0:
            padded = np.hstack((imarray, np.zeros((dimx, -difference))))
    else:
        padded = imarray
    return padded
    

def interp_2d_nearest_neighbor(imarray, detid, smoothing = DEFAULT_SMOOTHING):
    """
    Return a background frame for imarray. 

    The background level is computed by masking pixels in imarray located
    near powder peaks and replacing their values using a 2d-interpolation 
    between non-masked regions of the frame.

    Keyword arguments:
        -smoothing: standard deviation of gaussian smoothing kernel to apply
        to the interpolated  background.
        -method: interpolation mode for scipy.interpolate.griddata
    """

    
    # TODO: use a better 2d-interpolation than nearest neighbor
    geometry_params = get_detid_parameters(detid)
    dimx, dimy = np.shape(imarray)
    gridx, gridy = map(lambda arr: 1. * arr, get_x_y(imarray, *geometry_params))

    # flattened values of all pixels
    x, y = gridx.flatten(), gridy.flatten()
    z = imarray.flatten()

    z_good = np.where(z != 0)[0]
    if len(z_good) > 0:
        resampled = griddata(np.array([x[z_good], y[z_good]]).T, z[z_good], (gridx, gridy), method = 'nearest')
    else:
        resampled = imarray
    smoothed = gaussian_filter(resampled, smoothing)
    return smoothed, resampled

@utils.eager_persist_to_file('cache/xrd/CTinterpolation')
def CTinterpolation(imarray, detid, smoothing = 10):
    """
    Do a 2d interpolation to fill in zero values of a 2d ndarray.
    
    Uses scipy.interpolate import CloughTocher2DInterpolator.
    
    Arguments:
        imarray : np.ndarray
        detid : string
        smoothing : numeric
    """
    from scipy.interpolate import CloughTocher2DInterpolator as ct
    geometry_params = get_detid_parameters(detid)
    dimx, dimy = np.shape(imarray)
    gridx, gridy = map(lambda arr: 1. * arr, get_x_y(imarray, *geometry_params))
    

    def interp_2d(imarray):
        # flattened values of all pixels
        z = imarray.flatten()
        z_good = np.where(z != 0)[0]
        if len(z_good) == 0:
            return np.zeros_like(imarray)
        else:
            x, y = gridx.flatten(), gridy.flatten()
            xgood, ygood = x[z_good], y[z_good]

            points = np.vstack((xgood, ygood)).T
            values = z[z_good]
            interpolator = ct(points, values)
            return interpolator(x, y).reshape(imarray.shape)

    # Input to the CT interpolation is a smoothed NN interpolation
    # This pre-interpolation step, combined with a sufficiently large value of
    # smoothing, is often necessary to prevent the interpolation from
    # oscillating/overshooting.
    smoothNN, _ = interp_2d_nearest_neighbor(imarray, detid, smoothing = smoothing)
    smooth_masked = np.where(np.isclose(imarray, 0), 0., smoothNN)
    CTinterpolated = interp_2d(smooth_masked)
    
    # Fill in NAN values from outside the convex hull of the interpolated points
    combined = np.where(np.isnan(CTinterpolated), smoothNN, CTinterpolated)
    return combined


def get_background_full_frame(imarray, detid, compound_list, smoothing = DEFAULT_SMOOTHING,
        width = config.peak_width):
    # TODO: reorganize this and the other background-calculation function
    """
    Calculate a background frame from imarray and return the result. 

    Keyword arguments:
        -smoothing: standard deviation of gaussian smoothing kernel to apply
            to the interpolated  background.
        -width: angular width of regions (centered on powder peaks) that will
            be excluded from the source array from which the background is
            interpolated.
    """
    # If compound_list is empty the computed background will include all our
    # signal.
    if not compound_list:
        raise ValueError("compounds_list is empty")
    bgfit = imarray.copy()
    
    # mask based on good pixels
    pixel_mask = utils.combine_masks((bgfit != 0), [], transpose = True)

    # mask based on powder peak locations
    powder_mask = make_powder_ring_mask(detid, bgfit, compound_list, width = width)

    # union of the two masks
    combined_mask = powder_mask & pixel_mask
    bgfit[~combined_mask] = 0.

    # TODO: Need a mechanism for getting dataset-specific paths.
    if utils.isroot():
        np.save('powder_with_cutout.npy', bgfit)

    # compute interpolated background
    bg = CTinterpolation(bgfit, detid, smoothing = smoothing)

    # zero out bad/nonexistent pixels
    bg[~pixel_mask] = 0.
    return bg

def subtract_background_full_frame(imarray, detid, compound_list, smoothing = DEFAULT_SMOOTHING,
        width = config.peak_width):
    """
    Background-subtract imarray and return the result. 

    This function does not mutate imarray.

    Keyword arguments:
        -smoothing: standard deviation of gaussian smoothing kernel to apply
            to the interpolated  background.
        -width: angular width of regions (centered on powder peaks) that will
            be excluded from the source array from which the background is
            interpolated.
    """
    # TODO: might be good to log intermediate stages
    bg_smooth = get_background_full_frame(imarray, detid, compound_list, smoothing = smoothing, width = width)
    utils.save_image('detector_images/last_bg.png', bg_smooth)
    result = imarray - bg_smooth
    return result

def get_powder_angles(compound, peak_threshold = 0.02, filterfunc = lambda x: True):
    """
    Accessor function for powder data in config.py

    Returns a list of Bragg peak angles, filtered using filterfunc.
    """
    if compound in config.powder_angles:
        return filter(filterfunc, config.powder_angles[compound])
    else:
        energy = config.photon_energy
        fname = utils.resource_path('data/' + compound + '.csv')
        try:
            powder_q, intensities = np.genfromtxt(fname, delimiter = ',').T
        except IOError:
            raise IOError("Simulated diffraction file " + fname + ": not found")
        powder_q = powder_q[intensities > np.max(intensities) * peak_threshold]
        powder_angles = 2 * np.arcsin(powder_q * HBARC / (2 * energy))
        powder_angles = powder_angles[~np.isnan(powder_angles)]
        return filter(filterfunc, list(np.rad2deg(powder_angles)))

def make_powder_ring_mask(detid, imarray, compound_list, width = config.peak_width):
    """
    Given a detector ID, assembeled image data array, and list of
    polycrystalline compounds in the target, return a mask that
    excludes pixels located near powder peaks.
    """
    angles = []
    for compound in compound_list:
        try:
            compound_xrd = get_powder_angles(compound)
        except KeyError:
            raise KeyError("No XRD reference data found for compound: " + compound)
        if isinstance(compound_xrd, list): # compound_xrd is a list of angles
            angles = angles + compound_xrd
        else: # compound_xrd is a path
            # TODO: implement this
            raise NotImplementedError("compound_xrd path")

    # Initialize mask to all True
    mask = ma.make_mask(np.ones(np.shape(imarray)))
    (phi, x0, y0, alpha, r) = get_detid_parameters(detid)
    betas, rho = get_beta_rho(imarray, phi, x0, y0, alpha, r)
    for ang in angles:
        mask = np.where(np.logical_and(betas > ang - width/2., betas < ang + width/2.), False, mask)
    return mask
