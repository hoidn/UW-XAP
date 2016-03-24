from scipy import misc
import numpy as np
import pylab as plt
import os.path
import sys

# the basic approach is to look at each pixel in the png, look out n=avoidby pixels from that pixel and count how many zeros are observed. if the number of zeros is >= maxzeros, that pixel will be excluded. All others are included. 

# it should generate a file with the same path and start of the name as fname, then append avoidby_maxzeros.mask, so 40_quad1.png could become
# 40_quad1mask_2_2.mask

# parameters you might want to change are here

def nearestxys(x,y, shapex, shapey, n=2):
    shifts = np.sort(np.hstack((np.arange(n+1),-np.arange(n+1)[1:])))
    #for n=2 should make [-2, -1, 0, 1, 2]
    inds = []
    for shiftx in shifts:
        for shifty in shifts:
            newx = x+shiftx
            newy = y+shifty
            if newx>=0 and newx<=shapex-1 and newy>=0 and newy<=shapey-1:
                inds.append([newx, newy])
    return inds

def has_zero_neighbors(arr, x, y, shapex, shapey, n):
    shifts = np.sort(np.hstack((np.arange(n+1),-np.arange(n+1)[1:])))
    for shiftx in shifts:
        for shifty in shifts:
            newx = x+shiftx
            newy = y+shifty
            if newx>=0 and newx<=shapex-1 and newy>=0 and newy<=shapey-1:
                if arr[newx, newy] == 0:
                    return True
    return False

#def makemask(image, avoidby, maxzeros):
#	mask = np.zeros(image.shape, dtype="bool")
#	for x in xrange(image.shape[0]):
#		for y in xrange(image.shape[1]):
#			nearestinds = nearestxys(x,y, image.shape[0], image.shape[1], avoidby)
#			numzeros=0
#			for (xx, yy) in nearestinds:
#				if image[xx,yy]==0:
#					numzeros+=1
#			shoulduse=numzeros<=maxzeros
#			mask[x,y]=shoulduse
#	return mask

def makemask(image, avoidby):
    mask = np.zeros(image.shape, dtype="bool")
    for x in xrange(image.shape[0]):
        for y in xrange(image.shape[1]):
            if has_zero_neighbors(image, x, y, image.shape[0], image.shape[1], avoidby):
                shoulduse = False
            else:
                shoulduse = True
            mask[x,y]=shoulduse
    return mask
