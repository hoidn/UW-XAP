import sys
sys.path.append('/reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages')
sys.path.append('/reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages/pathos-0.2a1.dev0-py2.7.egg')
sys.path.append('/reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages/dataccess-1.0-py2.7.egg')
# The version of multiprocessing installed on the psana system is incompatible
# with pathogen. We need to install multiprocessing locally and push its
# to sys.path
#sys.path.insert(0, '/reg/neh/home/ohoidn/anaconda/lib/python2.7/site-packages/multiprocessing/')
