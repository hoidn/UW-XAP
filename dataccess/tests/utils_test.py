from dataccess import utils

from mpi4py import MPI

def test_mpimap():
    square = lambda x: x**2
    lst = range(5)
    result = utils.mpimap(square, lst)
    print result
    assert  result == map(square, lst)

test_mpimap()

#MPI.Finalize()
