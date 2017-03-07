import config
config.autobatch = False

from dataccess.batchjobs import *
from dataccess.utils import mpi_rank
from dataccess.utils import mpimap


def test_jobpool():
    @JobPool
    def get_rank(sleep = 1):
        time.sleep(2)
        return mpi_rank()
    return get_rank()

def test_jobpool_2():
    @JobPool
    def get_rank(sleep = 1):
        time.sleep(2)
        return mpimap(lambda x: mpi_rank(), [1, 1, 1, 1])
    result = get_rank()
    assert engines[0].check_busy() == True
    assert set(result.get()) == set(range(4))
    assert engines[0].check_busy() == False
    return result

def test_jobpool_3():
    import numpy as np
    @JobPool
    def square(arr):
        return mpimap(lambda x: x**2, arr)

    arr = np.array(range(10))
    result = square(arr).get()
    assert np.all(np.isclose(result, [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]))
    return result

def test_jobpool_4():
    import numpy as np
    @JobPool
    def square(arr):
        from dataccess import data_access
        return mpimap(lambda x: x**2, arr)

    arr = np.array(range(10))
    result = square(arr).get()
    assert np.all(np.isclose(result, [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]))
    return result

def test_blocking_delay():
    def mf():
        raise ValueError
    start = time.time()
    try:
        blocking_delay(mf, ValueError, maxdelay = .9)
    except:
        pass
    diff = time.time() - start
    assert .5 < diff < 1.5


def check_pid(pid):        
    """ Check For the existence of a unix pid. """
    import os
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True

#def test_kill_process():
#    host = bashrun("hostname")
#    import subprocess
#    pid = subprocess.Popen('sleep 100', shell = True).pid
#    assert check_pid(pid)
#    kill_process(host, pid)
#    time.sleep(0.1)
#    assert not check_pid(pid)

#def test_start_local_engines():
#    engine = Engine(10, 'testengine', ncores = 2)
#    start_engines(engine.profile_name, ncores = engine.ncores, mode = 'local')
#    dview = engine.get_view()
#    def tfunc():
#        return 1
#    assert dview.apply(tfunc).get() == [1, 1]


