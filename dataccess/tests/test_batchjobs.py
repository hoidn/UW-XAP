from dataccess.batchjobs import *
from dataccess.utils import mpi_rank

#def test_jobpool_2():
#    result = []
#    @JobPool
#    def delay1():
##        import time
##        time.sleep(10)
#        time.sleep(2)
#        return 1
#    result.append(delay1())
#    time.sleep(0.1)
#    return result

def test_jobpool_2():
    @JobPool
    def get_rank(sleep = 1):
        time.sleep(2)
        return mpi_rank()
    return get_rank()

def test_jobpool():
    @JobPool
    def get_rank(sleep = 1):
        time.sleep(2)
        return mpi_rank()
    return get_rank()


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

#def test_start_local_engines():
#    engine = Engine(10, 'testengine', ncores = 2)
#    start_engines(engine.profile_name, ncores = engine.ncores, mode = 'local')
#    dview = engine.get_view()
#    def tfunc():
#        return 1
#    assert dview.apply(tfunc).get() == [1, 1]

