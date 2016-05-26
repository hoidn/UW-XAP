from dataccess.batchjobs import *
from dataccess.utils import mpi_rank
import time

def test_jobpool():
    @JobPool
    def get_rank(sleep = 1):
        return mpi_rank()

    return get_rank()

def test_jobpool_2():
    @JobPool
    def delay1():
        time.sleep(300)
        return 1

    @JobPool
    def delay2():
        time.sleep(300)
        return 2


    @JobPool
    def delay3():
        time.sleep(300)
        return 3

    result = [delay1(), delay2(), delay3()]
    return result
    #assert len(JobPool.jobs) == 3

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
    
#def test_job():
    
