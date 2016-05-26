import os
from threading import Timer
from ipyparallel import Client
from ipyparallel import error

import pdb
import subprocess
import time

# TODO: refer to jobs by id instead of profile name
def blocking_delay(func, err, maxdelay = 5):
    """
    Evaluate func() once a second until it no longer raises err, or until
    the maximum delay has elapsed.

    Returns func().
    """
    try:
        return func()
    except err:
        if maxdelay <= 0:
            print func.__name__, "timed out."
            raise
        time.sleep(1)
        return blocking_delay(func, err, maxdelay = maxdelay - 1)

def bashrun(cmd, shell = True):
    return subprocess.check_output(cmd, shell=shell)

def batch_submit(cmd, ncores = 6, queue = 'psfehq'):
    return bashrun(r'bsub -n {0} -q {1} -o batch_logs/%J.log mpirun '.format(ncores, queue) + cmd)

def ipcluster_submit(name, ncores = 6, queue = 'psfehq'):
    """
    Launch (1) an ipcontroller (locally) and (2) the specified number of ipengines
    via an LSF batch job.
    """
    host_ip = bashrun('hostname --ip-address')[:-1]
    os.system("ipcontroller --profile={0} --location={1} --ip='*' --log-to-file > /dev/null 2>&1 &".format(name, host_ip))
    return batch_submit('ipengine --profile={0} --location={1}'.format(name, host_ip), ncores = ncores)


def bjobs_count(queue_name):
    return int(bashrun('bjobs -q %s -u all -p | wc -l' % queue_name))

class Job:
    def __init__(self, job_id, profile_name, ncores = 6, queue = 'pfehq',
                shutdown_seconds = 60, init_batch_job = False, jobpool = None):
        self.job_id = job_id
        self.profile_name = profile_name
        self.ncores = ncores
        self.queue = queue
        self.busy = False
        self.shutdown_seconds = shutdown_seconds
        self.jobpool = jobpool # parent JobPool instance

        if init_batch_job:
            #pdb.set_trace()
            self.init()
        
    def init(self):
        ipcluster_submit(self.profile_name, ncores = self.ncores, queue = self.queue)
        
        t = Timer(self.shutdown_seconds, self.terminate)
        t.start()
        
    def set_busy(self):
        self.busy = True
        
    def check_busy(self):
        return self.busy
        
    def unset_busy(self):
        self.busy = False
    
    def terminate(self, hub = False):
        self.get_rc().shutdown(hub = hub)
        if self.jobpool is not None:
            self.jobpool.remove_job(self.job_id)
    
    def get_rc(self, maxdelay = 5):
        def _get_rc():
            return Client(profile = self.profile_name)
        err = (IOError, error.TimeoutError)

        return blocking_delay(_get_rc, err, maxdelay = maxdelay)
            
    
    def get_view(self, maxdelay = 15):
        def _get_view():
            rc = self.get_rc()
            print "Waiting for engines..."
            dview = rc[:]
            dview.use_dill()
            return dview

        return blocking_delay(_get_view, error.NoEnginesRegistered, maxdelay = maxdelay)
    
class JobPool:
    """
    Decorator class for dispatching function calls to batch queue
    """
    default_jobs = 10
    jobname_prefix = 'mpi'
    jobs = {}
    
    def __init__(self, func):
        if not JobPool.jobs:
            self.add_jobs(init = True)
        self.func = func
        
    @staticmethod
    def _get_profile_name(jobid):
        return JobPool.jobname_prefix + str(jobid)
        
    def add_jobs(self, njobs = 2, ncores = 4, queue = 'psfehq', init = False):
        def add_job():
            if not self.jobs:
                newid = 0
            else:
                newid = max(self.jobs) + 1
            profile_name = self._get_profile_name(newid)
            JobPool.jobs[newid] = Job(newid, profile_name, ncores = ncores,
                  queue = queue, init_batch_job = init, jobpool = self)
            return JobPool.jobs[newid]
        
        return [add_job() for i in range(njobs)]

    def remove_job(self, job_id):
        """
        Remove a job from the pool.
        """
        del[self.jobs[job_id]]
        
    def _get_free_job(self):
        """Return a free Job instance, creating new ones if necessary"""
        pdb.set_trace()
        free = [job for jid, job in self.jobs.iteritems() if not job.check_busy()]
        if free:
            return free[0]
        else:
            return add_jobs()[0]

    def _get_job_view(self, job, maxdelay = 15):
        """Return a view, delaying, if necessary, until engines are available."""
        return job.get_view(maxdelay = maxdelay)
    
    def __call__(self, *args, **kwargs):
        job = self._get_free_job()
        dview = self._get_job_view(job)

        @dview.remote(block = False)
        def newfunc(*args, **kwargs):
            job.set_busy()
            result = self.func(*args, **kwargs)
            def cleanup(check_interval = 1):
                if result.done():
                    job.unset_busy()
                    print "unset busy"
                    #rc.shutdown(hub=True)
                else:
                    t = Timer(check_interval, cleanup)
                    t.start()
            return result

        return newfunc(*args, **kwargs)

    

class Bqueue:
    sizes = {'psanaq': 960., 'psfehq': 288., 'psnehq': 288.}
    def __init__(self, name):
        self.name = name
        if name == 'psfehq' or name == 'psnehq':
            self.upstream_q = Bqueue(name[:-1] + 'hiprioq')
        else:
            self.upstream_q = None
        
    def number_pending(self):
        if self.upstream_q is not None:
            return bjobs_count(self.name) + bjobs_count(self.upstream_q.name)
        else:
            return bjobs_count(self.name)
        
    def key(self):
        """
        Sorting key for this class.
        """
        return self.number_pending()/Bqueue.sizes[self.name]

def best_queue(usable_queues):
    """Return the name of the least-subscribed batch queue"""
    return min(usable_queues, key = lambda q: q.key())

usable_queues = map(Bqueue, ('psanaq', 'psfehq', 'psnehq'))
