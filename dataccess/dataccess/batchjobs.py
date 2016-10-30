import dill
import os
from threading import Timer
from ipyparallel import Client
from ipyparallel import error

import pdb
import subprocess
import time
    
engines = {}

from IPython.config import Application
log = Application.instance().log
log.level = 10

# TODO: refer to jobs by id instead of profile name
def blocking_delay(func, err, maxdelay = 5, timeout_func = None):
    """
    Evaluate func() once a second until it no longer raises err, or until
    the maximum delay has elapsed.
    timeout_func: None or a function of no arguments to excecute in case of timeout.
    Returns func().
    """
    try:
        return func()
    except err:
        if maxdelay <= 0:
            print func.__name__, "timed out."
            if timeout_func is not None:
                print "running timeout function"
                timeout_func()
            else:
                raise
        else:
            time.sleep(1)
        return blocking_delay(func, err, maxdelay = maxdelay - 1)

def bashrun(cmd, shell = True):
    return subprocess.check_output(cmd, shell=shell)

def batch_submit(cmd, ncores = 6, queue = 'psfehq'):
    return bashrun(r'bsubx -n {0} -q {1} -o batch_logs/%J.log mpirun '.format(ncores, queue) + cmd)

def add_engine(newid = None, ncores = 4, queue = 'psfehq', init = True):
    if newid is None:
        try:
            newid = max(engines) + 1
        except:
            newid = 0
    elif newid in engines:
        raise ValueError("Engine id: %d already exists" % newid)
    profile_name = JobPool._get_profile_name(newid)
    engines[newid] = Engine(newid, profile_name, ncores = ncores,
          queue = queue, init_batch_job = init)
    return engines[newid]

def start_engines(name, host_ip = None, ncores = 2, mode = 'batch'):
    if mode == 'batch':
        assert host_ip is not None
        return batch_submit('ipengine --profile={0} --location={1}'.format(name, host_ip), ncores = ncores)
    elif mode == 'local':
        import os
        os.system('mpirun -n 4 ipengine --profile={0} --log-to-file 2>&1 &'.format(name))
    else:
        raise ValueError("invalid mode: %s" % mode)

def ipcluster_submit(name, ncores = 6, queue = 'psfehq', mode = 'batch'):
    """
    Launch (1) an ipcontroller (locally) and (2) the specified number of ipengines
    via an LSF batch job.
    """
    host_ip = bashrun('hostname --ip-address')[:-1]
    os.system("ipcontroller --profile={0} --location={1} --ip='*' --log-to-file 2>&1 &".format(name, host_ip))
    #os.system('mpirun -n 4 ipengine --profile={0} --location={1} &'.format(name, host_ip))
    #return batch_submit('ipengine --profile={0} --location={1}'.format(name, host_ip), ncores = ncores)
    return start_engines(name, host_ip, ncores = ncores, mode = mode)


def bjobs_count(queue_name):
    return int(bashrun('bjobs -q %s -u all -p | wc -l' % queue_name))

class Engine:
    def __init__(self, job_id, profile_name, ncores = 6, queue = 'pfehq',
                shutdown_seconds = None, init_batch_job = True):
        """
        shutdown_seconds : int
            shut down the engine after this many seconds.
        TODO docstring
        """
        self.job_id = job_id
        self.profile_name = profile_name
        self.ncores = ncores
        self.queue = queue
        self.busy = False
        self.shutdown_seconds = shutdown_seconds

        if init_batch_job:
            #pdb.set_trace()
            self.init()
        
    def init(self):
        ipcluster_submit(self.profile_name, ncores = self.ncores, queue = self.queue)
        
        if self.shutdown_seconds is not None:
            t = Timer(self.shutdown_seconds, self.terminate)
            t.start()
        
    def set_busy(self):
        self.busy = True
        
    def check_busy(self):
        return self.busy
        
    def unset_busy(self):
        self.busy = False
    
    def terminate(self, hub = True):
        self.get_rc().shutdown(hub = hub)
	JobPool.remove_job(self.job_id)
    
    def get_rc(self, maxdelay = 5):
        def _get_rc():
            return Client(profile = self.profile_name)
        err = (IOError, error.TimeoutError)

        return blocking_delay(_get_rc, err, maxdelay = maxdelay)

    def get_view(self, maxdelay = 15):
        def _get_view():
            rc = self.get_rc()
            #sys.stdout.write('.')
            dview = rc[:]
            dview.use_dill()
            return dview
        import sys
        print("Waiting for engines...")

        result = blocking_delay(_get_view, error.NoEnginesRegistered, maxdelay = maxdelay)
        print ''
        return result
#        except error.NoEnginesRegistered:
#            print "caught exception: error.NoEnginesRegistered"
#            start_engines(self.profile_name, ncores = self.ncores, mode = 'local')
#            return self.get_view()

class JobPool:
    """
    Decorator class for dispatching function calls to batch queue
    """
    jobname_prefix = 'mpi'

    def __init__(self, func, shutdown = False):
        self.func = func
        self.shutdown = shutdown

    @staticmethod
    def _get_profile_name(jobid):
        return JobPool.jobname_prefix + str(jobid)
        
    @staticmethod
    def launch_engines(nengines = 1, ncores = 4, queue = 'psfehq'):
        return [add_engine(ncores = ncores, queue = queue) for i in range(nengines)]

    @staticmethod
    def remove_job(job_id):
        """
        Remove a job from the pool.
        """
        del[engines[job_id]]
        
    @staticmethod
    def _get_free_engine():
        """Return a free Engine instance, creating new ones if necessary"""
        #pdb.set_trace()
        free = [engine for jid, engine in engines.iteritems() if not engine.check_busy()]
        if free:
            return free[0]
        else:
            return JobPool.launch_engines()[0]


    def __call__(self, *args, **kwargs):
        engine = JobPool._get_free_engine()
        engine.set_busy()
        try:
            dview = engine.get_view()
        except:
            print "Could not obtain batch engine. Attempting to run on local host"
            start_engines(engine.profile_name, ncores = engine.ncores, mode = 'local')
            dview = engine.get_view()

        @dview.remote(block = False)
        def newfunc(*args, **kwargs):
            import logging
            import os

            LOG_FILENAME = '{0}.log'.format(os.getpid())
            logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

            logging.debug('This message should go to the log file')

            return self.func(*args, **kwargs)

#            def cleanup():
#                import sys
#                sys.exit()
#            result = self.func(*args, **kwargs)
#            if self.shutdown:
#                t = Timer(1, cleanup)
#                t.start()
#            return result

            #jengine.set_busy()
            #result = self.func(*args, **kwargs)
#            def cleanup(check_interval = 1):
#                if result.done():
#                    engine.unset_busy()
#                    print "unset busy"
                    #rc.shutdown(hub=True)
#                else:
#                    t = Timer(check_interval, cleanup)
#                    t.start()
#            return result
        result = newfunc(*args, **kwargs).get()
        if self.shutdown:
            engine.terminate()
        else:
            engine.unset_busy()
        engine.get_rc().purge_everything()
        return result

    

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

def init():
    JobPool.launch_engines(nengines = 6, ncores = 4)
    import dill
    with open('.engines', 'wb') as f:
        dill.dump(engines.keys(), f)

try:
    print ("looking for engines file...")
    with open('.engines', 'rb') as f:
        engine_ids = dill.load(f)
        [add_engine(newid = i, init = False) for i in engine_ids]
except IOError:
    print ("File .engines not found; launching batch jobs")
    init()
print "engines:", engines
