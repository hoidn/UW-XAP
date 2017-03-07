from ipyparallel import error
import pdb
import os
from threading import Timer
import ipyparallel
from ipyparallel import Client
from ipyparallel import error

import subprocess
import time
    
engines = []
DEFAULT_NCORES = 8
LOCAL_NCORES = 8

from dataccess import utils
from IPython.config import Application
from dataccess.output import log
#log = Application.instance().log
#log.level = 10


import atexit

def shutdown():
    if not utils.is_mpi():
        for engine in engines:
            engine.terminate(remove = False)

atexit.register(shutdown)

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

def batch_submit(cmd, ncores = DEFAULT_NCORES, queue = 'psfehq'):
    return bashrun(r'bsubx -n {0} -q {1} -o batch_logs/%J.log mpirun '.format(ncores, queue) + cmd)

def add_engine(newid = None, ncores = DEFAULT_NCORES, queue = 'psfehq', mode = 'batch'):
    if newid is None:
        try:
            newid = max([e.job_id for e in engines]) + 1
        except:
            newid = 0
    elif newid in engines:
        raise ValueError("Engine id: %d already exists" % newid)
    engine = Engine(newid, ncores = ncores,
                      queue = queue, init_controller = True, mode = mode)
    engines.append(engine) 
    return engine

def start_engines(name, ncores = DEFAULT_NCORES, mode = 'batch', queue = 'psfehq'):
    """
    Start ipyparallel engines.

    If mode == 'batch', launch the engines in a batch job and return the batch job's id.
    If mode == 'local', run the engines locally and return None.
    """
    def get_batch_id(bsub_stdout):
        import re
        pat = re.compile(r"Job <([0-9]+)>.*")
        return int(pat.search(bsub_stdout).groups()[0])

    host_ip = bashrun('hostname --ip-address')[:-1]
    if mode == 'batch':
        return get_batch_id(batch_submit('ipengine --profile={0} --location={1}'.format(name, host_ip), ncores = ncores, queue = queue))
    elif mode == 'local':
        subprocess.Popen('mpirun -n {0} ipengine --profile={1} >> mpi_log_file 2>&1 &'.format(min(ncores, LOCAL_NCORES), name),
                shell = True)
    else:
        raise ValueError("invalid mode: %s" % mode)

def ipcluster_submit(name, ncores = DEFAULT_NCORES, queue = 'psfehq', mode = 'batch'):
    """
    Launch (1) an ipcontroller (locally) and (2) the specified number of ipengines
    via an LSF batch job.
    """
    ipcontroller_path = os.path.expanduser('~/.ipython/profile_{0}/security/ipcontroller-engine.json'.format(name))
    # TODO:L this check makes it unecessary for the user to specify whether to
    # launch the controllers or not
    #if not os.path.exists(ipcontroller_path):
    host_ip = bashrun('hostname --ip-address')[:-1]
    subprocess.Popen("ipcontroller --profile={0} --location={1} --ip='*' --log-to-file 2>&1 &".format(name, host_ip), shell = True)
    log('launching ipcontroller')
    #else:
    #    log('ipcontroller found')
    batch_id = start_engines(name, ncores = ncores, mode = mode)
    print batch_id
    return batch_id


def bjobs_count(queue_name):
    return int(bashrun('bjobs -q %s -u all -p | wc -l' % queue_name))

#def kill_process(host, pid):
#    print host,pid
#    #subprocess.Popen("ssh {0} \"kill {1}\"".format(host, pid), shell = True)

def kill_job(batch_id):
    try:
        return bashrun('bkill {0}'.format(batch_id))
    except subprocess.CalledProcessError, e:
        print e

def kill_engines():
    """
    kill all running batch jobs.
    """
    subprocess.Popen("bjobs -w | sed -r 's/[ \t\*]+/:/g' | cut -d: -f7 | tail -n +2 | uniq | xargs -I {} -n 1 ssh -o StrictHostKeyChecking=no {} \"ps -u ohoidn | grep python2.7 | sed -r 's/^ +//g' | cut -d' ' -f1 | xargs kill\" >> .ssh_log 2>&1", shell = True)
    subprocess.Popen("ps -aux | egrep ohoidn.*ipcontroller | sed -r 's/ +/:/g' | cut -d: -f2 | xargs kill", shell = True)
    subprocess.Popen("rm .engines", shell = True)


class Engine:
    jobname_prefix = 'mpi'
    def __init__(self, job_id, ncores = DEFAULT_NCORES, queue = 'pfehq',
                shutdown_seconds = None, init_controller = True, mode = 'batch'):
        """
        shutdown_seconds : int
            shut down the engine after this many seconds.
        TODO docstring
        """
        self.job_id = job_id
        self.profile_name = Engine.jobname_prefix + str(self.job_id)
        self.ncores = ncores
        self.queue = queue
        self.busy = False
        self.shutdown_seconds = shutdown_seconds
        self.mode = mode

        if init_controller:
            self.batch_id = ipcluster_submit(self.profile_name, ncores = self.ncores, queue = self.queue, mode = mode)
            
#            if self.shutdown_seconds is not None:
#                t = Timer(self.shutdown_seconds, self.terminate)
#                t.start()
        
    def __eq__(self, other):
        try:
            return other.batch_id == self.batch_id
        except AttributeError:
            return other == self.batch_id

    def set_busy(self):
        self.busy = True
        
    def check_busy(self):
        return self.busy
        
    def unset_busy(self):
        self.busy = False
    
    def terminate(self, remove = True):
        if self.mode == 'batch':
            kill_job(self.batch_id)
        if remove:
            engines.remove(self)

    def restart(self):
        print 'restarting'
        if self.mode == 'batch':
            print self.batch_id
        self.terminate(remove = True)
        engines.append(self)
        self.batch_id = start_engines(self.profile_name, ncores = self.ncores, queue = self.queue,
                mode = self.mode)

    
    def get_rc(self, maxdelay = 5):
        def _get_rc():
            try:
                return self._rc
            except AttributeError:
                self._rc = Client(profile = self.profile_name)
                return self._rc
        err = (IOError, error.TimeoutError)

        return blocking_delay(_get_rc, err, maxdelay = maxdelay)

    def get_view(self, maxdelay = 15):
        def _get_view():
            rc = self.get_rc()
            #sys.stdout.write('.')
            dview = rc[:]
            dview.use_dill()
            return dview
        print("Waiting for engines...")

        result = blocking_delay(_get_view, error.NoEnginesRegistered, maxdelay = maxdelay)
        print ''
        return result
class JobResult:
    def __init__(self, engine, output):
        self.raw_output = output
        self.engine = engine
        if type(output) != ipyparallel.client.asyncresult.AsyncResult:
            self.result = self._process_output(output)

    def _reset_engine(self):
        self.engine.restart()
        self.engine.unset_busy()

    def _process_output(self, output_list):
        """
        Find output from the rank 0 worker and return it. Also resets self.engine.
        """
        hosts = []
        pids = []
        for host, pid, rank, result in output_list:
            hosts.append(host)
            pids.append(pid)
            if rank == 0:
                rank0_result = result
        self._reset_engine()
        return rank0_result


    def get(self):
        while True:
            try:
                return self.result
            except AttributeError:
                try:
                    self.result = self._process_output(self.raw_output.get())
                    return self.result
                except error.TimeoutError:
                    time.sleep(1)
            except error.EngineError:
                self.engine._reset_engine()
                log('restarting engines due to EngineError exception')
                raise

class JobPool:
    """
    Decorator class for dispatching function calls to batch queue
    """

    def __init__(self, func, shutdown = False):
        self.func = func
        self.shutdown = shutdown

    @staticmethod
    def launch_engines(nengines = 1, ncores = DEFAULT_NCORES, queue = 'psfehq', mode = 'batch'):
        return [add_engine(ncores = ncores, queue = queue, mode = mode) for i in range(nengines)]

    @staticmethod
    def remove_job(job_id):
        """
        Remove a job from the pool.
        """
        del[engines[job_id]]
        
    @staticmethod
    def _get_free_engine(mode = 'batch'):
        """Return a free Engine instance, creating new ones if necessary"""
        free = [engine for engine in engines if not engine.check_busy()]
        if free:
            engine = free[0]
        else:
            engine = JobPool.launch_engines(mode = mode)[0]
        return engine


    def __call__(self, *args, **kwargs):
        import config
        if config.autobatch:
            engine = JobPool._get_free_engine(mode = 'batch')
            try:
                dview = engine.get_view()
                print "engine batch: ", engine.batch_id
            except:
                print "Could not obtain batch engine. Attempting to run on local host"
                start_engines(engine.profile_name, ncores = engine.ncores, mode = 'local')
                dview = engine.get_view()
        else:
            try:
                engine = JobPool._get_free_engine(mode = 'local')
                dview = engine.get_view()
            except error.NoEnginesRegistered:
                start_engines(engine.profile_name, ncores = engine.ncores, mode = 'local')
                dview = engine.get_view()
        engine.set_busy()


        @dview.remote(block = False)
        def newfunc(*args, **kwargs):
            #import logging
            host = os.environ['HOSTNAME']
            pid = os.getpid()
            rank = utils.mpi_rank()
            result = self.func(*args, **kwargs)

            #utils.mpi_finalize()
            return host, pid, rank, result

        return JobResult(engine, newfunc(*args, **kwargs))

    
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
    import config
    if config.autobatch:
        JobPool.launch_engines(nengines = 4, ncores = DEFAULT_NCORES)

# launch ipyparallel controller and engines if this is the controlling process.
if not utils.is_mpi():
    init()
