import config
from dataccess import output
from contextlib import contextmanager
from StringIO import StringIO

@contextmanager
def captured_output():
    """
    From http://stackoverflow.com/questions/4219717/how-to-assert-output-with-nosetest-unittest-in-python
    """
    import sys
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

def test_conditional_print():
    def stringproc(sio):
        return sio.getvalue().strip()
    def test1():
        output.log("hello")
    def test2():
        output.log("world")
    config.stdout_to_file = False
    with captured_output() as (out, err):
        test1()
        assert stringproc(out) == 'hello'
    config.stdout_to_file = True
    reload(output)
    with captured_output() as (out, err):
        test2()
        assert stringproc(out) == ''

