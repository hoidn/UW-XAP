#!/usr/bin/env python2.7

import os; activate_this=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'activate_this.py'); exec(compile(open(activate_this).read(), activate_this, 'exec'), dict(__file__=activate_this)); del os, activate_this

# EASY-INSTALL-SCRIPT: 'dill==0.2.5','get_objgraph.py'
__requires__ = 'dill==0.2.5'
__import__('pkg_resources').run_script('dill==0.2.5', 'get_objgraph.py')