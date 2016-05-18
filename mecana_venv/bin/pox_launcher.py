#!/usr/bin/env python2.7

import os; activate_this=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'activate_this.py'); exec(compile(open(activate_this).read(), activate_this, 'exec'), dict(__file__=activate_this)); del os, activate_this

# EASY-INSTALL-SCRIPT: 'pox==0.2.2','pox_launcher.py'
__requires__ = 'pox==0.2.2'
__import__('pkg_resources').run_script('pox==0.2.2', 'pox_launcher.py')