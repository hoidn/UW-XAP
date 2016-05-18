#!/usr/bin/env python2.7

import os; activate_this=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'activate_this.py'); exec(compile(open(activate_this).read(), activate_this, 'exec'), dict(__file__=activate_this)); del os, activate_this

# EASY-INSTALL-SCRIPT: 'ppft==1.6.4.6','ppserver.py'
__requires__ = 'ppft==1.6.4.6'
__import__('pkg_resources').run_script('ppft==1.6.4.6', 'ppserver.py')