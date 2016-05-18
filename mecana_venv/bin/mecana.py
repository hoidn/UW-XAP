#!/usr/bin/env python2.7

import os; activate_this=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'activate_this.py'); exec(compile(open(activate_this).read(), activate_this, 'exec'), dict(__file__=activate_this)); del os, activate_this

# EASY-INSTALL-SCRIPT: 'dataccess==1.0','mecana.py'
__requires__ = 'dataccess==1.0'
__import__('pkg_resources').run_script('dataccess==1.0', 'mecana.py')