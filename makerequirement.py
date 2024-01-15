import os
import platform
import sys


# find pwd
project_root = os.path.dirname(os.path.realpath(__file__))
print('current path' + project_root)


# code for Linux or Windows system
if platform.system() == 'Linux':
    command = sys.executable + ' -m pip freeze > ' + project_root + '/requirements.txt'
if platform.system() == 'Windows':
    command = '"' + sys.executable + '"' + ' -m pip freeze > "' + project_root + '\\requirements.txt"'

print(command)


# execute
os.popen(command)