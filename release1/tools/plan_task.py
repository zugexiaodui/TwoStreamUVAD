#!/usr/bin/python3
import os
import time
import argparse
from typing import Tuple

parser = argparse.ArgumentParser("Schedule task.")
parser.add_argument('cmd', type=str)
parser.add_argument('pid', type=int, nargs="+")
parser.add_argument('--cond', '-c', type=str, default='all', choices=('all', 'any'))
parser.add_argument('--wtime', '-t', type=int, default=60)

args = parser.parse_args()


def pid_exisits(pid: int):
    pid_exists = None
    try:
        os.kill(pid, 0)
        pid_exists = True
    except:
        pid_exists = False

    return pid_exists


cmd: str = args.cmd
dst_pid_list: Tuple[int] = args.pid
cond_str: str = args.cond
wtime: int = args.wtime

cond_type = None
if cond_str == 'all':
    cond_type = all
elif cond_str == 'any':
    cond_type = any
else:
    raise NameError(cond_str)

while True:
    print(time.ctime())
    if cond_type([not pid_exisits(dst_pid) for dst_pid in dst_pid_list]):
        print(cmd)
        os.system(cmd)
        break

    time.sleep(wtime)
