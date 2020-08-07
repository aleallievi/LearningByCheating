import psutil
import os

PROCNAME = "Carla"

for proc in psutil.process_iter():
    if PROCNAME in proc.name():
        pid = proc.pid
        os.kill(pid, 9)
