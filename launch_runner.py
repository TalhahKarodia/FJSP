"""Launch runners as fully detached Windows processes."""
import subprocess
import sys
import os

runners = [
    ("run_scenario1.py", "results/log_s1.txt", "results/log_s1.err.txt"),
    ("run_scenario2.py", "results/log_s2.txt", "results/log_s2.err.txt"),
    ("run_baselines.py", "results/log_baselines.txt", "results/log_baselines.err.txt"),
]

# Windows flags for full detachment
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

for script, logfile, errfile in runners:
    with open(logfile, 'w') as out, open(errfile, 'w') as err:
        p = subprocess.Popen(
            [sys.executable, "-u", script],
            stdout=out,
            stderr=err,
            stdin=subprocess.DEVNULL,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB,
            close_fds=True,
            cwd=os.getcwd(),
        )
    print(f"Launched {script} (PID {p.pid})")

print("All launched as fully detached processes.")
print("These will survive terminal close, VS Code close, and logoff.")
