import time
from contextlib import contextmanager
from datetime import datetime


@contextmanager
def _timer(label: str):
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start = time.time()
    print(f"[TIMER] {label} start={start_ts}")
    try:
        yield
    finally:
        end = time.time()
        end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[TIMER] {label} end={end_ts} elapsed_s={end - start:.3f}")
