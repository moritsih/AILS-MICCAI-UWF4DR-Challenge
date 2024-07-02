from enum import Enum
import time
from contextlib import contextmanager

class Timings(Enum):
    DATA_LOADING = "DATA_LOADING"
    FORWARD_PASS = "FORWARD_PASS"
    CALC_LOSS = "CALC_LOSS"
    BACKWARD_PASS = "BACKWARD_PASS"
    OPTIMIZER_STEP = "OPTIMIZER_STEP"

class Timer:
    def __init__(self):
        self.timings = {timing: 0 for timing in Timings}
    
    @contextmanager
    def time(self, timing):
        start_time = time.time()
        yield
        end_time = time.time()
        self.timings[timing] += end_time - start_time
        
    def __str__(self):
        return ", ".join(f"{timing.name}_{elapsed_time:.2f}s" for timing, elapsed_time in self.timings.items())

# Example usage:
# timer = Timer()
# with timer.time(Timings.DATA_LOADING):
#     # your code block here
