import time
import numpy as np


def profile(N: int, func: callable, *args, **kwargs):
    """Profiles a function and returns the output"""
    # Warmup and store result
    result = func(*args, **kwargs)

    times = []

    for _ in range(N):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    median_time = np.median(times)
    print(f"Median time out of {N}: {median_time}")
    return result
