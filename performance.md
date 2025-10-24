`IFF_PROCESSING.PY`
    `pushPullredux`: 31.9s (vectorization + gpu) -> 3.62s
    `iffRedux`:      41.2s (vectorization + gpu) -> 27s (I/O parallelization & prefetching (8 workers - 1 mode prefetch))  -> 14.9s
    Total process time for the `process` function: 42s -> 19s
