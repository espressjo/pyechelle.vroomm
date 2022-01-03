Performance considerations
==========================

PyEchelle simulates individual photons. Therefore, the simulation problem scales linearly with the numbers of photons
that are launched. In particular, this means that the simulation time scales linearly with the chosen exposure time.

There are *three* ways of running PyEchelle: *using a single CPU core, multiple CPU cores at once or a CUDA compatible GPU.*

Single CPU
----------
Running it on a single CPU core (by using the flag *--max_cpu 1*) has the least overhead and the lowest memory requirements.

Multiple CPUs
-------------
By using multiple CPUs (e.g. by using the flag *--max_cpu 8*, a significant speedup can be achieved.
However, to avoid memory race conditions, PyEchelle is multithreading the simulation (grating) order by order and summing up the
different grating orders in the end. For large detectors, this requires a fairly large amount of memory (roughly the number
of cores times the size of a detector frame. E.g. for a 10k x 10k, the requirement is > 400MB x #cores.)
Also, it means that there is no speedup once the number of used CPU cores is larger than the number of simulated orders.

CUDA
----
The CUDA implementation (by using the flag *--cuda*) differs from the multi-CPU implementation.
The parallelization happens on the photon level while the orders are still simulated sequentially just like in the case of a single CPU.

Recommendations
---------------
Due to the large speedup of running PyEchelle with the help of a GPU, we **recommend using CUDA if available**.

Otherwise it depends on the use case:
 * For a large number of simulations with different parameters, we recommend using a single core and starting pyechelle in parallel for the different simulations.
 * For a single simulation (with multiple grating orders) we recommend using the multi-core version

Benchmark
---------

The following graph shows the simulation times on a AMD Ryzen 5950 X

.. raw:: html
   :file: _static/plots/benchmark.html

