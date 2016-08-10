from __future__ import absolute_import, print_function, division
import sys
import cProfile
import pstats
from timeit import default_timer as timer

from six.moves import range

import theano
import numpy as np
from numpy.testing import assert_allclose

from platoon.worker import Worker

SEED = 567
np.random.seed(SEED)

worker = Worker(control_port=5567)


def profile(shape=(1000, 1000), dtype='float64', rng=(-1, 1)):
    print("\n### Profiling worker")
    print()
    print("### shape =", shape)
    print("### dtype =", dtype)
    print("### range =", sorted(rng))

    rang = abs(rng[1] - rng[0])
    inp = np.random.random(shape) * rang + min(rng)
    inp = inp.astype(dtype)
    sinp = theano.shared(inp)
    out = np.empty_like(inp)
    sout = theano.shared(out)

    print("\n### Profiling worker.all_reduce")
    print("## First call to worker.all_reduce")
    cProfile.runctx("worker.all_reduce(sinp, '+', sout)", globals(), locals(),
                    filename="worker.prof")
    s = pstats.Stats("worker.prof")
    s.strip_dirs().sort_stats("time").print_stats()
    assert_allclose(inp * worker.global_size, sout.get_value())

    print("## Second call to worker.all_reduce")
    cProfile.runctx("worker.all_reduce(sinp, '+', sout)", globals(), locals(),
                    filename="worker.prof")
    s = pstats.Stats("worker.prof")
    s.strip_dirs().sort_stats("time").print_stats()
    assert_allclose(inp * worker.global_size, sout.get_value())
    if worker._multinode:
        print("## Note that there must be difference between the first and")
        print("## the second call as a result of the extra call to worker.shared")
        print("## during the first time.")


def benchmark(shape=(1000, 1000), dtype='float64', rng=(-1, 1), number=10):
    print("\n### Benchmarking worker")
    print()
    print("### shape =", shape)
    print("### dtype =", dtype)
    print("### range =", sorted(rng))
    print("### num of iterations =", number)

    rang = abs(rng[1] - rng[0])
    inp = np.random.random(shape) * rang + min(rng)
    inp = inp.astype(dtype)
    sinp = theano.shared(inp)
    out = np.empty_like(inp)
    sout = theano.shared(out)

    print("\n## Benchmarking worker.shared")
    print("# First call")
    start = timer()
    worker.shared(sinp)
    end = timer()
    print("Time:", end - start)
    print("# Second call")
    start = timer()
    worker.shared(sinp)
    end = timer()
    print("Time:", end - start)

    print("\n## Benchmarking worker.all_reduce")
    print("# First call to worker.all_reduce")
    print("# Contains call to worker.shared internally, if multi-node")
    start = timer()
    worker.all_reduce(sinp, '+', sout)
    end = timer()
    print("Time:", end - start)
    assert_allclose(inp * worker.global_size, sout.get_value())

    print("# Timing worker.all_reduce w/o calls to worker.shared")
    ttime = 0
    for _ in range(number):
        start = timer()
        worker.all_reduce(sinp, '+', sout)
        end = timer()
        ttime += end - start
        assert_allclose(inp * worker.global_size, sout.get_value())
    print("Mean time:", ttime / number)


if __name__ == '__main__':
    try:
        profile()
        benchmark()
    except Exception as exc:
        print(exc, file=sys.stderr)
    finally:
        worker.close()
