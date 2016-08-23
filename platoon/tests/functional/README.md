To functional test the *all_reduce* worker interface, you need to:

1. Export the environmental variable `PLATOON_TEST_WORKERS_NUM` to be equal to
   the total number of workers (GPUs) to be spawned across hosts in the
   functional test.
2. Call `platoon-launcher test` to start the test while being in the same
   directory as `test_worker.py` file. You can configure the multi-GPU/node
   procedure in any possible way as long as the total number of workers, which
   was set in the previous step, is respected.

The procedure exits with 0 for success. If this does not hold, please check
`platoon-launcher`, in order to see a high-level description of the return
code, and `PLATOON_LOGS` of the late procedure in current directory.

To profile and benchmark the new worker interface, you need to run
`platoon-launcher time` in current directory. Results are written in
`PLATOON_LOGS`.

To test and profile the Theano Ops of worker interface, you need to run
`platoon-launcher test_ops` in current directory.

To test implementations of global dynamics, please run
`platoon-launcher test_global_dynamics` in current directory.

**Note**: Depending on your hardware configuration, launching on defaults
Platoon may not suffice for a successful execution. Please check the
documentation and *platoonrc.conf* on how to configure Platoon.
