To functional test the new worker interface, you need to:

1. Export the environmental variable `PLATOON_TEST_WORKERS_NUM` to be equal to
   the total number of workers (GPUs) to be spawned across hosts in the
   functional test.
2. Call `platoon2-launcher test` to start the test while being in the same
   directory as `test_worker.py` file. You can configure the multi-GPU/node
   procedure in any possible way as long as the total number of workers, which
   was set in the previous step, is respected.

The procedure exits with 0 for success. If this does not hold, please check
`platoon2-launcher`, in order to see a high-level description of the return
code, and `PLATOON_LOGS` of the late procedure in current directory.

To profile and benchmark the new worker interface, you need to run
`platoon2-launcher time` in current directory. Results are written in
PLATOON_LOGS.
