## GOAL ##
The goal of this example is to showcase Platoon's functionality in the simplest way possible.


## CONTENT ##
- README.txt : This file!
- batched_pixel_sum.py : A simple theano pixel-wise sum on MNIST
- batched_pixel_sum_master.py : A platoon implementation of batched_pixel_sum.py
  batched_pixel_sum_worker.py


## HOW TO USE ##
1) The example should be ran from that folder
   `cd platoon/example/simple_batched_pixel_sum/`

2) Start the controller
   `THEANO_FLAGS='device=cpu' python -u batched_pixel_sum_controller.py`

3) Start the main worker with the init flag, this is create the shared memory.
   `THEANO_FLAGS='device=gpu0' python -u batched_pixel_sum_worker.py --init=True`

4) Start extra workers.
   `THEANO_FLAGS='device=gpu1' python -u batched_pixel_sum_worker.py`


## NOTE ##
- Using more than 1 worker causes problem at the moment for THIS particular example.
  The reason is that we are using the "dataset handled by the controller" feature which is not quite ready yet.
- The concept of a "main worker" that initialized the shared memory will be removed in a future release.
  All worker will be equal, which make more sense.
