## GOAL ##
The goal of this example is to showcase Platoon's functionality in the simplest way possible.


## CONTENT ##
- README.txt : This file!
- batched_pixel_sum.py : A simple Theano pixel-wise sum on MNIST
- batched_pixel_sum_master.py : A platoon implementation of batched_pixel_sum.py
  batched_pixel_sum_worker.py


## HOW TO USE ##
# USING THE LAUNCHER
1) Assuming you are in the simple_batched_pixel_sum folder.
   `cd platoon/example/simple_batched_pixel_sum/`

2) Launch the experiment on 1 gpu using the platoon_launcher script.
   All the outputs will be saved in a newly created `PLATOON_LOGS` folder.
   `platoon_launcher batched_pixel_sum gpu0`

# MANUALLY
1) Assuming you are in the simple_batched_pixel_sum folder.
   `cd platoon/example/simple_batched_pixel_sum/`

2) Start the controller.
   `THEANO_FLAGS='device=cpu' python -u batched_pixel_sum_controller.py`

3) Start the worker.
   `THEANO_FLAGS='device=gpu0' python -u batched_pixel_sum_worker.py`


## NOTE ##
- Using more than 1 worker causes problem at the moment for THIS particular example.
  The reason is that we are using the "dataset handled by the controller" feature which is not quite ready yet.
