## GOAL ##
LSTM example using Platoon *param sync* interface


## CONTENT ##
- README.txt
- lstm_controller.py
- lstm_worker.py
- imdb.py


## HOW TO USE ##
# USING THE LAUNCHER
1) Assuming you are in the lstm folder.
   `cd platoon/example/lstm/`

1) Launch the experiment on 2 GPUs using the platoon-launcher script.
   `platoon-launcher lstm -D cuda0 cuda2`

To see all controller parameters do: `python lstm_controller.py -h`
To pass them via the platoon-launcher script: `platoon-launcher lstm -D cuda0 cuda2 -c=...`

To see all worker parameters do: `python lstm_worker.py -h`
To pass them via the platoon-launcher script: `platoon-launcher lstm -D cuda0 cuda2 -w=...`

# MANUALLY
1) Assuming you are in the lstm folder.
   `cd platoon/example/lstm/`

2) Start the controller.
   `THEANO_FLAGS='device=cpu' python -u lstm_controller.py`

3) Start the worker. Repeat as needed changing the GPU id.
   `THEANO_FLAGS='device=gpu0' python -u lstm_worker.py`


## NOTE ##
If you use the MANUAL way, you may want to run them in different windows of screen or tmux.
They all expect to be in the foreground.


## Timing ##
This timing was done with 2 k80.
The timing is about efficiency of computation, not efficiency of
training.  So the parameter alpha is constant. The number of mini-batches
is fixed as the hyper-parameter. The sync is also fixed to be after 10
mini-batch of computation.

With 1 worker, Platoon does not give you any advantage. This is
there just to show the overhead of the EASGD implementation.  Normal
is without this framework and with SGD, also there for overhead evaluation.

Normal | 1 GPU | 2 GPUs | 3 GPUs | 4 GPUs
-------|-------|--------|--------|-------
  870s |  912s |  477s  |  329s  |  254s
 1.00x | 0.95x | 1.82x  | 2.65x  | 3.42x
