## GOAL ##
LSTM example using Platoon *all reduce* interface


## CONTENT ##
- README.txt
- lstm_controller.py
- lstm_worker.py
- imdb.py
It is assumed that imdb.pkl is in the same foler, otherwise it will be downloaded.


## HOW TO USE ##
# USING THE LAUNCHER
When the launcher is used, the outputs and errors of controller and workers are automatically
stored in an auto-generated folder of PLATOON_LOGS/lstm/DATE_TIME.

1) Assuming you are in the synchronous_lstm folder.
   `cd platoon/example/synchronous_lstm/`

2) Launch the experiment on 2 GPUs using the platoon-launcher script.
   `platoon-launcher lstm -D cuda0 cuda1`

To see all controller parameters do: `python lstm_controller.py -h`
To pass them via the platoon-launcher script: `platoon-launcher lstm -D cuda0 cuda1 -c=...`

To see all worker parameters do: `python lstm_worker.py -h`
To pass them via the platoon-launcher script: `platoon-launcher lstm -D cuda0 cuda1 -w=...`


For setting THEANO_FLAGS for the workers, you can use the
following command which sets floatX to float32 for all the workers:
`THEANO_FLAGS=floatX=float32 platoon-launcher lstm -D cuda0 cuda1`

# USING THE SCRIPTS
When the scripts are used the path to store the outputs can be given.

1) Assuming you are in the synchronous_lstm folder.
   `cd platoon/example/synchronous_lstm/`

2) Launch the experiment. Platoon will automatically find all the available GPUs
   and run the workers on them:
   THEANO_FLAGS=floatX=float32 python lstm_controller.py --single lstm PATH/TO/OUTPUT

--single indicates the GPUs are all on the same machine.
lstm is the name of the experiment. It will look for an lstm_worker.py to run the workers.
THEANO_FLAGS are set for all the workers and not the controller. The controller should use
the CPU.


## TIMING ##
These timings were done using the Nvidia DGX-1 and by averaging results from 
two runs for each setup. 
1 GPU : 5.698 seconds / epoch
2 GPU : 2.230 seconds / epoch
