# platoon
Experimental multi-GPU mini-framework for Theano

This framework is still a prototype. It's interface is not polished and it is
likely to undergo changes in the future.

In Platoon, there are two main components : Soldiers, and Lieutenants.
Soldiers do the bulk of the work (training, monitoring, ...). Lieutenants are
controllers. They interact with multiple soldiers to coordinate their work,
collect the results and decide how to act on them. To use Platoon, you will
need to implement both your soldiers and your lieutenant but Platoon provides
helper classes to facilitate this.

The steps below describe what needs to be done to use Platoon for
data-parallelism. The LSTM example in the folder 'example' was implemented
following these steps and should be referred to for guidance.

Implementing a lieutenant
-------------------------

These steps describe how to implement the Python script that will launch
your Lieutenant. In the included LSTM example, both of these steps are done
in the file lstm_master.py

1) Define which commands your Lieutenant can receive and how it responds to
them.

This is done by creating a new class that inherits from channel.Lieutenant
and having it override the method 'handle_control()' which will be called
whenever your Lieutenant receives a request from a Soldier.

2) Instantiate and launch your custom lieutenant.

Create a script that will instantiate your custom lieutenant. Once this is
done, define the port on which the lieutenant should listen by calling the
function 'init_control'. Finally, call your lieutenant's 'serve' method which
will make him ready to receive requests from Soldiers.

Implementing the soldiers
-------------------------

These steps describe how to start with a script that performs stand-alone
training of a machine learning model and adapt it to serve as a worker in
Platoon.

1) Add a new parameter to the script which will be used during execution to
know whether the worker is the first one to be launched and should create the
central parameters or not.

2) Before entering the main loop, the script must create an instance of the
class channel.Soldier, providing it with the same port number as used to
initialize the lieutenant. It is not necessary to sub-class Soldier, you can
instantiate it directly. This object will provide the necessary methods to
handle communication with the lieutenant.

3) After the model has been built and the parameters initialized,
initialize the central parameters by calling the Soldier's
init_shared_params() method. Every worker should call this method but only
the first worker to be launched should provide the parameter cleanup=True,
the other workers should pass False or provide no value for this parameter.

4) In the main loop, instead of deciding when to train and when to monitor
performance, the worker should send control request to the Lieutenant to know
what action it should take, according to the communication protocol
established in the Lieutenant's 'handle_control()' method.

5) In the main loop, whenever the worker has performed 'N' (a hyper-parameter)
iterations of training, it should synchronize it's parameters with the central
parameters using it's Soldier's 'sync_params()' method.

Putting it all together
-----------------------

1) Launch your lieutenant. Wait until it has reached the point where it is
ready to serve requests.

2) Launch the first soldier. Wait until it has performed the initialization of
the central parameters.

3) Launch additional soldiers if desired.
