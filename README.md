# platoon
Experimental multi-GPU mini-framework for Theano

It supports **data-parallelism** inside one compute node, not
model-parallelism. For model-parallelism check [Theano multiple GPUs
tutorial](http://deeplearning.net/software/theano/tutorial/using_multi_gpu.html).

In Platoon, there are two main components : workers, and controllers.
Workers do the bulk of the work (training, monitoring, ...). Controllers
interact with multiple workers to coordinate their work, collect the results
and decide how to act on them. To use Platoon, you will need to write code which
uses a worker. You can also extend the functionality of a worker or a controller by
implementing your own. Platoon provides helper classes to
facilitate this.

This framework is under development. Its interface is not polished and it is
likely to undergo changes in the future.

The framework provides two separate worker interfaces that allow user to implement
multiple data-parallel algorithms: *param_sync* and *all_reduce*. The default interface
is *param_sync*. Installing optional dependencies listed in the features table below
will make *all_reduce* interface available too.

Interface | sync type | multi-node                  | Theano Ops | extra dependencies
----------|-----------|-----------------------------|------------|--------------------
param_sync| sync/async|  no                         | no         | no
all_reduce| sync only | yes (if [mpi4py](https://github.com/mpi4py/mpi4py) is installed)| yes        | [NCCL](https://github.com/NVIDIA/nccl), [pygpu](https://github.com/Theano/libgpuarray), [Theano](https://github.com/Theano/Theano)

There are currently two algorithms for distributed gradient descent implemented with
*param_sync* interface and three with *all_reduce* interface.

* **param_sync**: [EASGD](http://arxiv.org/abs/1412.6651) and ASGD.
* **all_reduce**: Synchronous sum/average SGD, EASGD and a synchronous variant of [Downpour](http://research.google.com/archive/large_deep_networks_nips2012.html)

There are working examples in the examples directory.

The steps below describe what needs to be done to use Platoon for
data-parallelism. The LSTM example in the folder 'example' was implemented
following these steps and should be referred to for guidance.


## Install
You can simply install it using pip.
`pip install git+https://github.com/mila-udem/platoon`


If you would like to use the examples or help develop platoon first you have to clone the repo.

`git clone https://github.com/mila-udem/platoon`

Then install what you just cloned.

`pip install -e <path-to-platoon-folder>`


## Usage
The simplest way to launch a multi-gpu experiment is to first implement a controller and a worker as described below and then launch it using the `platoon-launcher`. It is not necessary that you have implemented a controller file if you want
to use the existing controller functionality.

The launcher assume that you named both files as such: `<experiment-name>_controller.py` and `<experiment-name>_worker.py`.

Then to launch the experiment you just need to specify the experiment name and GPUs you want to use:

`platoon-launcher <experiment-name> -D gpu0 gpu1`

You can also omit the `-D` argument and let launcher find all available CUDA GPUs to use
in the single-node experiment:

`platoon-launcher <experiment-name>`

For more configuration options, see `platoon-launcher -h`.


### Implementing a controller
These steps describe how to implement the Python script that will launch
your controller. In the included LSTM example, both of these steps are done
in the file `lstm_controller.py`

1) Define which commands your controller can receive and how it responds to
them. Commands starting by "platoon-" are reserved by platoon.

This is done by creating a new class that inherits from channel.Controller
and having it override the method `handle_control()` which will be called
whenever your controller receives a request from a worker.

2) Instantiate and launch your custom controller.

Create a script that will instantiate your custom controller. Once this is
done, define the port on which the controller should listen by calling the
function `init_control`. Finally, call your controller's `serve` method which
will make him ready to receive requests from workers.

### Implementing the workers
These steps describe how to start with a script that performs stand-alone
training of a machine learning model and adapt it to serve as a worker in
Platoon.

1) Add a new parameter to the script which will be used during execution to
know whether the worker is the first one to be launched and should create the
central parameters or not.

2) Before entering the main loop, the script must create an instance of the
class channel.Worker, providing it with the same port number as used to
initialize the controller. It is not necessary to sub-class Worker, you can
instantiate it directly. This object will provide the necessary methods to
handle communication with the controller.

3) After the model has been built and the parameters initialized,
initialize the central parameters by calling the Worker's
`init_shared_params()` method. Every worker should call this method.

4) In the main loop, instead of deciding when to train and when to monitor
performance, the worker should send control request to the controller to know
what action it should take, according to the communication protocol
established in the controller's `handle_control()` method.

5) In the main loop, whenever the worker has performed `N` (a hyper-parameter)
iterations of training, it should synchronize it's parameters with the central
parameters using it's Worker's `sync_params()` method.


### Real usage consideration
The optimal (as in more efficient for learning) hyper-parameters values are
dependent on the number of workers. At least, consider tuning the
learning rate and the alpha parameter of EASGD.

How to change the alpha hyper-parameter isn't clear. An alpha of 0.5
for the LSTM example with 2 workers seem to have good training
efficiency for this model/dataset/hyper-parameter combination.

Using alpha = 1/N (with N being the number of workers) might be a
reasonable guideline but the experiments performed with Platoon are
insufficient to conclude anything.

In the EASGD paper it is shown that in some cases a larger number of
workers can result in a better test error.

## Examples
For *param sync* interface, see `example/lstm/` [folder](https://github.com/mila-udem/platoon/tree/master/example/lstm).

For *all reduce* interface, see `example/synchronous_lstm/` [folder](https://github.com/mila-udem/platoon/tree/master/example/synchronous_lstm).

