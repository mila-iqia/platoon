 
To use the provided example, follow the steps below : 

1) Run the script lstm_master.py in a shell and leave it running until it
   prints "Lieutenant is ready".  This shouldn't take a long time.

2) Run "python lstm.py init" until it prints "Param init done".  

3) At this point, you can run other workers (with lstm.py).  You may want to
   run a single worker until it starts training to do all the compilation
   required and not have all the workers fight for the compile lock.


IMPORTANT

-You have to manually assign GPUs to the workers (via THEANO_FLAGS=device=???).

-You may want to run them in different windows of screen or tmux since they all
 expect to be in the foreground.