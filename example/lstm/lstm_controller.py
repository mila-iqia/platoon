from __future__ import print_function
import numpy
import time

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from platoon.channel import Controller


class LSTMController(Controller):
    """
    This multi-process controller implements patience-based early-stopping SGD
    """

    def __init__(self, control_port, max_mb, patience, validFreq):
        """
        Initialize the LSTMController

        Parameters
        ----------
        max_mb : int
            Max number of minibatches to train on.
        patience: : int
            Training stops when this many minibatches have been trained on
            without any reported improvement.
        validFreq : int
            Number of minibatches to train on between every monitoring step.
        """

        Controller.__init__(self, control_port)
        self.patience = patience
        self.max_mb = int(max_mb)

        self.validFreq = validFreq
        self.uidx = 0
        self.eidx = 0
        self.history_errs = []
        self.bad_counter = 0

        self.valid = False
        self.start_time = None

    def handle_control(self, req, worker_id):
        """
        Handles a control_request received from a worker

        Parameters
        ----------
        req : str or dict
            Control request received from a worker.
            The control request can be one of the following
            1) "next" : request by a worker to be informed of its next action
               to perform. The answers from the server can be 'train' (the
               worker should keep training on its training data), 'valid' (the
               worker should perform monitoring on its validation set and test
               set) or 'stop' (the worker should stop training).
            2) dict of format {"done":N} : used by a worker to inform the
                server that is has performed N more training iterations and
                synced its parameters. The server will respond 'stop' if the
                maximum number of training minibatches has been reached.
            3) dict of format {"valid_err":x, "test_err":x2} : used by a worker
                to inform the server that it has performed a monitoring step
                and obtained the included errors on the monitoring datasets.
                The server will respond "best" if this is the best reported
                validation error so far, otherwise it will respond 'stop' if
                the patience has been exceeded.
        """
        control_response = ""

        if req == 'next':
            if self.start_time is None:
                self.start_time = time.time()

            if self.valid:
                self.valid = False
                control_response = 'valid'
            else:
                control_response = 'train'
        elif 'done' in req:
            self.uidx += req['done']

            if numpy.mod(self.uidx, self.validFreq) == 0:
                self.valid = True
        elif 'valid_err' in req:
            valid_err = req['valid_err']
            test_err = req['test_err']
            self.history_errs.append([valid_err, test_err])
            harr = numpy.array(self.history_errs)[:, 0]

            if valid_err <= harr.min():
                self.bad_counter = 0
                control_response = 'best'
                print("Best error valid:", valid_err, "test:", test_err)
            elif (len(self.history_errs) > self.patience and valid_err >= harr[:-self.patience].min()):
                self.bad_counter += 1

        if self.uidx > self.max_mb or self.bad_counter > self.patience:
            control_response = 'stop'
            self.worker_is_done(worker_id)
            print("Training time {:.4f}s".format(time.time() - self.start_time))
            print("Number of samples:", self.uidx)

        return control_response


def lstm_control(dataset='imdb',
                 patience=10,
                 max_epochs=5000,
                 validFreq=370,
                 saveFreq=1110,
                 saveto=None):

    # TODO: have a better way to set max_mb
    l = LSTMController(control_port=5567, max_mb=(5000*1998)/10, patience=patience, validFreq=validFreq)

    print("Controller is ready")
    l.serve()

if __name__ == '__main__':
    lstm_control()
