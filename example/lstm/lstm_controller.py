from __future__ import print_function
import os
import sys
import time

import numpy

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from platoon.channel import Controller


class LSTMController(Controller):
    """
    This multi-process controller implements patience-based early-stopping SGD
    """

    def __init__(self, max_mb, patience, valid_freq, default_args):
        """
        Initialize the LSTMController

        Parameters
        ----------
        max_mb : int
            Max number of minibatches to train on.
        patience: : int
            Training stops when this many minibatches have been trained on
            without any reported improvement.
        valid_freq : int
            Number of minibatches to train on between every monitoring step.
        default_args : dict
            Arguments of default class Controller
        """

        super(LSTMController, self).__init__(**default_args)
        self.patience = patience
        self.max_mb = int(max_mb)

        self.valid_freq = valid_freq
        self.uidx = 0
        self.eidx = 0
        self.history_errs = []
        self.bad_counter = 0

        self.valid = False
        self.start_time = None
        self._should_stop = False

    def handle_control(self, req, worker_id, req_info):
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
            if not self._should_stop:
                if self.start_time is None:
                    self.start_time = time.time()

                if self.valid:
                    self.valid = False
                    control_response = 'valid'
                else:
                    control_response = 'train'
            else:
                control_response = 'stop'
        elif req == 'done':
            self.uidx += req_info['train_len']

            if numpy.mod(self.uidx, self.valid_freq) == 0:
                self.valid = True
        elif req == 'pred_errors':
            valid_err = req_info['valid_err']
            test_err = req_info['test_err']
            self.history_errs.append([valid_err, test_err])
            harr = numpy.array(self.history_errs)[:, 0]

            if valid_err <= harr.min():
                self.bad_counter = 0
                control_response = 'best'
                print("Best error valid:", valid_err, "test:", test_err)
            elif (len(self.history_errs) > self.patience and valid_err >= harr[:-self.patience].min()):
                self.bad_counter += 1

        if self.uidx > self.max_mb or self.bad_counter > self.patience:
            if not self._should_stop:
                print("Training time {:.4f}s".format(time.time() - self.start_time))
                print("Number of samples:", self.uidx)
            self._should_stop = True

        return control_response


def lstm_control(saveFreq=1110, saveto=None):
    parser = Controller.default_parser()
    parser.add_argument('--max-mb', default=((5000 * 1998) / 10), type=int,
                        required=False, help='Maximum mini-batches to train upon in total.')
    parser.add_argument('--patience', default=10, type=int,
                        required=False, help='Maximum patience when failing to get better validation results.')
    parser.add_argument('--valid-freq', default=370, type=int,
                        required=False, help='How often in mini-batches prediction function should get validated.')
    args = parser.parse_args()

    l = LSTMController(max_mb=args.max_mb,
                       patience=args.patience,
                       valid_freq=args.valid_freq,
                       default_args=Controller.default_arguments(args))

    print("Controller is ready")
    return l.serve()

if __name__ == '__main__':
    rcode = lstm_control()
    if rcode != 0:
        sys.exit(rcode)
