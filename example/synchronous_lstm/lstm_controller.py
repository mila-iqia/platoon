from __future__ import absolute_import, print_function
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

    def __init__(self, seed, patience, default_args):
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
        self.nb_worker = len(self._devices)
        # map ids to members of range(nb_worker)
        self.worker_ids_dict = dict(zip(self._workers, [i for i in range(len(self._workers))]))

        self.patience = patience
        self.seed = seed

        self.valid_history_errs = [[None for i in range(self.nb_worker)]]
        self.test_history_errs = [[None for i in range(self.nb_worker)]]
        self.bad_counter = 0
        self._epoch = 0
        self.best_dict = dict(best__epoch=-1, best_valid=numpy.inf)


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
        worker_id = self.worker_ids_dict[worker_id]

        if req == 'pred_errors':
            if self.valid_history_errs[self._epoch][worker_id] is not None:
                # if a worker tries to add a valid error where there is no None
                # it means it tries to index after or before current _epoch
                raise RuntimeError('Worker got out of synch!')
            self.valid_history_errs[self._epoch][worker_id] = req_info['valid_err']
            self.test_history_errs[self._epoch][worker_id] = req_info['test_err']

            if not any([i is None for i in self.valid_history_errs[self._epoch]]):
                print('Epoch %d is done'%req_info['epoch'])
                valid_err = sum(self.valid_history_errs[self._epoch]) / float(self.nb_worker)

                if valid_err <= self.best_dict['best_valid']:
                    self.best_dict['best_epoch'] = self._epoch
                    self.best_dict['best_valid'] = valid_err
                    self.bad_counter = 0
                    control_response = 'best'
                    print("Best error valid:", valid_err)
                else:
                    self.bad_counter += 1
                self.valid_history_errs += [[None for i in range(self.nb_worker)]]
                self.test_history_errs += [[None for i in range(self.nb_worker)]]
                self._epoch += 1

        elif req == 'splits':
            # the controller never loads the dataset but the worker doesn't
            # know how many workers there are
            train_len = req_info['train_len'] // self.nb_worker
            valid_len = req_info['valid_len'] // self.nb_worker
            test_len = req_info['test_len'] // self.nb_worker
            splits = dict(train_splits=[train_len * worker_id, train_len * (worker_id + 1)],
                          valid_splits=[valid_len * worker_id, valid_len * (worker_id + 1)],
                          test_splits=[test_len * worker_id, test_len * (worker_id + 1)])
            control_response = splits

            # kind of when the training start but not really
            self.start_time = time.time()

        elif req == 'seed':
            control_response = self.seed

        if self.bad_counter > self.patience:
            print("Early stopping!")
            end_time = time.time()
            # should terminate with best printing and best dumping of params
            # and then close everything
            print("Best error valid:", self.best_dict['best_valid'])
            test_err = sum(self.test_history_errs[self.best_dict['best_epoch']]) / \
                    float(self.nb_worker)
            print("Best error test:", test_err)
            print( ("Training took %.1fs" % (end_time - self.start_time)), file=sys.stderr)
            control_response = 'stop'
            self._close()

        return control_response


def lstm_control(saveFreq=1110, saveto=None):
    parser = Controller.default_parser()
    parser.add_argument('--seed', default=1234, type=int,
                        required=False, help='Maximum mini-batches to train upon in total.')
    parser.add_argument('--patience', default=10, type=int, required=False,
                        help='Maximum patience when failing to get better validation results.')
    args = parser.parse_args()

    l = LSTMController(seed=args.seed,
                       patience=args.patience,
                       default_args=Controller.default_arguments(args))

    print("Controller is ready")
    return l.serve()

if __name__ == '__main__':
    rcode = lstm_control()
    if rcode != 0:
        sys.exit(rcode)
