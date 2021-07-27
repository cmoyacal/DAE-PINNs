import sys
import time
import numpy as np

class Event(object):
    """
    Events base class
    Args:
        :model instance of supervisor
    """
    def __init__(self):
        self.model = None
    
    def set_model(self, model):
        if model is not self.model:
            self.model = model
            self.init()

    def init(self):
        """
        init after setting the model
        """

    def on_epoch_started(self):
        """
        called at the beginning of every epoch
        """
    
    def on_epoch_completed(self):
        """
        called at the end of the epoch
        """

    def on_train_started(self):
        """
        called at the beginning of training
        """

    def on_train_completed(self):
        """
        called at the end of training
        """

    def on_predict_started(self):
        """
        called at the beginning of prediction
        """

    def on_predict_completed(self):
        """
        called at the end of prediction
        """

class EventList(Event):
    """
    contained abstracting a list of events
    """
    def __init__(self, events=None):
        events = events or []
        self.events = [e for e in events]
        self.model = None

    def set_model(self, model):
        self.model = model
        for event in self.events:
            event.set_model(model)

    def on_epoch_started(self):
        for event in self.events:
            event.on_epoch_started()

    def on_epoch_completed(self):
        for event in self.events:
            event.on_epoch_completed()

    def on_train_started(self):
        for event in self.events:
            event.on_train_started()

    def on_train_completed(self):
        for event in self.events:
            event.on_train_completed()
    
    def on_predict_started(self):
        for event in self.events:
            event.on_predict_started()

    def on_predict_completed(self):
        for event in self.events:
            event.on_predict_completed()

    def append(self, event):
        if not isinstance(event, Event):
            raise Exception(str(event) + " is an invalid Event object")
        self.events.append(event)

class ModelCheckPoint(Event):
    """
    save the model after every epoch
    Args:
        :filepath
        :save_better_only
        :every: interval (number of epochs) between checkpoints
    """
    def __init__(self, filepath, verbose=0, save_better_only=False, every=1, monitor="train loss"):
        super(ModelCheckPoint, self).__init__()
        self.filepath = filepath
        self.verbose = verbose
        self.save_better_only = save_better_only
        self.period = every

        self.monitor = monitor
        self.epochs_since_last_save = 0
        self.best = np.Inf

    def on_epoch_completed(self):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save < self.period:
            return
        self.epochs_since_last_save = 0
        if self.save_better_only:
            if self.monitor == "train loss":
                current = self.model.state.best_loss_train
            elif self.monitor == "test loss":
                current = self.model.state.best_loss_test
            if current < self.best:
                if self.verbose > 0:
                    print(
                        "Epoch {epoch}: {} improved from {:.2e} to {:.2e}, saving model to {}-{epoch} ...\n".format(
                            self.monitor,
                            self.best,
                            current,
                            self.filepath,
                            epoch=self.model.state.epoch,
                    ))
                self.best = current
                self.model.save(self.filepath, verbose=0)
        else:
            self.model.save(self.filepath, verbose=self.verbose)

class EarlyStopping(Event):
    """
    stop training when a monitored quantity (training loss) has stopped improving.
    Args
        :min_delta: minimum change in the monitored quantity to qualify as an improvement
        :patience: number of epochs with no improvement after which training will be stopped
        :baseline: baseline value for the monitored quantity to reach. Training will stop if the model doesn't show improvement over the baseline
    """
    def __init__(self, min_delta=0, patience=0, baseline=None, monitor="train loss"):
        super(EarlyStopping, self).__init__()

        self.baseline = baseline
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.min_delta *= -1
        self.monitor = monitor

    def on_train_started(self):
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf

    def on_epoch_completed(self):
        if self.monitor == "train loss":
            current = self.model.state.loss_train
        elif self.monitor == "test loss":
            current = self.model.state.loss_test

        if current - self.min_delta < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.model.state.epoch
                self.model.stop_training = True

    def on_train_complete(self):
        if self.stopped_epoch > 0:
            print("Epoch {}: early stopping".format(self.stopped_epoch))


class Timer(Event):
    """
    stop training when training time reaches the threshold
    Args
        :available_time (float): Total time in minutes available for training
    """
    def __init__(self, available_time):
        super(Timer, self).__init__()

        self.threshold = available_time * 60  # convert to seconds
        self.t_start = None

    def on_train_started(self):
        if self.t_start is None:
            self.t_start = time.time()

    def on_epoch_completed(self):
        if time.time() - self.t_start > self.threshold:
            self.model.stop_training = True
            print(
                "\nStop training as time used up. time used: {:.1f} mins, epoch trained: {}".format(
                    (time.time() - self.t_start) / 60, self.model.train_state.epoch
                )
            )