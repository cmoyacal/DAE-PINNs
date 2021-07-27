import sys

from .utils import list_to_str

class TrainingDisplay:
    """
    Display training progress.
    """
    def __init__(self):
        self.len_train = None
        self.len_test = None
        self.len_metric = None
        self.is_header_print = False
    
    def print_one(self, s1, s2, s3, s4):
        print(
            "{:{l1}s}{:{l2}s}{:{l3}s}{:{l4}s}".format(
                s1,
                s2,
                s3,
                s4,
                l1=10,
                l2=self.len_train,
                l3=self.len_test,
                l4=self.len_metric,
            )
        )
        sys.stdout.flush()

    def header(self):
        self.print_one("Step", "Train loss", "Test loss", "Test metric")
        self.is_header_print = True

    def __call__(self, state):
        if not self.is_header_print:
            self.len_train = len(state.loss_train) * 10 + 4
            self.len_test = len(state.loss_test) * 10 + 4
            self.len_metric = len(state.metrics_test) * 10 + 4
            self.header()
        self.print_one(
            str(state.step),
            list_to_str(state.loss_train),
            list_to_str(state.loss_test),
            list_to_str(state.metrics_test),
        )

    def summary(self, state):
        print("Best model at step {:d}:".format(state.best_step))
        print("  train loss: {:.3e}".format(state.best_loss_train))
        print("  test loss: {:.3e}".format(state.best_loss_test))
        print("  test metric: {:s}".format(list_to_str(state.best_metrics)))
        print("")
        self.is_header_print = False

training_display = TrainingDisplay()