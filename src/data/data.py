import abc

class Data(object):
    """data base class"""
    def __init__(self):
        pass

    @abc.abstractmethod
    def loss_fn(self, targets, outputs, model):
        """returns a list of losses"""

    @abc.abstractmethod
    def train_next_batch(self, batch_size=None):
        """returns a training dataset of size [batch_size, *]"""

    @abc.abstractmethod
    def test(self):
        """returns a test dataset"""