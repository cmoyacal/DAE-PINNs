import numpy as np
import torch
from tqdm import tqdm

from utils import display
from utils.utils import timing
from events import EventList
import metrics as metrics_module

class supervisor(object):
    """
    the supervisor class trains a map on a data
    Args:
        :data (instance of data)
        :map (instance of maps or models)
        :device (cuda or cpu)
    """
    def __init__(self, data, net, device="cpu"):
        self.data = data
        self.net = net
        self.device = device

        self.optimizer = None
        self.scheduler = None
        self.batch_size = None

        self.state = State(device=self.device)    # training state
        self.loss_history = LossHistory()         # loss history
        self.stop_training = False
        self.events = None

    @timing 
    def compile(self, optimizer, metrics=None, loss_weights=None, scheduler=None, scheduler_type=None):
        """
        configures the supervisor for training
        Args:
            :optimizer: torch optimizer
            :metrics: list of metrics (not yet supported for DAE-PINNs)
            :loss_weights (list): a list of scalar coefficients to weight the loss contribution
            :scheduler: torch scheduler
            :scheduler_type (str): type of scheduler
        """
        print("compiling the supervisor...\n")
        self.optimizer = optimizer
        # metrics is not yet supported for DAE-PINNs.
        # To this end, we must collect simulation data from MATLAB or Scipy.
        metrics = metrics or []
        self.metrics = [metrics_module.get(m) for m in metrics]
        self.scheduler = scheduler
        self.scheduler_type = scheduler_type
        self.loss_history.set_loss_weights(loss_weights)

    @timing 
    def train(
        self, 
        epochs=None,
        batch_size=None,
        test_every=1000,
        num_val=10,
        disregard_previous_best=False,
        events=None,
        model_restore_path=None,
        model_save_path=None,
        use_tqdm=True,
        ):
        """
        training function
        Args:
            :epochs (int): number of epochs
            :batch_size (int or None): batch size
            :test_every (int): test the function every
            :num_val (int): number of validation examples
            :disregard_previous_best (bool): if True, disregard the previous best model
            :events (list): a list of event instances
            :model_restore_path (str): path where model.parameters() were previously saved
            :model_save_path (str): filename for checkpoint
            :use_tqdm (bool): prints training evolution
        """
        self.batch_size = batch_size
        self.num_val = num_val
        self.events = EventList(events=events)
        self.events.set_model(self)
        # disregard previous best model
        if disregard_previous_best:
            self.state.disregard_best()
        # restore model from path if required
        if model_restore_path is not None:
            check_point = self.restore(model_restore_path)
            state_dict = check_point['state_dict']
            self.net.load_state_dict(state_dict)
            # we always need to have training data and net parameters on device memory
            self.net.to(self.device)
        # start training
        print("training model...\n")
        self.stop_training = False
        # get the training, testing, and validation data - this should set the dataloaders
        self.state.set_data_train(*self.data.train_next_batch(self.batch_size))
        self.state.set_data_test(*self.data.test())     # we do not use batches while testing
        self.state.set_data_val(*self.data.train_next_batch(self.num_val))

        self.events.on_train_started()
        self._train(epochs, test_every, use_tqdm)
        self.events.on_train_completed()

        # print training results
        print("")
        display.training_display.summary(self.state)

        # save model ### check if this is not overwriting saving the BEST model
        #if model_save_path is not None:
        #    self.save(model_save_path, verbose=1)
        return self.loss_history, self.state

    def _train(self, epochs, test_every, use_tqdm):
        if use_tqdm:
            range_epochs = tqdm(range(epochs))
        else:
            range_epochs = range(epochs) 

        for epoch in range_epochs:
            self.events.on_epoch_started()
            self.net.train()
            loss_record_epoch = []
            for x_batch, _ in self.state.train_loader:
                self.optimizer.zero_grad()
                # compute losses
                loss_list = self.data.loss_fn(x_batch, model=self.net)
                if not isinstance(loss_list, list):
                    loss_list = [loss_list]       
                if self.loss_history.loss_weights is not None:                
                    for k in range(len(loss_list)):
                        loss_list[k] *= self.loss_history.loss_weights[k]
                loss = sum(loss_list)
                # optimize parameters
                loss.backward()
                self.optimizer.step()
                # detecting gradient explosion
                if loss.item() > 1e10:
                    print("gradient explosion detected")
                    self.stop_training = True
                # saving current batch loss
                loss_record_epoch.append(loss.item())
            try:
                avg_loss_epoch = sum(loss_record_epoch) / len(loss_record_epoch)
            except ZeroDivisionError as e:
                print("error: ", e, "batch size larger than number of training examples")
            # save epoch loss
            self.state.loss_train = [avg_loss_epoch]

            if self.scheduler is not None:
                if self.scheduler_type == "plateau":
                    if (epoch % 1 == 0):
                        self.net.eval()
                        # get validation data
                        val_data_device, _ = self.state.get_val_data()
                        with torch.no_grad():
                            # compute losses
                            loss_list = self.data.loss_fn(val_data_device, model=self.net)
                        if not isinstance(loss_list, list):
                            loss_list = [loss_list]
                        if self.loss_history.loss_weights is not None:                
                            for k in range(len(loss_list)):
                                loss_list[k] *= self.loss_history.loss_weights[k]
                        loss_val = sum(loss_list)
                        self.scheduler.step(loss_val.cpu().data.numpy())
                else:
                    self.scheduler.step()

            self.state.epoch += 1
            self.state.step += 1
            # testing the model
            if self.state.step % test_every == 0 or epoch + 1 == epochs:
                self._test()
            self.events.on_epoch_completed()

            if self.stop_training:
                break

    def _test(self):
        # batch testing is not supported
        self.net.eval()
        # compute losses
        with torch.no_grad():
            loss_list = self.data.loss_fn(self.state.X_test, model=self.net)
        if not isinstance(loss_list, list):
            loss_list = [loss_list]
        if self.loss_history.loss_weights is not None:                
            for k in range(len(loss_list)):
                loss_list[k] *= self.loss_history.loss_weights[k]
        loss = sum(loss_list)
        self.state.loss_test = [loss.item()]
        # DAE-PINNs does not support metrics during testing yet
        y_pred_test = None
        self.state.metrics_test = [
            m(self.state.y_test_np, y_pred_test.cpu().data.numpy()) for m in self.metrics
            ]
        self.state.update_best()
        self.loss_history.append(
            self.state.step,
            self.state.loss_train,
            self.state.loss_test,
            self.state.metrics_test,
        )
        display.training_display(self.state)

    def predict(self, input, events=None, model_restore_path=None):
        """
        generate output predictions for the given input samples
        Args:
            :input (numpy Tensor or list of tensors)
            :events (list of event instances)
            :model_restore_path (str) path where model.parameters() were previously saved
        Returns:
            :y (numpy Tensor)
        """
        if str(self.device) == "cpu":
            TensorFloat = torch.FloatTensor
        else:
            TensorFloat = torch.cuda.FloatTensor
        X = TensorFloat(input)
        self.events = EventList(events=events)
        self.events.set_model(self)
        self.events.on_predict_started()

        if model_restore_path is not None:
            check_point = self.restore(model_restore_path)
            state_dict = check_point['state_dict']
            self.net.load_state_dict(state_dict)
            # we always need to have training data and net parameters on device's memory
            self.net.to(self.device)
        
        self.net.eval()
        with torch.no_grad():
            # forward pass
            vel1, vel2, ang2, ang3, v3 = self.net(X)
        y_pred = np.vstack((vel1.cpu().numpy(),vel2.cpu().numpy(), ang2.cpu().numpy(), ang3.cpu().numpy(), v3.cpu().numpy()))
        self.events.on_predict_completed()
        return y_pred

    # @timing
    def integrate(self, X0, N=1, dyn_state_dim=4, model_restore_path=None):
        """
        Integrates the power network dynamics for N time steps
        Args:
            :X0 (numpy.array): \in [1, dyn_state_dim]
        Returns:
            :y_pred (numpy.array): \in [1, dyn_state_dim + alg_state_dim]
        """
        #print("integrating with DAE-PINNs...")
        yn = X0
        soln = []
        for _ in range(N):
            y_pred_n = self.predict(yn.reshape(1, -1), model_restore_path=model_restore_path)
            soln.append(y_pred_n)
            yn = y_pred_n[:dyn_state_dim, -1]
        return np.hstack(soln)
            
    def save(self, save_path, verbose=0):
        """
        save model to save_path
        """
        if verbose > 0:
            print(
                "Epoch {}: saving model to {} ...\n".format(
                    self.state.epoch, save_path
                )
            )
        state = {
            'state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
                }
        torch.save(state, save_path)

    def restore(self, restore_path, verbose=0):
        """
        restore model from restore_path
        """
        if verbose > 0:
            print("Restoring model from {} ...".format(restore_path))
        return torch.load(restore_path)

class State(object):
    def __init__(self, device="cpu"):
        self.epoch, self.step = 0, 0
        self.device = device

        # data
        self.loss_train = None
        self.loss_test = None
        self.metrics_test = None
        self.loss_val = None

        # the best results correspond to the min train loss
        # we can change to min test loss
        self.best_step = 0
        self.best_loss_train, self.best_loss_test = np.inf, np.inf
        self.best_metrics = None

        # data loaders
        self.train_loader = None

    def set_data_train(self, X, y, batch_size, shuffle=True):
        if str(self.device) == "cpu":
            TensorFloat = torch.FloatTensor
        else:
            TensorFloat = torch.cuda.FloatTensor
        X, y = TensorFloat(X), TensorFloat(y)
        data = torch.utils.data.TensorDataset(X, y)
        self.train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    def set_data_val(self, X, y, num_val):
        self.num_val = num_val
        if str(self.device) == "cpu":
            TensorFloat = torch.FloatTensor
        else:
            TensorFloat = torch.cuda.FloatTensor
        self.X_val, self.y_val = TensorFloat(X), TensorFloat(y)
    
    def get_val_data(self):
        num_train = self.X_val.shape[0]
        if self.num_val > num_train:
            self.num_val = num_train
        val_indices = torch.randperm(num_train)[:self.num_val].tolist()
        X_val_device = self.X_val[val_indices,:].to(self.device)
        y_val_device = self.y_val[val_indices,:].to(self.device)
        return X_val_device, y_val_device

    def set_data_test(self, X, y):
        self.y_test_np = y
        if str(self.device) == "cpu":
            TensorFloat = torch.FloatTensor
        else:
            TensorFloat = torch.cuda.FloatTensor
        self.X_test, self.y_test = TensorFloat(X), TensorFloat(y)

    def disregard_best(self):
        self.best_loss_train = np.inf

    def update_best(self):
        if self.best_loss_train > np.sum(self.loss_train):
            self.best_step = self.step
            self.best_loss_train = np.sum(self.loss_train)
            self.best_loss_test = np.sum(self.loss_test)
            self.best_metrics = self.metrics_test

class LossHistory(object):
    def __init__(self):
        self.steps = []
        self.loss_train = []
        self.loss_test = []
        self.metrics_test = []
        self.loss_weights = 1      
        
    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights

    def append(self, step, loss_train, loss_test, metrics_test):
        self.steps.append(step)
        self.loss_train.append(loss_train)
        if loss_test is None:
            loss_test = self.loss_test[-1]
        if metrics_test is None:
            metrics_test = self.metrics_test[-1]
        self.loss_test.append(loss_test)
        self.metrics_test.append(metrics_test)
