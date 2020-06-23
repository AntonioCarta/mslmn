import pickle
import torch
from mslmn.cannon.utils import cuda_move
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torch.optim.lr_scheduler import ReduceLROnPlateau


class TrainingCallback:
    def __init__(self):
        pass

    def before_training(self, model_trainer):
        pass

    def after_epoch(self, model_trainer, train_data, validation_data):
        pass

    def after_train_before_validate(self, model_trainer):
        pass

    def after_training(self, model_trainer):
        pass

    def after_training_interrupted(self, model_trainer):
        pass

    def after_backward(self, model_trainer):
        pass

    def __str__(self):
        return self.__class__.__name__ + '(TrainingCallback)'


def save_training_checkpoint(self, e):
    torch.save(self.model, self.log_dir + 'model_e.pt')
    if self.best_vl_metric < self.val_accs[-1]:
        self.best_result = {
            'tr_loss': self.train_losses[-1],
            'tr_acc': self.train_accs[-1],
            'vl_loss': self.val_losses[-1],
            'vl_acc': self.val_accs[-1]
        }
        self.best_vl_metric = self.val_accs[-1]
        self.best_epoch = e
        torch.save(self.model, self.log_dir + 'best_model.pt')
    train_params = self._train_dict()
    d = {
        'model_params': self.model.params_dict(),
        'train_params': train_params,
        'best_result': self.best_result,
        'tr_loss': self.train_losses,
        'vl_loss': self.val_losses,
        'tr_accs': self.train_accs,
        'vl_accs': self.val_accs
    }
    with open(self.log_dir + 'checkpoint.pickle', 'wb') as f:
        pickle.dump(d, f)


class LRDecayCallback(TrainingCallback):
    """ Periodically reduce the learning rate if the validation performance is not improving.

    Attributes:
        decay_rate (float): multiplicative factor used to compute the new lr. must be < 1.
    """
    def __init__(self, decay_rate, patience):
        super().__init__()
        self.scheduler = None
        self.decay_rate = decay_rate
        self.patience = patience
        self.prev_lr = None

    def before_training(self, model_trainer):
        self.scheduler = ReduceLROnPlateau(model_trainer.opt, 'min', factor=self.decay_rate, patience=self.patience,
                                           verbose=True)
        self.prev_lr = model_trainer.opt.param_groups[0]['lr']

    def after_epoch(self, model_trainer, train_data, validation_data):
        curr_val_loss = model_trainer.val_losses[-1]
        self.scheduler.step(curr_val_loss)
        new_lr = model_trainer.opt.param_groups[0]['lr']
        if self.prev_lr > new_lr:
            self.prev_lr = new_lr
            model_trainer.logger.info("learning rate decreased to {:5e}".format(new_lr))

    def __str__(self):
        return "LRDecayCallback(decay_rate={}))".format(self.decay_rate)


class LearningCurveCallback(TrainingCallback):
    """
        Plot the learning curve for train/validation sets loss and metric and stores it into the log_dir folder.
    """
    def after_epoch(self, model_trainer, train_data, validation_data):
        plot_dir = model_trainer.log_dir + 'plots/'
        os.makedirs(plot_dir, exist_ok=True)

        fig = Figure()
        ax = fig.add_subplot(111)
        ax.plot(model_trainer.train_losses, label='train')
        ax.plot(model_trainer.val_losses, label='valid')
        ax.set_title("Loss")
        ax.set_xlabel("#epochs")
        ax.set_xlabel("loss")
        ax.legend()
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure('fig')
        fig.savefig(plot_dir + 'lc_loss.png')
        plt.close(fig)

        fig = Figure()
        ax = fig.add_subplot(111)
        ax.plot(model_trainer.train_metrics, label='train')
        ax.plot(model_trainer.val_metrics, label='valid')
        ax.set_title("Metric")
        ax.set_xlabel("#epochs")
        ax.set_xlabel("metric")
        ax.legend()
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure('fig')
        fig.savefig(plot_dir + 'lc_metric.png')
        plt.close(fig)

    def __str__(self):
        return "LearningCurveCallback(TrainingCallback)"


class EarlyStoppingCallback(TrainingCallback):
    def __init__(self, patience):
        super().__init__()
        self.patience = patience

    def after_epoch(self, model_trainer, train_data, validation_data):
        e = model_trainer.global_step
        model_trainer.logger.debug(f"EarlyStoppingCallback patience={self.patience}, waiting={e - model_trainer.best_epoch}")
        if not model_trainer.is_improved_performance(model_trainer) and \
                e - model_trainer.best_epoch > self.patience:
            model_trainer.logger.info("Early stopping at epoch {}".format(e + 1))
            model_trainer.stop_train()

    def __str__(self):
        return f"EarlyStoppingCallback(patience={self.patience})"


class ModelCheckpoint(TrainingCallback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir

    def after_train_before_validate(self, model_trainer):
        model_name = self.log_dir + 'best_model'
        if os.path.isfile(model_name + '.pt'):
            # model_trainer.model = cuda_move(torch.load(model_name + '.pt'))
            model_trainer.model.load_state_dict(torch.load(model_name + '.pt'))
            model_trainer.logger.info("Loaded best model checkpoint before final validation.")
        elif os.path.isfile(model_name + '.ptj'):
            try:
                model_trainer.model = cuda_move(torch.jit.load(model_name + '.ptj'))
                model_trainer.logger.info("Loaded best model checkpoint before final validation.")
            except BaseException as e:
                model_trainer.logger.info(str(e))
                model_trainer.logger.info("Could not load model. Checkpoint is corrupted.")
        else:
            model_trainer.logger.info("Checkpoint file not found. Using current model for final validation.")

    def after_epoch(self, model_trainer, train_data, validation_data):
        def try_save(model_name):
            if isinstance(model_trainer.model, torch.jit.ScriptModule):
                # ScriptModule should be checked first because it is a subclass of nn.Module.
                try:
                    model_trainer.model.save(model_name + '.ptj')
                except Exception as e:
                    torch.save(model_trainer.model, model_name + '.pt')
            elif isinstance(model_trainer.model, torch.nn.Module):
                # torch.save(model_trainer.model, model_name + '.pt')
                torch.save(model_trainer.model.state_dict(), model_name + '.pt')
            else:
                raise TypeError("Unrecognized model type. Cannot serialize.")

        try:
            try_save(self.log_dir + 'model_e')
            if model_trainer.best_epoch == model_trainer.global_step:
                try_save(self.log_dir + 'best_model')
        except Exception as err:
            model_trainer.logger.debug(err)
            model_trainer.logger.info('Error during model checkpoint phase.')
            assert False


class OrthogonalInit(TrainingCallback):
    def __init__(self, ortho_params, gain=1):
        super().__init__()
        self.ortho_params = ortho_params
        self.gain = gain

    def before_training(self, model_trainer):
        for p in self.ortho_params:
            torch.nn.init.orthogonal_(p, gain=1)
