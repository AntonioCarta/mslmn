import torch
from mslmn.cannon.torch_trainer import TorchTrainer


class SequentialTaskTrainer(TorchTrainer):
    def __init__(self, model, optimizer, n_epochs, log_dir=None, regularizers=None, callbacks=None, use_tqdm=True, **kwargs):
        super().__init__(model, n_epochs, log_dir, **kwargs)
        self.opt = optimizer
        self.append_hyperparam_dict({
            'optimizer': optimizer
        })
        self.regularizers = [] if regularizers is None else regularizers
        if not use_tqdm:
            global tqdm
            tqdm = lambda x: x
        if callbacks:
            self.callbacks.extend(callbacks)

    def compute_metrics(self, data):
        with torch.no_grad():
            self.model.eval()
            err = 0
            acc = 0
            bi = 0
            for batch in tqdm(data.iter()):
                y_pred = self.model(batch[0])
                err += data.loss_score(batch, y_pred)
                acc += data.metric_score(batch, y_pred)
                bi += 1
            err = err / bi
            acc = acc / bi

            if torch.isnan(err):
                self.logger.info("NaN loss. Stopping training...")
                self.stop_train()
        return float(err), float(acc)

    def fit_epoch(self, train_data):
        self.model.train()
        for batch in tqdm(train_data.iter()):
            y_pred = self.model(batch[0])
            err = train_data.loss_score(batch, y_pred)

            reg = 0.0
            for reg_foo in self.regularizers:
                reg += reg_foo(self.model, None, None)
            err = err + reg

            self.model.zero_grad()
            err.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            self.opt.step()

            for cb in self.callbacks:
                cb.after_backward(batch)
