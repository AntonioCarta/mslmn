from mslmn.cannon.callbacks import TrainingCallback
import torch
from mslmn.cannon.laes import LinearAutoencoder


class IncrementalTrainingCallback(TrainingCallback):
    def __init__(self, params, rnn_list, train_data, val_data, test_data, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.params = params
        self.ms_rnns = rnn_list
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.num_modules = 0

    def before_training(self, model_trainer):
        model_trainer.logger.info("PRETRAINING: pretraining initial network")
        self.pretrain_module(model_trainer)

    def after_epoch(self, model_trainer, train_data, validation_data):
        if (model_trainer.global_step + 1) % self.params['pretrain_every'] == 0 and \
                self.num_modules < self.params['num_modules']:
            model_trainer.logger.info("PRETRAINING: training new module")
            self.pretrain_module(model_trainer)

    def pretrain_module(self, model_trainer):
        encoded_data = self.encode_data(model_trainer.model, self.train_data)
        model_trainer.logger.info("PRETRAINING: building linear autoencoder")

        laes_list = []
        for i, h_model in enumerate(encoded_data):
            lin_ae = LinearAutoencoder(self.params['memory_size'], whiten=False)
            lin_ae.fit(h_model, svd_algo='cols', approx_k=5, t_max=500, verbose=True)
            lin_ae.save(self.log_dir + f'clock_{self.clock_period}_c{i}_lin_ae.pkl')
            laes_list.append(lin_ae)

        model_trainer.logger.info(f"PRETRAINING: ****************************************************")
        if self.num_modules > 0:
            te, ta = model_trainer.compute_metrics(self.train_data)
            ve, va = model_trainer.compute_metrics(self.val_data)
            model_trainer.logger.info(f"PRETRAINING: before pretraining TRAIN: err {te:.5f}, acc {ta:.5f}")
            model_trainer.logger.info(f"PRETRAINING: before pretraining VALID: err {ve:.5f}, acc {va:.5f}")
        self.add_new_module(model_trainer, laes_list)
        self.update_readout(model_trainer)
        te, ta = model_trainer.compute_metrics(self.train_data)
        ve, va = model_trainer.compute_metrics(self.val_data)
        model_trainer.logger.info(f"PRETRAINING: after pretraining TRAIN: err {te:.5f}, acc {ta:.5f}")
        model_trainer.logger.info(f"PRETRAINING: after pretraining VALID: err {ve:.5f}, acc {va:.5f}")
        model_trainer.logger.info(f"PRETRAINING: ****************************************************")

    def encode_data(self, model, data):
        hs_rnns = [[] for cell in self.ms_rnns]
        for x, (y, t_x, t_y) in data.iter():
            model(x)
            for i, cell in enumerate(self.ms_rnns):
                h_currs = torch.stack(cell._h, dim=0).detach().cpu().numpy()
                for ii in range(h_currs.shape[1]):  # subsampling
                    hs_rnns[i].append(h_currs[:t_x[ii]:self.clock_period, ii])
        return hs_rnns

    def add_new_module(self, model_trainer, laes_list):
        M = self.params['memory_size']

        for laes, model in zip(laes_list, self.ms_rnns):
            model = model.rnn_cell
            A, B = laes.A, laes.B
            model.update_num_modules(model.num_modules + 1)
            self.num_modules = model.num_modules
            g = model.num_modules
            model.Whm.data[(g - 1) * M:] = torch.zeros_like(model.Whm.data[(g - 1) * M:])
            model.Whm.data[(g-1)*M:g*M] = torch.from_numpy(A)
            model.Wmh.data[:, (g - 1) * M:] = torch.zeros_like(model.Wmh.data[:, (g - 1) * M:])
            model.Wmm.data[(g-1)*M:g*M, :g*M] = torch.zeros_like(model.Wmm.data[(g-1)*M:g*M, :g*M])
            model.Wmm.data[:g*M, (g-1)*M:g*M] = torch.zeros_like(model.Wmm.data[:g*M, (g-1)*M:g*M])
            model.Wmm.data[(g-1)*M:g*M, (g-1)*M:g*M] = torch.from_numpy(B)
            model.Wmm.data[(g-1)*M:g*M, (g-1)*M:g*M] = torch.from_numpy(B)
            model.bm.data[(g-1)*M: g*M] = 0
        model_trainer.model.Wo.data[:, (g-1)*M:] = torch.zeros_like(model_trainer.model.Wo.data[:, (g-1)*M:])

    @property
    def clock_period(self):
        return 2 ** self.num_modules
