import argparse

from mslmn.cannon.callbacks import OrthogonalInit
from mslmn.cannon.rnn import LSTMLayer, LMNLayer
from mslmn.cannon.utils import cuda_move, set_gpu, set_allow_cuda
import torch.nn.init
from mslmn.cannon.regularizers import OrthogonalPenalty
from mslmn.container import ItemClassifier
from mslmn.task_trainer import SequentialTaskTrainer
from mslmn.iamondb import IAMOnDB
from mslmn.cannon.model_selection import ParamListTrainer
from mslmn.models import MultiScaleLMN, ClockworkRNN
from mslmn.incremental_train import IncrementalTrainingCallback
import shutil
from torch import nn
from mslmn.cannon.utils import standard_init
import os
set_gpu()


class BidirectionalRNN(nn.Module):
    def __init__(self, fw_layer, bw_layer):
        super().__init__()
        self.fw_layer = fw_layer
        self.bw_layer = bw_layer
        standard_init(self.parameters())

    def init_hidden(self, batch_size: int):
        return [
            self.fw_layer.init_hidden(batch_size),
            self.bw_layer.init_hidden(batch_size)
        ]

    def forward(self, x, m_prev):
        assert len(x.shape) == 3
        fw_out, m_prev[0] = self.fw_layer(x, m_prev[0])
        bw_out, m_prev[1] = self.bw_layer(torch.stack(list(x.unbind(0))[::-1], dim=0), m_prev[1])
        bw_out = torch.stack(bw_out.unbind(0)[::-1], dim=0)
        out = torch.cat([fw_out, bw_out], dim=2)
        return out, m_prev


def train_foo(log_dir, params):
    DEBUG = params['DEBUG']
    model_type = params['model_type']
    train_data = IAMOnDB(data_dir, 'train', params['batch_size'], debug=DEBUG)
    val_data = IAMOnDB(data_dir, 'valid', params['batch_size'], debug=DEBUG)
    test_data = IAMOnDB(data_dir, 'test', params['batch_size'], debug=DEBUG)

    if DEBUG:
        params['n_epochs'] = 15
        params['pretrain_every'] = 5
        params['k_unroll'] = 3
        params['ulm_n_epochs'] = 4

    ins, hs, ms = params['in_size'], params['hidden_size'], params['memory_size']
    if model_type == 'lstm':
        rnn = BidirectionalRNN(LSTMLayer(ins, hs), LSTMLayer(ins, hs))
        ro_in = 2 * hs
        cbs, regs = [], []
    elif model_type == 'lmn':
        rnn = BidirectionalRNN(LMNLayer(ins, hs, ms), LMNLayer(ins, hs, ms))
        ro_in = 2 * ms
        cbs = [OrthogonalInit([rnn.fw_layer.Wmm, rnn.bw_layer.Wmm])]
        regs = [OrthogonalPenalty(params['ortho_penalty'], [rnn.fw_layer.Wmm, rnn.bw_layer.Wmm])]
    elif model_type == 'ms_lmn':
        rnn = BidirectionalRNN(
            MultiScaleLMN(ins, hs, ms, num_modules=params['num_modules']),
            MultiScaleLMN(ins, hs, ms, num_modules=params['num_modules'])
        )
        ro_in = 2 * params['num_modules'] * ms
        cbs, regs = [], []
    elif model_type == 'cw_rnn':
        rnn = BidirectionalRNN(
            ClockworkRNN(ins, hs, params['num_modules']),
            ClockworkRNN(ins, hs, params['num_modules'])
        )
        ro_in = 2 * hs * params['num_modules']
        cbs, regs = [], []
    elif model_type == 'pret_ms_lmn':
        rnn = BidirectionalRNN(
            MultiScaleLMN(ins, hs, ms, num_modules=0, max_modules=params['num_modules']),
            MultiScaleLMN(ins, hs, ms, num_modules=0, max_modules=params['num_modules'])
        )
        ro_in = 2 * ms * params['num_modules']
        cbs = [IncrementalTrainingCallback(params, [rnn.bw_layer, rnn.fw_layer], train_data, val_data, test_data, log_dir)]
        regs = []

    model = cuda_move(ItemClassifier(
        rnn=rnn,
        hidden_size=ro_in,
        output_size=params['output_size']
    ))

    if params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=params['learning_rate'],
                                     weight_decay=params['l2_decay'])

    trainer = SequentialTaskTrainer(
        cuda_move(model),
        n_epochs=params['n_epochs'],
        optimizer=optimizer,
        regularizers=regs,
        log_dir=log_dir,
        callbacks=cbs,
        use_tqdm=True,
        validation_steps=1,
        patience=params['patience'],
        checkpoint_mode='loss',
        debug=DEBUG,
    )
    trainer.append_hyperparam_dict(params)
    trainer.fit(train_data, val_data)
    e, a = trainer.compute_metrics(test_data)
    trainer.logger.info(f"Final performance on TEST set: loss {e}, acc {a}")
    trainer.logger.info(trainer.best_result)
    return trainer.best_result


if __name__ == '__main__':
    set_allow_cuda(True)
    DEBUG = False
    model_type = 'pret_ms_lmn'
    data_dir = '/home/USER/data/iamondb'
    log_dir = f'./logs/iamondb/gs4/'

    if DEBUG:
        log_dir = './logs/debug/'
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    params = {
        'DEBUG': DEBUG,
        'model_type': model_type,
        'log_dir': log_dir,
        'in_size': 4,
        'num_modules': 9,
        'memory_size': 100,
        'output_size': 58,
        'optimizer': 'adam',
        'batch_size': 32,
        'learning_rate': 3.e-4,
        'l2_decay': 0.0,
        'ortho_penalty': 0.0,
        'n_epochs': 2000,
        'patience': 50,
        # pretraining
        'pretrain_every': 50
    }

    hs_params = {  # params for bidirectional models
        'lstm': [33, 77, 110],
        'lmn': [40, 91, 129],
        'cw_rnn': [11, 24, 33],
        'ms_lmn': [(45, 5), (102, 11), (144, 16)],
        'pret_ms_lmn': [(45, 5), (102, 11), (144, 16)]
    }

    param_list = []
    # for model_type in ['lstm', 'lmn', 'cw_rnn', 'ms_lmn', 'pret_ms_lmn']:
    for hs in hs_params[model_type]:
        for k in range(5):
            pi = params.copy()
            pi['model_type'] = model_type
            if 'ms' in model_type:
                pi['hidden_size'] = hs[0]
                pi['memory_size'] = hs[1]
            else:
                pi['hidden_size'] = hs
                pi['memory_size'] = hs
            param_list.append(pi)

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', metavar='N', type=int, help='model selection ID')
    args = parser.parse_args()

    if DEBUG:
        log_dir = './logs/debug/'

    list_trainer = ParamListTrainer(log_dir, param_list, train_foo, resume_ok=True, id=args.id)
    list_trainer.run()
