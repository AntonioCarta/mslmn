import torch.nn
from torch import jit, nn


class JitItemClassifier(jit.ScriptModule):
    def __init__(self, rnn, hidden_size, output_size):
        super().__init__()
        self.rnn = rnn
        self.Wo = torch.nn.Parameter(0.01 * torch.rand(output_size, hidden_size))
        self.bo = torch.nn.Parameter(0.01 * torch.rand(output_size))
        for p in self.parameters():
            if len(p.shape) == 2:
                torch.nn.init.xavier_uniform_(p)

    @jit.script_method
    def forward(self, x):
        h0 = self.rnn.init_hidden(x.shape[1])
        h_list = self.rnn(x, h0)[0]
        y = torch.matmul(h_list, self.Wo.t()) + self.bo
        return y


class ItemClassifier(nn.Module):
    def __init__(self, rnn, hidden_size, output_size):
        super().__init__()
        self.rnn = rnn
        self.Wo = torch.nn.Parameter(0.01 * torch.rand(output_size, hidden_size))
        self.bo = torch.nn.Parameter(0.01 * torch.rand(output_size))

        for p in self.parameters():
            if len(p.shape) == 2:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        h0 = self.rnn.init_hidden(x.shape[1])
        h_list = self.rnn(x, h0)[0]
        if h_list.shape[2] == 0:  # used during the first incremental training step for MS-LMN
            return
        # output size is dynamic during incremental training
        y = torch.matmul(h_list, self.Wo[:, :h_list.shape[-1]].t()) + self.bo
        return y
