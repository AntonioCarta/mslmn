import torch
from torch import nn
from torch import Tensor
from mslmn.cannon.utils import standard_init, cuda_move


class MultiScaleLMN(nn.Module):
    """
    Recurerent implementation for Multiscale Linear Memory Network.

    Args:
        input_size:
        hidden_size:
        memory_size:
        num_modules: initial number of modules
        max_modules: maximum number of modules added during the incremental training
        alt_mode: wheter to use memory or functional output (Default: False, memory output)
    """
    def __init__(self, input_size, hidden_size, memory_size, num_modules, max_modules=None, alt_mode=False, learn_h0=False):
        super().__init__()
        self.rnn_cell = MultiScaleLMNCell(input_size, hidden_size, memory_size, num_modules, max_modules, alt_mode, learn_h0)
        self._h = None

    def init_hidden(self, batch_size: int) -> Tensor:
        return self.rnn_cell.init_hidden(batch_size)

    def forward(self, x, m_prev):
        assert len(x.shape) == 3
        out = []
        self._h = []
        x = x.unbind(0)
        for t in range(len(x)):
            xt = x[t]
            h_prev, m_prev = self.rnn_cell(xt, m_prev, t)
            out.append(h_prev)
            self._h.append(self.rnn_cell._h)
        return torch.stack(out), m_prev


class MultiScaleLMNCell(nn.Module):
    """
    Cell for Multiscale Linear Memory Network.

    Args:
        input_size:
        hidden_size:
        memory_size:
        num_modules: initial number of modules
        max_modules: maximum number of modules added during the incremental training
        alt_mode: wheter to use memory or functional output (Default: False, memory output)
    """
    def __init__(self, input_size, hidden_size, memory_size, num_modules, max_modules=None, alt_mode=False, learn_h0=False):
        super().__init__()
        # print(f"num_modules: {num_modules}, max_modules: {max_modules}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.num_modules = num_modules
        self.max_modules = num_modules if max_modules is None else max_modules
        self.alt_mode = alt_mode
        self.learn_h0 = learn_h0
        self.fc_activation = torch.tanh

        self.Wxh = nn.Parameter(torch.zeros(input_size, hidden_size))
        self.Whm = nn.Parameter(torch.zeros(self.max_modules * memory_size, hidden_size))
        self.Wmm = nn.Parameter(torch.zeros(self.max_modules * memory_size, self.max_modules * memory_size))
        self.Wmh = nn.Parameter(torch.zeros(hidden_size, self.max_modules * memory_size))
        self.bm = nn.Parameter(torch.zeros(memory_size * self.max_modules))
        self.bh = nn.Parameter(torch.zeros(hidden_size))

        if self.learn_h0:
            nmods = max_modules if max_modules is not None else num_modules
            self.h0 = torch.nn.Parameter(torch.zeros(memory_size * nmods))

        if num_modules > 0:
            standard_init(self.parameters())
        self._update_mask()
        self._h = None

    def update_num_modules(self, num_modules):
        assert num_modules <= self.max_modules
        self.num_modules = num_modules
        self._update_mask()

    def _update_mask(self):
        memory_size = self.memory_size
        max_modules = self.max_modules
        print(f"updating ms_lmn mask for {max_modules} modules")
        if not hasattr(self, 'm_mask'):
            self.register_buffer('m_mask', torch.zeros(memory_size * max_modules, memory_size * max_modules))
            self.mask_hook = self.Wmm.register_hook(lambda grad: grad * self.m_mask)

        torch.zero_(self.m_mask)
        for mi in range(self.num_modules):
            self.m_mask[mi*memory_size:(mi+1)*memory_size, mi*memory_size:] = 1
        self.Wmm.data = self.Wmm.data * self.m_mask

    def init_hidden(self, batch_size: int) -> Tensor:
        if self.learn_h0:
            return self.h0[:self.memory_size*self.num_modules].unsqueeze(0).reshape(batch_size, -1)
        else:
            return cuda_move(torch.zeros(batch_size, self.memory_size * self.num_modules))

    def forward(self, x_prev, m_prev, t_clock, get_hidden=False):
        assert len(x_prev.shape) == 2
        is_out_h = get_hidden or self.alt_mode
        self.Wmm.data = self.Wmm.data * self.m_mask
        max_active_module = 0
        while t_clock % (2 ** max_active_module) == 0 and max_active_module < self.num_modules:
            max_active_module += 1

        # print(f"t_clock: {t_clock}, max_active_module: {max_active_module}, num_modules: {self.num_modules}")
        Wmh_view = self.Wmh[:, :self.num_modules * self.memory_size]
        h_t = self.fc_activation(torch.mm(x_prev, self.Wxh) +
                                 torch.mm(m_prev, Wmh_view.t()) + self.bh)

        Wmm_view = self.Wmm[:self.memory_size*max_active_module, :self.memory_size*self.num_modules]
        m_t_update = torch.mm(m_prev, Wmm_view.t()) + \
                     torch.mm(h_t, self.Whm[:self.memory_size*max_active_module].t()) + \
                     self.bm[:self.memory_size*max_active_module]

        m_t = torch.cat([m_t_update, m_prev[:, self.memory_size*max_active_module:]], dim=1)
        out = h_t if is_out_h else m_t
        self._h = h_t
        return out, m_t


class ClockworkRNN(nn.Module):
    """
    Recurrent implementation for Clockwork RNN.

    Args:
        input_size:
        hidden_size:
        num_modules: initial number of modules
    """
    def __init__(self, input_size, hidden_size, num_modules):
        super().__init__()
        self.rnn_cell = ClockworkRNNCell(input_size, hidden_size, num_modules)

    def init_hidden(self, batch_size: int) -> Tensor:
        return self.rnn_cell.init_hidden(batch_size)

    def forward(self, x, h_prev):
        assert len(x.shape) == 3
        out = []
        x = x.unbind(0)
        for t in range(len(x)):
            xt = x[t]
            h_prev, h_prev = self.rnn_cell(xt, h_prev, t)
            out.append(h_prev)
        return torch.stack(out), h_prev


class ClockworkRNNCell(nn.Module):
    """
    Cell for Clockwork RNN.

    Args:
        input_size:
        hidden_size:
        num_modules: initial number of modules
    """
    def __init__(self, input_size, hidden_size, num_modules):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_modules = num_modules

        self.Wxh = nn.Parameter(torch.zeros(self.num_modules * hidden_size, input_size))
        self.Whh = nn.Parameter(torch.zeros(self.num_modules * hidden_size, self.num_modules * hidden_size))
        self.bh = nn.Parameter(torch.zeros(self.num_modules * hidden_size))
        standard_init(self.parameters())

        self.register_buffer('h_mask', torch.zeros(hidden_size * num_modules, hidden_size * num_modules))
        self.mask_hook = self.Whh.register_hook(lambda grad: grad * self.h_mask)
        torch.zero_(self.h_mask)
        for mi in range(self.num_modules):
            self.h_mask[mi*hidden_size:(mi+1)*hidden_size, mi*hidden_size:] = 1
        self.Whh.data = self.Whh.data * self.h_mask

    def init_hidden(self, batch_size: int) -> Tensor:
        return cuda_move(torch.zeros(batch_size, self.hidden_size * self.num_modules))

    def forward(self, x_prev, h_prev, t_clock):
        assert len(x_prev.shape) == 2
        self.Whh.data = self.Whh.data * self.h_mask
        max_active_module = 0
        while t_clock % (2 ** max_active_module) == 0 and max_active_module < self.num_modules:
            max_active_module += 1

        Whh_view = self.Whh[:self.hidden_size*max_active_module, :self.hidden_size*self.num_modules]
        h_t_update = torch.mm(h_prev, Whh_view.t()) + \
                     torch.mm(x_prev, self.Wxh[:self.hidden_size*max_active_module].t()) + \
                     self.bh[:self.hidden_size*max_active_module]

        h_t = torch.cat([h_t_update, h_prev[:, self.hidden_size*max_active_module:]], dim=1)
        return h_t, h_t


if __name__ == '__main__':
    import torch.nn.functional as F

    # test model update equivalence
    b, i, h, M = 13, 3, 5, 7
    num_modules = 11
    x = cuda_move(torch.randn(4, b, i))
    model = cuda_move(MultiScaleLMN(i, h, M, num_modules=1, max_modules=3))

    m_prev = model.init_hidden(b)
    y_old = model(x, m_prev)

    model.rnn_cell.update_num_modules(model.rnn_cell.num_modules + 1)
    g = model.rnn_cell.num_modules
    model.rnn_cell.Whm.data[(g - 1) * M:] = torch.zeros_like(model.rnn_cell.Whm.data[(g - 1) * M:])
    model.rnn_cell.Whm.data[(g - 1) * M:g * M] = torch.randn_like(model.rnn_cell.Whm.data[(g - 1) * M:g * M])
    model.rnn_cell.Wmh.data[:,(g - 1) * M:] = torch.zeros_like(model.rnn_cell.Wmh.data[:,(g - 1) * M:])
    model.rnn_cell.Wmm.data[(g - 1) * M:g * M, :g * M] = torch.zeros_like(model.rnn_cell.Wmm.data[(g - 1) * M:g * M, :g * M])
    model.rnn_cell.Wmm.data[:g * M, (g - 1) * M:g * M] = torch.zeros_like(model.rnn_cell.Wmm.data[:g * M, (g - 1) * M:g * M])
    model.rnn_cell.Wmm.data[(g - 1) * M:g * M, (g - 1) * M:g * M] = torch.randn_like(model.rnn_cell.Wmm.data[(g - 1) * M:g * M, (g - 1) * M:g * M])
    model.rnn_cell.bm.data[(g -1)*M: g*M] = 0

    m_prev = model.init_hidden(b)
    y_new = model(x, m_prev)

    assert (y_new[0][:,:,:M] - y_old[0]).reshape(-1, 7).detach().cpu().numpy().sum() == 0
    # Other tests

    b, i, h, m = 13, 3, 5, 7
    num_modules = 11
    clmn = MultiScaleLMNCell(i, h, m, num_modules)
    x = torch.randn(b, i)
    y = torch.randn(b, m * num_modules)

    clmn.zero_grad()
    m0 = torch.randn(b, m * num_modules)
    y_pred, m0 = clmn(x, m0, 0)
    e = F.mse_loss(y_pred, y)
    e.backward()
    assert torch.norm(clmn.Wmm.grad[m:]).item() != 0

    clmn.zero_grad()
    m0 = m0.detach()
    y_pred, m0 = clmn(x, m0, 1)
    e = F.mse_loss(y_pred, y)
    e.backward()
    assert torch.norm(clmn.Wmm.grad[m:]).item() == 0

    # Controlla i moduli attivi ad ogni timestep
    T = 128
    x = cuda_move(torch.randn(T, 1, i))
    clmn = cuda_move(MultiScaleLMN(i, h, m, 2, 8))
    m0 = clmn.init_hidden(1)
    y_pred, m0 = clmn(x, m0)

    # Controlla laes struttura a blocchi dell'output
    clmn = cuda_move(MultiScaleLMN(i, h, m, num_modules))
    m0 = clmn.init_hidden(1)
    y_pred, m0 = clmn(x, m0)

    assert False
