import torch
import numpy as np
from torch.autograd import Variable
import os
import pickle

ALLOW_CUDA = True  # Global variable to control cuda_move allocation behavior


def standard_init(params):
    for p in params:
        if len(p.shape) == 2:
            torch.nn.init.xavier_normal_(p)


def set_allow_cuda(b):
    global ALLOW_CUDA
    ALLOW_CUDA = b
    if b:
        print("CUDA enabled.")
    else:
        print("CUDA disabled.")


def set_gpu():
    import os
    try:
        import gpustat
    except ImportError as e:
        print("gpustat module is not installed. No GPU allocated.")

    try:
        stats = gpustat.GPUStatCollection.new_query()
        ids = map(lambda gpu: int(gpu.entry['index']), stats)
        ratios = map(lambda gpu: float(gpu.entry['memory.used']) / float(gpu.entry['memory.total']), stats)
        bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]

        print("Setting GPU to: {}".format(bestGPU))
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(bestGPU)
    except BaseException as e:
        print("GPU not available: " + str(e))


def cuda_move(args):
    """ Move a sequence of tensors to CUDA if the system supports it. """
    if not ALLOW_CUDA:
        return args.cpu()
    b = torch.cuda.is_available()
    # for t in args:
    #     if b:
    #         yield t.cuda()
    #     else:
    #         yield t
    if b:
        return args.cuda()
    else:
        return args


def gradient_clipping(model, clip=1):
    """ Clip the value of each gradient component.
        Args:
            clip: maximum absolute value
    """
    for name, p in model.named_parameters():
        try:
            p.grad.data.clamp_(min=-clip, max=clip)
        except AttributeError as e:
            # print("Parameter {} has no gradient.".format(name))
            # raise e
            pass


def cosine_similarity(x1: Variable, x2: Variable, dim: int=1, eps=1e-8) -> Variable:
    """ Compute cosine similarity along given dim. """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def is_nan(v: Variable):
    return np.isnan(np.sum(v.data.cpu().numpy()))


def assert_equals(a, b, eps=1e-6):
    assert torch.sum(torch.ge(torch.abs(a - b), eps)) == 0


def assert_relative_equals(a, b, perc=0.01):
    diff = torch.sum(torch.abs(a - b))
    tot = torch.sum(a + b)
    assert diff / tot < perc


def load_dir_results(log_dir):
    print(f"Reporting results in {log_dir}")
    res = []
    for file in os.scandir(log_dir):
        if os.path.isdir(file):
            log_file = log_dir + file.name + '/checkpoint.pickle'

            try:
                with open(log_file, 'rb') as f:
                    d = pickle.load(f)
                    best_result = d['best_result']
                    train_par = d['train_params']
                    # model_par = d['model_params']

                res.append((best_result, train_par)) #, model_par))
            except EOFError:
                print(f"could not open {log_file}")
    return res
