import torch
from collections import defaultdict

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def calc_final_size(input_dim, num_conv_blocks, num_filters, conv_block_scale=0.5):
    """
    Inputs:
    - input_dim: one dimension of input images ie (for 32x32 imgs, input_dim=32)
    - num_conv_blocks: number of conv_blocks used
    - num_filters: number of filters used in conv_blocks
    - conv_block_scale: dimension scaling due to conv block 
    Returns:
    - size of final output of conv_blocks
    >>> calc_final_size(84, 4, 32, conv_block_scale=0.5)
    800
    >>> calc_final_size(84, 4, 64, conv_block_scale=0.5)
    1600
    """
    # each conv_block scales input dims by 0.5 for learner.conv_block (conv-batchnorm-relu-maxpool)
    final_height = int(input_dim * (0.5)**num_conv_blocks)
    return final_height**2 * num_filters

if __name__ == "__main__":
    import doctest
    doctest.testmod()
