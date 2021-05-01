from mtftorch.testing._internal.common_utils import TestCase, run_tests

import mtftorch as torch
from mtftorch import tf, nn
import torch as pytorch

import functools


def compare_return_values(self, x, y, func, *args, **kws):
    if hasattr(x, 'numpy'):
        x = x.numpy()
    if hasattr(y, 'numpy'):
        y = y.numpy()
    if hasattr(x, 'indices') or hasattr(x, 'values'):
        if hasattr(x, 'indices'):
            compare_return_values(self, x.indices, y.indices, func, *args, **kws)
        if hasattr(x, 'values'):
            compare_return_values(self, x.values, y.values, func, *args, **kws)
    elif isinstance(x, tuple) and isinstance(y, tuple):
        for i, (A, B) in enumerate(zip(x, y)):
            compare_return_values(self, A, B, func, *args, **kws)
    else:
        self.assertEqual(x, y, rtol=1e-3, atol=1e-3)


def compare(f):
    @functools.wraps(f)
    def wrap(self, *args, **kws):
        A = f(self, pytorch, *args, **kws, tensor=pytorch.tensor)
        B = f(self, torch, *args, **kws, tensor=torch.tensor)
        compare_return_values(self, A, B, f, *args, **kws)
    return wrap

class TestModule(TestCase):
    def setUp(self):
        super().setUp()

    @compare
    def test_arange(self, torch, tensor):
        return torch.arange(4).view(2, 2)

    @compare
    def test_tensor(self, torch, tensor):
        return tensor([[4, 3, 5], [6, 7, 8]])

    @compare
    def test_take(self, torch, tensor):
        src = tensor([[4, 3, 5],
                      [6, 7, 8]])
        return torch.take(src, tensor([0, 2, 5]))

    @compare
    def test_argmax(self, torch, tensor):
        # a = torch.randn(4, 4)
        a = tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [ 0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092,  0.5419, -0.2993,  0.3195]])
        return a, torch.argmax(a), torch.argmax(a, dim=1)

    @compare
    def test_mean(self, torch, tensor):
        # a = torch.randn(4, 4)
        a = tensor([[-0.3841,  0.6320,  0.4254, -0.7384],
                    [-0.9644,  1.0131, -0.6549, -1.4279],
                    [-0.2951, -1.3350, -0.7694,  0.5600],
                    [ 1.0842, -0.9580,  0.3623,  0.2343]])
        return a, torch.mean(a), torch.mean(a, 1), torch.mean(a, 1, True)

    @compare
    def test_min(self, torch, tensor):
        a = tensor([[ 0.6750,  1.0857,  1.7197]])
        return a, torch.min(a)

    @compare
    def test_min_dim(self, torch, tensor):
        # a = torch.randn(4, 4)
        a = tensor([[-0.6248,  1.1334, -1.1899, -0.2803],
                    [-1.4644, -0.2635, -0.3651,  0.6134],
                    [ 0.2457,  0.0384,  1.0128,  0.7015],
                    [-0.1153,  2.9849,  2.1458,  0.5788]])
        return a, torch.min(a, 1)

    @compare
    def test_minimum(self, torch, tensor):
        a = torch.tensor((1, 2, -1))
        b = torch.tensor((3, 0, 4))
        return a, b, torch.minimum(a, b)

    @compare
    def test_amin(self, torch, tensor):
        # a = torch.randn(4, 4)
        a = tensor([[ 0.6451, -0.4866,  0.2987, -1.3312],
                    [-0.5744,  1.2980,  1.8397, -0.2713],
                    [ 0.9128,  0.9214, -1.7268, -0.2995],
                    [ 0.9023,  0.4853,  0.9075, -1.6165]])
        return a, torch.amin(a, 1)

    @compare
    def test_mm(self, torch, tensor):
        # >>> mat1 = torch.randn(2, 3)
        mat1 = tensor([[-0.6552,  0.0376,  0.8117],
                       [-2.2243,  0.0596,  1.2111]])
        # >>> mat2 = torch.randn(3, 3)
        mat2 = tensor([[ 1.4362,  0.0861, -0.6185],
                       [ 1.4264,  2.3783,  0.3660],
                       [-1.2079,  0.1771,  0.6216]])
        # >>> torch.mm(mat1, mat2)
        # tensor([[-1.8678,  0.1768,  0.9235],
        #         [-4.5724,  0.1647,  2.1504]])
        return mat1, mat2, torch.mm(mat1, mat2)



if __name__ == '__main__':
    run_tests()
