from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Function


class StraightThroughEstimator(Function):
    @staticmethod
    def forward(ctx, input, prob):
        rand_samp = input.new(input.size()).random_()
        mul = torch.lt(rand_samp, prob).float()
        ctx.save_for_backward(input, mul)
        return input * mul

    @staticmethod
    def backward(ctx, grad_output):
        input, mul = ctx.saved_variables
        grad_input = grad_output * mul
        grad_prob = grad_output * input
        return grad_input, grad_prob
