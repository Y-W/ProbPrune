from __future__ import absolute_import, division, print_function

from torch.autograd import Function
import torch.nn as nn


class ProbPrune(Function):
    @staticmethod
    def forward(ctx, input, prob):
        rand_samp = input.new().random_()
        mul = rand_samp.lt(prob).float()
        ctx.save_for_backward(input, mul)
        return input * mul

    @staticmethod
    def backward(ctx, grad_output):
        input, mul = ctx.saved_variables
        grad_input = grad_output * mul
        grad_prob = grad_output * input
        return grad_input, grad_prob


def prob_prune(input, prob):
    ProbPrune.apply(input, prob)


class BinaryPrune(Function):
    @staticmethod
    def forward(ctx, input, prob):
        mul = prob.gt(0.5).float()
        ctx.save_for_backward(input, mul)
        return input * mul

    @staticmethod
    def backward(ctx, grad_output):
        input, mul = ctx.saved_variables
        grad_input = grad_output * mul
        grad_prob = grad_output * input
        return grad_input, grad_prob


def binary_prune(input, prob):
    BinaryPrune.apply(input, prob)
