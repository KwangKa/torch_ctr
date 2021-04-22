# -*- coding: utf-8 -*-
#  @Time    : 2021/4/22 15:23
#  @Author  : kaka


import numpy as np
import torch


class FM(torch.nn.Module):
    def __init__(self, field_dims, n_hidden):
        super().__init__()
        embed_in = sum(field_dims)
        offset = np.cumsum(field_dims)[:-1]
        offset = np.array([0, *offset], dtype=np.long)
        self.offset = offset                                  # 每个特征field对应的index offset
        self.fc = torch.nn.Embedding(embed_in, 1)             # 一次项系数
        self.embed = torch.nn.Embedding(embed_in, n_hidden)   # 二次项fm embedding
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """

        :param x:  [batch_size, filed]
        :return:
        """
        x = x + x.new_tensor(self.offset, dtype=torch.long).unsqueeze(0)
        x_linear = self.fc(x)
        x_linear = torch.sum(x_linear, dim=1).squeeze(1)

        x_embed = self.embed(x)
        x_sum_and_square = torch.sum(x_embed, dim=1).square()
        x_square_and_sum = torch.sum(x_embed.square(), dim=1)
        x_fm = 0.5 * (x_sum_and_square - x_square_and_sum)
        x_fm = torch.sum(x_fm, dim=1)

        out = x_fm + x_linear + self.bias
        out = torch.sigmoid(out)
        return out
