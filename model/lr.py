# -*- coding: utf-8 -*-
# @Time    : 2021/4/22 10:26
# @Author  : kaka

import numpy as np
import torch


class LR(torch.nn.Module):
    def __init__(self, field_dims):
        super().__init__()
        dim_in = sum(field_dims)
        self.fc = torch.nn.Embedding(dim_in, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        fea_idx_offsets = np.cumsum(field_dims)[:-1]
        self.fea_idx_offsets = np.array([0, *fea_idx_offsets], dtype=np.long)

    def forward(self, x):
        """

        :param x: [batch, field_num]
        :return:
        """
        x = self.fc(x + x.new_tensor(self.fea_idx_offsets).unsqueeze(0))
        x = torch.sum(x, dim=1) + self.bias
        out = torch.sigmoid(x.squeeze(1))
        return out
