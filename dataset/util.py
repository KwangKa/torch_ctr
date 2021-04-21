# -*- coding: utf-8 -*-
# @Time    : 2021/4/21 23:13
# @Author  : kaka

from torch.utils.data import random_split, DataLoader
from dataset.criteo import CriteoDataset


def get_data(args):
    criteo_ds = CriteoDataset(dataset_path=args.datapath, rebuild_cache=False, min_threshold=args.min_thres)
    train_num = int(len(criteo_ds) * 0.8)
    val_num = int(len(criteo_ds) * 0.1)
    test_num = int(len(criteo_ds) - train_num - val_num)
    train_ds, val_ds, test_ds = random_split(criteo_ds, [train_num, val_num, test_num])
    train_data_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    val_data_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    return train_data_loader, val_data_loader, test_data_loader
