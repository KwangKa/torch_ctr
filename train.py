# -*- coding: utf-8 -*-
# @Time    : 2021/4/21 17:31
# @Author  : kaka

import argparse
from dataset.util import get_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="./data/criteo_sampled_data_10k.csv", help="criteo data path")
    parser.add_argument("--num_workers", type=int, default=4, help="data loader worker num")
    parser.add_argument("--min_thres", type=int, default=10, help="feature count less than min_thres will be merged")
    parser.add_argument("--batch_size", type=int, default=128, help="batch_size")
    parser.add_argument("--epoch", type=int, default=20, help="epoch")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    train_iter, val_iter, test_iter = get_data(args)

    for feature, label in train_iter:
        print(feature.shape)
        print(feature)
        print(label.shape)
        print(label)
        break


if __name__ == "__main__":
    main()
