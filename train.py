# -*- coding: utf-8 -*-
# @Time    : 2021/4/21 17:31
# @Author  : kaka

import argparse
import logging
from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score

from dataset.util import get_data
from model.lr import LR
from model.fm import FM


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", type=str, default="lr", help="lr|fm|ffm|deepfm")
    parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda:0")
    parser.add_argument("--datapath", type=str, default="./data/criteo_sampled_data_10k.csv", help="criteo data path")
    parser.add_argument("--num_workers", type=int, default=0, help="data loader worker num")
    parser.add_argument("--min_thres", type=int, default=10, help="feature count less than min_thres will be merged")
    parser.add_argument("--log_interval", type=int, default=10, help="display loss during training")
    parser.add_argument("--early_stop_tolerance", type=int, default=3, help="early stop")
    parser.add_argument("--batch_size", type=int, default=256, help="batch_size")
    parser.add_argument("--epoch", type=int, default=100, help="epoch")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay")

    parser.add_argument("--fm_hidden", type=int, default=20, help="fm embedding dim")
    args = parser.parse_args()
    return args


def get_model(args, field_dims):
    model_name = args.model_name
    if model_name == "lr":
        model = LR(field_dims)
    elif model_name == "fm":
        model = FM(field_dims, args.fm_hidden)
    else:
        raise ValueError("Invalid model name:{0}".format(args.model_name))
    return model


def train_model(model, train_iter, val_iter, device, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCELoss()
    model.to(device)
    best_auc = 0.0
    tolerance = 0
    for epoch_idx in range(1, args.epoch + 1):
        model.train()
        tk0 = tqdm(train_iter, smoothing=0, mininterval=1.0)
        total_loss = 0.0
        for i, (fea, label) in enumerate(tk0):
            optimizer.zero_grad()
            fea.to(device)
            label.to(device)
            pred = model(fea)
            loss = criterion(pred, label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % args.log_interval == 0:
                tk0.set_postfix(loss=total_loss / args.log_interval)
                total_loss = 0.0
        auc = eval_model(model, val_iter, device)
        logging.info("epoch {0}, auc {1:.4f}".format(epoch_idx, auc))
        if args.early_stop_tolerance > 0:
            # early stop check
            if auc > best_auc:
                best_auc = auc
                tolerance = 0
            else:
                tolerance += 1
                if tolerance > args.early_stop_tolerance:
                    logging.info("Early stop")
                    return


def eval_model(model, data_iter, device):
    model.eval()
    with torch.no_grad():
        target = []
        pred = []
        for fea, label in data_iter:
            fea.to(device)
            label.to(device)
            cur_pred = model(fea)
            target.extend(label.tolist())
            pred.extend(cur_pred.tolist())
    auc = roc_auc_score(target, pred)
    return auc


def main():
    log_fmt = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    args = parse_args()
    logging.info(args)

    device = torch.device(args.device)
    train_iter, val_iter, test_iter, field_dims = get_data(args)
    model = get_model(args, field_dims)
    train_model(model, train_iter, val_iter, device, args)

    test_auc = eval_model(model, test_iter, device)
    logging.info("test auc :{0:.4f}".format(test_auc))


if __name__ == "__main__":
    main()
