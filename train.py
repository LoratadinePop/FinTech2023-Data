import argparse
import datetime
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from clean_dataset import clean_train_set
from dataset import CustomerPortraitDataset, collate_data_fn
from focal_loss import FocalLoss
from model_all import ModelAll


def get_args():
    parser = argparse.ArgumentParser(
        description='Training Script for CMB FinTech 2023 Data Analysis Competition.'
    )

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=2)
    # user info
    parser.add_argument('--gender_embedding_dim', type=int, default=2)
    parser.add_argument('--age_embedding_dim', type=int, default=4)
    parser.add_argument('--user_info_hidden_dim', type=int, default=8)

    # app history
    parser.add_argument('--page_id_embedding_dim', type=int, default=16)
    parser.add_argument('--acs_day_embedding_dim', type=int, default=8)
    # parser.add_argument('--app_his_hidden_dim', type=int, default=64)

    # trx history
    parser.add_argument('--trx_type_embedding_dim', type=int, default=16)
    parser.add_argument('--trx_amt_embedding_dim', type=int, default=4)
    parser.add_argument('--trx_day_embedding_dim', type=int, default=8)
    # parser.add_argument('--trx_his_hidden_dim', type=int, default=)

    # transformer args
    parser.add_argument('--transformer_hidden_dim', type=int, default=32)
    parser.add_argument('--transformer_nhead', type=int, default=4)
    parser.add_argument('--transformer_nlayers', type=int, default=4)

    parser.add_argument('--feat_dim', type=int, default=64)

    args = parser.parse_args()

    return args


def make_optimizer(model, args):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = args.wd
        if "bias" in key or "pos_embed" in key or "cls_token" in key:
            weight_decay = 0
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd
    )
    return optimizer


def val(model, val_loader):
    print("Start validation...")
    binary_pred = np.array([])
    binary_label = np.array([])
    multi_pred = np.array([])
    multi_label = np.array([])

    with torch.no_grad():
        for i, (batch_x, labels) in enumerate(val_loader):
            binary_logits, multi_class_logits = model(batch_x)
            labels = labels.detach().cpu().numpy()

            # 计算是否会购买
            b_label = np.copy(labels)
            b_label[np.where(b_label > 0)[0]] = 1
            binary_label = np.hstack((binary_label, b_label))

            binary_prediction = F.softmax(binary_logits, dim=-1)
            binary_prediction = torch.argmax(binary_prediction, dim=-1).cpu().numpy()
            binary_pred = np.hstack((binary_pred, binary_prediction))

            # 如果预测为购买，计算具体购买的时机
            multi_prediction = F.softmax(multi_class_logits, dim=-1)
            # 0-13 -> 1-14
            multi_prediction = torch.argmax(multi_prediction, dim=-1).cpu().numpy() + 1
            multi_pred = np.hstack(
                (multi_pred, multi_prediction[np.where(binary_prediction == 1)[0]])
            )
            multi_label = np.hstack(
                (multi_label, labels[np.where(binary_prediction == 1)[0]])
            )

    import sklearn
    from sklearn.metrics import precision_score, recall_score

    precision = precision_score(binary_label, binary_pred)
    recall = recall_score(binary_label, binary_pred)
    F2 = (5 * precision * recall) / (4 * precision + recall)
    print(f"F2 score: {F2}")
    mape = sum(abs(multi_pred - multi_label) / (multi_pred * len(multi_pred)))
    print(f"MAPE score: {mape}")
    score = F2 - 0.2 * mape
    print(f"val result: {score}")
    return score


def train(args):
    train_dataset = CustomerPortraitDataset(mode='train')
    val_dataset = CustomerPortraitDataset(mode='val')
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_data_fn,
        num_workers=2,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_data_fn,
        num_workers=2,
    )

    model = ModelAll(args)
    print(model)
    # todo: set bias wd = 0
    optimizer = make_optimizer(model, args)
    # todo: init parameters
    total_iter = len(train_loader) * args.epoch
    warmup_iter = len(train_loader) / 10
    print(
        f"len train_loader: {len(train_loader)}, total_iter: {total_iter}, warmup_iter: {warmup_iter}"
    )

    def lr_schedule(iter):
        if iter < warmup_iter:
            lr = args.lr * ((iter + 1) / warmup_iter)
        else:
            lr = (
                warmup_iter
                * 0.5
                * (
                    1
                    + math.cos(
                        (iter - warmup_iter) / (total_iter - warmup_iter) * math.pi
                    )
                )
            )
        return lr

    scheduler = LambdaLR(optimizer, lr_schedule)
    ce_loss = nn.CrossEntropyLoss()
    class_weight = [
        0.05713879,
        0.08300559,
        0.07816764,
        0.13556457,
        0.44754464,
        0.7972167,
        0.12758511,
        0.1016219,
        0.12484433,
        0.09356043,
        0.21127503,
        0.78015564,
        1.0,
        0.13976995,
    ]
    # class_weight = [1., 1.4527013, 1.36803119, 2.37254902, 7.83258929, 13.95228628, 2.2328985, 1.77850988, 2.18493151, 1.63742417, 3.6975764, 13.6536965, 17.50124688, 2.44614848]
    focal_loss = FocalLoss(alpha=class_weight, num_classes=14, size_average=True)

    best_val = -10000.0
    print(f"Pre val result: {val(model, val_loader)}")
    print("Start training!")
    for epoch in range(args.epoch):
        for i, (batch_x, labels) in enumerate(train_loader):
            # 0,1 | 0~13
            binary_logits, multi_class_logits = model(batch_x)

            # 0~14
            nonzero_idx = torch.nonzero(labels > 0).squeeze(-1)

            # modify > 0 labels into 1, [0, 1]
            binary_labels = labels.clone()
            binary_labels[nonzero_idx] = 1
            binary_loss = ce_loss(binary_logits, binary_labels.long())

            # 1-14
            multi_class_labels = labels[nonzero_idx].long()
            # change into 0-13, indicating the (0-13) + 1 days
            # ToRun: 这里要减去1，因为这个head只负责预测1-14，映射为0-13！！！预测的时候记得加回来！
            multi_class_labels = multi_class_labels - 1
            multi_class_loss = focal_loss(
                multi_class_logits[nonzero_idx], multi_class_labels
            )

            # 平衡 loss
            loss = binary_loss + 2 * multi_class_loss

            if i % 10 == 0:
                print(
                    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} epoch {epoch}, iter {i}, loss {loss}, binary loss {binary_loss} multiclass loss {2 * multi_class_loss}",
                    flush=True,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        torch.save(model.state_dict(), f"/work/model_{epoch}.pt")
        val_result = val(model, val_loader)
        if val_result > best_val:
            best_val = val_result
            torch.save(model.state_dict(), '/work/model_best.pt')


if __name__ == '__main__':
    clean_train_set()
    args = get_args()
    print(args)
    train(args)
