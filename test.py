import argparse
import datetime
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from clean_dataset import clean_test_set, clean_train_set
from dataset import CustomerPortraitDataset, collate_data_fn
from focal_loss import FocalLoss
from model_all import ModelAll


def get_args():
    parser = argparse.ArgumentParser(
        description='Training Script for CMB FinTech 2023 Data Analysis Competition.'
    )

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=2)
    # user info
    parser.add_argument('--gender_embedding_dim', type=int, default=2)
    parser.add_argument('--age_embedding_dim', type=int, default=4)
    parser.add_argument('--user_info_hidden_dim', type=int, default=8)

    # app history
    parser.add_argument('--page_id_embedding_dim', type=int, default=32)
    parser.add_argument('--acs_day_embedding_dim', type=int, default=16)
    # parser.add_argument('--app_his_hidden_dim', type=int, default=64)

    # trx history
    parser.add_argument('--trx_type_embedding_dim', type=int, default=32)
    parser.add_argument('--trx_amt_embedding_dim', type=int, default=4)
    parser.add_argument('--trx_day_embedding_dim', type=int, default=16)
    # parser.add_argument('--trx_his_hidden_dim', type=int, default=)

    # transformer args
    parser.add_argument('--transformer_hidden_dim', type=int, default=32)
    parser.add_argument('--transformer_nhead', type=int, default=4)
    parser.add_argument('--transformer_nlayers', type=int, default=2)

    parser.add_argument('--feat_dim', type=int, default=64)

    args = parser.parse_args()

    return args


def val(args):
    clean_train_set()
    clean_test_set()
    test_dataset = CustomerPortraitDataset(mode='test')
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_data_fn,
        num_workers=2,
    )

    model = ModelAll(args)
    model.load_state_dict(torch.load('/tasks/10952/model_0.pt'))
    print(model)
    # todo: set bias wd = 0

    import pandas as pd

    output_df = pd.DataFrame(columns=['cust_wid', 'label'])

    print(f"Total iters: {len(test_loader)}")

    with torch.no_grad():
        for i, (batch_x, labels) in enumerate(test_loader):
            if i % 20 == 0:
                print(
                    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} iter {i}"
                )
            binary_logits, multi_class_logits = model(batch_x)
            # labels = labels.detach().cpu().numpy()

            binary_prediction = F.softmax(binary_logits, dim=-1)
            binary_prediction = torch.argmax(binary_prediction, dim=-1).cpu().numpy()
            multi_prediction = F.softmax(multi_class_logits, dim=-1)
            multi_prediction = torch.argmax(multi_prediction, dim=-1).cpu().numpy() + 1
            multi_prediction[np.where(binary_prediction == 0)[0]] = 0

            cust_wid = batch_x['cust_wid']
            new_rows = pd.DataFrame(
                {'cust_wid': cust_wid, 'label': multi_prediction.astype(int)}
            )
            output_df = pd.concat([output_df, new_rows], ignore_index=True)
            # output_df = output_df.append(new_rows, ignore_index=True)

    output_df.to_csv("/work/output.csv")


if __name__ == '__main__':
    args = get_args()
    print(args)
    val(args)
