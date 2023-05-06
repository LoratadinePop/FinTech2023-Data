import pickle

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CustomerPortraitDataset(Dataset):
    def __init__(self, mode="train"):
        super(CustomerPortraitDataset).__init__()
        self.mode = mode
        assert self.mode in ["train", "val", "test"], "Dataset mode error!"
        print(f"Init dataset with mode={self.mode}")
        
        if self.mode in ["train", "val"]:
            # self.customer_base_info_df = pd.read_csv('train_userinfo.csv')
            # self.app_histrory_df = pd.read_csv('trian_app_his.csv')
            # self.transaction_df = pd.read_csv('train_trx_his.csv')
            self.customer_base_info_df = pd.read_parquet('/work/train_userinfo.parquet')
            self.app_histrory_df = pd.read_parquet('/work/trian_app_his.parquet')
            self.transaction_df = pd.read_parquet('/work/train_trx_his.parquet')
        else:
            # self.customer_base_info_df = pd.read_csv('test_userinfo.csv')
            # self.app_histrory_df = pd.read_csv('test_app_his.csv')
            # self.transaction_df = pd.read_csv('test_trx_his.csv')

            self.customer_base_info_df = pd.read_parquet('/work/test_userinfo.parquet')
            self.app_histrory_df = pd.read_parquet('/work/test_app_his.parquet')
            self.transaction_df = pd.read_parquet('/work/test_trx_his.parquet')

        with open('train_page_id_map.pickle', 'rb') as f:
            self.page_id_map = pickle.load(f)
            print('page_id', len(list(self.page_id_map.keys())))
        with open('train_acs_day_map.pickle', 'rb') as f:
            self.acs_day_map = pickle.load(f)
            print('acs_day', len(list(self.acs_day_map.keys())))
        with open('train_trx_type_map.pickle', 'rb') as f:
            self.trx_type_map = pickle.load(f)
            print('trx_type', len(list(self.trx_type_map.keys())))
        with open('train_trx_day_map.pickle', 'rb') as f:
            self.trx_day_map = pickle.load(f)
            print('trx_day', len(list(self.trx_day_map.keys())))

        self.cust_wids = sorted(self.customer_base_info_df['cust_wid'].unique())
        if self.mode == "val":
            import random

            random.shuffle(self.cust_wids)
            self.cust_wids = self.cust_wids[:1000]
        print("End dataset initialization!")

    def __len__(self):
        return len(self.cust_wids)

    def __getitem__(self, idx):
        cust_wid = self.cust_wids[idx]
        # 'cust_wid', 'label', 'age', 'gdr_cd'
        info_df = self.customer_base_info_df.loc[
            self.customer_base_info_df['cust_wid'] == cust_wid
        ]
        # 'cust_wid', 'page_id', 'acs_day'
        app_his_df = self.app_histrory_df.loc[
            self.app_histrory_df['cust_wid'] == cust_wid
        ]
        # 'cust_wid', 'trx_cd', 'trx_amt', 'trx_day'
        trx_his_df = self.transaction_df.loc[
            self.transaction_df['cust_wid'] == cust_wid
        ]

        age = self.age_map(float(info_df['age'].values[0]))
        # print(info_df['age'].values[0], age)
        gender = self.gender_map(info_df['gdr_cd'].values[0])

        page_ids = []
        acs_days = []
        for pageid, acsday in zip(
            app_his_df['page_id'].values, app_his_df['acs_day'].values
        ):
            page_ids.append(self.page_id_map[pageid])
            acs_days.append(self.acs_day_map[acsday])

        trx_types = []
        trx_amts = []
        trx_days = []
        for trxtype, trxamt, trxday in zip(
            trx_his_df['trx_cd'].values,
            trx_his_df['trx_amt'].values,
            trx_his_df['trx_day'],
        ):
            trx_types.append(self.trx_type_map[trxtype])
            trx_amts.append(trxamt)
            trx_days.append(self.trx_day_map[trxday])
        # trx_types = torch.LongTensor(trx_types)
        # trx_amts = torch.tensor(trx_amts)
        # trx_days = torch.LongTensor(trx_days)

        return_dict = {
            'cust_wid': cust_wid,
            'age': age,
            'gender': gender,
            'page_id': page_ids,
            'acs_day': acs_days,
            'trx_type': trx_types,
            'trx_amt': trx_amts,
            'trx_day': trx_days,
        }

        if self.mode in ["train", "val"]:
            label = torch.tensor(info_df['label'].values[0])
        else:
            label = -1
        return return_dict, label

    def gender_map(self, gender):
        return 1 if gender == 'M' else 0

    def age_map(self, age):
        if age < 20:
            return 0
        elif age >= 20 and age < 30:
            return 1
        elif age >= 30 and age < 40:
            return 2
        elif age >= 40 and age < 50:
            return 3
        elif age >= 50 and age < 60:
            return 4
        else:
            return 5


def collate_data_fn(batch):
    # user info data
    cust_wid = [x[0]['cust_wid'] for x in batch]
    age = [x[0]['age'] for x in batch]
    age = torch.LongTensor(age)
    gender = [x[0]['gender'] for x in batch]
    gender = torch.LongTensor(gender)

    # App view history data
    page_id = [x[0]['page_id'] for x in batch]
    acs_day = [x[0]['acs_day'] for x in batch]
    max_app_history_len = max([len(x) for x in page_id])
    # pad with index 0
    page_id = [
        F.pad(torch.LongTensor(x), (0, max_app_history_len - len(x)), value=0)
        for x in page_id
    ]
    page_id = torch.stack(page_id)
    acs_day = [
        F.pad(torch.LongTensor(x), (0, max_app_history_len - len(x)), value=0)
        for x in acs_day
    ]
    acs_day = torch.stack(acs_day)
    mask_app = torch.tensor(
        [
            [1] * torch.nonzero(x).size(0)
            + [0] * (max_app_history_len - torch.nonzero(x).size(0))
            for x in page_id
        ]
    )

    # Trx history data
    trx_type = [x[0]['trx_type'] for x in batch]
    trx_amt = [x[0]['trx_amt'] for x in batch]
    trx_day = [x[0]['trx_day'] for x in batch]
    max_trx_len = max([len(x) for x in trx_type])
    # pad with index 0
    trx_type = [
        F.pad(torch.LongTensor(x), (0, max_trx_len - len(x)), value=0) for x in trx_type
    ]
    trx_type = torch.stack(trx_type)
    trx_amt = [
        F.pad(torch.tensor(x, dtype=torch.float32), (0, max_trx_len - len(x)), value=0)
        for x in trx_amt
    ]
    trx_amt = torch.stack(trx_amt)
    trx_day = [
        F.pad(torch.LongTensor(x), (0, max_trx_len - len(x)), value=0) for x in trx_day
    ]
    trx_day = torch.stack(trx_day)
    mask_trx = torch.tensor(
        [
            [1] * torch.nonzero(x).size(0)
            + [0] * (max_trx_len - torch.nonzero(x).size(0))
            for x in trx_type
        ]
    )

    labels = [x[1] for x in batch]
    labels = torch.tensor(labels)

    return {
        'cust_wid': cust_wid,
        'age': age,
        'gender': gender,
        'page_id': page_id,
        'acs_day': acs_day,
        'mask_app': mask_app,
        'trx_type': trx_type,
        'trx_amt': trx_amt,
        'trx_day': trx_day,
        'mask_trx': mask_trx,
    }, labels


# # 创建数据集
# dataset = CustomerPortraitDataset(is_train=True)

# # 创建 DataLoader，并指定 collate_fn
# dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_data_fn)

# # 训练模型
# num_epochs = 1
# for epoch in range(num_epochs):
#     for i, (batch_data, batch_labels) in enumerate(dataloader):
#         print(i, batch_data, batch_labels)
#         if i == 2:
#             break
