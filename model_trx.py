import torch
import torch.nn as nn

from transformer import TransformerEncoder

"""
trx_cd: 41
trx_amt: 连续值 需要normalize一下~
trx_tm: 31
"""


class TransactionModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.trx_type_embedding_dim = args.trx_type_embedding_dim
        self.trx_amt_embedding_dim = args.trx_amt_embedding_dim
        self.trx_day_embedding_dim = args.trx_day_embedding_dim
        # self.trx_his_hidden_dim = args.trx_his_hidden_dim

        # 41 种交易类型
        self.trx_type_embedding = nn.Embedding(
            26 + 1, self.trx_type_embedding_dim, padding_idx=0
        )
        # 31 个日期类型
        self.trx_day_embedding = nn.Embedding(
            31 + 1, self.trx_day_embedding_dim, padding_idx=0
        )
        self.fc_trx_amt = nn.Linear(1, self.trx_amt_embedding_dim)
        self.fc_merge = nn.Linear(
            self.trx_type_embedding_dim
            + self.trx_amt_embedding_dim
            + self.trx_day_embedding_dim,
            args.transformer_hidden_dim,
        )
        self.encoder = TransformerEncoder(
            args.transformer_hidden_dim,
            args.transformer_hidden_dim,
            args.transformer_nhead,
            args.transformer_nlayers,
            args.transformer_hidden_dim * 4,
            0.2,
        )

        torch.nn.init.xavier_normal_(self.fc_trx_amt.weight)
        torch.nn.init.constant_(self.fc_trx_amt.bias, 0)

        torch.nn.init.xavier_normal_(self.fc_merge.weight)
        torch.nn.init.constant_(self.fc_merge.bias, 0)

    def forward(self, x):
        trx_type, trx_amt, trx_day, mask_trx = (
            x['trx_type'],
            x['trx_amt'],
            x['trx_day'],
            x['mask_trx'],
        )
        trx_type = self.trx_type_embedding(trx_type)
        # print(trx_type.shape)
        # print(trx_amt.unsqueeze(-1).shape)
        # print(trx_day.shape)
        trx_amt = self.fc_trx_amt(trx_amt.unsqueeze(-1)).squeeze(1)
        trx_day = self.trx_day_embedding(trx_day)
        res = torch.cat([trx_type, trx_amt, trx_day], dim=-1)
        res = self.fc_merge(res)
        res = self.encoder(res, mask=mask_trx)
        return res
