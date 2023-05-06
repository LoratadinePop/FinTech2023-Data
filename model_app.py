import torch
import torch.nn as nn

from transformer import TransformerEncoder

"""
page_id: 64 个 
acs_tm
"""


class AppHistoryModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.page_id_embedding_dim = args.page_id_embedding_dim
        self.acs_day_embedding_dim = args.acs_day_embedding_dim
        # self.app_his_hidden_dim = args.app_his_hidden_dim

        # 82 种页面类型
        self.page_id_embedding = nn.Embedding(
            82 + 1, self.page_id_embedding_dim, padding_idx=0
        )
        # 31 个日期类型
        self.acs_day_embedding = nn.Embedding(
            31 + 1, self.acs_day_embedding_dim, padding_idx=0
        )
        # + 1 为 padding token
        self.fc_merge = nn.Linear(
            self.page_id_embedding_dim + self.acs_day_embedding_dim,
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

        torch.nn.init.xavier_normal_(self.fc_merge.weight)
        torch.nn.init.constant_(self.fc_merge.bias, 0)

    def forward(self, x):
        page_id, acs_day, mask_app = x['page_id'], x['acs_day'], x['mask_app']
        page_id, acs_day = self.page_id_embedding(page_id), self.acs_day_embedding(
            acs_day
        )
        res = torch.cat([page_id, acs_day], dim=-1)
        res = self.fc_merge(res)
        res = self.encoder(res, mask=mask_app)
        return res
