import torch
import torch.nn as nn

from model_app import AppHistoryModel
from model_trx import TransactionModel
from model_user import UserInfoModel


class ModelAll(nn.Module):
    def __init__(self, args):
        super().__init__()
        # feature extractor
        self.user_info_encoder = UserInfoModel(args)
        self.app_his_encoder = AppHistoryModel(args)
        self.trx_his_encoder = TransactionModel(args)
        # predictor
        self.pooler1 = nn.Linear(
            args.user_info_hidden_dim
            + args.transformer_hidden_dim
            + args.transformer_hidden_dim,
            args.feat_dim,
        )
        self.pooler2 = nn.Linear(args.feat_dim, args.feat_dim)
        self.buy_predictor = nn.Linear(args.feat_dim, 2)
        self.buy_day_predictor = nn.Linear(
            args.feat_dim, 14
        )  # 0-13 (need mapping to 1-14)

        for layer in [
            self.pooler1,
            self.pooler2,
            self.buy_predictor,
            self.buy_day_predictor,
        ]:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # 'age' : age,
        # 'gender' : gender,
        # 'page_id' : page_id,
        # 'acs_day' : acs_day,
        # 'mask_app' : mask_app,
        # 'trx_type' : trx_type,
        # 'trx_amt' : trx_amt,
        # 'trx_day' : trx_day,
        # 'mask_trx' : mask_trx
        userinfo_feat = self.user_info_encoder(x)
        app_his_feat = self.app_his_encoder(x)
        trx_his_feat = self.trx_his_encoder(x)
        overall_feat = torch.cat([userinfo_feat, app_his_feat, trx_his_feat], dim=-1)
        overall_feat = self.pooler1(overall_feat)
        overall_feat = self.pooler2(overall_feat)
        return self.buy_predictor(overall_feat), self.buy_day_predictor(overall_feat)
