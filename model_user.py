import torch
import torch.nn as nn

"""
gender: M, F
age: 20-; 20-30; 30-40; 40-50; 50-60; 60+
"""


class UserInfoModel(nn.Module):
    def __init__(self, args):
        super(UserInfoModel, self).__init__()
        self.gender_embedding_dim = args.gender_embedding_dim
        self.age_embedding_dim = args.age_embedding_dim
        self.user_info_hidden_dim = args.user_info_hidden_dim

        self.gender_embedding = nn.Embedding(2, self.gender_embedding_dim)
        self.age_embedding = nn.Embedding(6, self.age_embedding_dim)

        self.fc_gender = nn.Linear(self.gender_embedding_dim, self.user_info_hidden_dim)
        self.bn_gender = nn.BatchNorm1d(self.user_info_hidden_dim)
        self.fc_age = nn.Linear(self.age_embedding_dim, self.user_info_hidden_dim)
        self.bn_age = nn.BatchNorm1d(self.user_info_hidden_dim)
        self.fc_merge = nn.Linear(
            self.user_info_hidden_dim * 2, self.user_info_hidden_dim
        )
        self.bn_merge = nn.BatchNorm1d(self.user_info_hidden_dim)

        self.fc_merge2 = nn.Linear(self.user_info_hidden_dim, self.user_info_hidden_dim)

        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

        for layer in [self.fc_gender, self.fc_age, self.fc_merge, self.fc_merge2]:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x_gender, x_age = x['gender'], x['age']
        x_gender = self.gender_embedding(x_gender)
        x_age = self.age_embedding(x_age)

        x_gender = self.fc_gender(x_gender)
        x_gender = self.bn_gender(x_gender)
        x_gender = self.act_fn(x_gender)
        x_gender = self.dropout(x_gender)

        x_age = self.fc_age(x_age)
        x_age = self.bn_age(x_age)
        x_age = self.act_fn(x_age)
        x_age = self.dropout(x_age)

        x_merge = torch.cat([x_gender, x_age], dim=1)
        x_merge = self.fc_merge(x_merge)
        x_merge = self.bn_merge(x_merge)
        x_merge = self.act_fn(x_merge)
        x_merge = self.dropout(x_merge)

        x_merge = self.fc_merge2(x_merge)

        return x_merge
