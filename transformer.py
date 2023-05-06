import torch
import torch.nn as nn


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=mask)
        x = residual + self.drop1(x)
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.drop2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout_rate
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate

        self.embedding = nn.Linear(self.input_dim, self.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, 300, self.d_model)
        )  # max_len = 64
        self.pos_dropout = nn.Dropout(self.dropout_rate)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    self.d_model,
                    self.nhead,
                    self.dim_feedforward,
                    dropout=self.dropout_rate,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.d_model)

        init_weights(self.embedding)
        for layer in self.layers:
            init_weights(layer)

    def forward(self, x, mask=None):
        B, seq_len, _ = x.shape

        # embedding layer
        x = self.embedding(x)

        # add positional embeddings
        pos_embedding = self.pos_embedding[0, :seq_len, :]
        x = x + pos_embedding

        # add then concat
        # cls_token = self.cls_token.repeat(x.size(0), 1, 1).to(x.device)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = self.pos_dropout(x)

        if mask is not None:
            # B, 1 + seq_len
            mask = torch.cat([torch.ones((B,)).unsqueeze(1), mask], dim=1)
            # B, 1 + seq, 1 + seq
            mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            mask.diagonal(dim1=-2, dim2=-1).copy_(1)
            mask = ~torch.repeat_interleave(mask, self.nhead, dim=0).bool()

        # transformer encoder layers
        for layer in self.layers:
            x = layer(x, mask=mask)

        # # remove cls token
        # x = x[:, 1:, :]

        # # layer normalization
        # x = self.norm(x)

        # return the cls token embedding
        return x[:, 0, :]
