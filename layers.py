# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_attention(distances, scale, width):
    # distances: tensor of topological divergences
    masks = torch.exp(-(((distances - scale) ** 2) / width))
    return masks

class GaussianNoise(nn.Module):
    def __init__(self, std=0.001):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


class DropOut(nn.Module):
    def __init__(self, p=0.05):
        super(DropOut, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            return nn.Dropout(self.p)(x)
        return x


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFN, self).__init__()
        self.ffn = nn.Sequential(
            # nn.LayerNorm(input_dim),
            GaussianNoise(),
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            DropOut(),
            nn.Linear(hidden_dim, output_dim),
            # nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.ffn(x)

class SoftAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        initial_scale,
        initial_width,
    ):
        # initial_scales: list of initial scale values for each head
        #  [scale1, scale2, ...]
        # TODO: initial_scales를 리스트로 만들기
        super(SoftAttention, self).__init__()
        self.scale = nn.Parameter(
            torch.tensor(initial_scale, dtype=torch.float32), requires_grad=True
        )
        self.width = nn.Parameter(
            torch.tensor(initial_width, dtype=torch.float32), requires_grad=True
        )
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        assert (
            self.head_dim * num_heads == output_dim
        ), "hidden_dim must be divisible by num_heads"
        # self.qkv = nn.Linear(input_dim, output_dim * 3)
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(K.size(-1), dtype=torch.float32)
        )
        masks = soft_attention(adj, self.scale, self.width)
        masked_scores = scores * masks
        attention_weights = F.softmax(masked_scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output

class DeepInteractLayer_Base(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, initial_scale):
        super(DeepInteractLayer, self).__init__()
        self.residual = nn.Parameter(
            torch.tensor(0.5, dtype=torch.float32), requires_grad=True
        )
        self.attention = SoftAttention(input_dim, output_dim, num_heads, initial_scale)
        self.ffn = FFN(output_dim, output_dim, output_dim)
        self.projection_for_residual = nn.Linear(input_dim, output_dim)
        # GRU for residual connection
        # self.gru = nn.GRU(
        #     input_size=output_dim, hidden_size=output_dim, batch_first=True
        # )

    def forward(self, x, adj):
        h = self.attention(x, adj)
        h = self.ffn(h)
        x = self.projection_for_residual(x)
        # h = self.gru(x, h)
        h = self.residual * h + (1 - self.residual) * x
        return h

class DeepInteractLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, initial_scale, initial_width):
        super(DeepInteractLayer, self).__init__()
        self.residual = nn.Parameter(
            torch.tensor(0.5, dtype=torch.float32), requires_grad=True
        )
        self.attention = SoftAttention(input_dim, output_dim, num_heads, initial_scale, initial_width)
        self.ffn = FFN(output_dim, output_dim, output_dim)
        self.projection_for_residual = nn.Linear(input_dim, output_dim)
        # GRU for residual connection
        # self.gru = nn.GRU(
        #     input_size=output_dim, hidden_size=output_dim, batch_first=True
        # )

    def forward(self, x, adj):
        h = self.attention(x, adj)
        h = self.ffn(h)
        x = self.projection_for_residual(x)
        # h = self.gru(x, h)
        h = self.residual * h + (1 - self.residual) * x
        return h
