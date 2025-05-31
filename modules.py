import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


class CharTokenizer:
    def __init__(self, vocabs: list[str]) -> None:
        self.nvocab = len(vocabs)
        self.encoder = {char: idx for idx, char in enumerate(vocabs)}
        self.decoder = {idx: char for char, idx in self.encoder.items()}

    def encode(self, text: str) -> list[int]:
        return [self.encoder[char] for char in text]

    def decode(self, indices: list[int]) -> str:
        return ''.join(self.decoder[idx] for idx in indices)


@dataclass
class SysConfig:
    V: int  # number of vocabs, V
    B: int  # batch size, B
    T: int  # max number of tokens, T
    C: int  # embedding dimension (column), C
    H: int  # number of heads, H


# thin wrapper to include loss in the model
class NeuralNet(nn.Module):
    def __init__(self, context_length: int) -> None:
        super().__init__()
        self._context_length = context_length

    def loss(self, logits: torch.Tensor, target_indx: torch.Tensor) -> torch.Tensor:
        # default loss estimator with cross-entropy
        return F.cross_entropy(input=logits.view(-1, logits.size(-1)), target=target_indx.view(-1))

# Embedding & Input Prep Moldules


class TokenEmbedding(nn.Module):
    """
    Create token embedding for the LLM. 
    Trainable params: 
    - token embedding matrix of shape (V, D)
    """

    def __init__(self, config: SysConfig) -> None:
        super().__init__()
        self.params_ = nn.Parameter(torch.randn((config.V, config.C)))

    def forward(self, token_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for token embedding.

        :param torch.Tensor token_index: Indices of the tokens to embed
        :return torch.Tensor: Token embeddings of shape (B, T, D)
        """
        return self.params_[token_index]


class PositionalEmbedding(nn.Module):
    def __init__(self, config: SysConfig) -> None:
        """
        Initialize positional embedding.

        Trainable params:
        - positional embedding matrix of shape (1, T, D)
        """
        super().__init__()
        self.params_ = nn.Parameter(torch.randn((1, config.T,
                                                config.C)))  # TxV should be enough

    def forward(self, length: int) -> torch.Tensor:
        """
        Forward pass for positional embedding.

        :param torch.Tensor token_indices: Token indices of shape (B, T)
        :return torch.Tensor: Positional embeddings of shape (B, T, D)
        """
        return self.params_[:, :length, :]


class InputEmbedding(nn.Module):
    def __init__(self, config: SysConfig) -> None:
        super().__init__()
        self.token_embed = TokenEmbedding(config=config)
        self.pos_embed = PositionalEmbedding(config=config)

    def forward(self, token_indices: torch.Tensor) -> torch.Tensor:
        return self.token_embed(token_indices) + self.pos_embed(token_indices.size(1))


class LayerNorm(nn.Module):
    def __init__(self, config: SysConfig) -> None:
        super().__init__()
        self.beta_ = nn.Parameter(torch.zeros((1, config.C)))
        self.gamma_ = nn.Parameter(torch.ones((1, config.C)))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        mean = X.mean(dim=-1, keepdim=True)  # column-wise mean
        var = X.var(dim=-1, keepdim=True, unbiased=False)  # column-wise variance
        return self.gamma_ * (X - mean) / torch.sqrt(var + 1e-5) + self.beta_


class AttentionHead(nn.Module):
    def __init__(self, config: SysConfig) -> None:
        super().__init__()
        # Weights and biases for the Q, K, V matrices (see forward)
        # Each cases have weights of size [C, C] and biases of size [C, 1]
        assert config.C % config.H == 0, "Num of attn. head must be multiple of model dimension"
        self.head_num = config.H
        self.head_dim = config.C // self.head_num
        # projection of input X into Q, K, V matrices
        self.Q_proj_ = nn.Linear(config.C, config.C, bias=True)  # in: [B,T,C], out: [B,T,C]
        self.K_proj_ = nn.Linear(config.C, config.C, bias=True)  # in: [B,T,C], out: [B,T,C]
        self.V_proj_ = nn.Linear(config.C, config.C, bias=True)  # in: [B,T,C], out: [B,T,C]
        # projection of attention output to final output (in: [B,T,C], out: [B,T,C])
        self.A_proj = nn.Linear(config.C, config.C, bias=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, T, C = X.size()
        # reshape projected X into (B, T, head_num, head_dim)
        # after that, transpose to (B, head_num, T, head_dim)
        Q = self.Q_proj_(X).view(B, T, self.head_num, self.head_dim).transpose(1, 2)
        K = self.K_proj_(X).view(B, T, self.head_num, self.head_dim).transpose(1, 2)
        V = self.V_proj_(X).view(B, T, self.head_num, self.head_dim).transpose(1, 2)

        # compute attention scores
        A = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)  # (B, head_num, T, T)
        # apply mask for causal attention
        mask = torch.tril(torch.ones((T, T), device=X.device)).unsqueeze(0).unsqueeze(0)
        A_weights = F.softmax(A.masked_fill(mask == 0, float('-inf')),
                              dim=-1)  # (B, head_num, T, T)
        # compute attention output
        A_output = A_weights @ V  # (B, head_num, T, head_dim)
        # reshape back to (B, T, C) ... merge the outputs
        A_output = A_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.A_proj(A_output)


class TransformerBlock(nn.Module):
    def __init__(self, config: SysConfig) -> None:
        super().__init__()
        self.ln_inp_ = LayerNorm(config)
        self.attn_ = AttentionHead(config=config)  # C x T
        self.ffnn_ = nn.Sequential(nn.Linear(config.C, config.C * config.H),
                                   nn.GELU(),
                                   nn.Linear(config.C * config.H, config.C))
        self.ln_out_ = LayerNorm(config)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln_inp_(X)
        attn_res = self.attn_(x_norm)  # (B, C, T)?
        x_comp = attn_res + x_norm  # residual connection
        ffnn_output = self.ffnn_(self.ln_out_(x_comp))
        tf_out = ffnn_output + x_comp  # second residual connection
        return tf_out

# Output Layers


class OutputLayer(nn.Module):
    def __init__(self, config: SysConfig, token_embed: TokenEmbedding) -> None:
        super().__init__()
        self._token_embed = token_embed
        self.ln_inp_ = LayerNorm(config)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_n = self.ln_inp_(X)
        return X_n @ self._token_embed.params_.T

# Composition
