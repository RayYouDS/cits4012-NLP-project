"""M2: BiGRU with dual-channel attention for PIQA binary choice.

Input contract (M1 -> M2)
    goal_ids   (B, L_g)      long    goal token IDs
    goal_mask  (B, L_g)      bool    True = valid token, False = padding
    sol_ids    (B, 2, L_s)   long    candidate token IDs
    sol_mask   (B, 2, L_s)   bool
    labels     (B,)          long    0 or 1, passed to the loss function

    Default vocab_size = 8000 and pad_id = 3; match these to the tokenizer.

    The loader must guarantee right padding only, at least one valid token per
    row, and mask[i,t] == False  <=>  ids[i,t] == pad_id.
    Sequence lengths are read from the tensors, so other values work unchanged.

Output
    logits (B, 2)  raw scores for nn.CrossEntropyLoss
    attn   dict    "goal": (B, 2, L_s, L_g), "comp": (B, 2, L_s, L_s)
    Dictionary values are None for disabled channels or return_attn=False.

Usage
    logits, _ = model(goal_ids, goal_mask, sol_ids, sol_mask, return_attn=False)
    loss = nn.CrossEntropyLoss()(logits, labels)
"""

import math
import random
import warnings

import numpy as np
import torch
import torch.nn as nn


SEED = 4012


def set_seed(seed: int = SEED) -> None:
    """Set random seeds before constructing the model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Attention(nn.Module):
    """
    Supported attention scoring methods:
        dot         score = q . k
        scaled_dot  score = q . k / sqrt(d)
        general     score = q . W . k
        additive    score = v^T tanh(W1 q + W2 k)  (Bahdanau)
    """

    def __init__(self, dim: int, attention_type: str = "scaled_dot"):
        super().__init__()
        assert attention_type in {"dot", "scaled_dot", "general", "additive"}
        self.attention_type = attention_type
        self.dim = dim

        if attention_type == "general":
            self.W = nn.Linear(dim, dim, bias=False)
        elif attention_type == "additive":
            self.W1 = nn.Linear(dim, dim, bias=False)
            self.W2 = nn.Linear(dim, dim, bias=False)
            self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, query, keys, key_mask):
        """
        query    (N, Lq, D)
        keys     (N, Lk, D)
        key_mask (N, Lk)   bool, True = valid token, False = padding
        return   ctx (N, Lq, D),  attn (N, Lq, Lk)
        """
        if self.attention_type == "additive":
            e = torch.tanh(self.W1(query).unsqueeze(2) + self.W2(keys).unsqueeze(1))
            scores = self.v(e).squeeze(-1)                        # (N, Lq, Lk)
        else:
            k = self.W(keys) if self.attention_type == "general" else keys
            scores = torch.bmm(query, k.transpose(1, 2))          # (N, Lq, Lk)
            if self.attention_type == "scaled_dot":
                scores = scores / math.sqrt(self.dim)

        # Mask padded keys before applying softmax.
        neg = torch.finfo(scores.dtype).min / 2
        scores = scores.masked_fill(~key_mask.unsqueeze(1), neg)
        attn = torch.softmax(scores, dim=-1)                      # (N, Lq, Lk)

        # Set attention weights to zero when all keys are masked.
        has_key = key_mask.any(dim=1).view(-1, 1, 1)              # (N, 1, 1)
        attn = attn * has_key

        ctx = torch.bmm(attn, keys)                               # (N, Lq, D)
        return ctx, attn


class BiGRUAttention(nn.Module):
    """Encode the goal and candidates, then score each candidate."""

    N_CAND = 2

    def __init__(
        self,
        vocab_size: int = 8000,
        pad_id: int = 3,
        emb_dim: int = 200,
        hidden_dim: int = 256,
        num_layers: int = 1,
        rnn_type: str = "gru",            # gru | lstm
        bidirectional: bool = True,
        use_goal_attn: bool = True,       # Channel A: sol <- goal
        use_candidate_comp: bool = True,  # Channel B: sol_i <- sol_j
        attention_type: str = "scaled_dot",
        pooling: str = "mean+max",  # mean | max | mean+max
        use_fusion_gate: bool = True,
        block_norm: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert rnn_type in {"gru", "lstm"}
        assert pooling in {"mean", "max", "mean+max"}


        self.pad_id = pad_id
        self.use_goal_attn = use_goal_attn
        self.use_candidate_comp = use_candidate_comp
        self.pooling = pooling

        if use_fusion_gate and not use_goal_attn:
            warnings.warn(
                "Fusion gate is disabled because goal attention is disabled.",
                stacklevel=2,
            )
        self.use_fusion_gate = use_fusion_gate and use_goal_attn

        # Shared embedding layer for the goal and both candidates.
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)

        # Shared recurrent encoder for all input sequences.

        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.enc_dim = hidden_dim * (2 if bidirectional else 1)

        # Create the enabled attention channels.
        if use_goal_attn:
            self.attn_goal = Attention(self.enc_dim, attention_type)
        if use_candidate_comp:
            self.attn_comp = Attention(self.enc_dim, attention_type)


        if self.use_fusion_gate:
            self.gate = nn.Linear(2 * self.enc_dim, self.enc_dim)

        # Calculate the feature size after concatenation and pooling.
        n_blocks = 1 + int(use_goal_attn) + 3 * int(use_candidate_comp)
        n_pool = 2 if pooling == "mean+max" else 1
        feat_dim = self.enc_dim * n_blocks * n_pool
        self.n_blocks = n_blocks

        # Optionally normalise each feature block before concatenation.
        if block_norm:
            self.block_norms = nn.ModuleList(
                [nn.LayerNorm(self.enc_dim) for _ in range(n_blocks)])
        self.block_norm = block_norm
        self.feat_dim = feat_dim

        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1, bias=False),
        )

    def _encode(self, ids, mask):
        """Encode right-padded sequences of shape (N, L) into (N, L, D)."""
        x = self.embedding(ids)
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        # Pack valid tokens before passing them to the recurrent encoder.
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=ids.size(1)
        )
        return out

    @staticmethod
    def _masked_pool(x, mask, mode):
        """Pool valid tokens using mean, max, or both; return zero for empty rows."""
        m = mask.unsqueeze(-1).to(x.dtype)                    # (N, L, 1)
        outs = []
        if mode in ("mean", "mean+max"):
            outs.append((x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0))
        if mode in ("max", "mean+max"):
            neg = torch.finfo(x.dtype).min
            mx = x.masked_fill(~mask.unsqueeze(-1), neg).max(dim=1).values
            has_tok = mask.any(dim=1, keepdim=True).to(x.dtype)
            outs.append(mx * has_tok)
        return torch.cat(outs, dim=-1)

    def forward(self, goal_ids, goal_mask, sol_ids, sol_mask, return_attn: bool = True):
        """Score two candidate solutions for each goal.

        Inputs:
            goal_ids, goal_mask: (B, L_g)
            sol_ids, sol_mask: (B, 2, L_s)
            Masks are boolean, with True for valid tokens.
            Sequences must be right-padded and contain at least one valid token.

        Returns:
            Raw logits (B, 2) and a dictionary of attention weights:
            goal: (B, 2, L_s, L_g); comp: (B, 2, L_s, L_s).
            Values are None for disabled channels or when return_attn is False.
            Padded query rows must be excluded when plotting attention.
        """
        B, n_cand, L_s = sol_ids.shape
        assert n_cand == self.N_CAND, "Expected exactly two candidates."
        L_g = goal_ids.size(1)
        N = B * n_cand

        # Encode both candidates together as a batch of size B * 2.
        Sm = sol_mask.reshape(N, L_s)
        S = self._encode(sol_ids.reshape(N, L_s), Sm)

        feats = [S]
        attn_out = {"goal": None, "comp": None}


        if self.use_goal_attn:
            # Repeat the goal representation for each candidate.
            G = self._encode(goal_ids, goal_mask)                 # (B, L_g, D)
            G = G.unsqueeze(1).expand(B, n_cand, L_g, self.enc_dim)
            G = G.reshape(N, L_g, self.enc_dim)
            Gm = goal_mask.unsqueeze(1).expand(B, n_cand, L_g).reshape(N, L_g)

            ctx_g, a_g = self.attn_goal(S, G, Gm)                 # (N,L_s,D)
            if self.use_fusion_gate:
                # Blend candidate states with goal context.
                g = torch.sigmoid(self.gate(torch.cat([S, ctx_g], dim=-1)))
                ctx_g = g * ctx_g + (1.0 - g) * S
            feats.append(ctx_g)
            if return_attn:
                attn_out["goal"] = a_g.view(B, n_cand, L_s, L_g)

        if self.use_candidate_comp:
            # Swap candidates to attend to the other solution.
            other = S.view(B, n_cand, L_s, self.enc_dim).flip(1).reshape(N, L_s, -1)
            other_m = sol_mask.flip(1).reshape(N, L_s)
            ctx_c, a_c = self.attn_comp(S, other, other_m)        # (N,L_s,D)
            # Add comparison context, absolute differences, and products.
            feats += [ctx_c, (S - ctx_c).abs(), S * ctx_c]
            if return_attn:
                attn_out["comp"] = a_c.view(B, n_cand, L_s, L_s)


        if self.block_norm:
            feats = [ln(f) for ln, f in zip(self.block_norms, feats)]
        H = torch.cat(feats, dim=-1)                              # (N,L_s,feat)
        # Pool valid candidate tokens and compute one score per candidate.
        pooled = self._masked_pool(self.dropout(H), Sm, self.pooling)
        logits = self.scorer(pooled).squeeze(-1).view(B, n_cand)  # (B, 2)
        return logits, attn_out

    @torch.no_grad()
    def predict(self, goal_ids, goal_mask, sol_ids, sol_mask):
        """Return candidate indices (0 or 1) with dropout disabled."""
        was_training = self.training
        self.eval()
        try:
            logits, _ = self.forward(goal_ids, goal_mask, sol_ids, sol_mask,
                                     return_attn=False)
        finally:
            # Restore the model's previous training mode.
            self.train(was_training)
        return logits.argmax(dim=1)
