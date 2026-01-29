import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _overlap_add_1d(
    patches: torch.Tensor,  # [B, C, Np, P]
    T_out: int,
    stride: int,
) -> torch.Tensor:
    """
    Reconstruct a [B, C, T] sequence from overlapped patches using simple averaging.
    """
    B, C, Np, P = patches.shape
    device = patches.device
    out = patches.new_zeros(B, C, T_out)
    cnt = patches.new_zeros(B, C, T_out)

    for i in range(Np):
        t0 = i * stride
        t1 = t0 + P
        out[:, :, t0:t1] += patches[:, :, i, :]
        cnt[:, :, t0:t1] += 1.0

    # avoid divide-by-zero (can only happen on padded tail)
    cnt = torch.where(cnt == 0, torch.ones_like(cnt), cnt)
    return out / cnt


def _maybe_pad_to_fit(T: int, P: int) -> Tuple[int, int]:
    """
    Returns (pad_right, T_padded) so that T_padded >= P.
    """
    if T >= P:
        return 0, T
    pad = P - T
    return pad, T + pad


# ----------------------------
# Classic sinusoidal PE (seq-first)
# ----------------------------
class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, 1, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def _extend(self, T_needed: int, device, dtype):
        T_have = self.pe.size(0)
        if T_needed <= T_have:
            return
        # rebuild larger PE
        pe = torch.zeros(T_needed, 1, self.d_model, dtype=torch.float32, device=device)
        position = torch.arange(0, T_needed, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pe = pe.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, d_model]
        T = x.size(0)
        self._extend(T, x.device, x.dtype)
        return x + self.pe[:T].to(device=x.device, dtype=x.dtype)


# ----------------------------
# PatchTST (minimal, channel-independent)
# ----------------------------
class Patchtst(nn.Module):
    """
    PatchTST-style encoder with time-patch tokenization and de-patching
    back to the full sequence. Matches the simple LSTM/Linear/Informer
    interface you shared:

      - ctor(*, nx, ny, hidden_size=256, dr=0.1, n_heads=4, e_layers=2, d_ff=None, ...)
      - forward(x) where x is [T, B, nx] and returns [T, B, ny]
      - attributes: name, nx, ny, hidden_size, ct, n_layers

    Key knobs:
      patch_len (P): length of each time patch
      patch_stride (S): hop length between patches
    """

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        hidden_size: Optional[int] = 256,
        dr: Optional[float] = 0.1,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: Optional[int] = None,
        max_pos_len: int = 5000,
        patch_len: int = 16,
        patch_stride: int = 8,
    ) -> None:
        super().__init__()
        self.name = "PatchTST"
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size or 256
        self.ct = 0
        self.n_layers = e_layers

        d_model = self.hidden_size
        d_ff = d_ff or 4 * d_model
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)

        # Tokenizer: per-channel patches of length P → d_model
        # (shared across channels to keep it lean)
        self.token_proj = nn.Linear(self.patch_len, d_model)

        # Positional encoding over the patch sequence
        self.pos_enc = _PositionalEncoding(d_model, max_len=max_pos_len)

        # Transformer over patches (seq-first)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dr or 0.0,
            batch_first=False,  # seq-first
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=e_layers)

        # Head: project token → patch reconstruction (per-channel)
        self.token_to_patch = nn.Linear(d_model, self.patch_len)

        # If ny != nx, add a thin mixing layer applied per time step
        self.mix_out = None
        if ny != nx:
            self.mix_out = nn.Linear(nx, ny, bias=True)

        self.dropout = nn.Dropout(dr or 0.0)

        # Init (xavier uniform for linears)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -------------- helpers --------------
    def _patchify(self, x_bct: torch.Tensor) -> torch.Tensor:
        """
        x_bct: [B, C, T]
        returns patches: [B, C, Np, P]
        """
        B, C, T = x_bct.shape
        P = self.patch_len
        S = self.patch_stride

        # Ensure we can unfold at least one patch
        pad_right, T_pad = _maybe_pad_to_fit(T, P)
        if pad_right > 0:
            x_bct = F.pad(x_bct, (0, pad_right))  # pad on the right along T

        # Unfold along last dim: [B, C, Np, P]
        # torch.Tensor.unfold works on any dimension
        patches = x_bct.unfold(dimension=2, size=P, step=S)  # [B, C, Np, P]
        return patches

    def _tokens_from_patches(self, patches: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        patches: [B, C, Np, P]
        returns tokens: [Np, B*C, d_model], and (B*C, Np)
        """
        B, C, Np, P = patches.shape
        # merge B and C so transformer runs across patch sequence for each (B,C)
        pc = patches.reshape(B * C, Np, P)      # [B*C, Np, P]
        tok = self.token_proj(pc)               # [B*C, Np, d_model]
        tok = self.dropout(tok)
        # to seq-first for encoder
        tok = tok.transpose(0, 1).contiguous()  # [Np, B*C, d_model]
        return tok, B, C

    def _patches_from_tokens(self, tok: torch.Tensor, B: int, C: int) -> torch.Tensor:
        """
        tok: [Np, B*C, d_model]
        returns patches_recon: [B, C, Np, P]
        """
        Np, BC, _ = tok.shape
        # back to [B*C, Np, d_model]
        tok = tok.transpose(0, 1).contiguous()
        # project each token back to a patch
        patches_rc = self.token_to_patch(tok)  # [B*C, Np, P]
        patches_rc = self.dropout(patches_rc)
        patches_rc = patches_rc.view(B, C, Np, self.patch_len)
        return patches_rc

    # -------------- forward --------------
    def forward(
        self,
        x: torch.Tensor,                 # [T, B, nx]
        do_drop_mc: Optional[bool] = False,
        dr_false: Optional[bool] = False,
    ) -> torch.Tensor:
        # Expect seq-first
        T, B, C_in = x.shape
        assert C_in == self.nx, f"Expected nx={self.nx}, got {C_in}"
        P, S = self.patch_len, self.patch_stride

        # Rearrange to [B, C, T]
        x_bct = x.permute(1, 2, 0).contiguous()

        # Tokenize (patchify → linear)
        patches = self._patchify(x_bct)                    # [B, C, Np, P]
        Np = patches.size(2)

        # Project patches to tokens and add positional encoding
        tokens, B_eff, C_eff = self._tokens_from_patches(patches)  # [Np, B*C, d_model]
        tokens = self.pos_enc(tokens)                              # [Np, B*C, d_model]

        # Transformer encoder over patch sequence
        z = self.encoder(tokens)                                   # [Np, B*C, d_model]

        # Back to patches and overlap-add
        patches_rc = self._patches_from_tokens(z, B_eff, C_eff)    # [B, C, Np, P]

        # Compute output length after de-patching (cover original T)
        # Effective coverage length:
        # if Np patches with length P and stride S, last end index:
        # L = (Np - 1) * S + P
        T_cover = (Np - 1) * S + P
        y_bct = _overlap_add_1d(patches_rc, T_cover, S)            # [B, C, T_cover]

        # Trim to original T (drop padded tail); if T_cover < T (shouldn’t happen), pad
        if y_bct.size(-1) >= T:
            y_bct = y_bct[..., :T]
        else:
            y_bct = F.pad(y_bct, (0, T - y_bct.size(-1)))

        # Optional channel mixing if ny != nx
        if self.mix_out is not None:
            # time-wise linear mix: reshape to [B*T, C] → [B*T, ny]
            BT = B * T
            y_bt_c = y_bct.permute(0, 2, 1).reshape(BT, self.nx)   # [B*T, nx]
            y_bt_o = self.mix_out(y_bt_c)                          # [B*T, ny]
            y_bto = y_bt_o.view(B, T, self.ny)                     # [B, T, ny]
            y_tbo = y_bto.permute(1, 0, 2).contiguous()            # [T, B, ny]
            return y_tbo

        # Else, keep channels = nx
        y_tbc = y_bct.permute(2, 0, 1).contiguous()                # [T, B, nx]
        return y_tbc
