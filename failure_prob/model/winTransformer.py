import torch
import torch.nn as nn

from .base import BaseModel
from .utils import get_time_weight
from failure_prob.conf import Config

import numpy as np

def get_model(cfg: Config, input_dim: int) -> BaseModel:
    return WinTransformerModel(cfg, input_dim)


class WinTransformerModel(BaseModel):
    """
    Causal-style window Transformer.
    For each timestep t, use last n_history_steps to predict a score.
    """

    def __init__(self, cfg: Config, input_dim: int):

        np.random.seed(cfg.train.seed)
        torch.manual_seed(cfg.train.seed)

        super().__init__(cfg, input_dim)

        self.n_history = cfg.model.n_history_steps
        self.if_use_vl = cfg.model.if_use_vl

        self.hidden_dim = cfg.model.hidden_dim
        self.n_layers = cfg.model.n_layers
        self.n_heads = cfg.model.n_heads
        self.dropout = cfg.model.dropout

        # ------------------------------------------------
        # Input dimensions
        # ------------------------------------------------
        self.hidden_state_dim = input_dim            # 1024
        self.vl_state_dim = 2 * input_dim            # 2048 (only if use_vl)

        # ------------------------------------------------
        # Input projections
        # ------------------------------------------------
        self.hidden_proj = nn.Linear(
            self.hidden_state_dim, self.hidden_dim
        )

        if self.if_use_vl:
            self.vl_proj = nn.Linear(
                self.vl_state_dim, self.hidden_dim
            )

            # Fusion after projection
            self.fusion_proj = nn.Linear(
                self.hidden_dim * 2, self.hidden_dim
            )

        # ------------------------------------------------
        # Transformer encoder
        # ------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True,
            activation="relu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
        )

        # ------------------------------------------------
        # Output head
        # ------------------------------------------------
        self.output_head = nn.Linear(self.hidden_dim, 1)

        if cfg.model.final_act_layer == "sigmoid":
            self.final_act = nn.Sigmoid()
        elif cfg.model.final_act_layer == "relu":
            self.final_act = nn.ReLU()
        elif cfg.model.final_act_layer == "none":
            self.final_act = nn.Identity()
        else:
            raise ValueError(f"Unknown final activation: {cfg.model.final_act_layer}")

    # ==================================================
    # Forward
    # ==================================================
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        batch["features"]  : (B, T, 1024)
        batch["vl_states"] : (B, T, 2048) if use_vl
        """
        hidden_states = batch["features"]  # (B, T, 1024)
        B, T, _ = hidden_states.shape

        if self.if_use_vl:
            vl_states = batch["vl"]  # (B, T, 2048)

        scores = []

        for t in range(T):
            # ------------------------------------------------
            # Build time window
            # ------------------------------------------------
            if t < self.n_history - 1:
                hs_window = hidden_states[:, t:t+1, :].repeat(
                    1, self.n_history, 1
                )
                if self.if_use_vl:
                    vl_window = vl_states[:, t:t+1, :].repeat(
                        1, self.n_history, 1
                    )
            else:
                hs_window = hidden_states[
                    :, t - self.n_history + 1 : t + 1, :
                ]
                if self.if_use_vl:
                    vl_window = vl_states[
                        :, t - self.n_history + 1 : t + 1, :
                    ]

            # ------------------------------------------------
            # Projection
            # ------------------------------------------------
            hs_emb = self.hidden_proj(hs_window)  # (B, n, H)

            if self.if_use_vl:
                vl_emb = self.vl_proj(vl_window)  # (B, n, H)

                fused = torch.cat([hs_emb, vl_emb], dim=-1)  # (B, n, 2H)
                x = self.fusion_proj(fused)                  # (B, n, H)
            else:
                x = hs_emb                                   # (B, n, H)

            # ------------------------------------------------
            # Transformer
            # ------------------------------------------------
            h = self.transformer(x)          # (B, n, H)
            h_t = h[:, -1, :]                # (B, H)

            score_t = self.output_head(h_t)  # (B, 1)
            score_t = self.final_act(score_t)

            scores.append(score_t)

        scores = torch.stack(scores, dim=1)  # (B, T, 1)
        return scores

    # ==================================================
    # Loss
    # ==================================================
    def forward_compute_loss_step(
        self,
        batch: dict[str, torch.Tensor],
    ):
        """
        Regression loss between scores and step_labels
        Only valid steps (label != -1 and mask == 1) are counted
        """
        masks = batch["masks"]                # (B, T)
        step_labels = batch["step_labels"]    # (B, T)

        scores = self(batch).squeeze(-1)      # (B, T)

        # ------------------------------------------------
        # Valid mask
        # ------------------------------------------------
        label_mask = (step_labels != -1) & masks.bool()  # (B, T)

        # ------------------------------------------------
        # Regression loss (MSE)
        # ------------------------------------------------
        diff = scores - step_labels
        per_step_loss = diff ** 2             # (B, T)

        masked_loss = per_step_loss * label_mask.float()

        total_loss = masked_loss.sum() / (label_mask.sum() + 1e-8)

        # ------------------------------------------------
        # Logs
        # ------------------------------------------------
        logs = {
            "loss": total_loss.item(),
            "n_valid_steps": label_mask.sum().item(),
            "mse": masked_loss.sum().item() / (label_mask.sum().item() + 1e-8),
        }

        return total_loss, logs
