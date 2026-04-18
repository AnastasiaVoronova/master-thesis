from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


@dataclass
class ModelConfig:
    model_name: str
    stage: str  # "binary" | "multiclass"
    num_labels: int
    dropout: float = 0.1
    use_score_feature: bool = True


class ENPSClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.stage = cfg.stage
        self.num_labels = cfg.num_labels

        transformer_cfg = AutoConfig.from_pretrained(cfg.model_name)
        backbone = AutoModel.from_pretrained(cfg.model_name, config=transformer_cfg)
        # Seq2seq backbones (for example T5-based) require decoder inputs when called directly.
        # For classification we only need encoder representations.
        if getattr(transformer_cfg, "is_encoder_decoder", False) and hasattr(backbone, "get_encoder"):
            self.encoder = backbone.get_encoder()
        else:
            self.encoder = backbone

        hidden = getattr(transformer_cfg, "hidden_size", None)
        if hidden is None:
            hidden = getattr(transformer_cfg, "d_model", None)
        if hidden is None:
            raise ValueError(f"Cannot infer hidden size from config for model {cfg.model_name}")

        if cfg.stage == "binary":
            self.text_proj = nn.Sequential(
                nn.Linear(hidden, 128),
                nn.SiLU(),
            )
            self.score_proj = nn.Sequential(
                nn.Linear(1, 32),
                nn.SiLU(),
                nn.Linear(32, 64),
                nn.SiLU(),
            )
            self.dropout = nn.Dropout(cfg.dropout)
            self.head = nn.Linear(192, 1)
        elif cfg.stage == "multiclass":
            # Повторяет идею из ноутбука stage-2.
            self.classifier = nn.Sequential(
                nn.Linear(hidden, 256),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 512),
                nn.SiLU(),
                nn.Dropout(0.05),
                nn.Linear(512, 512),
                nn.SiLU(),
                nn.Dropout(0.05),
                nn.Linear(512, cfg.num_labels),
            )
        else:
            raise ValueError(f"Unknown stage: {cfg.stage}")

    def _pooled_output(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self._pooled_output(input_ids, attention_mask)
        if self.stage == "binary":
            x = self.text_proj(x)
            if scores is None:
                raise ValueError("scores tensor is required for binary stage")
            s = self.score_proj(scores.unsqueeze(1).float())
            x = torch.cat((x, s), dim=1)
            x = self.dropout(x)
            return self.head(x).squeeze(-1)

        return self.classifier(x)
