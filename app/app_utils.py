from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel


# ── Model ─────────────────────────────────────────────────────────────────────

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


# ── Dataset ───────────────────────────────────────────────────────────────────

class ENPSDataset(Dataset):
    def __init__(self, encodings: dict[str, np.ndarray], scores: np.ndarray, labels: Optional[np.ndarray] = None):
        self.input_ids = torch.tensor(encodings["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(encodings["attention_mask"], dtype=torch.long)
        self.scores = torch.tensor(scores, dtype=torch.float32)
        self.labels = None if labels is None else torch.tensor(labels)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "scores": self.scores[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


# ── Preprocessing ─────────────────────────────────────────────────────────────

CATEGORY_TO_INDEX: dict[str, int] = {
    "нет конкретного ответа": 0,
    "?": 0,
    "ок": 1,
    "work-life balance": 2,
    "адекватные планы и количество метрик": 3,
    "адекватные планы и кол-во метрик": 3,
    "бесплатное питание": 4,
    "бесплатная еда": 4,
    "бюрократия": 5,
    "взаимодействие": 6,
    "взаимодействие ": 6,
    "внерабочие активности": 7,
    "график работы": 8,
    "график": 8,
    "дополнительные сотрудники": 9,
    "идея по продукту": 10,
    "идеи по продукту": 10,
    "карьерный рост": 11,
    "клиенты": 12,
    "конкурсы": 13,
    "культура обратной связи": 14,
    "культура обратной связи ": 14,
    "лояльность к сотрудникам": 15,
    "льготы": 16,
    "ль": 16,
    "спортивный зал": 16,
    "бассейн": 16,
    "мерч": 17,
    "нездоровая атмосфера": 18,
    "обучение": 19,
    "оплата труда": 20,
    "оплата": 20,
    "оплата трудв": 20,
    "офисное пространство": 21,
    "подарки на праздники": 22,
    "подарки по праздникам": 22,
    "подарки детям": 22,
    "премии": 23,
    "Премии": 23,
    "процессы": 24,
    "сложность работы": 25,
    "техника/ит": 26,
    "технологии/ит": 26,
    "удаленная работа": 27,
    "работа из заграницы": 27,
    "работа из других стран": 27,
    "оплата сверхурочного труда": 28,
    "руководитель": 29,
}

INDEX_TO_CATEGORY: dict[int, str] = {
    0: "нет конкретного ответа",
    1: "ок",
    2: "work-life balance",
    3: "адекватные планы и количество метрик",
    4: "бесплатное питание",
    5: "бюрократия",
    6: "взаимодействие",
    7: "внерабочие активности",
    8: "график работы",
    9: "дополнительные сотрудники",
    10: "идея по продукту",
    11: "карьерный рост",
    12: "клиенты",
    13: "конкурсы",
    14: "культура обратной связи",
    15: "лояльность к сотрудникам",
    16: "льготы",
    17: "мерч",
    18: "нездоровая атмосфера",
    19: "обучение",
    20: "оплата труда",
    21: "офисное пространство",
    22: "подарки на праздники",
    23: "премии",
    24: "процессы",
    25: "сложность работы",
    26: "техника/ит",
    27: "удаленная работа",
    28: "оплата сверхурочного труда",
    29: "руководитель",
}

NON_ALPHA_RE = re.compile(r"[^a-zа-яё]", flags=re.IGNORECASE)
ONE_CHAR_WORD_RE = re.compile(r"\b\w\b", flags=re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s+")


@dataclass
class TextPreprocessConfig:
    syntax_correction: bool = False
    lemmatization: bool = False
    stopwords_removal: bool = False
    russian_words_path: Optional[str] = None
    symspell_dict_path: Optional[str] = None


def _normalize_category(category: object) -> Optional[str]:
    if category is None or (isinstance(category, float) and np.isnan(category)):
        return None
    return str(category).strip().lower()


def category_to_index(category: object) -> Optional[int]:
    normalized = _normalize_category(category)
    if normalized is None:
        return None
    return CATEGORY_TO_INDEX.get(normalized)


def index_to_category(index: int) -> Optional[str]:
    return INDEX_TO_CATEGORY.get(index)


def delete_non_alpha(text: object) -> str:
    text = str(text).lower()
    text = NON_ALPHA_RE.sub(" ", text)
    text = ONE_CHAR_WORD_RE.sub(" ", text)
    return MULTISPACE_RE.sub(" ", text).strip()


def get_stopwords() -> set[str]:
    import nltk

    try:
        stopwords = nltk.corpus.stopwords.words("russian")
    except LookupError:
        nltk.download("stopwords")
        stopwords = nltk.corpus.stopwords.words("russian")

    anti_stopwords = {
        "не", "нет", "ни", "ничего", "без", "никогда",
        "нельзя", "всегда", "конечно", "надо", "хорошо",
        "лучше", "больше", "более",
    }
    return {w for w in stopwords if w not in anti_stopwords}


def _load_word_set(path: str) -> set[str]:
    words = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                words.add(line)
    return words


def _build_symspell(path: str):
    try:
        from symspellpy import SymSpell
    except ImportError as exc:
        raise ImportError("symspellpy is required for syntax_correction=True") from exc

    sym_spell = SymSpell()
    loaded = sym_spell.load_dictionary(path, term_index=0, count_index=1, separator="\t")
    if not loaded:
        raise ValueError(f"Unable to load symspell dictionary from {path}")
    return sym_spell


def _correct_text(words: Iterable[str], full_dict: set[str], sym_spell) -> list[str]:
    from symspellpy.symspellpy import Verbosity

    corrected: list[str] = []
    for word in words:
        if word in full_dict:
            corrected.append(word)
            continue
        suggestions = sym_spell.lookup(word, Verbosity.TOP)
        corrected.append(suggestions[0].term if suggestions else word)
    return corrected


def _lemmatize(words: Iterable[str], morph) -> list[str]:
    return [morph.parse(word)[0].normal_form for word in words]


def preprocess_text_series(series: pd.Series, cfg: TextPreprocessConfig) -> pd.Series:
    tokens = series.astype(str).map(delete_non_alpha).map(str.split)

    if cfg.syntax_correction:
        if not cfg.russian_words_path or not cfg.symspell_dict_path:
            raise ValueError(
                "For syntax_correction=True provide --russian-words-path and --symspell-dict-path"
            )
        full_dict = _load_word_set(cfg.russian_words_path)
        sym_spell = _build_symspell(cfg.symspell_dict_path)
        tokens = tokens.map(lambda words: _correct_text(words, full_dict, sym_spell))

    if cfg.lemmatization:
        try:
            import pymorphy3
        except ImportError as exc:
            raise ImportError("pymorphy3 is required for lemmatization=True") from exc
        morph = pymorphy3.analyzer.MorphAnalyzer()
        tokens = tokens.map(lambda words: _lemmatize(words, morph))

    if cfg.stopwords_removal:
        stopwords = get_stopwords()
        tokens = tokens.map(lambda words: [w for w in words if w not in stopwords])

    text = tokens.map(lambda words: " ".join(words) if words else None)
    return text.replace("nan", np.nan)


# ── Inference helpers ─────────────────────────────────────────────────────────

def _load_model_from_checkpoint(path: str, device: torch.device, expected_stage: str) -> ENPSClassifier:
    payload = torch.load(path, map_location=device)
    cfg = ModelConfig(**payload["model_config"])
    if cfg.stage != expected_stage:
        raise ValueError(f"Checkpoint {path} has stage='{cfg.stage}', expected '{expected_stage}'.")
    model = ENPSClassifier(cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _tokenize(tokenizer, texts: list[str], max_length: int) -> dict[str, np.ndarray]:
    encoded = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


def _make_inference_loader(
    tokenizer,
    texts: list[str],
    scores: np.ndarray,
    max_length: int,
    batch_size: int,
) -> DataLoader:
    enc = _tokenize(tokenizer, texts, max_length=max_length)
    ds = ENPSDataset(encodings=enc, scores=scores, labels=None)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
