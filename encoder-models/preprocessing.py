from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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
        "не",
        "нет",
        "ни",
        "ничего",
        "без",
        "никогда",
        "нельзя",
        "всегда",
        "конечно",
        "надо",
        "хорошо",
        "лучше",
        "больше",
        "более",
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


def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".xlsx"):
        return pd.read_excel(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def load_training_pairs(path: str, preprocess_cfg: TextPreprocessConfig) -> pd.DataFrame:
    df = load_data(path)
    left = df.loc[:, ["Score", "A1", "C1"]].rename(columns={"Score": "score", "A1": "text", "C1": "category"})
    right = df.loc[:, ["Score", "A2", "C2"]].rename(columns={"Score": "score", "A2": "text", "C2": "category"})
    pairs = pd.concat([left, right], axis=0, ignore_index=True).dropna(subset=["text", "category"])
    pairs["label"] = pairs["category"].map(category_to_index)
    pairs = pairs.dropna(subset=["label"]).copy()
    pairs["label"] = pairs["label"].astype(int)
    pairs["category"] = pairs["label"].map(index_to_category)
    pairs["text"] = preprocess_text_series(pairs["text"], preprocess_cfg)
    pairs = pairs.dropna(subset=["text"]).reset_index(drop=True)
    return pairs


def load_inference_pairs(path: str, text_col: str, preprocess_cfg: TextPreprocessConfig) -> pd.DataFrame:
    df = load_data(path)
    pairs = df.loc[:, ["Score", text_col]].copy().rename(columns={"Score": "score", text_col: "text"})
    pairs["text"] = preprocess_text_series(pairs["text"], preprocess_cfg)
    pairs = pairs.dropna(subset=["text"])
    return pairs


def split_binary(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    train_indices_path: Optional[str] = None,
    test_indices_path: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    if train_indices_path and test_indices_path:
        train_idx = np.load(train_indices_path)
        test_idx = np.load(test_indices_path)
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["label"],
        )
        train_idx = train_df.index.to_numpy()
        test_idx = test_df.index.to_numpy()

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["binary_label"] = (train_df["label"] > 1).astype(np.float32)
    test_df["binary_label"] = (test_df["label"] > 1).astype(np.float32)
    return train_df, test_df, train_idx, test_idx


def build_multiclass_from_binary_split(
    full_df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = full_df.iloc[train_idx].copy()
    test_df = full_df.iloc[test_idx].copy()
    train_df = train_df[train_df["label"] > 1].copy()
    test_df = test_df[test_df["label"] > 1].copy()
    train_df["multi_label"] = (train_df["label"] - 2).astype(int)
    test_df["multi_label"] = (test_df["label"] - 2).astype(int)
    return train_df, test_df


def save_split_indices(train_idx: np.ndarray, test_idx: np.ndarray, output_dir: str) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train_indices.npy"
    test_path = out / "test_indices.npy"
    np.save(train_path, train_idx)
    np.save(test_path, test_idx)
    return train_path, test_path

