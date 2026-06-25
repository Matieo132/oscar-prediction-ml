import re
import numpy as np
import pandas as pd


PRE_OSCAR = [
    "BAFTA - Best film", "Golden Globes - Best movie", "PGA - Best Picutre",
    "BAFTA - Best Director", "Golden Globes - Best Director",
    "BAFTA - Best Screenplay", "Golden Globes - Best Screenplay",
    "BAFTA - Best Montage", "BAFTA - Best Cinematography",
]
OSCAR_NIGHT = [
    "Oscars - Best Director", "Oscars - Best Screenplay",
    "Oscars - Best Montage", "Oscars - Best Cinematography",
]
NUMERIC = [
    "Year", "Runtime_min", "IMDb_Rating", "IMDb_Votes",
    "Metascore", "Rotten_Tomatoes", "total_awards",
]
TOP_GENRES = [
    "Drama", "Comedy", "Thriller", "Biography", "History", "Romance",
    "Crime", "Adventure", "Action", "War", "Music", "Mystery", "Sport",
    "Sci-Fi", "Horror", "Fantasy", "Animation", "Western", "Family", "Musical",
]
TARGET = "Oscars - Best Picture"
NUMERIC_IDX = list(range(len(NUMERIC)))   # pierwsze kolumny feature_cols to NUMERIC


def load_oscar_night(data_path="oscary_dane.csv", exclude_year=None,
                     include_genres=False, include_critic=False):
    df = pd.read_csv(data_path, sep=";")

    df["IMDb_Rating"] = df["IMDb_Rating"].apply(
        lambda v: float(str(v).replace(",", ".")) if pd.notna(v) else np.nan
    )
    for col in ["Metascore", "Rotten_Tomatoes", "Runtime_min", "IMDb_Votes", "total_awards"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in PRE_OSCAR + OSCAR_NIGHT:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Usuń Best Picture z total_awards — bez tego data leakage
    df["total_awards"] = (
        pd.to_numeric(df["total_awards"], errors="coerce").fillna(0)
        - pd.to_numeric(df[TARGET], errors="coerce").fillna(0)
    ).clip(lower=0)

    if include_critic:
        df["critic_vs_audience"] = df["Metascore"].fillna(50) / 10 - df["IMDb_Rating"].fillna(7)

    if include_genres:
        for g in TOP_GENRES:
            df[f"Genre_{g}"] = df["Genre"].fillna("").apply(
                lambda x: int(bool(re.search(rf"\b{g}\b", x, re.IGNORECASE)))
            )

    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    feature_cols = (
        NUMERIC
        + [c for c in PRE_OSCAR + OSCAR_NIGHT if c in df.columns]
    )
    if include_critic:
        feature_cols += ["critic_vs_audience"]
    if include_genres:
        feature_cols += [f"Genre_{g}" for g in TOP_GENRES]

    for col in NUMERIC:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if exclude_year is not None:
        train_df = df[df["Year"] != exclude_year].copy()
        pred_df  = df[df["Year"] == exclude_year].copy()
    else:
        train_df = df.copy()
        pred_df  = None

    return train_df, pred_df, feature_cols, TARGET
