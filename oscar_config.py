BG     = "#0A0A0A"
PANEL  = "#141414"
GOLD   = "#F5C518"
BLUE   = "#1E6DB5"
WHITE  = "#F0F0F0"
GRID   = "#2A2A2A"
RED    = "#E63946"
TEAL   = "#2EC4B6"
DIM    = "#444"
PALETTE = [GOLD, BLUE, TEAL, RED, "#A8DADC"]

LABELS = {
    "BAFTA - Best film":               "BAFTA: Najlepszy Film",
    "Golden Globes - Best movie":      "Złote Globy: Najlepszy Film",
    "PGA - Best Picutre":              "PGA: Najlepszy Film",
    "BAFTA - Best Director":           "BAFTA: Najlepszy Reżyser",
    "Golden Globes - Best Director":   "Złote Globy: Najlepszy Reżyser",
    "BAFTA - Best Screenplay":         "BAFTA: Najlepszy Scenariusz",
    "Golden Globes - Best Screenplay": "Złote Globy: Najlepszy Scenariusz",
    "BAFTA - Best Montage":            "BAFTA: Najlepszy Montaż",
    "BAFTA - Best Cinematography":     "BAFTA: Najlepsze Zdjęcia",
    "Oscars - Best Director":          "Oscar: Najlepszy Reżyser",
    "Oscars - Best Screenplay":        "Oscar: Najlepszy Scenariusz",
    "Oscars - Best Montage":           "Oscar: Najlepszy Montaż",
    "Oscars - Best Cinematography":    "Oscar: Najlepsze Zdjęcia",
    "Year":                            "Rok produkcji",
    "Runtime_min":                     "Czas trwania (min)",
    "IMDb_Rating":                     "Ocena IMDb",
    "IMDb_Votes":                      "Liczba głosów IMDb",
    "Metascore":                       "Metascore (krytycy)",
    "Rotten_Tomatoes":                 "Rotten Tomatoes (%)",
    "total_awards":                    "Łączna liczba nagród",
    "critic_vs_audience":              "Krytycy vs. widzowie",
}


def lbl(f):
    if f in LABELS:
        return LABELS[f]
    if f.startswith("Genre_"):
        return f"Gatunek: {f[6:]}"
    return f
