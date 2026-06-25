import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
from oscar_config import BG, PANEL, GOLD, WHITE, TEAL, RED

os.makedirs("wyniki", exist_ok=True)

TITLE = "WNIOSKI"

bullet_texts = [
    (GOLD,
     "LDA osiąga najwyższy F1 = 0,78 i AUC = 0,97 (5-fold CV)",
     "— modele liniowe (LDA, SGD, Regresja Logistyczna) dominują w rankingu, co potwierdza, że "
     "liniowe zależności między nagrodami branżowymi a Oscarem skutecznie opisują problem klasyfikacji."),
    (TEAL,
     "Testy statystyczne (sparowany t-test Studenta, 5 foldów)",
     "potwierdzają, że LDA istotnie przewyższa 7 modeli (p < 0,05), w tym Extra Tree (p < 0,001, ***) "
     "i KNN (p < 0,01, **). Różnice wobec pozostałych 17 modeli są nieistotne statystycznie — "
     "klasa modeli liniowych osiąga porównywalne wyniki."),
    (GOLD,
     "Najsilniejszy predyktor: Oscar za Reżyserię",
     "— wygranie tej nagrody podczas gali zwiększa szansę na Best Picture o ponad 29 pp (AME). "
     "Filmy, które wygrały BAFTA, Złote Globy i PGA, zdobywają Oscara za Najlepszy Film w 75% przypadków."),
    (GOLD,
     "Więcej cech nie oznacza lepszego modelu.",
     "Największy przyrost F1 (+0,03 do +0,10) dają wyniki Oscarów ogłoszone tej samej nocy. "
     "Dodanie kodowania gatunków filmowych i cechy „krytycy vs. widzowie‟ nie poprawia "
     "wyników — na zbiorze n=469 wprowadzają szum. Optymalny wynik to 20 cech "
     "(numeryczne + Pre-Oscar + Oscar Night)."),
    (RED,
     "Uwaga metodologiczna:",
     "zmienna total_awards w wersji surowej zawierała samą wygraną Best Picture, co stanowiło "
     "data leakage sztucznie zawyżający F1 o 8–13 pp. Po korekcie wyniki odzwierciedlają "
     "rzeczywistą trudność zadania."),
]

fig_w, fig_h = 9.0, 5.8
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.add_patch(mpatches.FancyBboxPatch(
    (0.0, 0.0), 1.0, 1.0, boxstyle="square,pad=0",
    linewidth=1.5, edgecolor="#444", facecolor=PANEL,
    transform=ax.transAxes, zorder=0))

ax.text(0.5, 0.965, TITLE,
        ha="center", va="top", fontsize=15, color=GOLD,
        fontweight="bold", transform=ax.transAxes)
ax.axhline(y=0.915, xmin=0.02, xmax=0.98,
           color=GOLD, linewidth=0.8, alpha=0.6, transform=ax.transAxes)

y_positions = [0.855, 0.695, 0.535, 0.375, 0.185]

for (dot_color, bold_txt, cont_txt), y0 in zip(bullet_texts, y_positions):
    ax.text(0.015, y0, "●", ha="left", va="top",
            fontsize=9, color=dot_color, transform=ax.transAxes)
    ax.text(0.045, y0, bold_txt, ha="left", va="top",
            fontsize=8.0, color=dot_color, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.045, y0 - 0.072, cont_txt, ha="left", va="top",
            fontsize=7.9, color=WHITE, transform=ax.transAxes,
            wrap=True)

fig.tight_layout(pad=0.3)
out = "wyniki/poster_wnioski.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"[wykres] {out}")
