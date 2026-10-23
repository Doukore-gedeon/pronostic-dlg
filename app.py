import math
import pandas as pd
import gradio as gr

def poisson(k, moyenne):
    return math.exp(-moyenne) * moyenne**k / math.factorial(k)

def charger_csv(fichier):
    if fichier is None:
        return None, [], "Importe d'abord le fichier CSV de résultats."
    try:
        df = pd.read_csv(fichier.name)
        requis = ["Domicile", "Exterieur", "Buts_dom", "Buts_ext"]
        if not all(c in df.columns for c in requis):
            return None, [], (
                "Colonnes requises : Domicile, Exterieur, Buts_dom, Buts_ext. "
                f"Colonnes trouvées : {list(df.columns)}"
            )
        df["Domicile"] = df["Domicile"].astype(str).str.strip()
        df["Exterieur"] = df["Exterieur"].astype(str).str.strip()
        df["Buts_dom"] = pd.to_numeric(df["Buts_dom"], errors="coerce")
        df["Buts_ext"] = pd.to_numeric(df["Buts_ext"], errors="coerce")
        df = df.dropna(subset=requis)
        df = df[(df["Buts_dom"] >= 0) & (df["Buts_ext"] >= 0)]
        teams = sorted(set(df["Domicile"]) | set(df["Exterieur"]))
        if df.empty or len(teams) < 2:
            return None, [], "Le fichier ne contient pas assez de résultats valides."
        return df, teams, f"✅ {len(df)} matchs chargés et {len(teams)} équipes trouvées."
    except Exception as e:
        return None, [], f"Impossible de lire le CSV : {type(e).__name__}: {e}"

def analyser(fichier, domicile, exterieur):
    df, teams, status = charger_csv(fichier)
    if df is None:
        return status
    if not domicile or not exterieur:
        return "Sélectionne les deux équipes."
    if domicile == exterieur:
        return "Choisis deux équipes différentes."
    md = df[df["Domicile"] == domicile]
    me = df[df["Exterieur"] == exterieur]
    if md.empty:
        return f"Aucun match à domicile trouvé pour {domicile}."
    if me.empty:
        return f"Aucun match à l'extérieur trouvé pour {exterieur}."

    home_avg = float(df["Buts_dom"].mean())
    away_avg = float(df["Buts_ext"].mean())
    home_scored = float(md["Buts_dom"].mean())
    home_conceded = float(md["Buts_ext"].mean())
    away_scored = float(me["Buts_ext"].mean())
    away_conceded = float(me["Buts_dom"].mean())
    lam_home = max(0.05, min(6.0, home_scored * away_conceded / home_avg if home_avg else home_scored))
    lam_away = max(0.05, min(6.0, away_scored * home_conceded / away_avg if away_avg else away_scored))

    grid = [(i, j, poisson(i, lam_home)*poisson(j, lam_away))
            for i in range(16) for j in range(16)]
    total = sum(p for _, _, p in grid)
    grid = [(i, j, p/total) for i, j, p in grid]
    p1 = sum(p for i,j,p in grid if i > j)
    pn = sum(p for i,j,p in grid if i == j)
    p2 = sum(p for i,j,p in grid if i < j)
    over05 = sum(p for i,j,p in grid if i+j >= 1)
    over15 = sum(p for i,j,p in grid if i+j >= 2)
    over25 = sum(p for i,j,p in grid if i+j >= 3)
    btts = sum(p for i,j,p in grid if i > 0 and j > 0)
    home05 = sum(p for i,j,p in grid if i >= 1)
    home15 = sum(p for i,j,p in grid if i >= 2)
    away05 = sum(p for i,j,p in grid if j >= 1)
    away15 = sum(p for i,j,p in grid if j >= 2)
    top = sorted(grid, key=lambda x:x[2], reverse=True)[:5]
    return f"""# ⚽ PRONOSTIC DLG
### {domicile} — {exterieur}

*Estimations expérimentales calculées à partir du CSV fourni ({len(df)} matchs).*

**Buts attendus**
- {domicile} : {lam_home:.2f}
- {exterieur} : {lam_away:.2f}

**Résultat**
- Victoire domicile (1) : **{p1:.1%}**
- Match nul (N) : **{pn:.1%}**
- Victoire extérieur (2) : **{p2:.1%}**

**Double chance**
- 1N : **{p1+pn:.1%}**
- 12 : **{p1+p2:.1%}**
- N2 : **{pn+p2:.1%}**

**Total de buts**
- Plus de 0,5 : **{over05:.1%}**
- Plus de 1,5 : **{over15:.1%}**
- Plus de 2,5 : **{over25:.1%}**
- Moins de 2,5 : **{1-over25:.1%}**

**Buts par équipe**
- {domicile}, au moins 1 : **{home05:.1%}**
- {domicile}, au moins 2 : **{home15:.1%}**
- {exterieur}, au moins 1 : **{away05:.1%}**
- {exterieur}, au moins 2 : **{away15:.1%}**

**Les deux équipes marquent**
- Oui : **{btts:.1%}**
- Non : **{1-btts:.1%}**

**Cinq scores les plus probables**
""" + "\n".join(f"- **{i}–{j}** : {p:.1%}" for i,j,p in top) + """

---
*Ces probabilités sont des estimations statistiques, pas des garanties. Les marchés de corners/cartons nécessitent des données dédiées.*
"""

with gr.Blocks(title="PRONOSTIC DLG") as demo:
    gr.Markdown("# ⚽ PRONOSTIC DLG\n### Analyse statistique à partir de résultats historiques")
    gr.Markdown("1. Importe ton fichier CSV historique. 2. Choisis les équipes. 3. Lance l'analyse.")
    fichier = gr.File(label="Fichier CSV historique", file_types=[".csv"], type="filepath")
    status = gr.Markdown("CSV non chargé.")
    with gr.Row():
        domicile = gr.Dropdown(label="Équipe à domicile", choices=[], interactive=True)
        exterieur = gr.Dropdown(label="Équipe à l'extérieur", choices=[], interactive=True)
    bouton_charger = gr.Button("Charger les équipes")
    bouton_analyser = gr.Button("Analyser le match", variant="primary")
    sortie = gr.Markdown()
    def update_teams(path):
        df, teams, msg = charger_csv(type("Upload", (), {"name": path})() if path else None)
        if not teams:
            return msg, gr.update(choices=[], value=None), gr.update(choices=[], value=None)
        return msg, gr.update(choices=teams, value=teams[0]), gr.update(choices=teams, value=teams[1] if len(teams)>1 else teams[0])
    bouton_charger.click(update_teams, inputs=fichier, outputs=[status, domicile, exterieur])
    bouton_analyser.click(analyser, inputs=[fichier, domicile, exterieur], outputs=sortie)

demo.launch()
