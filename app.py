
import math
import os
import re
import unicodedata

import pandas as pd
import gradio as gr

MAX_BUTS = 15

CHAMPIONNATS = [
    "Premier League — Angleterre",
    "Championship — Angleterre",
    "LaLiga — Espagne",
    "Bundesliga — Allemagne",
    "Ligue 1 — France",
    "Serie A — Italie",
    "Eredivisie — Pays-Bas",
    "Liga Portugal — Portugal",
]


def sans_accents(texte):
    texte = unicodedata.normalize("NFKD", str(texte))
    return "".join(c for c in texte if not unicodedata.combining(c))


def texte_normalise(texte):
    texte = sans_accents(texte).lower().strip()
    return re.sub(r"[^a-z0-9]+", " ", texte).strip()


def reconnaitre_championnat(valeur):
    """Associe les noms courants aux huit championnats."""
    t = texte_normalise(valeur)

    correspondances = {
        "Premier League — Angleterre": [
            "premier league", "england premier league",
            "angleterre premier league", "epl"
        ],
        "Championship — Angleterre": [
            "championship", "england championship",
            "angleterre championship"
        ],
        "LaLiga — Espagne": [
            "laliga", "la liga", "spain la liga",
            "espagne laliga", "espagne la liga"
        ],
        "Bundesliga — Allemagne": [
            "bundesliga", "germany bundesliga",
            "allemagne bundesliga"
        ],
        "Ligue 1 — France": [
            "ligue 1", "france ligue 1", "ligue1"
        ],
        "Serie A — Italie": [
            "serie a", "italy serie a", "italie serie a"
        ],
        "Eredivisie — Pays-Bas": [
            "eredivisie", "netherlands eredivisie",
            "pays bas eredivisie", "holland eredivisie"
        ],
        "Liga Portugal — Portugal": [
            "liga portugal", "portugal liga portugal",
            "portuguese league", "primeira liga"
        ],
    }

    for championnat, alias in correspondances.items():
        if any(texte_normalise(a) in t for a in alias):
            return championnat

    return None


def normaliser_colonnes(df):
    df.columns = [
        texte_normalise(c).replace(" ", "_")
        for c in df.columns
    ]

    alias = {
        "domicile": "Domicile",
        "equipe_domicile": "Domicile",
        "equipe_a_domicile": "Domicile",
        "home_team": "Domicile",
        "exterieur": "Exterieur",
        "equipe_exterieure": "Exterieur",
        "equipe_a_exterieur": "Exterieur",
        "away_team": "Exterieur",
        "buts_dom": "Buts_dom",
        "buts_domicile": "Buts_dom",
        "home_goals": "Buts_dom",
        "buts_ext": "Buts_ext",
        "buts_exterieur": "Buts_ext",
        "away_goals": "Buts_ext",
        "score_final": "score_final",
        "score_final_indique": "score_final",
        "championnat": "Championnat",
        "league": "Championnat",
        "competition": "Championnat",
    }

    return df.rename(columns=alias)


def charger_csv(fichier):
    if fichier is None:
        return None, "Importe d'abord un fichier CSV."

    try:
        chemin = fichier if isinstance(fichier, str) else fichier.name

        try:
            df = pd.read_csv(
                chemin, sep=None, engine="python",
                encoding="utf-8-sig"
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                chemin, sep=None, engine="python",
                encoding="latin-1"
            )

        if df.empty:
            return None, "Le fichier CSV est vide."

        df = normaliser_colonnes(df)

        if "score_final" in df.columns:
            scores = df["score_final"].astype(str).str.extract(
                r"^\s*(\d{1,2})\s*[-–—:]\s*(\d{1,2})"
            )
            if "Buts_dom" not in df.columns:
                df["Buts_dom"] = pd.to_numeric(
                    scores[0], errors="coerce"
                )
            if "Buts_ext" not in df.columns:
                df["Buts_ext"] = pd.to_numeric(
                    scores[1], errors="coerce"
                )

        requis = ["Domicile", "Exterieur", "Buts_dom", "Buts_ext"]
        manquantes = [c for c in requis if c not in df.columns]

        if manquantes:
            return None, (
                f"Colonnes manquantes : {manquantes}. "
                f"Colonnes détectées : {list(df.columns)}"
            )

        for col in ["Domicile", "Exterieur"]:
            df[col] = df[col].astype(str).str.strip()

        df["Buts_dom"] = pd.to_numeric(df["Buts_dom"], errors="coerce")
        df["Buts_ext"] = pd.to_numeric(df["Buts_ext"], errors="coerce")

        df = df.dropna(subset=requis).copy()
        df = df[
            (df["Domicile"] != "")
            & (df["Exterieur"] != "")
            & (df["Domicile"].str.lower() != "nan")
            & (df["Exterieur"].str.lower() != "nan")
            & (df["Buts_dom"] >= 0)
            & (df["Buts_ext"] >= 0)
            & (df["Buts_dom"] % 1 == 0)
            & (df["Buts_ext"] % 1 == 0)
        ].copy()

        if df.empty:
            return None, "Aucun résultat valide trouvé dans le CSV."

        if "Championnat" in df.columns:
            df["Championnat_original"] = df["Championnat"].astype(str)
            df["Championnat"] = df["Championnat"].apply(
                reconnaitre_championnat
            )
            non_reconnus = df["Championnat"].isna().sum()
            df = df[df["Championnat"].notna()].copy()

            if df.empty:
                return None, (
                    "Les scores sont présents, mais aucun nom de "
                    "championnat n'a été reconnu. Vérifie les valeurs "
                    "de la colonne championnat dans ton CSV."
                )

            message = (
                f"{len(df)} matchs valides chargés. "
                f"{df['Championnat'].nunique()} championnats reconnus."
            )
            if non_reconnus:
                message += (
                    f" {non_reconnus} lignes ignorées car leur "
                    "championnat n'a pas été reconnu."
                )
        else:
            message = (
                f"{len(df)} matchs chargés, mais aucune colonne "
                "'championnat' n'a été trouvée. Le filtrage par ligue "
                "ne sera pas disponible tant que cette colonne "
                "n'existera pas dans le CSV."
            )

        return df, message

    except Exception as e:
        return None, f"Erreur de lecture : {type(e).__name__}: {e}"


def poisson(k, moyenne):
    if moyenne < 0:
        return 0.0
    if moyenne == 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-moyenne) * moyenne**k / math.factorial(k)


def analyser_match(df, equipe_dom, equipe_ext):
    if df is None or df.empty:
        return "Charge un CSV valide et choisis un championnat."
    if not equipe_dom or not equipe_ext:
        return "Choisis les deux équipes."
    if equipe_dom == equipe_ext:
        return "Choisis deux équipes différentes."

    matchs_dom = df[df["Domicile"] == equipe_dom]
    matchs_ext = df[df["Exterieur"] == equipe_ext]

    moyenne_dom = df["Buts_dom"].mean()
    moyenne_ext = df["Buts_ext"].mean()

    attaque_dom = (
        matchs_dom["Buts_dom"].mean()
        if not matchs_dom.empty else moyenne_dom
    )
    defense_ext = (
        matchs_ext["Buts_dom"].mean()
        if not matchs_ext.empty else moyenne_dom
    )
    attaque_ext = (
        matchs_ext["Buts_ext"].mean()
        if not matchs_ext.empty else moyenne_ext
    )
    defense_dom = (
        matchs_dom["Buts_ext"].mean()
        if not matchs_dom.empty else moyenne_ext
    )

    lambda_dom = max(0.05, (attaque_dom + defense_ext) / 2)
    lambda_ext = max(0.05, (attaque_ext + defense_dom) / 2)

    scores = {
        (a, b): poisson(a, lambda_dom) * poisson(b, lambda_ext)
        for a in range(MAX_BUTS + 1)
        for b in range(MAX_BUTS + 1)
    }
    total = sum(scores.values())
    scores = {s: p / total for s, p in scores.items()}

    p1 = sum(p for (a, b), p in scores.items() if a > b)
    pn = sum(p for (a, b), p in scores.items() if a == b)
    p2 = sum(p for (a, b), p in scores.items() if a < b)

    def prob(condition):
        return sum(p for (a, b), p in scores.items() if condition(a, b))

    over05 = prob(lambda a, b: a + b >= 1)
    over15 = prob(lambda a, b: a + b >= 2)
    over25 = prob(lambda a, b: a + b >= 3)
    over35 = prob(lambda a, b: a + b >= 4)
    btts = prob(lambda a, b: a >= 1 and b >= 1)

    lignes = [
        f"PRONOSTIC DLG — {equipe_dom} vs {equipe_ext}",
        "",
        "BUTS ATTENDUS",
        f"{equipe_dom} : {lambda_dom:.2f}",
        f"{equipe_ext} : {lambda_ext:.2f}",
        "",
        "RÉSULTAT 1N2",
        f"1 — Victoire domicile : {p1:.2%}",
        f"N — Match nul : {pn:.2%}",
        f"2 — Victoire extérieur : {p2:.2%}",
        "",
        "DOUBLE CHANCE",
        f"1N : {p1 + pn:.2%}",
        f"12 : {p1 + p2:.2%}",
        f"N2 : {pn + p2:.2%}",
        "",
        "TOTAL DE BUTS",
        f"Plus de 0,5 : {over05:.2%}",
        f"Moins de 0,5 : {1 - over05:.2%}",
        f"Plus de 1,5 : {over15:.2%}",
        f"Moins de 1,5 : {1 - over15:.2%}",
        f"Plus de 2,5 : {over25:.2%}",
        f"Moins de 2,5 : {1 - over25:.2%}",
        f"Plus de 3,5 : {over35:.2%}",
        f"Moins de 3,5 : {1 - over35:.2%}",
        "",
        "BUTS PAR ÉQUIPE",
        f"{equipe_dom} — plus de 0,5 : {prob(lambda a,b: a >= 1):.2%}",
        f"{equipe_dom} — plus de 1,5 : {prob(lambda a,b: a >= 2):.2%}",
        f"{equipe_dom} — plus de 2,5 : {prob(lambda a,b: a >= 3):.2%}",
        f"{equipe_ext} — plus de 0,5 : {prob(lambda a,b: b >= 1):.2%}",
        f"{equipe_ext} — plus de 1,5 : {prob(lambda a,b: b >= 2):.2%}",
        f"{equipe_ext} — plus de 2,5 : {prob(lambda a,b: b >= 3):.2%}",
        "",
        "LES DEUX ÉQUIPES MARQUENT",
        f"Oui : {btts:.2%}",
        f"Non : {1 - btts:.2%}",
        "",
        "TOP 5 SCORES EXACTS",
    ]

    for (a, b), p in sorted(
        scores.items(), key=lambda item: item[1], reverse=True
    )[:5]:
        lignes.append(f"{a} - {b} : {p:.2%}")

    lignes.extend([
        "",
        "Attention : ces probabilités sont des estimations, pas des garanties.",
        "La fiabilité dépend de la qualité et du volume des données historiques."
    ])
    return "\n".join(lignes)


def actualiser_championnats(df):
    if df is None or "Championnat" not in df.columns:
        return gr.update(choices=[], value=None), (
            "Le CSV doit contenir une colonne championnat."
        )
    presents = sorted(df["Championnat"].dropna().unique().tolist())
    return gr.update(choices=presents, value=None), (
        f"Championnats présents dans le fichier : {len(presents)}. "
        "Seuls ceux disposant de données apparaissent."
    )


def actualiser_equipes(df, championnat):
    if df is None or not championnat:
        return None, gr.update(choices=[], value=None), gr.update(choices=[], value=None), (
            "Choisis un championnat avec des données."
        )

    filtre = df[df["Championnat"] == championnat].copy()
    if filtre.empty:
        return filtre, gr.update(choices=[], value=None), gr.update(choices=[], value=None), (
            f"Aucun match historique pour {championnat}."
        )

    equipes = sorted(set(filtre["Domicile"]) | set(filtre["Exterieur"]))
    return (
        filtre,
        gr.update(choices=equipes, value=None),
        gr.update(choices=equipes, value=None),
        f"{championnat} : {len(filtre)} matchs et {len(equipes)} équipes."
    )


def charger_interface(fichier):
    df, message = charger_csv(fichier)
    if df is None:
        return (
            None,
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            message
        )

    choix = (
        sorted(df["Championnat"].dropna().unique().tolist())
        if "Championnat" in df.columns else []
    )
    return (
        df,
        gr.update(choices=choix, value=None),
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        message
    )


with gr.Blocks(title="PRONOSTIC DLG") as demo:
    gr.Markdown(
        "# ⚽ PRONOSTIC DLG\n"
        "### Analyse statistique des matchs historiques avec Poisson\n"
        "Importe un CSV, charge les données, choisis un championnat, "
        "puis sélectionne les deux équipes."
    )

    fichier = gr.File(
        label="Fichier CSV historique",
        file_types=[".csv"],
        type="filepath"
    )
    charger = gr.Button("Charger les données", variant="primary")
    message = gr.Textbox(label="État du chargement", interactive=False)

    donnees_completes = gr.State(None)
    donnees_ligue = gr.State(None)

    championnat = gr.Dropdown(
        label="Championnat",
        choices=[],
        interactive=True
    )

    etat_ligue = gr.Textbox(
        label="Données du championnat",
        interactive=False
    )

    with gr.Row():
        equipe_dom = gr.Dropdown(
            label="Équipe à domicile", choices=[], interactive=True
        )
        equipe_ext = gr.Dropdown(
            label="Équipe à l'extérieur", choices=[], interactive=True
        )

    analyser = gr.Button("Analyser le match", variant="primary")
    resultat = gr.Textbox(
        label="Analyse PRONOSTIC DLG", lines=32, interactive=False
    )

    charger.click(
        fn=charger_interface,
        inputs=[fichier],
        outputs=[
            donnees_completes, championnat,
            equipe_dom, equipe_ext, message
        ]
    )

    championnat.change(
        fn=actualiser_equipes,
        inputs=[donnees_completes, championnat],
        outputs=[donnees_ligue, equipe_dom, equipe_ext, etat_ligue]
    )

    analyser.click(
        fn=analyser_match,
        inputs=[donnees_ligue, equipe_dom, equipe_ext],
        outputs=[resultat]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )


