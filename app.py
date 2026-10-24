
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
    return "".join(
        c for c in texte if not unicodedata.combining(c)
    )


def texte_normalise(texte):
    texte = sans_accents(texte).lower().strip()
    return re.sub(r"[^a-z0-9]+", " ", texte).strip()


def reconnaitre_championnat(valeur):
    t = texte_normalise(valeur)

    correspondances = {
        "Premier League — Angleterre": [
            "premier league", "epl"
        ],
        "Championship — Angleterre": [
            "championship"
        ],
        "LaLiga — Espagne": [
            "laliga", "la liga"
        ],
        "Bundesliga — Allemagne": [
            "bundesliga"
        ],
        "Ligue 1 — France": [
            "ligue 1", "ligue1"
        ],
        "Serie A — Italie": [
            "serie a"
        ],
        "Eredivisie — Pays-Bas": [
            "eredivisie"
        ],
        "Liga Portugal — Portugal": [
            "liga portugal", "primeira liga"
        ],
    }

    # Vérifier les noms spécifiques avant les noms génériques.
    if "championship" in t:
        return "Championship — Angleterre"

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
        "date": "Date",
    }

    return df.rename(columns=alias)


def charger_csv(fichier):
    if fichier is None:
        return None, "Importe d'abord ton fichier CSV."

    try:
        chemin = (
            fichier if isinstance(fichier, str)
            else fichier.name
        )

        try:
            df = pd.read_csv(
                chemin,
                sep=None,
                engine="python",
                encoding="utf-8-sig"
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                chemin,
                sep=None,
                engine="python",
                encoding="latin-1"
            )

        if df.empty:
            return None, "Le fichier CSV est vide."

        df = normaliser_colonnes(df)

        # Vérifier les colonnes nécessaires pour afficher les matchs.
        requises = ["Domicile", "Exterieur"]
        manquantes = [
            c for c in requises if c not in df.columns
        ]

        if manquantes:
            return None, (
                f"Colonnes manquantes : {manquantes}. "
                f"Colonnes trouvées : {list(df.columns)}"
            )

        # Les scores sont facultatifs pour charger les rencontres.
        if "Buts_dom" not in df.columns:
            df["Buts_dom"] = float("nan")

        if "Buts_ext" not in df.columns:
            df["Buts_ext"] = float("nan")

        # Extraire les buts depuis un score du type 2-1.
        if "score_final" in df.columns:
            scores = df["score_final"].astype(str).str.extract(
                r"^\s*(\d{1,2})\s*[-–—:]\s*(\d{1,2})"
            )

            buts_dom = pd.to_numeric(
                scores[0], errors="coerce"
            )
            buts_ext = pd.to_numeric(
                scores[1], errors="coerce"
            )

            df["Buts_dom"] = pd.to_numeric(
                df["Buts_dom"], errors="coerce"
            ).fillna(buts_dom)

            df["Buts_ext"] = pd.to_numeric(
                df["Buts_ext"], errors="coerce"
            ).fillna(buts_ext)

        for col in ["Domicile", "Exterieur"]:
            df[col] = df[col].fillna("").astype(str).str.strip()

        df = df[
            (df["Domicile"] != "")
            & (df["Exterieur"] != "")
            & (df["Domicile"].str.lower() != "nan")
            & (df["Exterieur"].str.lower() != "nan")
        ].copy()

        if df.empty:
            return None, "Aucune rencontre avec deux équipes valides."

        # Reconnaître les championnats sans supprimer les matchs
        # dont le score n'est pas encore renseigné.
        if "Championnat" in df.columns:
            df["Championnat_original"] = (
                df["Championnat"].astype(str)
            )
            df["Championnat"] = df["Championnat"].apply(
                reconnaitre_championnat
            )
            non_reconnus = int(df["Championnat"].isna().sum())
            df = df[df["Championnat"].notna()].copy()

            if df.empty:
                return None, (
                    "Aucun championnat reconnu. Vérifie la "
                    "colonne Championnat de ton CSV."
                )
        else:
            return None, (
                "La colonne Championnat est absente du CSV."
            )

        # Les colonnes de buts restent numériques.
        df["Buts_dom"] = pd.to_numeric(
            df["Buts_dom"], errors="coerce"
        )
        df["Buts_ext"] = pd.to_numeric(
            df["Buts_ext"], errors="coerce"
        )

        # Seules les lignes avec deux scores entiers valides
        # seront utilisables pour le calcul statistique.
        scores_valides = (
            df["Buts_dom"].notna()
            & df["Buts_ext"].notna()
            & (df["Buts_dom"] >= 0)
            & (df["Buts_ext"] >= 0)
            & (df["Buts_dom"] % 1 == 0)
            & (df["Buts_ext"] % 1 == 0)
        )

        nb_matchs = len(df)
        nb_resultats = int(scores_valides.sum())
        nb_championnats = df["Championnat"].nunique()

        message = (
            f"Chargement réussi : {nb_matchs} rencontres, "
            f"{nb_championnats} championnats reconnus, "
            f"{nb_resultats} résultats avec scores réels."
        )

        if nb_resultats == 0:
            message += (
                "\nLes rencontres sont chargées, mais aucun "
                "pronostic Poisson ne peut encore être calculé : "
                "il faut ajouter des résultats historiques."
            )

        if non_reconnus:
            message += (
                f"\n{non_reconnus} lignes ignorées car leur "
                "championnat n'a pas été reconnu."
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
        return "Charge d'abord un CSV valide."

    if not equipe_dom or not equipe_ext:
        return "Choisis les deux équipes."

    if equipe_dom == equipe_ext:
        return "Choisis deux équipes différentes."

    # Garder uniquement les matchs dont les scores sont valides.
    historiques = df[
        df["Buts_dom"].notna()
        & df["Buts_ext"].notna()
        & (df["Buts_dom"] >= 0)
        & (df["Buts_ext"] >= 0)
        & (df["Buts_dom"] % 1 == 0)
        & (df["Buts_ext"] % 1 == 0)
    ].copy()

    if historiques.empty:
        return (
            "ANALYSE IMPOSSIBLE POUR LE MOMENT\n\n"
            "Ton CSV contient les rencontres, mais aucun score "
            "historique exploitable.\n\n"
            "Ajoute des matchs déjà terminés avec les colonnes "
            "Buts_dom et Buts_ext, puis recharge le CSV.\n\n"
            "PRONOSTIC DLG ne va pas inventer des probabilités."
        )

    # Utiliser seulement les résultats du championnat sélectionné.
    if "Championnat" in df.columns and "Championnat" in historiques.columns:
        ligues_selectionnees = df["Championnat"].dropna().unique()

        if len(ligues_selectionnees) == 1:
            historique_ligue = historiques[
                historiques["Championnat"] == ligues_selectionnees[0]
            ]
        else:
            historique_ligue = historiques
    else:
        historique_ligue = historiques

    if len(historique_ligue) < 5:
        return (
            "Données historiques insuffisantes dans ce championnat.\n\n"
            f"Nombre de résultats disponibles : "
            f"{len(historique_ligue)}.\n"
            "Ajoute davantage de matchs terminés avant de calculer "
            "des probabilités."
        )

    matchs_dom = historique_ligue[
        historique_ligue["Domicile"] == equipe_dom
    ]
    matchs_ext = historique_ligue[
        historique_ligue["Exterieur"] == equipe_ext
    ]

    moyenne_dom = historique_ligue["Buts_dom"].mean()
    moyenne_ext = historique_ligue["Buts_ext"].mean()

    attaque_dom = (
        matchs_dom["Buts_dom"].mean()
        if len(matchs_dom) >= 2 else moyenne_dom
    )
    defense_ext = (
        matchs_ext["Buts_dom"].mean()
        if len(matchs_ext) >= 2 else moyenne_dom
    )
    attaque_ext = (
        matchs_ext["Buts_ext"].mean()
        if len(matchs_ext) >= 2 else moyenne_ext
    )
    defense_dom = (
        matchs_dom["Buts_ext"].mean()
        if len(matchs_dom) >= 2 else moyenne_ext
    )

    lambda_dom = max(0.05, (attaque_dom + defense_ext) / 2)
    lambda_ext = max(0.05, (attaque_ext + defense_dom) / 2)

    scores = {
        (a, b): poisson(a, lambda_dom) * poisson(b, lambda_ext)
        for a in range(MAX_BUTS + 1)
        for b in range(MAX_BUTS + 1)
    }

    total = sum(scores.values())
    if total <= 0:
        return "Impossible de calculer les probabilités avec ces données."

    scores = {s: p / total for s, p in scores.items()}

    def prob(condition):
        return sum(
            p for (a, b), p in scores.items()
            if condition(a, b)
        )

    p1 = prob(lambda a, b: a > b)
    pn = prob(lambda a, b: a == b)
    p2 = prob(lambda a, b: a < b)

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
        f"Résultats historiques utilisés : {len(historique_ligue)}",
        "Attention : ces probabilités sont des estimations, "
        "pas des garanties."
    ])

    return "\n".join(lignes)


def actualiser_equipes(df, championnat):
    if df is None or not championnat:
        return (
            None,
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            "Choisis un championnat."
        )

    filtre = df[df["Championnat"] == championnat].copy()

    if filtre.empty:
        return (
            filtre,
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            f"Aucune rencontre trouvée pour {championnat}."
        )

    equipes = sorted(
        set(filtre["Domicile"]) | set(filtre["Exterieur"])
    )

    historiques = filtre[
        filtre["Buts_dom"].notna()
        & filtre["Buts_ext"].notna()
    ]

    return (
        filtre,
        gr.update(choices=equipes, value=None),
        gr.update(choices=equipes, value=None),
        f"{championnat} : {len(filtre)} rencontres chargées ; "
        f"{len(historiques)} avec scores historiques."
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

    choix = sorted(df["Championnat"].dropna().unique().tolist())

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
        "### Analyse statistique des matchs avec le modèle de Poisson\n"
        "Importe ton CSV, charge les rencontres, choisis un championnat "
        "et sélectionne deux équipes.\n\n"
        "**Important :** pour obtenir des probabilités, il faut aussi "
        "des résultats historiques avec les scores réels."
    )

    fichier = gr.File(
        label="Fichier CSV des rencontres et/ou résultats historiques",
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
            label="Équipe à domicile",
            choices=[],
            interactive=True
        )
        equipe_ext = gr.Dropdown(
            label="Équipe à l'extérieur",
            choices=[],
            interactive=True
        )

    analyser = gr.Button("Analyser le match", variant="primary")

    resultat = gr.Textbox(
        label="Analyse PRONOSTIC DLG",
        lines=32,
        interactive=False
    )

    charger.click(
        fn=charger_interface,
        inputs=[fichier],
        outputs=[
            donnees_completes,
            championnat,
            equipe_dom,
            equipe_ext,
            message
        ]
    )

    championnat.change(
        fn=actualiser_equipes,
        inputs=[donnees_completes, championnat],
        outputs=[
            donnees_ligue,
            equipe_dom,
            equipe_ext,
            etat_ligue
        ]
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
