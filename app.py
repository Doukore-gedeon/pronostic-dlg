
import math
import os
import re
import pandas as pd
import gradio as gr


# ============================================================
# PRONOSTIC DLG
# Analyse statistique des résultats historiques avec Poisson
# ============================================================

MAX_BUTS = 15


def poisson(k, moyenne):
    """Probabilité de marquer exactement k buts."""
    if moyenne < 0:
        return 0.0
    if moyenne == 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-moyenne) * moyenne**k / math.factorial(k)


def normaliser_colonnes(df):
    """Uniformise les noms de colonnes du fichier CSV."""
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace("é", "e", regex=False)
        .str.replace("è", "e", regex=False)
        .str.replace("ê", "e", regex=False)
        .str.replace(" ", "_", regex=False)
    )

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
    }

    return df.rename(columns=alias)


def charger_csv(fichier):
    """Charge un CSV historique et prépare les données."""
    if fichier is None:
        return None, [], "Importe d'abord un fichier CSV historique."

    try:
        chemin = fichier if isinstance(fichier, str) else fichier.name

        # Lecture du CSV, avec prise en charge de plusieurs encodages.
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
            return None, [], "Le fichier CSV ne contient aucune ligne."

        df = normaliser_colonnes(df)

        # Extraire les buts du score si les colonnes de buts manquent.
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
        manquantes = [col for col in requis if col not in df.columns]

        if manquantes:
            return None, [], (
                f"Colonnes manquantes : {manquantes}. "
                f"Colonnes détectées : {list(df.columns)}"
            )

        df["Domicile"] = df["Domicile"].astype(str).str.strip()
        df["Exterieur"] = df["Exterieur"].astype(str).str.strip()
        df["Buts_dom"] = pd.to_numeric(df["Buts_dom"], errors="coerce")
        df["Buts_ext"] = pd.to_numeric(df["Buts_ext"], errors="coerce")

        # Éliminer les lignes incomplètes ou contenant des buts impossibles.
        df = df.dropna(subset=requis)
        df = df[
            (df["Domicile"] != "")
            & (df["Exterieur"] != "")
            & (df["Domicile"].str.lower() != "nan")
            & (df["Exterieur"].str.lower() != "nan")
            & (df["Buts_dom"] >= 0)
            & (df["Buts_ext"] >= 0)
        ].copy()

        if df.empty:
            return None, [], (
                "Aucun résultat valide. Vérifie les noms des équipes "
                "et les scores dans ton CSV."
            )

        # Les buts doivent être des nombres entiers.
        df = df[
            (df["Buts_dom"] % 1 == 0)
            & (df["Buts_ext"] % 1 == 0)
        ].copy()

        equipes = sorted(set(df["Domicile"]) | set(df["Exterieur"]))

        if df.empty or len(equipes) < 2:
            return None, [], "Il faut au moins deux équipes et un résultat valide."

        return df, equipes, (
            f"✅ {len(df)} matchs historiques chargés. "
            f"{len(equipes)} équipes trouvées."
        )

    except Exception as e:
        return None, [], (
            f"Erreur de lecture : {type(e).__name__}: {e}"
        )


def analyser_match(df, equipe_dom, equipe_ext):
    """Calcule les probabilités à partir des résultats historiques."""
    if df is None or df.empty:
        return "❌ Charge d'abord un CSV historique valide."

    if not equipe_dom or not equipe_ext:
        return "❌ Choisis les deux équipes."

    if equipe_dom == equipe_ext:
        return "❌ Choisis deux équipes différentes."

    matchs_dom = df[df["Domicile"] == equipe_dom]
    matchs_ext = df[df["Exterieur"] == equipe_ext]

    # Moyennes générales utilisées si l'équipe a peu de matchs.
    moyenne_dom_generale = df["Buts_dom"].mean()
    moyenne_ext_generale = df["Buts_ext"].mean()

    # Attaque à domicile et défense de l'adversaire à l'extérieur.
    attaque_dom = (
        matchs_dom["Buts_dom"].mean()
        if not matchs_dom.empty
        else moyenne_dom_generale
    )

    defense_ext = (
        matchs_ext["Buts_dom"].mean()
        if not matchs_ext.empty
        else moyenne_dom_generale
    )

    # Attaque à l'extérieur et défense de l'équipe à domicile.
    attaque_ext = (
        matchs_ext["Buts_ext"].mean()
        if not matchs_ext.empty
        else moyenne_ext_generale
    )

    defense_dom = (
        matchs_dom["Buts_ext"].mean()
        if not matchs_dom.empty
        else moyenne_ext_generale
    )

    # Combinaison simple des statistiques d'attaque et de défense.
    buts_attendus_dom = max(0.05, (attaque_dom + defense_ext) / 2)
    buts_attendus_ext = max(0.05, (attaque_ext + defense_dom) / 2)

    # Distribution des scores exacts.
    probabilites_scores = {}
    for bdom in range(MAX_BUTS + 1):
        for bext in range(MAX_BUTS + 1):
            probabilites_scores[(bdom, bext)] = (
                poisson(bdom, buts_attendus_dom)
                * poisson(bext, buts_attendus_ext)
            )

    total = sum(probabilites_scores.values())
    probabilites_scores = {
        score: proba / total
        for score, proba in probabilites_scores.items()
    }

    p1 = sum(
        p for (a, b), p in probabilites_scores.items() if a > b
    )
    pn = sum(
        p for (a, b), p in probabilites_scores.items() if a == b
    )
    p2 = sum(
        p for (a, b), p in probabilites_scores.items() if a < b
    )

    over_05 = sum(
        p for (a, b), p in probabilites_scores.items() if a + b >= 1
    )
    over_15 = sum(
        p for (a, b), p in probabilites_scores.items() if a + b >= 2
    )
    over_25 = sum(
        p for (a, b), p in probabilites_scores.items() if a + b >= 3
    )
    over_35 = sum(
        p for (a, b), p in probabilites_scores.items() if a + b >= 4
    )

    btts_oui = sum(
        p for (a, b), p in probabilites_scores.items() if a >= 1 and b >= 1
    )

    # Marchés de buts par équipe.
    dom_over_05 = sum(
        p for (a, b), p in probabilites_scores.items() if a >= 1
    )
    dom_over_15 = sum(
        p for (a, b), p in probabilites_scores.items() if a >= 2
    )
    ext_over_05 = sum(
        p for (a, b), p in probabilites_scores.items() if b >= 1
    )
    ext_over_15 = sum(
        p for (a, b), p in probabilites_scores.items() if b >= 2
    )

    top_scores = sorted(
        probabilites_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )[:5]

    def pct(x):
        return f"{x * 100:.2f}%"

    lignes = [
        f"⚽ PRONOSTIC DLG — {equipe_dom} vs {equipe_ext}",
        "",
        "📊 BUTS ATTENDUS",
        f"{equipe_dom} : {buts_attendus_dom:.2f}",
        f"{equipe_ext} : {buts_attendus_ext:.2f}",
        "",
        "🏆 RÉSULTAT FINAL — 1N2",
        f"1 — Victoire {equipe_dom} : {pct(p1)}",
        f"N — Match nul : {pct(pn)}",
        f"2 — Victoire {equipe_ext} : {pct(p2)}",
        "",
        "🔁 DOUBLE CHANCE",
        f"1N : {pct(p1 + pn)}",
        f"12 : {pct(p1 + p2)}",
        f"N2 : {pct(pn + p2)}",
        "",
        "⚽ TOTAL DE BUTS",
        f"Plus de 0,5 but : {pct(over_05)}",
        f"Plus de 1,5 buts : {pct(over_15)}",
        f"Plus de 2,5 buts : {pct(over_25)}",
        f"Plus de 3,5 buts : {pct(over_35)}",
        f"Moins de 2,5 buts : {pct(1 - over_25)}",
        f"Moins de 3,5 buts : {pct(1 - over_35)}",
        "",
        "🥅 BUTS PAR ÉQUIPE",
        f"{equipe_dom} — plus de 0,5 but : {pct(dom_over_05)}",
        f"{equipe_dom} — plus de 1,5 buts : {pct(dom_over_15)}",
        f"{equipe_ext} — plus de 0,5 but : {pct(ext_over_05)}",
        f"{equipe_ext} — plus de 1,5 buts : {pct(ext_over_15)}",
        "",
        "🤝 LES DEUX ÉQUIPES MARQUENT",
        f"Oui : {pct(btts_oui)}",
        f"Non : {pct(1 - btts_oui)}",
        "",
        "🎯 TOP 5 DES SCORES EXACTS",
    ]

    for (a, b), p in top_scores:
        lignes.append(f"{a} - {b} : {pct(p)}")

    lignes.extend([
        "",
        "⚠️ Ces pourcentages sont des estimations statistiques, "
        "pas des garanties de résultat.",
        "La qualité dépend du volume et de la fiabilité des données historiques."
    ])

    return "\n".join(lignes)


def interface_charger(fichier):
    df, equipes, message = charger_csv(fichier)
    if df is None:
        return None, gr.update(choices=[], value=None), gr.update(choices=[], value=None), message

    return (
        df,
        gr.update(choices=equipes, value=None),
        gr.update(choices=equipes, value=None),
        message
    )


def interface_analyser(df, equipe_dom, equipe_ext):
    return analyser_match(df, equipe_dom, equipe_ext)


with gr.Blocks(title="PRONOSTIC DLG") as demo:
    gr.Markdown(
        "# ⚽ PRONOSTIC DLG\n"
        "### Analyse statistique à partir de résultats historiques\n"
        "1. Importe ton CSV historique. 2. Charge les équipes. "
        "3. Lance l'analyse."
    )

    fichier = gr.File(
        label="Fichier CSV historique",
        file_types=[".csv"],
        type="filepath"
    )

    charger = gr.Button("Charger les équipes", variant="primary")
    message = gr.Textbox(label="État du chargement", interactive=False)

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
        lines=30,
        interactive=False
    )

    donnees = gr.State(None)

    charger.click(
        fn=interface_charger,
        inputs=[fichier],
        outputs=[donnees, equipe_dom, equipe_ext, message]
    )

    analyser.click(
        fn=interface_analyser,
        inputs=[donnees, equipe_dom, equipe_ext],
        outputs=[resultat]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )

