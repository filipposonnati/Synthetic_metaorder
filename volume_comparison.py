"""
Confronto delle distribuzioni di volume (normalizzate rispetto al volume
giornaliero) tra più file CSV. Il file di review rimane fisso, mentre per le
altre cartelle vengono estratti N file casuali e uniti in un'unica 
distribuzione per ciascuna cartella.

ASSUNZIONI SUI FILE CSV
-----------------------
- La TERZA colonna (indice 2) contiene il volume.
- La PRIMA colonna (indice 0) e' il tempo, in uno di questi due formati
  (rilevato automaticamente):
    a) secondi dalla mezzanotte (numero, es. 34200.0 = 09:30:00)
    b) una data/timestamp leggibile (es. "2023-01-03 09:30:00")
- Se non viene trovata nessuna colonna data/tempo valida, l'intero file
  viene trattato come un unico "giorno" (normalizzazione sul totale del file).
- Se i CSV hanno un header, viene rilevato automaticamente.

COSA PRODUCE LO SCRIPT
-----------------------
  1. Istogramma "a gradini" (step, non riempito) in scala log-log
  2. Boxplot colorato -> confronto rapido di mediana, quartili, code
  3. ECDF in scala log sull'asse x
  4. Profilo intraday (volume medio normalizzato per fascia oraria)
     -> SOLO per i dataset marcati come "includi_intraday": True.

Tutti i grafici vengono salvati in: images/volume_comparison/
"""
from __future__ import annotations
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


CARTELLA_OUTPUT = os.path.join("images", "volume_comparison")

# Palette colori coerente per tutti i grafici (stessa serie = stesso colore ovunque)
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]


def carica_e_normalizza(path: str, col_volume_idx: int = 2, col_data_idx: int = 0) -> pd.DataFrame:
    """
    Legge un CSV, individua la colonna volume (terza colonna) ed
    eventualmente una colonna data/ora, e restituisce un DataFrame con:
      - volume_norm: volume normalizzato rispetto al totale del proprio giorno
      - ora_ore: (opzionale) orario espresso in ore decimali (0-24), utile
        per costruire il profilo intraday
    """
    try:
        df = pd.read_csv(path)
        if df.shape[1] <= col_volume_idx or not pd.api.types.is_numeric_dtype(
            pd.to_numeric(df.iloc[:, col_volume_idx], errors="coerce")
        ):
            df = pd.read_csv(path, header=None)
    except Exception:
        df = pd.read_csv(path, header=None)

    if df.shape[1] <= col_volume_idx:
        raise ValueError(f"Il file {path} non ha una terza colonna di volume.")

    volumi = pd.to_numeric(df.iloc[:, col_volume_idx], errors="coerce")

    giorno = None
    ora_ore = None

    if df.shape[1] > col_data_idx:
        colonna_tempo = df.iloc[:, col_data_idx]

        # --- Caso a) secondi dalla mezzanotte (colonna puramente numerica) ---
        tempo_numerico = pd.to_numeric(colonna_tempo, errors="coerce")
        frazione_numerica = tempo_numerico.notna().mean()

        if frazione_numerica > 0.9 and (tempo_numerico.dropna() >= 0).all():
            if tempo_numerico.max() > 86400:
                giorno = (tempo_numerico // 86400).astype("Int64")
                secondi_nel_giorno = tempo_numerico % 86400
            else:
                giorno = None  # unico giorno per l'intero file
                secondi_nel_giorno = tempo_numerico
            ora_ore = secondi_nel_giorno / 3600.0

        else:
            # --- Caso b) data/timestamp leggibile da pandas ---
            possibile_data = pd.to_datetime(colonna_tempo, errors="coerce")
            if possibile_data.notna().mean() > 0.8:
                giorno = possibile_data.dt.date
                if possibile_data.dt.time.nunique() > 1:
                    t = possibile_data.dt.time
                    ora_ore = t.map(lambda x: x.hour + x.minute / 60 + x.second / 3600)

    tmp = pd.DataFrame({"volume": volumi})
    tmp["giorno"] = giorno if giorno is not None else "unico_giorno"
    if ora_ore is not None:
        tmp["ora_ore"] = ora_ore

    tmp = tmp.dropna(subset=["volume"])
    tmp["volume_norm"] = tmp.groupby("giorno")["volume"].transform(lambda x: x / x.sum())

    return tmp.reset_index(drop=True)


def _salva(fig, nome_file: str):
    os.makedirs(CARTELLA_OUTPUT, exist_ok=True)
    path_completo = os.path.join(CARTELLA_OUTPUT, nome_file)
    fig.savefig(path_completo, dpi=150, bbox_inches="tight")
    print(f"Grafico salvato in: {path_completo}")


def confronta_distribuzioni(dati_dict: dict[str, pd.DataFrame], intraday_flags: dict[str, bool] | None = None):
    """
    Confronta piu' distribuzioni normalizzate.
    """
    nomi = list(dati_dict.keys())
    serie_dict = {nome: df["volume_norm"] for nome, df in dati_dict.items()}
    colori = {nome: PALETTE[i % len(PALETTE)] for i, nome in enumerate(nomi)}

    if intraday_flags is None:
        intraday_flags = {nome: True for nome in nomi}

    # --- Statistiche descrittive ---
    print("=== Statistiche descrittive (volumi normalizzati) ===")
    for nome, s in serie_dict.items():
        print(f"\n{nome}:")
        print(s.describe())

    # --- Test KS a coppie ---
    print("\n=== Test Kolmogorov-Smirnov a coppie ===")
    for i in range(len(nomi)):
        for j in range(i + 1, len(nomi)):
            a, b = nomi[i], nomi[j]
            stat, pvalue = stats.ks_2samp(serie_dict[a], serie_dict[b])
            print(f"{a} vs {b}: statistic={stat:.4f}, p-value={pvalue:.4g}"
                  f"  -> {'distribuzioni diverse' if pvalue < 0.05 else 'nessuna differenza significativa'}")

    # --- 1. Istogramma a gradini in scala log-log ---
    fig, ax = plt.subplots(figsize=(10, 6))
    tutti_valori = np.concatenate([s[s > 0].values for s in serie_dict.values()])
    bins = np.logspace(np.log10(tutti_valori.min()), np.log10(tutti_valori.max()), 60)
    for nome, s in serie_dict.items():
        s_pos = s[s > 0]
        ax.hist(s_pos, bins=bins, density=True, histtype="step",
                linewidth=2.2, label=nome, color=colori[nome])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Normalized Volume")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    _salva(fig, "histograms.png")
    plt.close(fig)

    # --- 2. Boxplot comparativo ---
    fig, ax = plt.subplots(figsize=(1.8 * len(nomi) + 2, 6))
    bp = ax.boxplot(
        [serie_dict[nome].values for nome in nomi],
        labels=nomi,
        patch_artist=True,
        showfliers=True,
        widths=0.55,
        flierprops=dict(marker="o", markersize=3, alpha=0.25, markerfacecolor="gray", markeredgecolor="none"),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="#555555", linewidth=1.3),
        capprops=dict(color="#555555", linewidth=1.3),
        boxprops=dict(linewidth=1.3),
    )
    for patch, nome in zip(bp["boxes"], nomi):
        patch.set_facecolor(colori[nome])
        patch.set_alpha(0.55)
        patch.set_edgecolor(colori[nome])

    ax.set_yscale("log")
    ax.set_ylabel("Normalized Volume")
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    _salva(fig, "boxplot.png")
    plt.close(fig)

    # --- 3. ECDF in scala log sull'asse x ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for nome, s in serie_dict.items():
        s_sorted = np.sort(s[s > 0])
        y = np.arange(1, len(s_sorted) + 1) / len(s_sorted)
        ax.plot(s_sorted, y, label=nome, color=colori[nome], linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Normalized Volume")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("ECDF")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    _salva(fig, "ecdf.png")
    plt.close(fig)

    # --- 4. Profilo intraday ---
    nomi_da_includere = [
        nome for nome in nomi
        if intraday_flags.get(nome, True) and "ora_ore" in dati_dict[nome].columns
    ]
    esclusi = [nome for nome in nomi if nome not in nomi_da_includere]

    if nomi_da_includere:
        fig, ax = plt.subplots(figsize=(12, 6))
        for nome in nomi_da_includere:
            df = dati_dict[nome]
            minuto = (df["ora_ore"] * 60).round() / 60.0
            profilo = df.groupby(minuto)["volume_norm"].mean().sort_index()
            ax.plot(profilo.index, profilo.values, label=nome, color=colori[nome], linewidth=2)
        ax.set_xlabel("Time")
        ax.set_ylabel("Normalized Volume")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        _salva(fig, "intraday_profile.png")
        plt.close(fig)
        if esclusi:
            print(f"\n(Esclusi dal profilo intraday: {', '.join(esclusi)})")
    else:
        print("\n(Nessun dataset abilitato/con orario valido: salto il grafico del profilo intraday)")


def estrai_file_casuali(cartella_path: str, n: int) -> list[str]:
    """
    Sceglie casualmente n file CSV all'interno di una cartella specifica.
    Se nella cartella ci sono meno file di n, li prende tutti.
    """
    if not os.path.exists(cartella_path):
        print(f"Errore: La cartella '{cartella_path}' non esiste.")
        return []
    
    files = [f for f in os.listdir(cartella_path) if f.endswith(".csv") and os.path.isfile(os.path.join(cartella_path, f))]
    
    if not files:
        print(f"Attenzione: Nessun file CSV trovato nella cartella '{cartella_path}'.")
        return []
        
    n_da_estrarre = min(n, len(files))
    file_scelti = random.sample(files, n_da_estrarre)
    
    return [os.path.join(cartella_path, f) for f in file_scelti]


if __name__ == "__main__":
    # =========================================================================
    # CONFIGURAZIONE GENERALE: QUI MODIFICHI IL VALORE DI N
    # =========================================================================
    N_FILES = 3  # Quanti file estrarre a caso e UNIRE da ciascuna cartella (Real, AR, MEM)

    dati = {}
    intraday_flags = {}
    
    # 1. Caricamento del file fisso di REVIEW (singolo e specifico)
    path_review_fisso = "database/review/review_20260708_153517.csv"
    if os.path.exists(path_review_fisso):
        try:
            dati["GAN review"] = carica_e_normalizza(path_review_fisso)
            intraday_flags["GAN review"] = True
            print(f"- File specifico caricato per [GAN review]: {path_review_fisso}")
        except Exception as e:
            print(f"Errore nel caricamento del file di review: {e}")
    else:
        print(f"Attenzione: File di review '{path_review_fisso}' non trovato.")

    # 2. Struttura delle cartelle esplicite da cui estrarre N file e unirli
    STRUTTURA_CARTELLE = {
        "Real": {
            "cartella": "database/data",
            "includi_intraday": True,
        },
        "AR": {
            "cartella": "database/data_ar_1000",
            "includi_intraday": False,
        },
        "MEM": {
            "cartella": r"database\data_lmf_tim_sqrt_mem_1.5_50", # Preservato raw string originale
            "includi_intraday": False,
        },
    }

    print(f"\n=== Estrazione casuale di {N_FILES} file per cartella (Uniti insieme) ===")
    for etichetta, info in STRUTTURA_CARTELLE.items():
        files_estratti = estrai_file_casuali(info["cartella"], N_FILES)
        
        dfs_da_unire = []
        for file_estratto in files_estratti:
            print(f"- Per [{etichetta}] estratto casualmente: {file_estratto}")
            try:
                df_singolo = carica_e_normalizza(file_estratto)
                dfs_da_unire.append(df_singolo)
            except Exception as e:
                print(f"Errore nel caricamento di {file_estratto}: {e}")
        
        # Se abbiamo caricato con successo almeno un file, li uniamo in un unico blocco dati
        if dfs_da_unire:
            dati[etichetta] = pd.concat(dfs_da_unire, ignore_index=True)
            intraday_flags[etichetta] = info["includi_intraday"]

    print("====================================================================\n")

    if dati:
        confronta_distribuzioni(dati, intraday_flags)
    else:
        print("Impossibile procedere: configurazione o dati assenti.")