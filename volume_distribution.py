import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
import os

# --- PARAMETRI CONFIGURABILI ---
# Puoi modificare questo esponente per testare diverse pendenze sulla coda (es. -3.0 per la legge del cubo)
POWER_LAW_EXPONENT = -2.5

data_dir = 'database/data'
os.makedirs('images', exist_ok=True)
paths = np.array(sorted([f for f in listdir(data_dir) if f.endswith('.csv') or f.endswith('.dat')]))

# Lista per accumulare tutti i volumi normalizzati
all_normalized_volumes = []

for path in paths:
    try:
        # Lettura trade singoli (volume alla colonna indice 2)
        trades = pd.read_csv(f"{data_dir}/{path}", header=None)
        
        # Estrazione e conversione dei volumi
        volumes = pd.to_numeric(trades[2], errors='coerce').dropna().to_numpy()
        volumes = volumes[volumes > 0]
        
        if len(volumes) == 0:
            continue
        
        # Normalizzazione sul volume totale della giornata corrente
        daily_total_volume = np.sum(volumes)
        normalized_volumes = volumes / daily_total_volume
        
        all_normalized_volumes.extend(normalized_volumes)

    except Exception as e:
        print(f"Errore nella lettura del file {path}: {e}")
        continue

all_normalized_volumes = np.array(all_normalized_volumes)

if len(all_normalized_volumes) > 0:
    # --- CALCOLO DELLA CCDF UNICA ---
    x_sorted = np.sort(all_normalized_volumes)
    n = len(x_sorted)
    
    # Calcolo della probabilità cumulata complementare P(V_norm > v)
    y_ccdf = 1.0 - (np.arange(1, n + 1) / n)
    
    # Filtro per la scala logaritmica (evita log(0))
    mask = y_ccdf > 0
    x_sorted = x_sorted[mask]
    y_ccdf = y_ccdf[mask]

    # Sottocampionamento geometrico per rendere la curva fluida
    if len(x_sorted) > 30000:
        idx = np.unique(np.logspace(0, np.log10(len(x_sorted) - 1), 20000).astype(int))
        x_sorted = x_sorted[idx]
        y_ccdf = y_ccdf[idx]

    # Setup Figure (Stessa dimensione di meta_volume_dist.py)
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Colore Deep Navy/Midnight Blue con marker piccoli per altissima risoluzione
    data_color = '#1e3d59' 
    ax.plot(x_sorted, y_ccdf, color=data_color, marker='o', markersize=1.5, 
            linestyle='-', linewidth=1.0, alpha=0.85, label='Empirical Data (Trades)')
    
    # --- AGGIUNTA POWER LAW RIFERIMENTO (Esponente parametrizzato sulla coda) ---
    # Definiamo il range della coda per il plot teorico (dall'ultimo 20% dei dati in poi)
    x_tail_start = x_sorted[int(len(x_sorted) * 0.9)]
    x_tail_end = x_sorted.max()
    x_power_law = np.logspace(np.log10(x_tail_start), np.log10(x_tail_end), 100)
    
    # Scaliamo la retta teorica usando il parametro configurato
    y_tail_start = y_ccdf[int(len(y_ccdf) * 0.9)]
    y_power_law = y_tail_start * (x_power_law / x_tail_start)**(POWER_LAW_EXPONENT)
    
    # Stringa dinamica per la legenda in base all'esponente scelto
    label_power_law = rf'Power Law Coda $\propto v^{{{POWER_LAW_EXPONENT}}}$'
    
    # Plot della Power Law teorica di confronto
    ax.plot(x_power_law, y_power_law, color='#e056fd', linestyle='--', 
            linewidth=2.0, label=label_power_law)
    
    # Axis Formatting (Identico a meta_volume_dist.py)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle=':', alpha=0.5)
    
    # Etichette assi formattate in LaTeX (fontsize=14)
    ax.set_xlabel(r'Normalized Trade Volume $q / V_{giorno}$', fontsize=14)
    ax.set_ylabel(r'CCDF $P(V_{norm} > v)$', fontsize=14)
    
    # Mostra la legenda (Stesso stile dell'altro script)
    ax.legend(title="Datasets & Models", loc='upper right', frameon=True, fontsize=11)
    
    fig.tight_layout()
    
    # Salvataggio dell'immagine coerente
    output_image = os.path.join('images', 'single_trade_normalized_dist_design.png')
    fig.savefig(output_image, dpi=300)
    print(f"Grafico salvato con successo: {output_image}")
else:
    print("Nessun dato trovato.")