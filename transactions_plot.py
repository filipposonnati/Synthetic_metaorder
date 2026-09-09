import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import numpy

# 1. Configurazione dello stile personalizzato
plt.rcParams.update({
    'font.size':       12,
    'axes.titlesize':  20,
    'axes.labelsize':  16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
})

# 2. Caricamento dati e selezione dell'intervallo generico
df = pd.read_csv('database\\data\\data_2023-01-03_type4_aggregato_auction.csv', header=None, names=['timestamp', 'midprice', 'volume', 'sign'])

start_idx = 50
end_idx = 100
df = df.iloc[start_idx:end_idx].copy()

df['timestamp'] = df['timestamp'] - numpy.min(df['timestamp'])

# 3. Calcolo dell'intervallo temporale tra ogni ordine e il successivo (dt)
df['t_next'] = df['timestamp'].shift(-1)
# Per l'ultima barra assegniamo la durata media degli intervalli per completare il range
df.loc[df.index[-1], 't_next'] = df['timestamp'].iloc[-1] + df['timestamp'].diff().mean()

df['dt'] = df['t_next'] - df['timestamp']
df['t_mid'] = df['timestamp'] + df['dt'] / 2  # Posizione centrale della barra per ax.bar

# Mappatura dei colori in base al segno (+1.0 Verde per Buy, -1.0 Rosso per Sell)
colors = ['#2ca02c' if s > 0 else '#d62728' for s in df['sign']]

# 4. Creazione dei sotto-grafici sovrapposti
fig, (ax_price, ax_vol) = plt.subplots(
    2, 1, 
    figsize=(12, 8), 
    sharex=True, 
    gridspec_kw={'height_ratios': [2, 1]}
)

# Pannello Superiore: Linea del Midprice
ax_price.plot(df['timestamp'], df['midprice'], color='black', marker='o', markersize=4, linewidth=1.5, label='Midprice')
ax_price.set_ylabel('Midprice')
ax_price.grid(True, linestyle=':', alpha=0.6)
ax_price.legend(loc='upper left')

# Pannello Inferiore: Barre dei Volumi adiacenti senza spazi (width = dt)
ax_vol.bar(
    x=df['t_mid'],           # Posizionamento al centro dell'intervallo temporale
    height=df['volume'], 
    width=df['dt'],          # Larghezza esatta fino all'ordine successivo
    color=colors, 
    edgecolor='black', 
    linewidth=0.5, 
    alpha=0.85
)
ax_vol.set_xlabel('Timestamp (s)')
ax_vol.set_ylabel('Volume')
ax_vol.grid(True, linestyle=':', alpha=0.6)

# Allineamento dei margini dell'asse X
ax_vol.set_xlim(df['timestamp'].min(), df['t_next'].max())

# Legenda personalizzata
legend_elements = [
    Patch(facecolor='#2ca02c', edgecolor='black', label='Buy (+1)'),
    Patch(facecolor='#d62728', edgecolor='black', label='Sell (-1)')
]
ax_vol.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()

# Salvataggio dell'immagine
plt.savefig('images\\transactions_time.png')
#plt.show()

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

# 1. Configurazione dello stile personalizzato
plt.rcParams.update({
    'font.size':       12,
    'axes.titlesize':  20,
    'axes.labelsize':  16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
})

# 2. Caricamento dati e selezione del range generico
df = pd.read_csv('database\\data\\data_2023-01-03_type4_aggregato_auction.csv', header=None, names=['timestamp', 'midprice', 'volume', 'sign'])

start_idx = 0
end_idx = 100
df = df.iloc[start_idx:end_idx].copy()

# 3. Calcolo dell'asse del volume accumulato (inizio e fine di ogni ordine)
df['cum_vol_end'] = df['volume'].cumsum()
df['cum_vol_start'] = df['cum_vol_end'] - df['volume']

# 4. Creazione del grafico
fig, ax = plt.subplots(figsize=(12, 6))

# Linea guida tratteggiata per unificare la traiettoria del prezzo
ax.step(df['cum_vol_start'], df['midprice'], where='post', color='#888888', linestyle='--', alpha=0.5)

# Disegno di un segmento orizzontale colorato per ciascun ordine
for idx, row in df.iterrows():
    color = '#2ca02c' if row['sign'] > 0 else '#d62728'
    
    # Segmento orizzontale con larghezza pari al volume dell'ordine
    ax.hlines(
        y=row['midprice'],
        xmin=row['cum_vol_start'],
        xmax=row['cum_vol_end'],
        colors=color,
        linewidth=4.5,
        zorder=3
    )
    
    # Linea verticale discreta per collegare il cambio di prezzo tra un ordine e il successivo
    if idx < len(df) - 1:
        next_price = df.iloc[df.index.get_loc(idx) + 1]['midprice']
        if row['midprice'] != next_price:
            ax.vlines(
                x=row['cum_vol_end'],
                ymin=min(row['midprice'], next_price),
                ymax=max(row['midprice'], next_price),
                colors='#a0a0a0',
                linestyle=':',
                linewidth=1,
                zorder=2
            )

# Formattazione assi e titolo
#ax.set_title('Esecuzione Ordini in Tempo di Volume')
ax.set_xlabel('Transaction Time')
ax.set_ylabel('Midprice')
ax.grid(True, linestyle=':', alpha=0.6)

# Legenda
legend_elements = [
    Line2D([0], [0], color='#888888', linestyle='--', label='Midprice'),
    Patch(facecolor='#2ca02c', label='Buy (+1)'),
    Patch(facecolor='#d62728', label='Sell (-1)')
]
ax.legend(handles=legend_elements, loc='best', frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()

# Salvataggio dell'immagine nel path richiesto
plt.savefig('images\\transactions.png')
#plt.show()