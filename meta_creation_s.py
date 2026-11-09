import numpy as np
import os
from os import listdir
import gc

import methods_s

model = 'lmf_emp_tim_param_0.3'
#model = 'var_1000'
#model = 'lmf_emp_tim_0.3'
#model = ''

iterations = 1
n_filter = 2

# --- configurazione metaordini ---
mode = 'hmm'       # 'hmm' (lunghezze geometriche) | 'hmm_pareto' (lunghezze Pareto)
                          # | 'hmm_lambda' (numero di metaordini attivi variabile) | 'truth' (id veri)
nb_traders = 4            # N (modelli a N fisso); puo' essere una lista, es. [2, 3, 4]: scelta per verosimiglianza
pareto_exp = 1.5          # esponente della Pareto per la lunghezza dei metaordini
n_types = 6               # (hmm_pareto) n. di componenti della miscela di geometriche (5-6 consigliato)
lam = 0.2                 # (hmm_lambda) prob. di arrivo di un nuovo metaordine per passo; anche lista
max_pool = 30             # (hmm_lambda) max metaordini attivi nello stato nascosto
eps = 0.0                 # frazione di trade di rumore; anche lista, es. [0.0, 0.05, 0.1]
block_size = None         # parametri scelti per blocchi di block_size trade (None = per giorno)
drop_prob = 0.0           # probabilita' di scartare un metaordine con MetaImpact < 0 (0 = mai)
pre_window = 1000         # n. di trade prima del primo figlio per la colonna PreImbalance

data_dir = 'database\\data'
if model != "":
    data_dir = data_dir + "_" + model

meta_dir = 'database\\meta'
if model != "":
    meta_dir = meta_dir + "_" + model

if not os.path.exists(meta_dir):
    os.makedirs(meta_dir)

paths = np.array(sorted(listdir(data_dir)))


def tag(x):
    """Valore o lista -> stringa per il nome del file (es. [2, 3] -> '2-3')."""
    xs = x if isinstance(x, (list, tuple, np.ndarray)) else [x]
    return '-'.join(f'{v:g}' if isinstance(v, (float, np.floating)) else str(v) for v in xs)


# --- nome del file: riflette tutti i parametri che cambiano il risultato ---
if mode == 'truth':
    filename = 'meta_s_truth'
else:
    if mode == 'hmm':
        filename = f'meta_s_{tag(nb_traders)}_hmm_{pareto_exp}'
    elif mode == 'hmm_pareto':
        filename = f'meta_s_{tag(nb_traders)}_hmmP{n_types}_{pareto_exp}'
    else:
        filename = f'meta_s_lam{tag(lam)}_hmmL_{pareto_exp}'
    if np.any(np.asarray(eps) > 0):
        filename += f'_eps{tag(eps)}'
    if block_size:
        filename += f'_blk{block_size}'

if drop_prob > 0:
    filename = filename + f'_drop{drop_prob}'

if n_filter > 2:
    filename = filename + f'_{n_filter}'

filename = filename + '.csv'

file_path = os.path.join(meta_dir, filename)

print(file_path, filename)

if os.path.exists(file_path):
    print(f"Skipping configuration ({mode}, N={nb_traders}, pareto={pareto_exp}) — {filename} already exists.")
    exit()

l = 0
first = True

for path in paths:
    print(path)
    for it in range(iterations):
        meta, _ = methods_s.generate(path, nb_traders, pareto_exp, l, data_dir,
                                     mode=mode, drop_prob=drop_prob,
                                     pre_window=pre_window, n_types=n_types,
                                     eps=eps, lam=lam, block_size=block_size,
                                     max_pool=max_pool)
        meta = meta[meta['NbChild'] >= n_filter]
        l += len(meta)

        meta.to_csv(file_path, mode='a', index=False, header=first)
        first = False

        del meta
        gc.collect()