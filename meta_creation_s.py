import numpy as np
import os
from os import listdir
import gc

import methods_s

#model = 'lmf_emp_tim_param_1.0'
#model = 'var_1000'
#model = 'lmf_emp_tim_0.3'
model = ''

iterations = 1
n_filter = 2

# --- configurazione metaordini ---
mode = 'mcmc'       # 'hmm' (lunghezze geometriche) | 'hmm_pareto' (lunghezze Pareto)
                          # | 'hmm_lambda' (numero di metaordini attivi variabile)
                          # | 'mcmc' (linking MCMC, lunghezze Pareto) | 'truth' (id veri)
nb_traders = 3            # N (modelli a N fisso); puo' essere una lista, es. [2, 3, 4]: scelta per verosimiglianza
pareto_exp = 1.5          # esponente della Pareto per la lunghezza dei metaordini
n_types = 6               # (hmm_pareto) n. di componenti della miscela di geometriche (5-6 consigliato)
lam = 0.3                 # (hmm_lambda) prob. di arrivo di un nuovo metaordine per passo; anche lista
max_pool = 30             # (hmm_lambda) max metaordini attivi nello stato nascosto
eps = 0.0                 # frazione di trade di rumore; anche lista, es. [0.0, 0.05, 0.1]
block_size = None         # parametri scelti per blocchi di block_size trade (None = per giorno)
drop_prob = 0.0           # probabilita' di scartare un metaordine con MetaImpact < 0 (0 = mai)
mcmc_K = 300              # (mcmc) finestra massima (in trade) tra due figli consecutivi dello stesso metaordine
mcmc_em = 30              # (mcmc) iterazioni di EM stocastico per apprendere lam(eta') e nu[n] (0 = nessun apprendimento)
mcmc_sweeps = 40          # (mcmc) sweep per iterazione EM
mcmc_stat = 10            # (mcmc) sweep usati per le statistiche per iterazione EM
mcmc_final = 200          # (mcmc) sweep finali di campionamento
mcmc_rho = 0.6            # (mcmc) parametro della proposta geometrica sul rango
mcmc_learn_files = 5      # (mcmc) >0: apprende i parametri una volta su questo numero di giorni (equispaziati nel
                          # periodo) e li congela; 0 = comportamento di mcmc_relearn (primo file / ogni file)
mcmc_relearn = False      # (mcmc) True: riapprende i parametri su ogni file (default: appresi sul primo e riusati)
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
    if mode == 'mcmc':
        filename = f'meta_s_mcmc_K{mcmc_K}_{pareto_exp}'
        if mcmc_learn_files > 0:
            filename += f'_L{mcmc_learn_files}'
    elif mode == 'hmm':
        filename = f'meta_s_{tag(nb_traders)}_hmm_{pareto_exp}'
    elif mode == 'hmm_pareto':
        filename = f'meta_s_{tag(nb_traders)}_hmmP{n_types}_{pareto_exp}'
    else:
        filename = f'meta_s_lam{tag(lam)}_hmmL_{pareto_exp}'
    if mode != 'mcmc' and np.any(np.asarray(eps) > 0):
        filename += f'_eps{tag(eps)}'
    if mode != 'mcmc' and block_size:
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

if mode == 'mcmc' and mcmc_learn_files > 0:
    sel = np.unique(np.linspace(0, len(paths) - 1, min(mcmc_learn_files, len(paths))).round().astype(int))
    print('file usati per apprendere i parametri:', list(paths[sel]))
    methods_s.learn_mcmc_params(list(paths[sel]), data_dir, pareto_exp, K=mcmc_K, n_em=mcmc_em,
                                n_sweeps=mcmc_sweeps, n_stat=mcmc_stat, rho=mcmc_rho)
    mcmc_relearn = False

for path in paths:
    print(path)
    for it in range(iterations):
        meta, _ = methods_s.generate(path, nb_traders, pareto_exp, l, data_dir,
                                     mode=mode, drop_prob=drop_prob,
                                     pre_window=pre_window, n_types=n_types,
                                     eps=eps, lam=lam, block_size=block_size,
                                     max_pool=max_pool, mcmc_K=mcmc_K, mcmc_em=mcmc_em,
                                     mcmc_sweeps=mcmc_sweeps, mcmc_stat=mcmc_stat,
                                     mcmc_final=mcmc_final, mcmc_rho=mcmc_rho,
                                     mcmc_relearn=mcmc_relearn)
        meta = meta[meta['NbChild'] >= n_filter]
        l += len(meta)

        meta.to_csv(file_path, mode='a', index=False, header=first)
        first = False

        del meta
        gc.collect()