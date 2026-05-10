import numpy as np
import pandas as pd
import os
from os import listdir
import methods
from lmf import simulate_lmf
from corr import generate_gaussian_signs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_filename(nb_traders: int, kind: str, exponent: float) -> str:
    """Return a deterministic filename stem for the given configuration."""
    if nb_traders == 1:
        return "1"
    elif kind == 'uniform':
        return f"{nb_traders}_{kind}"
    else:
        return f"{nb_traders}_{kind}_{exponent}"


def results_path(results_dir: str, stem: str) -> str:
    return os.path.join(results_dir, f"realizations_{stem}.npz")


def load_existing(filepath: str) -> tuple[int, np.ndarray]:
    """
    Load previously saved realizations from an .npz file.

    Returns
    -------
    saved_iterations : int
        Number of iterations already completed.
    data : np.ndarray
        Array of NbChild values accumulated so far.
    """
    archive = np.load(filepath)
    saved_iterations = int(archive['iterations'])
    data = archive['data']
    print(f"[load]  Found existing file: {filepath}")
    print(f"        Iterations stored : {saved_iterations}")
    print(f"        Realizations stored: {len(data):,}")
    return saved_iterations, data


def save_results(filepath: str, total_iterations: int, data: np.ndarray) -> None:
    """Persist the combined results to disk."""
    np.savez_compressed(filepath, iterations=total_iterations, data=data)
    print(f"[save]  Written to {filepath}")
    print(f"        Total iterations  : {total_iterations}")
    print(f"        Total realizations: {len(data):,}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def process_one_file(trades: pd.DataFrame, signs: np.ndarray,
                     nb_traders: int, kind: str, alpha: float) -> pd.Series:
    """
    Apply one (nb_traders, kind, alpha) configuration to a pre-loaded trade
    file using already-generated signs.  Returns the run-length Series.
    """
    traders = methods.mapping_function(trades, nb_traders, kind, alpha)
    db = pd.DataFrame({'sign': signs, 'trader': traders})
    sorted_trades = db.sort_values(['trader']).reset_index(drop=True)

    sorted_trades['metaid'] = np.where(
        (sorted_trades['trader'] != sorted_trades['trader'].shift()) |
        (sorted_trades['sign'].shift() != sorted_trades['sign']),
        1, 0
    ).cumsum()

    return sorted_trades.groupby('metaid')['trader'].count()


def run_generate_all(delta_map: dict[str, int],
                     configurations: list[dict],
                     signs_origin: str,
                     lmf_alpha: float = None,
                     lmf_nb_traders: int = None) -> dict[str, np.ndarray]:
    """
    Run iterations over all trade files, generating signs ONCE per
    (iteration, file) and reusing them across every configuration.
    Each config is accumulated only for its own required delta, so configs
    that needed fewer iterations are not over-run.

    Parameters
    ----------
    delta_map      : maps each stem -> number of NEW iterations it still needs
    configurations : list of dicts with keys 'nb_traders', 'kind', 'exponent'
    signs_origin   : 'lmf', 'gaussian', or '' (use real signs from the data file)
    lmf_alpha      : alpha parameter for LMF (required when signs_origin='lmf')
    lmf_nb_traders : nb_traders for LMF   (required when signs_origin='lmf')

    Returns
    -------
    dict mapping each config stem -> 1-D numpy array of metaorder run-lengths
    """
    data_dir = 'database/data'
    paths = np.array(listdir(data_dir))
    max_delta = max(delta_map.values())

    # Accumulate results per config as plain lists to avoid the FutureWarning
    # from pd.concat on empty DataFrames, then convert to numpy at the end.
    accumulators: dict[str, list] = {stem: [] for stem in delta_map}
    iter_count:   dict[str, int]  = {stem: 0  for stem in delta_map}

    for i in range(max_delta):
        print(f'\n=== Iteration {i + 1}/{max_delta} ===')

        # Configs that still need this iteration
        active_cfgs = [
            cfg for cfg in configurations
            if iter_count[build_filename(cfg['nb_traders'], cfg['kind'], cfg['exponent'])]
             < delta_map[build_filename(cfg['nb_traders'], cfg['kind'], cfg['exponent'])]
        ]

        for path in paths:
            trades = pd.read_csv(f"{data_dir}\\{path}", header=None)

            # ── Resolve signs for this (iteration, file) pair ────────────────
            if signs_origin == 'lmf':
                signs = simulate_lmf(lmf_alpha, lmf_nb_traders, len(trades))
            elif signs_origin == 'gaussian':
                signs = generate_gaussian_signs(
                    len(trades), p_plus=np.mean(np.array(trades[3])) + 0.5
                )
            elif signs_origin == '':
                # Real signs: column 3 contains raw signs in {0,1} or {-1,+1};
                # normalise to +1 / -1 so the metaid logic is consistent.
                raw = np.array(trades[3])
                signs = np.where(raw > 0, 1, -1)
            else:
                raise ValueError(f"Unknown signs_origin: {signs_origin!r}")

            # ── Apply only active configurations to the same signs ───────────
            for cfg in active_cfgs:
                nb_traders = cfg['nb_traders']
                kind       = cfg['kind']
                exponent   = cfg['exponent']
                stem       = build_filename(nb_traders, kind, exponent)

                run = process_one_file(trades, signs, nb_traders, kind, exponent)
                accumulators[stem].append(run.to_numpy())

            print(f'  [{path}] signs resolved, {len(active_cfgs)} active config(s) processed.')

        # Increment iteration counter for every active config
        for cfg in active_cfgs:
            stem = build_filename(cfg['nb_traders'], cfg['kind'], cfg['exponent'])
            iter_count[stem] += 1

    # Flatten each accumulator list into a single 1-D array
    return {stem: np.concatenate(chunks) for stem, chunks in accumulators.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # ── Signs settings ───────────────────────────────────────────────────────
    iterations     = 1      # TARGET number of iterations to reach
    signs_origin   = 'lmf'  # 'lmf', 'gaussian', or '' (real signs from data)
    
    lmf_alpha      = 1.4
    lmf_nb_traders = 10

    # ── Load configurations from CSV ─────────────────────────────────────────
    cfg_df = pd.read_csv('configurations.csv')
    configurations = cfg_df.to_dict(orient='records')
    # Ensure exponent is float
    for cfg in configurations:
        cfg['exponent'] = float(cfg['exponent'])

    print(f"Loaded {len(configurations)} configuration(s) from configurations.csv")

    # ── Results directory ─────────────────────────────────────────────────────
    if signs_origin == 'lmf':
        results_dir = f'database\\meta_child_dist_{signs_origin}_{lmf_alpha}_{lmf_nb_traders}'
    elif signs_origin == 'gaussian':
        results_dir = f'database\\meta_child_dist_{signs_origin}'
    elif signs_origin == '':
        results_dir = f'database\\meta_child_dist'

    os.makedirs(results_dir, exist_ok=True)

    # ── Determine which configs still need work and how many iterations ───────
    configs_to_run = []
    existing_data_map: dict[str, np.ndarray] = {}
    delta_map: dict[str, int] = {}

    for cfg in configurations:
        stem     = build_filename(cfg['nb_traders'], cfg['kind'], cfg['exponent'])
        filepath = results_path(results_dir, stem)

        if os.path.exists(filepath):
            saved_iterations, existing_data = load_existing(filepath)

            if iterations <= saved_iterations:
                print(f"[skip]  {stem}: target ({iterations}) already reached "
                      f"({saved_iterations} stored). Skipping.\n")
                continue

            delta = iterations - saved_iterations
            print(f"[append] {stem}: {saved_iterations} stored, "
                  f"need {delta} more to reach target {iterations}.\n")
            existing_data_map[stem] = existing_data
            delta_map[stem] = delta
        else:
            print(f"[create] {stem}: no existing file, "
                  f"will run {iterations} iteration(s).\n")
            delta_map[stem] = iterations

        configs_to_run.append(cfg)

    if not configs_to_run:
        print("\nAll configurations already at target. Nothing to do.")
        exit()

    max_delta = max(delta_map.values())
    print(f"\nMax delta across configs: {max_delta} iteration(s). "
          f"Each config will be run for exactly its own required delta.\n")

    # ── Run all configs together, signs generated once per (iter, file) ───────
    new_data_map = run_generate_all(
        delta_map      = delta_map,
        configurations = configs_to_run,
        signs_origin   = signs_origin,
        lmf_alpha      = lmf_alpha,
        lmf_nb_traders = lmf_nb_traders,
    )

    # ── Save results for each config ──────────────────────────────────────────
    for cfg in configs_to_run:
        stem     = build_filename(cfg['nb_traders'], cfg['kind'], cfg['exponent'])
        filepath = results_path(results_dir, stem)

        new_data = new_data_map[stem]

        if stem in existing_data_map:
            combined_data    = np.concatenate([existing_data_map[stem], new_data])
            # saved_iterations + delta = exact target for this config
            total_iterations = iterations
        else:
            combined_data    = new_data
            total_iterations = delta_map[stem]

        save_results(filepath, total_iterations, combined_data)