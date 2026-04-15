import numpy as np
import pandas as pd
import os
from os import listdir
import methods

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
data_dir    = 'database\\data_signs'
results_dir = 'database\\meta_child_dist_signs'  # folder where realization files are stored
iterations  = 10                            # TARGET number of iterations to reach
nb_traders  = 20
kind        = 'power'
exponent    = 2.0

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

def run_generate(data_dir: str, n_iter: int, nb_traders: int,
                 kind: str, exponent: float,
                 offset: int = 0) -> np.ndarray:
    """
    Run `n_iter` rounds of simulation and return the concatenated NbChild array.

    Parameters
    ----------
    offset : int
        Starting index for meta IDs (avoids collisions when appending).
    """
    paths    = np.array(listdir(data_dir))
    meta_tot = pd.DataFrame()

    for i in range(n_iter):
        print(f'  Iteration: {i + 1}/{n_iter}')
        for path in paths:
            l = len(meta_tot) + offset
            meta, _ = methods.generate(path, nb_traders, kind, exponent, l, data_dir)
            meta_tot = pd.concat([meta_tot, meta['NbChild']])

    return meta_tot['NbChild'].to_numpy()


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main():
    os.makedirs(results_dir, exist_ok=True)

    stem     = build_filename(nb_traders, kind, exponent)
    filepath = results_path(results_dir, stem)

    if os.path.exists(filepath):
        # ── File exists: check whether the target is already reached ────────
        saved_iterations, existing_data = load_existing(filepath)

        if iterations <= saved_iterations:
            print(f"\n[skip]  Target ({iterations}) already reached "
                  f"({saved_iterations} iterations stored). Nothing to do.\n")
            return saved_iterations, existing_data

        # ── Run only the missing delta to reach the target ──────────────────
        delta = iterations - saved_iterations
        print(f"\n[mode]  APPEND — {saved_iterations} stored, "
              f"running {delta} more to reach target {iterations}.\n")
        new_data = run_generate(
            data_dir, delta, nb_traders, kind, exponent,
            offset=len(existing_data)
        )

        combined_data    = np.concatenate([existing_data, new_data])
        total_iterations = iterations

    else:
        # ── File does not exist: create from scratch ─────────────────────────
        print(f"\n[mode]  CREATE — no existing file found, "
              f"running {iterations} iteration(s).\n")
        combined_data    = run_generate(
            data_dir, iterations, nb_traders, kind, exponent
        )
        total_iterations = iterations

    save_results(filepath, total_iterations, combined_data)
    return total_iterations, combined_data


if __name__ == '__main__':
    total_iterations, data = main()