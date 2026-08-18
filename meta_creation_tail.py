import json
import numpy as np
import pandas as pd
import os
from os import listdir
import methods

model = ""
N_REALIZATIONS = 1
MIN_CHILD = 1
MAX_PER_CHILD = 1_000_000  # cap per NbChild value; None = no cap

# Which configuration(s) to run, by row index (0-based) in configurations.csv.
# Examples:
#   CONFIG_INDICES = [2]        -> run only row 2
#   CONFIG_INDICES = [0, 3, 7]  -> run rows 0, 3 and 7
#   CONFIG_INDICES = "all"      -> run every row (old behavior)
CONFIG_INDICES = [6]

# Re-run a configuration even if its state file says it's already complete.
FORCE = False


def state_path(full_path):
    return full_path + ".state.json"


def load_state(full_path):
    """Load realization-tracking state. Returns dict with 'completed_realizations' and 'counts'."""
    sp = state_path(full_path)
    if os.path.exists(sp):
        with open(sp, "r") as f:
            return json.load(f)
    return {"completed_realizations": 0, "counts": {}}


def save_state(full_path, state):
    sp = state_path(full_path)
    tmp = sp + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, sp)  # atomic-ish write, avoids corrupt state file on crash


def buckets_full(counts):
    if MAX_PER_CHILD is None:
        return False
    if not counts:
        return False
    return all(v >= MAX_PER_CHILD for v in counts.values())


def run_configuration(configuration, data_dir, meta_dir, paths, force=False):
    nb_traders = configuration['nb_traders']
    kind       = configuration['kind']
    exponent   = configuration['exponent']

    print(f"\n=== nb_traders={nb_traders}, kind={kind}, exponent={exponent} ===")

    if nb_traders == 1:
        filename = f'meta_{nb_traders}.csv'
    elif kind == 'power':
        filename = f'meta_{nb_traders}_{kind}_{exponent}.csv'
    else:
        filename = f'meta_{nb_traders}_{kind}.csv'

    full_path = os.path.join(meta_dir, filename)

    state = load_state(full_path)
    completed = state["completed_realizations"]
    counts = {int(k): v for k, v in state["counts"].items()}
    first = not os.path.exists(full_path)

    if completed >= N_REALIZATIONS and not force:
        print(f"  Already complete: {completed}/{N_REALIZATIONS} realizations done. Skipping "
              f"(use --force to redo).")
        return

    if force:
        completed = 0

    print(f"  Resuming from realization {completed}/{N_REALIZATIONS} — "
          f"counts: { {k: v for k, v in sorted(counts.items())} }")

    for r in range(completed, N_REALIZATIONS):
        print(f"  Realization {r + 1}/{N_REALIZATIONS}")
        for path in paths:
            slim = methods.generate_slim(path, nb_traders, kind, exponent, data_dir)

            if slim.empty:
                continue

            # Filter 1: drop metaorders below minimum length
            slim = slim[slim['NbChild'] >= MIN_CHILD]

            # Filter 2: for each row, save it only if its bucket is not full
            if MAX_PER_CHILD is not None:
                mask = slim['NbChild'].map(lambda nb: counts.get(nb, 0) < MAX_PER_CHILD)
                slim = slim[mask]
                for nb, n in slim['NbChild'].value_counts().items():
                    counts[nb] = counts.get(nb, 0) + n

            if slim.empty:
                continue

            slim.to_csv(full_path, mode='a', index=False, header=first)
            first = False

        # persist progress after every realization, so a crash never loses more than
        # the realization currently in flight
        state = {"completed_realizations": r + 1, "counts": counts}
        save_state(full_path, state)

        if buckets_full(counts):
            print(f"  All buckets reached MAX_PER_CHILD={MAX_PER_CHILD}, "
                  f"stopping early at realization {r + 1}/{N_REALIZATIONS}.")
            break

    print(f"  Final counts: { {k: v for k, v in sorted(counts.items())} }")


def main():
    if CONFIG_INDICES != "all" and not CONFIG_INDICES:
        raise ValueError("CONFIG_INDICES must be 'all' or a non-empty list of row indices")

    data_dir = os.path.join('database', 'data')
    if model != "":
        data_dir = data_dir + "_" + model

    meta_dir = os.path.join('database', 'meta_tail')
    if model != "":
        meta_dir = meta_dir + "_" + model

    if not os.path.exists(meta_dir):
        os.makedirs(meta_dir)

    paths = np.array(listdir(data_dir))

    configurations = pd.read_csv("configurations.csv", header=0)

    if CONFIG_INDICES == "all":
        indices = configurations.index.tolist()
    else:
        indices = CONFIG_INDICES
        bad = [i for i in indices if i not in configurations.index]
        if bad:
            raise ValueError(
                f"config index {bad} out of range (configurations.csv has "
                f"{len(configurations)} rows, valid indices 0..{len(configurations) - 1})"
            )

    for index in indices:
        configuration = configurations.loc[index]
        run_configuration(configuration, data_dir, meta_dir, paths, force=FORCE)


if __name__ == "__main__":
    main()