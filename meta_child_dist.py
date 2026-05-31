import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import powerlaw
import os
from pathlib import Path
from scipy.stats import ks_2samp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR    = Path('database')           # root that contains all meta_child_dist* folders
REAL_SUBDIR = 'meta_child_dist'         # the real-signs baseline (signs_origin == '')
results_dir = str(BASE_DIR / REAL_SUBDIR)

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10
})

# ---------------------------------------------------------------------------
# Helpers (shared naming logic)
# ---------------------------------------------------------------------------

def build_filename(nb_traders: int, kind: str, exponent: float) -> str:
    if nb_traders == 1:
        return "1"
    elif kind == 'uniform':
        return f"{nb_traders}_{kind}"
    else:
        return f"{nb_traders}_{kind}_{exponent}"


def results_path(results_dir: str, stem: str) -> str:
    return os.path.join(results_dir, f"realizations_{stem}.npz")

def load_data(results_dir: str, nb_traders: int, kind: str,
              exponent: float) -> tuple[int, np.ndarray]:
    """Load iterations count and NbChild array from the .npz file."""
    stem     = build_filename(nb_traders, kind, exponent)
    filepath = results_path(results_dir, stem)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"No realization file found at '{filepath}'.\n"
            f"Run meta_child_dist_save.py first to generate the data."
        )

    archive    = np.load(filepath)
    iterations = int(archive['iterations'])
    # ADD .ravel() HERE TO FLATTEN IT
    data       = archive['data'].ravel() 
    
    print(f"[load]  {filepath}")
    print(f"        Iterations   : {iterations}")
    print(f"        Realizations : {len(data):,}")
    return iterations, data

def load_all(directory: str) -> dict[str, tuple[int, np.ndarray]]:
    """Scan directory for all .npz files and load each one."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' not found.")

    files = [f for f in os.listdir(directory) if f.endswith('.npz')]
    if not files:
        raise FileNotFoundError(f"No .npz files found in '{directory}'.")

    results = {}
    for fname in sorted(files):
        stem     = fname[len('realizations_'):-len('.npz')]
        filepath = os.path.join(directory, fname)
        archive  = np.load(filepath)
        iterations = int(archive['iterations'])
        # ADD .ravel() HERE TO FLATTEN IT
        data       = archive['data'].ravel()
        
        print(f"[load]  {filepath}  |  iter={iterations}  |  n={len(data):,}")
        results[stem] = (iterations, data)

    return results

# ---------------------------------------------------------------------------
# CCDF helper
# ---------------------------------------------------------------------------

def compute_ccdf(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, P(X >= x)) for integer-valued data."""
    bins              = np.arange(1, max(data) + 2)
    counts, bin_edges = np.histogram(data, bins=bins, density=True)
    x_values          = bin_edges[:-1]
    ccdf              = np.cumsum(counts[::-1])[::-1]
    return x_values, ccdf


# ---------------------------------------------------------------------------
# KS test helper
# ---------------------------------------------------------------------------

def run_ks_test(data_b: np.ndarray, data_s: np.ndarray) -> dict:
    """
    Run a two-sample KS test between data_b (real) and data_s (simulated).
    """
    # FORCE BOTH ARRAYS TO BE 1D TO PREVENT BROADCASTING MEMORY ERRORS
    data_b = data_b.ravel()
    data_s = data_s.ravel()

    stat, p = ks_2samp(data_b, data_s)

    # Locate the x where |ECDF_real - ECDF_sim| is maximised
    sorted_b = np.sort(data_b)
    sorted_s = np.sort(data_s)
    all_vals = np.sort(np.concatenate([data_b, data_s]))

    ecdf_b_vals = np.searchsorted(sorted_b, all_vals, side='right') / len(data_b)
    ecdf_s_vals = np.searchsorted(sorted_s, all_vals, side='right') / len(data_s)

    d_vals  = np.abs(ecdf_b_vals - ecdf_s_vals)
    max_idx = np.argmax(d_vals)

    return {
        'stat'  : stat,
        'p'     : p,
        'x_D'   : all_vals[max_idx],
        'ecdf_b': ecdf_b_vals[max_idx],
        'ecdf_s': ecdf_s_vals[max_idx],
    }

# ---------------------------------------------------------------------------
# Fit & plot (single configuration)
# ---------------------------------------------------------------------------

def dist_fit(data: np.ndarray, iterations: int, filename: str) -> None:
    print(len(data) / iterations)

    # --- Fit ---
    fit     = powerlaw.Fit(data, discrete=True)
    tpl     = fit.truncated_power_law
    alpha   = tpl.alpha
    Lambda  = tpl.Lambda
    x_min   = tpl.xmin

    pl       = fit.power_law
    alpha_pl = pl.alpha
    x_min_pl = pl.xmin

    ln       = fit.lognormal
    mu_ln    = ln.mu
    sigma_ln = ln.sigma

    R_tpl_vs_pl, p_tpl_vs_pl = fit.distribution_compare('truncated_power_law', 'power_law')
    R_tpl_vs_ln, p_tpl_vs_ln = fit.distribution_compare('truncated_power_law', 'lognormal')

    print("=" * 55)
    print("  RISULTATI DEL FIT")
    print("=" * 55)
    print(f"\n{'--- Truncated Power Law':}")
    print(f"  alpha  (esponente)   : {alpha:.4f}")
    print(f"  Lambda (decay rate)  : {Lambda:.6f}")
    print(f"  x_min                : {x_min}")
    print(f"\n{'--- Power Law pura':}")
    print(f"  alpha                : {alpha_pl:.4f}")
    print(f"  x_min                : {x_min_pl}")
    print(f"\n{'--- Lognormal':}")
    print(f"  mu                   : {mu_ln:.4f}")
    print(f"  sigma                : {sigma_ln:.4f}")
    print(f"\n{'--- Likelihood Ratio Tests (vs Truncated PL)':}")
    print(f"  TPL vs Power Law     : R = {R_tpl_vs_pl:+.3f},  p = {p_tpl_vs_pl:.4f}")
    print(f"  TPL vs Lognormal     : R = {R_tpl_vs_ln:+.3f},  p = {p_tpl_vs_ln:.4f}")
    print("=" * 55)

    # --- Plot ---
    fig = plt.figure(figsize=(13, 5.5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    def _style_ax(ax, title):
        ax.tick_params(labelsize=9)
        ax.set_title(title, fontsize=11, pad=10)
        ax.grid(True, linewidth=0.5, linestyle='--', alpha=0.6)

    ax1 = fig.add_subplot(gs[0])
    _style_ax(ax1, 'PDF')
    fit.plot_pdf(linewidth=0, marker='o', markersize=4, label='Dati empirici', ax=ax1)
    fit.truncated_power_law.plot_pdf(linewidth=2.2,
        label=f'Trunc. PL  α={alpha:.2f}, Λ={Lambda:.4f}', ax=ax1)
    fit.power_law.plot_pdf(linewidth=1.6, linestyle='--',
        label=f'Power Law  α={alpha_pl:.2f}', ax=ax1)
    fit.lognormal.plot_pdf(linewidth=1.6, linestyle=':',
        label=f'Lognormal  μ={mu_ln:.2f} σ={sigma_ln:.2f}', ax=ax1)
    ax1.axvline(x_min, color='grey', linewidth=1, linestyle=':', alpha=0.5,
                label=f'x_min = {x_min}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('P(x)')
    ax1.legend(fontsize=7.5, framealpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    _style_ax(ax2, 'CCDF')
    fit.plot_ccdf(linewidth=0, marker='o', markersize=4, label='Dati empirici', ax=ax2)
    fit.truncated_power_law.plot_ccdf(linewidth=2.2, label='Trunc. PL', ax=ax2)
    fit.power_law.plot_ccdf(linewidth=1.6, linestyle='--', label='Power Law', ax=ax2)
    fit.lognormal.plot_ccdf(linewidth=1.6, linestyle=':', label='Lognormal', ax=ax2)
    ax2.axvline(x_min, color='grey', linewidth=1, linestyle=':', alpha=0.5,
                label=f'x_min = {x_min}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('P(X ≥ x)')
    ax2.legend(fontsize=7.5, framealpha=0.3)

    os.makedirs('images', exist_ok=True)
    fig.savefig(f'images\\dist_meta_child_{filename}.png', dpi=300)
    plt.show()


# ---------------------------------------------------------------------------
# Plot all configurations together (single directory)
# ---------------------------------------------------------------------------

def plot_all(results_dir: str) -> None:
    """
    Load every .npz in results_dir and overlay their empirical PDFs and CCDFs
    on a single figure for visual comparison.
    """
    all_data = load_all(results_dir)
    colors   = plt.cm.tab10.colors
    markers  = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

    fig, (ax_pdf, ax_ccdf) = plt.subplots(1, 2, figsize=(12, 6))

    for ax, title in [(ax_pdf, 'PDF'),
                      (ax_ccdf, 'CCDF')]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('x')
        ax.grid(True, linewidth=0.5, linestyle='--', alpha=0.6)
        ax.set_title(title)

    ax_pdf.set_ylabel('P(x)')
    ax_ccdf.set_ylabel('P(X ≥ x)')

    for i, (stem, (iterations, data)) in enumerate(all_data.items()):
        color  = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        bins              = np.arange(1, max(data) + 2)
        counts, bin_edges = np.histogram(data, bins=bins, density=True)
        x_values          = bin_edges[:-1]
        x_ccdf, ccdf      = compute_ccdf(data)

        kw = dict(linestyle='', marker=marker, markersize=3, color=color, label=stem)
        ax_pdf.plot(x_values, counts, **kw)
        ax_ccdf.plot(x_ccdf, ccdf, **kw)

    ax_pdf.legend(framealpha=0.4)
    ax_ccdf.legend(framealpha=0.4)
    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    fig.savefig('images\\meta_child_dist\\dist_meta_child_all.png', dpi=300)
    plt.close()

# ---------------------------------------------------------------------------
# Compare real signs vs all simulated variants (Grid without KS test)
# ---------------------------------------------------------------------------

def compare_all_distributions_grid(base_dir: Path, real_subdir: str,
                                   nb_traders: int = None, kind: str = None,
                                   exponent: float = None) -> None:
    """
    Scan base_dir for every subdirectory whose name starts with 'meta_child_dist'.
    Plots all configurations together in a dynamically scaled multi-panel grid 
    (rows = configurations, cols = PDF | CCDF), omitting any KS tests.
    """
    # ── Resolve target stem when a specific configuration is requested ────────
    target_stem = build_filename(nb_traders, kind, exponent) if nb_traders is not None else None

    # ── Discover directories ─────────────────────────────────────────────────
    all_dirs = sorted(
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith('meta_child_dist')
    )

    real_dir  = base_dir / real_subdir
    sim_dirs  = [d for d in all_dirs if d != real_dir]

    if not real_dir.exists():
        raise FileNotFoundError(
            f"Real-signs directory '{real_dir}' not found."
        )
    if not sim_dirs:
        print("[compare] No simulated variant directories found. Nothing to compare.")
        return

    print(f"\n[compare] Baseline : {real_dir}")
    print(f"[compare] Variants  : {[d.name for d in sim_dirs]}\n")

    # ── Load data ────────────────────────────────────────────────────────────
    def _load_stem(directory: str) -> dict[str, np.ndarray]:
        if target_stem is not None:
            filepath = results_path(directory, target_stem)
            if not os.path.exists(filepath):
                return {}
            archive = np.load(filepath)
            return {target_stem: archive['data']}
        return {stem: data for stem, (_, data) in load_all(directory).items()}

    real_data: dict[str, np.ndarray] = _load_stem(str(real_dir))

    sim_datasets: dict[str, dict[str, np.ndarray]] = {}
    for sim_dir in sim_dirs:
        try:
            sim_datasets[sim_dir.name] = _load_stem(str(sim_dir))
        except Exception as e:
            print(f"[warn] {e}")

    # ── Union of stems present in the real data ───────────────────────────────
    stems = sorted(real_data.keys())
    if target_stem is not None:
        if target_stem not in stems:
            raise ValueError(f"Stem '{target_stem}' not found in real-signs directory.")
        stems = [target_stem]
    if not stems:
        print("[compare] No stems found in real-signs directory.")
        return

    # ── Colour palette setup ──────────────────────────────────────────────────
    variant_names  = list(sim_datasets.keys())
    variant_colors = plt.cm.tab10.colors
    variant_style  = {
        name: dict(linestyle='', marker='o', markersize=3.5,
                   color=variant_colors[i % len(variant_colors)], alpha=0.85,
                   label=name)
        for i, name in enumerate(variant_names)
    }
    real_style = dict(linestyle='', marker='o', markersize=3.5,
                      color='black', alpha=0.9, label=real_subdir)

    # ── Build figure with variable scaling to handle multiple rows ────────────
    n = len(stems)
    fig = plt.figure(figsize=(13, 4.5 * n))
    gs = gridspec.GridSpec(n, 2, figure=fig, hspace=0.35, wspace=0.25)

    def _pdf_ccdf(data: np.ndarray):
        bins              = np.arange(1, max(data) + 2)
        counts, bin_edges = np.histogram(data, bins=bins, density=True)
        x_pdf             = bin_edges[:-1]
        x_ccdf, ccdf      = compute_ccdf(data)
        return x_pdf, counts, x_ccdf, ccdf

    for row, stem in enumerate(stems):
        ax_pdf  = fig.add_subplot(gs[row, 0])
        ax_ccdf = fig.add_subplot(gs[row, 1])

        # Configure subplots, labels, and corrected CCDF LaTeX syntax
        for ax, ylabel, col_title in [
            (ax_pdf,  f'{stem}\nP(x)', f'PDF'),
            (ax_ccdf, r'$P(X \geq x)$',  f'CCDF'),
        ]:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('x')
            ax.set_ylabel(ylabel)
            ax.grid(True, linewidth=0.4, linestyle='--', alpha=0.6)
            if row == 0:
                ax.set_title(col_title, fontsize=12, pad=8)

        # ── Plot real distribution ────────────────────────────────────────
        data_real = real_data[stem]
        x_pdf_r, cnt_r, x_ccdf_r, ccdf_r = _pdf_ccdf(data_real)
        ax_pdf.plot(x_pdf_r,   cnt_r,  **real_style)
        ax_ccdf.plot(x_ccdf_r, ccdf_r, **real_style)

        # ── Plot each simulated variant ───────────────────────────────────
        for variant_name, sim_data_dict in sim_datasets.items():
            if stem not in sim_data_dict:
                continue

            data_sim = sim_data_dict[stem]
            style    = variant_style[variant_name]

            x_pdf_s, cnt_s, x_ccdf_s, ccdf_s = _pdf_ccdf(data_sim)
            ax_pdf.plot(x_pdf_s,   cnt_s,  **style)
            ax_ccdf.plot(x_ccdf_s, ccdf_s, **style)

        # Show legend ONLY in the first configuration row
        if row == 0:
            ax_pdf.legend(fontsize=8, framealpha=0.4, loc='lower left')
            ax_ccdf.legend(fontsize=8, framealpha=0.4, loc='lower left')

    # Resolve folder generation to prevent FileNotFoundError bugs on saving
    out_fname = f'meta_child_dist/compare_real_vs_simulated_{target_stem}.png' if target_stem else 'meta_child_dist/compare_real_vs_simulated_all.png'
    out_path = os.path.join('images', out_fname)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[save]  {out_path}")
    plt.show()

# ---------------------------------------------------------------------------
# Compare real signs vs all simulated variants  (with KS test)
# ---------------------------------------------------------------------------

def compare_all_distributions(base_dir: Path, real_subdir: str,
                              nb_traders: int = None, kind: str = None,
                              exponent: float = None) -> None:
    """
    Scan base_dir for every subdirectory whose name starts with
    'meta_child_dist'.  The one named exactly real_subdir is treated as the
    real-signs baseline; every other one is a simulated variant.

    For each (variant, stem) pair where the stem also exists in the baseline,
    a two-sample KS test is run and results are printed to the console.

    A multi-panel figure is produced:
      rows  = configurations (stems)
      cols  = 2  (PDF | CCDF)
    Each row overlays the real distribution (blue) against every simulated
    variant (one colour per variant).  The CCDF panel annotates the KS
    statistic D and p-value for each variant.

    Parameters
    ----------
    base_dir    : Path to the folder that contains all meta_child_dist* subdirs
    real_subdir : name of the subdirectory holding real-sign results
    nb_traders  : if given, restrict the comparison to this single configuration
    kind        : required when nb_traders is given ('power' or 'uniform')
    exponent    : required when nb_traders is given (ignored for 'uniform')
    """
    # ── Resolve target stem when a specific configuration is requested ────────
    target_stem = build_filename(nb_traders, kind, exponent) if nb_traders is not None else None

    # ── Discover directories ─────────────────────────────────────────────────
    all_dirs = sorted(
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith('meta_child_dist')
    )

    real_dir  = base_dir / real_subdir
    sim_dirs  = [d for d in all_dirs if d != real_dir]

    if not real_dir.exists():
        raise FileNotFoundError(
            f"Real-signs directory '{real_dir}' not found. "
            f"Run meta_child_dist_save.py with signs_origin='' first."
        )
    if not sim_dirs:
        print("[compare] No simulated variant directories found. Nothing to compare.")
        return

    print(f"\n[compare] Baseline : {real_dir}")
    print(f"[compare] Variants  : {[d.name for d in sim_dirs]}\n")

    # ── Load data ────────────────────────────────────────────────────────────
    def _load_stem(directory: str) -> dict[str, np.ndarray]:
        """Load only target_stem if specified, otherwise every stem."""
        if target_stem is not None:
            filepath = results_path(directory, target_stem)
            if not os.path.exists(filepath):
                return {}
            archive = np.load(filepath)
            print(f"[load]  {filepath}  |  "
                  f"iter={int(archive['iterations'])}  |  n={len(archive['data']):,}")
            return {target_stem: archive['data']}
        return {stem: data for stem, (_, data) in load_all(directory).items()}

    real_data: dict[str, np.ndarray] = _load_stem(str(real_dir))

    # sim_datasets[variant_name][stem] = data
    sim_datasets: dict[str, dict[str, np.ndarray]] = {}
    for sim_dir in sim_dirs:
        try:
            sim_datasets[sim_dir.name] = _load_stem(str(sim_dir))
        except Exception as e:
            print(f"[warn] {e}")

    # ── Union of stems present in the real data ───────────────────────────────
    stems = sorted(real_data.keys())
    if target_stem is not None:
        if target_stem not in stems:
            raise ValueError(
                f"Stem '{target_stem}' not found in real-signs directory. "
                f"Available stems: {stems}"
            )
        stems = [target_stem]
    if not stems:
        print("[compare] No stems found in real-signs directory.")
        return

    # ── Colour palette: one colour per simulated variant ─────────────────────
    variant_names  = list(sim_datasets.keys())
    variant_colors = plt.cm.tab10.colors
    variant_style  = {
        name: dict(linestyle='', marker='o', markersize=3.5,
                   color=variant_colors[i % len(variant_colors)], alpha=0.85,
                   label=name)
        for i, name in enumerate(variant_names)
    }
    real_style = dict(linestyle='', marker='o', markersize=3.5,
                      color='black', alpha=0.9, label=real_subdir)

    # ── Build figure ─────────────────────────────────────────────────────────
    n     = len(stems)
    fig   = plt.figure()
    gs    = gridspec.GridSpec(
        n, 2,
        figure=fig
    )

    def _pdf_ccdf(data: np.ndarray):
        bins              = np.arange(1, max(data) + 2)
        counts, bin_edges = np.histogram(data, bins=bins, density=True)
        x_pdf             = bin_edges[:-1]
        x_ccdf, ccdf      = compute_ccdf(data)
        return x_pdf, counts, x_ccdf, ccdf

    print("\n" + "=" * 65)
    print(f"  KS TEST  —  real ({real_subdir})  vs  simulated variants")
    print("=" * 65)

    for row, stem in enumerate(stems):
        ax_pdf  = fig.add_subplot(gs[row, 0])
        ax_ccdf = fig.add_subplot(gs[row, 1])

        for ax, ylabel, col_title in [
            (ax_pdf,  'P(x)',      'PDF'),
            (ax_ccdf, 'P(X ≥ x)', 'CCDF'),
        ]:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('x')
            ax.set_ylabel(ylabel)
            ax.grid(True, linewidth=0.4, linestyle='--', alpha=0.6)
            if row == 0:
                ax.set_title(col_title)

        ax_pdf.set_ylabel('P(x)')

        # ── Plot real distribution ────────────────────────────────────────
        data_real = real_data[stem]
        x_pdf_r, cnt_r, x_ccdf_r, ccdf_r = _pdf_ccdf(data_real)
        ax_pdf.plot(x_pdf_r,   cnt_r,  **real_style)
        ax_ccdf.plot(x_ccdf_r, ccdf_r, **real_style)

        # ── Plot each simulated variant & run KS test ─────────────────────
        for variant_name, sim_data_dict in sim_datasets.items():
            if stem not in sim_data_dict:
                continue

            data_sim = sim_data_dict[stem]
            style    = variant_style[variant_name]

            x_pdf_s, cnt_s, x_ccdf_s, ccdf_s = _pdf_ccdf(data_sim)
            ax_pdf.plot(x_pdf_s,   cnt_s,  **style)
            ax_ccdf.plot(x_ccdf_s, ccdf_s, **style)

            # KS test
            ks  = run_ks_test(data_real, data_sim)
            sig = ("***" if ks['p'] < 0.001 else
                   ("**"  if ks['p'] < 0.01  else
                    ("*"   if ks['p'] < 0.05  else "ns")))

            print(f"  {stem:20s}  vs  {variant_name:35s}"
                  f"  D = {ks['stat']:.5f}   p = {ks['p']:.2e}  {sig}")

            # Mark max-separation point on CCDF
            ax_ccdf.vlines(
                ks['x_D'],
                min(ks['ecdf_b'], ks['ecdf_s']),
                max(ks['ecdf_b'], ks['ecdf_s']),
                color=style['color'], linestyle='--', linewidth=1.2,
                alpha=0.8, zorder=5,
            )

        if row == 0:
            ax_pdf.legend(fontsize=7.5, framealpha=0.4)
            ax_ccdf.legend(fontsize=7.5, framealpha=0.4)

    print("=" * 65 + "\n")

    os.makedirs('images', exist_ok=True)
    out_fname = f'meta_child_dist/compare_real_vs_simulated_{target_stem}.png' if target_stem else 'meta_child_dist/compare_real_vs_simulated.png'
    out_path = os.path.join('images', out_fname)
    fig.savefig(out_path, dpi=300)
    print(f"[save]  {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    #plot_all(results_dir)

    compare_all_distributions_grid(BASE_DIR, REAL_SUBDIR)

    # To compare a single configuration:
    #compare_all_distributions(BASE_DIR, REAL_SUBDIR, nb_traders=20, kind='power', exponent=2.0)