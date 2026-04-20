import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import powerlaw
import os
from scipy.stats import ks_2samp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
results_dir = 'database\\meta_child_dist'  # must match meta_child_generate.py

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
            f"Run meta_child_generate.py first to generate the data."
        )

    archive    = np.load(filepath)
    iterations = int(archive['iterations'])
    data       = archive['data']
    print(f"[load]  {filepath}")
    print(f"        Iterations   : {iterations}")
    print(f"        Realizations : {len(data):,}")
    return iterations, data


def load_all(results_dir: str) -> dict[str, tuple[int, np.ndarray]]:
    """
    Scan results_dir for all .npz files and load each one.

    Returns
    -------
    dict mapping filename stem (e.g. '4_uniform') to (iterations, data).
    """
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Directory '{results_dir}' not found.")

    files = [f for f in os.listdir(results_dir) if f.endswith('.npz')]
    if not files:
        raise FileNotFoundError(f"No .npz files found in '{results_dir}'.")

    results = {}
    for fname in sorted(files):
        stem     = fname[len('realizations_'):-len('.npz')]
        filepath = os.path.join(results_dir, fname)
        archive  = np.load(filepath)
        iterations = int(archive['iterations'])
        data       = archive['data']
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
    Run a two-sample KS test between data_b and data_s.

    Returns a dict with:
      - stat   : KS statistic D
      - p      : p-value
      - x_D    : x-value where max separation occurs (on the shared ECDF grid)
      - ecdf_b : ECDF of base at x_D
      - ecdf_s : ECDF of signs at x_D
    """
    stat, p = ks_2samp(data_b, data_s)

    # Locate the x where |ECDF_b - ECDF_s| is maximised
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
# Plot all configurations together
# ---------------------------------------------------------------------------

def plot_all(results_dir: str) -> None:
    """
    Load every .npz in results_dir and overlay their empirical PDFs and CCDFs
    on a single figure for visual comparison.
    """
    all_data = load_all(results_dir)
    colors   = plt.cm.tab10.colors
    markers  = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

    fig, (ax_pdf, ax_ccdf) = plt.subplots(1, 2, figsize=(13, 5.5))

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
    fig.savefig('images\\dist_meta_child_all.png', dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Compare meta_child_dist vs meta_child_dist_signs  (with KS test)
# ---------------------------------------------------------------------------

def compare_all_distributions(
    results_dir_base:  str,
    results_dir_signs: str,
) -> None:
    """
    Load every configuration present in *either* directory and produce a
    multi-panel figure:  one row per configuration, two columns (PDF | CCDF).

    Each subplot overlays:
      • meta_child_dist       — blue circles
      • meta_child_dist_signs — orange squares

    When both datasets exist for a configuration, a two-sample KS test is run
    and the result is annotated on the CCDF panel:
      • A vertical dashed red line marks x where |ECDF_base - ECDF_signs| is
        maximised (the KS distance D), consistent with the ks.py visualisation.
      • A text box reports D and the p-value.

    Missing stems in one directory are silently skipped for that dataset.

    Parameters
    ----------
    results_dir_base  : path to the 'meta_child_dist'       output folder
    results_dir_signs : path to the 'meta_child_dist_signs' output folder
    """
    # Load all available data from both directories
    all_base  = load_all(results_dir_base)
    all_signs = load_all(results_dir_signs)

    # Union of stems, sorted for a stable row order
    stems = sorted(set(all_base) | set(all_signs))
    n     = len(stems)

    if n == 0:
        print("[compare_all] No data found in either directory.")
        return

    # --- compute empirical PDF & CCDF ---
    def _pdf_ccdf(data):
        bins              = np.arange(1, max(data) + 2)
        counts, bin_edges = np.histogram(data, bins=bins, density=True)
        x_pdf             = bin_edges[:-1]
        x_ccdf, ccdf      = compute_ccdf(data)
        return x_pdf, counts, x_ccdf, ccdf

    STYLE_BASE  = dict(linestyle='', marker='o', markersize=3.5,
                       color='C0', alpha=0.85, label='meta_child_dist')
    STYLE_SIGNS = dict(linestyle='', marker='o', markersize=3.5,
                       color='C1', alpha=0.85, label='meta_child_dist_signs')

    # --- build figure: n rows × 2 cols ---
    row_h = 3.8
    fig   = plt.figure(figsize=(13, row_h * n))
    gs    = gridspec.GridSpec(
        n, 2,
        figure=fig,
        hspace=0.2,
        wspace=0.2,
        left=0.08, right=0.98,
        top=0.98,  bottom=0.03,
    )

    for row, stem in enumerate(stems):
        ax_pdf  = fig.add_subplot(gs[row, 0])
        ax_ccdf = fig.add_subplot(gs[row, 1])

        # --- style both axes ---
        for ax, ylabel, col_title in [
            (ax_pdf,  'P(x)',      'PDF'),
            (ax_ccdf, 'P(X ≥ x)', 'CCDF'),
        ]:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('x')
            ax.set_ylabel(ylabel)
            ax.tick_params()
            ax.grid(True, linewidth=0.4, linestyle='--', alpha=0.6)
            # Column header only on first row
            if row == 0:
                ax.set_title(col_title)

        # Row label on the left spine
        ax_pdf.set_ylabel(f'{stem.replace("_", " ")}\nP(x)')

        # --- plot base ---
        data_b = None
        if stem in all_base:
            _, data_b = all_base[stem]
            x_pdf_b, cnt_b, x_ccdf_b, ccdf_b = _pdf_ccdf(data_b)
            ax_pdf.plot( x_pdf_b,  cnt_b,  **STYLE_BASE)
            ax_ccdf.plot(x_ccdf_b, ccdf_b, **STYLE_BASE)

        # --- plot signs ---
        data_s = None
        if stem in all_signs:
            _, data_s = all_signs[stem]
            x_pdf_s, cnt_s, x_ccdf_s, ccdf_s = _pdf_ccdf(data_s)
            ax_pdf.plot( x_pdf_s,  cnt_s,  **STYLE_SIGNS)
            ax_ccdf.plot(x_ccdf_s, ccdf_s, **STYLE_SIGNS)

        # --- KS test (only when both datasets are available) ---
        if data_b is not None and data_s is not None:
            ks = run_ks_test(data_b, data_s)

            # Print results to console
            sig = "***" if ks['p'] < 0.001 else ("**" if ks['p'] < 0.01
                  else ("*" if ks['p'] < 0.05 else "ns"))
            print(f"[KS {stem:12s}]  D = {ks['stat']:.5f}   p = {ks['p']:.2e}  {sig}")

            # *** — p < 0.001: extremely strong evidence against the null hypothesis
            # **  — p < 0.01: very strong evidence
            # *   — p < 0.05: standard significance threshold (the most commonly used cutoff)
            # ns  — "not significant": p ≥ 0.05, meaning you can't reject the null hypothesis that the two distributions are the same

            # Mark the point of maximum separation on the CCDF plot with a
            # vertical line spanning the two ECDF values at x_D
            # ax_ccdf.vlines(
            #     ks['x_D'],
            #     min(ks['ecdf_b'], ks['ecdf_s']),
            #     max(ks['ecdf_b'], ks['ecdf_s']),
            #     color='red', linestyle='--', linewidth=1.4,
            #     label=f'KS distance (D)',
            #     zorder=5,
            # )

            # # Annotate with D and p-value
            # ax_ccdf.text(
            #     0.97, 0.97,
            #     f"D = {ks['stat']:.4f}\np = {ks['p']:.2e}  {sig}",
            #     transform=ax_ccdf.transAxes,
            #     fontsize=8,
            #     verticalalignment='top',
            #     horizontalalignment='right',
            #     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
            #               edgecolor='red', alpha=0.75),
            # )

        # Legend only on first row to avoid clutter
        if row == 0:
            ax_pdf.legend()
            ax_ccdf.legend()

    os.makedirs('images', exist_ok=True)
    out_path = os.path.join('images', 'compare_dist_vs_signs_all.png')
    fig.savefig(out_path, dpi=300)
    print(f"[save]  {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Style & entry point
# ---------------------------------------------------------------------------
plot_all(results_dir)

# --- compare base vs signs for the current configuration ---
results_dir_signs = 'database\\meta_child_dist_signs'   # adjust if needed
compare_all_distributions(results_dir, results_dir_signs)