"""Fig. 1 of the BSPC manuscript: problem, sensor model and measurement-aware PINN.

a  domain with zero-flux walls, Gaussian release and the two electrode sites of
   design S2 drawn with their sensing disks (radius a = 0.5 um); grey crosses are
   40 of the 400 scattered observations
b  sensor model at the near site (r = 0.5 um), computed from the exact solution:
   concentration at the site -> tip average -> response lag -> offset + correlated
   noise -> recorded signal (10 ms pre-stimulus baseline)
c  measurement-aware PINN: the network output passes through the same sensor
   operator (W) and whitening/offset projection (M) before it meets the record

Usage:  python plot_schematic_fig.py <manuscript_dir> <figures_dir>
Writes Fig1.pdf and Figure_1.{eps,tif} at 174 mm width.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

MS_DIR, FIG_DIR = Path(sys.argv[1]), Path(sys.argv[2])
RC = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
      'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
      'mathtext.fontset': 'dejavusans'}
W_IN = 174 / 25.4

D, K, L, SIG, C0 = 0.32, 0.020, 5.0, 0.5, 1.0
A_TIP, TAU, TAU_N, ETA = 0.5, 2.0, 2.0, 0.02
SITES = [(0.5, 0.0), (0.0, 2.0)]
COL_REL, COL_SITE = '#d35400', '#1f4e79'
COL_PDE, COL_IC, COL_BC, COL_D, COL_OUT = '#fde2e1', '#e3f2e1', '#e6f0e1', '#dceaf7', '#fff6c7'


def exact(x, y, t):
    """Image-series solution on the bounded domain (Appendix A)."""
    x, y, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(t, float))
    v = SIG**2 + 2 * D * np.maximum(t, 0)
    m = np.arange(-6, 7).reshape((-1,) + (1,) * x.ndim)
    g = (np.exp(-(x[None] - m * L)**2 / (2 * v[None])).sum(0)
         * np.exp(-(y[None] - m * L)**2 / (2 * v[None])).sum(0))
    return np.where(t >= 0, C0 * SIG**2 / v * np.exp(-K * t) * g, 0.0)


# ---- signals at the near site ------------------------------------------------------
dt = 0.02
tf = np.arange(-10.0, 45.0 + 1e-9, dt)
sx, sy = SITES[0]
c_pt = exact(sx, sy, tf)
gu, gw = np.polynomial.legendre.leggauss(4)
rr = A_TIP * np.sqrt((gu + 1) / 2)
th = 2 * np.pi * np.arange(16) / 16
c_tip = sum(wi / 2 / 16 * exact(sx + ri * np.cos(t_), sy + ri * np.sin(t_), tf)
            for ri, wi in zip(rr, gw) for t_ in th)
a_ = np.exp(-dt / TAU)
c_lag = np.zeros_like(c_tip)
for i in range(1, tf.size):                       # exact first-order filter, trapezoid input
    c_lag[i] = a_ * c_lag[i - 1] + (1 - a_) * 0.5 * (c_tip[i] + c_tip[i - 1])
ts = np.r_[np.arange(-10.0, 0.0, 0.5), np.arange(0.5, 45.0 + 1e-9, 0.5)]
rng = np.random.default_rng(5001)
rho = np.exp(-np.diff(ts) / TAU_N)
e = np.empty(ts.size); e[0] = rng.normal()
for i in range(1, ts.size):
    e[i] = rho[i - 1] * e[i - 1] + np.sqrt(1 - rho[i - 1]**2) * rng.normal()
offset = 0.012
y_rec = np.interp(ts, tf, c_lag) + offset + ETA * C0 * e


def box(ax, x, y, w, h, text, fc, fs=6.3, ec='0.35', lw=0.7, ha='center'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.012',
                                fc=fc, ec=ec, lw=lw, transform=ax.transAxes, clip_on=False))
    tx = x + w / 2 if ha == 'center' else x + 0.012
    ax.text(tx, y + h / 2, text, ha=ha, va='center', fontsize=fs, transform=ax.transAxes)


def arrow(ax, p0, p1, color='0.3', lw=0.8, style='-|>', ms=6, cs='arc3'):
    ax.add_patch(FancyArrowPatch(p0, p1, transform=ax.transAxes, arrowstyle=style,
                                 mutation_scale=ms, color=color, lw=lw, clip_on=False,
                                 connectionstyle=cs))


with plt.rc_context(RC):
    fig = plt.figure(figsize=(W_IN, 5.15))

    # ---------------- (a) domain ----------------
    ax = fig.add_axes([0.045, 0.555, 0.25, 0.39])
    g = np.linspace(-L / 2, L / 2, 241)
    X, Y = np.meshgrid(g, g)
    C = C0 * np.exp(-(X**2 + Y**2) / (2 * SIG**2))
    ax.contourf(X, Y, C, levels=np.linspace(0.05, 1.0, 9), cmap='Oranges')
    obs = np.random.default_rng(4001)
    ax.plot(obs.uniform(-L/2, L/2, 40), obs.uniform(-L/2, L/2, 40), 'x', color='0.6', ms=2.6, mew=0.6)
    for (px, py), lab in zip(SITES, ('site 1, $r = 0.5$', 'site 2, $r = 2$')):
        ax.add_patch(Circle((px, py), A_TIP, fc=(0.12, 0.31, 0.47, 0.18), ec=COL_SITE, lw=1.0))
        ax.plot(px, py, 'o', color=COL_SITE, ms=2.2)
    lab_bb = dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85)
    ax.annotate('electrode site 1\n($r = 0.5\\,\\mu$m)', xy=(0.5 + 0.35, -0.35), xytext=(0.55, -1.95),
                fontsize=5.8, color=COL_SITE, bbox=lab_bb,
                arrowprops=dict(arrowstyle='-', color=COL_SITE, lw=0.6))
    ax.annotate('electrode site 2\n($r = 2\\,\\mu$m)', xy=(-0.4, 2.0), xytext=(-2.35, 1.2),
                fontsize=5.8, color=COL_SITE, bbox=lab_bb,
                arrowprops=dict(arrowstyle='-', color=COL_SITE, lw=0.6))
    ax.annotate('release', xy=(-0.25, -0.2), xytext=(-2.2, -1.3), fontsize=5.8, color=COL_REL, bbox=lab_bb,
                arrowprops=dict(arrowstyle='-', color=COL_REL, lw=0.6))
    for sp in ax.spines.values():
        sp.set_linewidth(2.0)
    ax.set_xlim(-L/2, L/2); ax.set_ylim(-L/2, L/2); ax.set_aspect('equal')
    ax.set_xticks([-2.5, 0, 2.5]); ax.set_yticks([-2.5, 0, 2.5])
    ax.set_xlabel(r'$x$ ($\mu$m)', labelpad=1); ax.set_ylabel(r'$y$ ($\mu$m)', labelpad=0)
    ax.set_title(r'zero-flux walls, $\partial_n C = 0$', fontsize=6.5, pad=3)
    fig.text(0.008, 0.965, 'a', fontsize=9, fontweight='bold')

    # ---------------- (b) sensor chain ----------------
    fig.text(0.345, 0.965, 'b', fontsize=9, fontweight='bold')
    fig.text(0.665, 0.962, 'Sensor model at electrode site 1', fontsize=7, ha='center')
    stages = [
        (tf, c_pt, 'k', '-', 'concentration\nat the site', r'$C(\mathbf{x}_s, t)$'),
        (tf, c_tip, '#7d3c98', '-', 'tip average,\n$a = 0.5\\,\\mu$m', r'$\overline{C}_a$'),
        (tf, c_lag, '#c0392b', '-', 'response lag,\n$\\tau = 2$ ms', r'$h * \overline{C}_a$'),
        (ts, y_rec, '0.45', '.', 'offset $b_s$ +\ncorrelated noise', r'$y_s(t_i)$'),
    ]
    x0, wpan, gap = 0.375, 0.128, 0.03
    ymax = c_pt.max() * 1.08
    for j, (tt, cc, col, ls, title, sym) in enumerate(stages):
        axb = fig.add_axes([x0 + j * (wpan + gap), 0.63, wpan, 0.24])
        if ls == '.':
            axb.plot(tt, cc, '.', color=col, ms=1.6)
        else:
            axb.plot(tt, cc, ls, color=col, lw=1.0)
        if j > 0:
            axb.plot(tf, c_pt, '-', color='0.8', lw=0.6, zorder=0)
        axb.axvspan(-10, 0, color='0.93', lw=0, zorder=-1)
        axb.set_xlim(-10, 45); axb.set_ylim(-0.06, ymax)
        axb.set_xticks([0, 20, 40]); axb.set_yticks([0, 0.2] if j == 0 else [])
        axb.tick_params(length=2, pad=1)
        if j == 0:
            axb.set_ylabel(r'$\mu$M', labelpad=1)
        axb.set_xlabel('t (ms)', labelpad=0.5)
        axb.set_title(title, fontsize=6.0, pad=2.5, linespacing=1.05)
        axb.text(0.96, 0.9, sym, transform=axb.transAxes, ha='right', va='top', fontsize=6.6, color=col)
        if j < 3:
            fig.patches.append(FancyArrowPatch(
                (x0 + j * (wpan + gap) + wpan + 0.003, 0.75), (x0 + (j + 1) * (wpan + gap) - 0.003, 0.75),
                transform=fig.transFigure, arrowstyle='-|>', mutation_scale=7, color='0.3', lw=0.8))
    fig.text(x0, 0.565, 'grey band: 10 ms pre-stimulus baseline;  light grey line: true concentration',
             fontsize=5.6, color='0.35')

    # ---------------- (c) measurement-aware PINN ----------------
    fig.text(0.008, 0.505, 'c', fontsize=9, fontweight='bold')
    axc = fig.add_axes([0.0, 0.0, 1.0, 0.5]); axc.axis('off')
    # network drawing
    layers = [3, 5, 5, 5, 5, 1]
    xs = np.linspace(0.06, 0.30, len(layers))
    pos = []
    for li, (n, xl) in enumerate(zip(layers, xs)):
        ys_ = np.linspace(0.30, 0.82, n) if n > 1 else np.array([0.56])
        if n == 3:
            ys_ = np.array([0.72, 0.56, 0.40])
        pos.append([(xl, yy) for yy in ys_])
    for a1, a2 in zip(pos[:-1], pos[1:]):
        for p in a1:
            for q in a2:
                axc.plot([p[0], q[0]], [p[1], q[1]], '-', color='0.82', lw=0.35, transform=axc.transAxes)
    for li, layer in enumerate(pos):
        for (xx, yy), lab in zip(layer, (['$x$', '$y$', '$t$'] if li == 0 else
                                         [r'$\hat{C}$'] if li == len(pos) - 1 else [''] * 5)):
            fc = '#d6e6f5' if li == 0 else '#f9d9b8' if li == len(pos) - 1 else '0.85'
            axc.scatter([xx], [yy], s=58, c=[fc], edgecolors='0.4', linewidths=0.5,
                        transform=axc.transAxes, zorder=3, clip_on=False)
            if lab:
                axc.text(xx, yy, lab, ha='center', va='center', fontsize=6.3, transform=axc.transAxes,
                         zorder=4)
    axc.text(0.18, 0.92, r'network $\mathcal{N}_\theta$: 4 hidden layers $\times$ 64, tanh',
             ha='center', fontsize=6.5, transform=axc.transAxes)
    axc.text(0.18, 0.17, 'automatic differentiation\n'
             r'$\partial_t\hat{C}$, $\nabla^2\hat{C}$, $\partial_n\hat{C}$',
             ha='center', fontsize=6.0, color='0.3', transform=axc.transAxes)

    # loss boxes
    bx, bw, bh = 0.40, 0.255, 0.115
    rows = [(0.78, COL_PDE, r'$\mathcal{L}_r$: PDE residual  $\partial_t\hat{C} - D\nabla^2\hat{C} + k\hat{C}$'),
            (0.615, COL_IC, r'$\mathcal{L}_i$: initial condition at $t = 0$'),
            (0.45, COL_BC, r'$\mathcal{L}_b$: zero-flux walls  $\partial_n\hat{C}$')]
    out = pos[-1][0]
    for yb, fc, txt in rows:
        box(axc, bx, yb, bw, bh, txt, fc, ha='left')
        arrow(axc, (out[0] + 0.015, out[1]), (bx - 0.004, yb + bh / 2))
    # sensor operator and data term
    box(axc, bx, 0.20, bw, 0.17,
        'sensor operator $W$: same tip average\nand lag as panel b (quadrature nodes)\n'
        r'$\hat{C}$ at nodes $\rightarrow$ predicted output $W\hat{\mathbf{c}}$',
        '#efe3f6', fs=5.9, ec='#7d3c98', lw=0.9, ha='left')
    arrow(axc, (out[0] + 0.015, out[1] - 0.01), (bx - 0.004, 0.285), color='#7d3c98')
    box(axc, 0.705, 0.20, 0.265, 0.17,
        r'$\mathcal{L}_d = \frac{1}{N}\,\| M\,(W\hat{\mathbf{c}} - \mathbf{y}) \|^2$' + '\n'
        r'$M$: whiten correlated noise,' + '\n' + 'project out the baseline offset',
        COL_D, fs=6.0, ec=COL_SITE, lw=0.9, ha='left')
    arrow(axc, (bx + bw + 0.004, 0.285), (0.701, 0.285), color='#7d3c98')
    # record from panel b into the data term
    fig.add_artist(matplotlib.lines.Line2D([0.966, 0.995, 0.995], [0.60, 0.60, 0.1425],
                                           transform=fig.transFigure, color=COL_SITE, lw=0.9))
    fig.patches.append(FancyArrowPatch((0.995, 0.1425), (0.972, 0.1425), transform=fig.transFigure,
                                       arrowstyle='-|>', mutation_scale=7, color=COL_SITE, lw=0.9))
    fig.text(1.004, 0.37, r'record $\mathbf{y}$', fontsize=6.2, color=COL_SITE, rotation=90,
             ha='center', va='center')
    # total loss / output
    box(axc, 0.705, 0.50, 0.265, 0.39,
        r'$\mathcal{L} = \lambda_r\mathcal{L}_r + \lambda_i\mathcal{L}_i + \lambda_b\mathcal{L}_b + \lambda_d\mathcal{L}_d$'
        '\n' r'$(\lambda_r, \lambda_i, \lambda_b, \lambda_d) = (3000, 1, 100, 5)$' '\n\n'
        r'learned jointly: $\theta$, $\log D$, $\log k$' '\n'
        r'Adam ($2\times10^4$ it.), then L-BFGS' '\n' r'$\Rightarrow$ recovered $(D, k)$',
        COL_OUT, fs=6.0, ha='left')
    for yb, _, _ in rows:
        arrow(axc, (bx + bw + 0.004, yb + bh / 2), (0.701, min(max(yb + bh / 2, 0.52), 0.87)),
              color='0.55', lw=0.6, ms=5)
    arrow(axc, (0.84, 0.375), (0.84, 0.497), color='0.55', lw=0.6, ms=5)
    axc.text(0.40, 0.09, 'Ideal sensor: $W = M = I$ and $\\mathcal{L}_d$ is the usual mean squared misfit.',
             fontsize=5.8, color='0.35', transform=axc.transAxes)

    fig.savefig(MS_DIR / 'Fig1.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'Figure_1.eps', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'Figure_1.tif', bbox_inches='tight', dpi=600,
                pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(Path(__file__).resolve().parent / 'schematic_fig_preview.png', bbox_inches='tight', dpi=200)
print('done')
