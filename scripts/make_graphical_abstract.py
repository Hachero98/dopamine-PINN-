"""Graphical abstract for the BSPC submission.

Story, left to right:
1  an electrode with a finite tip records near a dopamine release site
2  the sensor distorts the signal (tip averaging + response lag + noise)
3  the measurement-aware PINN fits the sensor output, not the concentration
4  naive fits are biased and overconfident; sensor-aware fits are unbiased with
   close to nominal coverage (reference estimator, 50 realizations per scenario)

Drawn at 13 cm width (the journal's readability width), exported >= 1328 px wide.
Usage:  python scripts/make_graphical_abstract.py results/manuscript_extensions/measurement_model.json <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

meas = json.loads(Path(sys.argv[1]).read_text())
OUT = Path(sys.argv[2])
S = meas['scenarios']

RC = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
      'font.size': 7, 'axes.labelsize': 6.5, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
      'mathtext.fontset': 'dejavusans', 'axes.linewidth': 0.6}
W, H = 13 / 2.54, 5.2 / 2.54
C_NAIVE, C_AWARE, C_SITE = '#c0392b', '#1e8449', '#1f4e79'

D, K, L, SIG, C0, A_TIP, TAU = 0.32, 0.020, 5.0, 0.5, 1.0, 0.5, 2.0


def exact(x, y, t):
    x, y, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(t, float))
    v = SIG**2 + 2 * D * np.maximum(t, 0)
    m = np.arange(-6, 7).reshape((-1,) + (1,) * x.ndim)
    g = (np.exp(-(x[None] - m * L)**2 / (2 * v[None])).sum(0)
         * np.exp(-(y[None] - m * L)**2 / (2 * v[None])).sum(0))
    return np.where(t >= 0, C0 * SIG**2 / v * np.exp(-K * t) * g, 0.0)


dt = 0.02
tf = np.arange(-5.0, 45.0 + 1e-9, dt)
c_pt = exact(0.5, 0.0, tf)
gu, gw = np.polynomial.legendre.leggauss(4)
rr = A_TIP * np.sqrt((gu + 1) / 2); th = 2 * np.pi * np.arange(16) / 16
c_tip = sum(wi / 32 * exact(0.5 + ri * np.cos(t_), ri * np.sin(t_), tf) for ri, wi in zip(rr, gw) for t_ in th)
a_ = np.exp(-dt / TAU); c_lag = np.zeros_like(c_tip)
for i in range(1, tf.size):
    c_lag[i] = a_ * c_lag[i - 1] + (1 - a_) * 0.5 * (c_tip[i] + c_tip[i - 1])
ts = np.arange(-5.0, 45.0, 0.5)
rng = np.random.default_rng(7)
y_rec = np.interp(ts, tf, c_lag) + 0.02 * rng.normal(size=ts.size)


def fig_arrow(fig, x0, x1, y):
    fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle='-|>',
                                       mutation_scale=9, color='0.35', lw=1.1))


with plt.rc_context(RC):
    fig = plt.figure(figsize=(W, H))
    ytitle = 0.93

    # 1  electrode near a release site
    ax1 = fig.add_axes([0.012, 0.13, 0.17, 0.66])
    g = np.linspace(-L / 2, L / 2, 201); X, Y = np.meshgrid(g, g)
    ax1.contourf(X, Y, np.exp(-(X**2 + Y**2) / (2 * SIG**2)), levels=np.linspace(0.05, 1, 8), cmap='Oranges')
    for (px, py) in [(0.5, 0.0), (0.0, 2.0)]:
        ax1.add_patch(Circle((px, py), A_TIP, fc=(0.12, 0.31, 0.47, 0.22), ec=C_SITE, lw=1.0))
        ax1.plot(px, py, 'o', color=C_SITE, ms=2)
    for sp in ax1.spines.values():
        sp.set_linewidth(1.6)
    ax1.set_xlim(-L / 2, L / 2); ax1.set_ylim(-L / 2, L / 2); ax1.set_aspect('equal')
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.text(0.0, -2.05, 'release site', ha='center', fontsize=5.8, color='#a04000')
    ax1.text(-0.35, 0.95, 'electrodes', ha='center', fontsize=5.8, color=C_SITE)
    fig.text(0.097, ytitle, 'Sparse electrode\nrecordings', ha='center', va='top', fontsize=7, fontweight='bold')
    fig_arrow(fig, 0.188, 0.215, 0.46)

    # 2  the sensor distorts the signal
    ax2 = fig.add_axes([0.235, 0.20, 0.19, 0.55])
    ax2.axvspan(-5, 0, color='0.92', lw=0)
    ax2.plot(ts, y_rec, '.', color='0.6', ms=1.6)
    ax2.plot(tf, c_pt, '-', color='k', lw=0.9, label='true concentration')
    ax2.plot(tf, c_lag, '-', color=C_NAIVE, lw=1.2, label='sensor output')
    ax2.set_xlim(-5, 45); ax2.set_ylim(-0.05, 0.38)
    ax2.set_xticks([0, 20, 40]); ax2.set_yticks([])
    ax2.set_xlabel('time (ms)', labelpad=1)
    ax2.tick_params(length=2, pad=1)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.03, 1.0), fontsize=5.4, frameon=False, handlelength=1.0)
    fig.text(0.33, ytitle, 'The sensor distorts\nthe signal', ha='center', va='top', fontsize=7, fontweight='bold')
    ax2.text(0.97, 0.55, 'tip averaging\nresponse lag\ncorrelated noise\noffset', transform=ax2.transAxes,
             ha='right', va='center', fontsize=5.2, color='0.35', linespacing=1.05)
    fig_arrow(fig, 0.435, 0.462, 0.46)

    # 3  measurement-aware PINN
    ax3 = fig.add_axes([0.468, 0.08, 0.205, 0.72]); ax3.axis('off')
    ax3.add_patch(FancyBboxPatch((0.02, 0.06), 0.96, 0.86, boxstyle='round,pad=0.01,rounding_size=0.05',
                                 fc='#eef4fa', ec=C_SITE, lw=0.9, transform=ax3.transAxes))
    lay = [3, 4, 4, 1]; xs = np.linspace(0.14, 0.52, len(lay)); pts = []
    for n, x in zip(lay, xs):
        pts.append([(x, y) for y in (np.linspace(0.40, 0.78, n) if n > 1 else [0.59])])
    for a, b in zip(pts[:-1], pts[1:]):
        for p in a:
            for q in b:
                ax3.plot([p[0], q[0]], [p[1], q[1]], color='0.75', lw=0.35, transform=ax3.transAxes)
    for li, layer in enumerate(pts):
        for p in layer:
            ax3.scatter(*p, s=16, c=['#f9d9b8' if li == len(pts) - 1 else '0.85'], edgecolors='0.4',
                        linewidths=0.4, transform=ax3.transAxes, zorder=3)
    ax3.text(0.5, 0.13, r'PDE: $\partial_t C = D\nabla^2 C - kC$', ha='center', fontsize=5.6,
             transform=ax3.transAxes)
    ax3.add_patch(FancyBboxPatch((0.62, 0.48), 0.31, 0.22, boxstyle='round,pad=0.01,rounding_size=0.03',
                                 fc='#efe3f6', ec='#7d3c98', lw=0.8, transform=ax3.transAxes))
    ax3.text(0.775, 0.59, 'sensor\nmodel', ha='center', va='center', fontsize=5.8, transform=ax3.transAxes)
    ax3.add_patch(FancyArrowPatch((0.545, 0.59), (0.615, 0.59), transform=ax3.transAxes, arrowstyle='-|>',
                                  mutation_scale=6, color='#7d3c98', lw=0.8))
    ax3.text(0.775, 0.31, r'fit to the' + '\n' + r'record $\to D, k$', ha='center', va='center',
             fontsize=5.8, transform=ax3.transAxes)
    ax3.add_patch(FancyArrowPatch((0.775, 0.47), (0.775, 0.38), transform=ax3.transAxes, arrowstyle='-|>',
                                  mutation_scale=6, color='0.35', lw=0.8))
    fig.text(0.57, ytitle, 'Measurement-aware\nPINN', ha='center', va='top', fontsize=7, fontweight='bold')
    fig_arrow(fig, 0.678, 0.705, 0.46)

    # 4  result: naive vs aware
    names = ['M1_tip', 'M2_lag', 'M5_all']
    labs = ['tip', 'lag', 'all']
    medD = {m: [float(np.median([d[m]['D_err'] for d in S[n]['ref'].values()])) for n in names]
            for m in ('naive', 'aware')}
    covK = {m: [100 * S[n]['summary'][m]['ref_cover_k'] / S[n]['summary'][m]['n_ref'] for n in
                ['M2_lag', 'M3_colour', 'M4_drift']] for m in ('naive', 'aware')}
    xi = np.arange(3)
    ax4 = fig.add_axes([0.742, 0.22, 0.092, 0.53])
    ax4.bar(xi - 0.19, medD['naive'], 0.36, color=C_NAIVE)
    ax4.bar(xi + 0.19, medD['aware'], 0.36, color=C_AWARE)
    ax4.axhline(0, color='k', lw=0.5)
    ax4.set_yscale('symlog', linthresh=10)
    ax4.set_ylim(-8, 260)
    ax4.set_yticks([0, 10, 100]); ax4.set_yticklabels(['0', '10', '100'])
    ax4.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax4.set_xticks(xi); ax4.set_xticklabels(labs)
    ax4.set_title('error in $D$ (%)', fontsize=6.0, pad=2)
    ax4.tick_params(length=2, pad=1)
    ax5 = fig.add_axes([0.878, 0.22, 0.106, 0.53])
    ax5.bar(xi - 0.19, covK['naive'], 0.36, color=C_NAIVE, label='sensor ignored')
    ax5.bar(xi + 0.19, covK['aware'], 0.36, color=C_AWARE, label='sensor modelled')
    ax5.axhline(95, color='k', ls=':', lw=0.7)
    ax5.set_ylim(0, 100); ax5.set_yticks([0, 50, 95])
    ax5.set_xticks(xi); ax5.set_xticklabels(['lag', 'noise', 'offset'], fontsize=5.1)
    ax5.set_title('coverage of $k$ (%)', fontsize=6.0, pad=2)
    ax5.tick_params(length=2, pad=1)
    fig.legend(*ax5.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.855, 0.0),
               ncol=2, fontsize=5.5, frameon=False, handlelength=0.9, columnspacing=0.7, handletextpad=0.4)
    fig.text(0.855, ytitle, 'Model the sensor:\nunbiased, honest intervals', ha='center', va='top',
             fontsize=7, fontweight='bold')

    fig.savefig(OUT / 'Graphical_Abstract_Hackman.pdf')
    fig.savefig(OUT / 'Graphical_Abstract_Hackman.tif', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(OUT / 'Graphical_Abstract_Hackman_preview.png', dpi=300)
print('done')
