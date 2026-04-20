#!/usr/bin/env python3
"""
V&D CFD Validation — Master Plotting Script v3
===============================================
All plots show POSITIVE HALF CHANNEL (symmetric about center).
All variables dimensionless: y* = y/Dc, D* = D/Dc, δ* = τ/(ρgSDc)

To update files: edit the FILE PATHS section below (lines ~30-50).
To add a new case: add entry to the cfd_cases dict (~line 60).

Run: python3 master_plot_v3.py
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================================================================
# FILE PATHS — EDIT THESE
# ================================================================
BASE_DIR = '.'

# V&D analytical solution (MATLAB output)
VD_FILE = os.path.join(BASE_DIR, 'VD_Geometryanddelta.csv')

# CFD case files — dict of {label: (filepath, color, linestyle, linewidth)}
# Comment out or delete lines for cases you don't have yet
CFD_CYCLIC = {
    'Smooth 2m':                (os.path.join(BASE_DIR, 'wss_profile_2000_cyclic_smooth_2m.csv'),     '#2196F3', '-',  2.0),
    'Rough 2m (OF12)':          (os.path.join(BASE_DIR, 'wss_profile_2000_cyclic_rough_2m.csv'),      '#F44336', '-',  2.5),
    'Rough 4m (OF12)':          (os.path.join(BASE_DIR, 'wss_profile_1500_cyclic_rough_4m.csv'),      '#9C27B0', '--', 2.0),
    'Rough 2m (v2312 atmNutk)': (os.path.join(BASE_DIR, 'wss_profile_2000_rough_2m_v2312.csv'),      '#FF9800', '-.', 2.5),
}

CFD_HPC_X45 = {
    '50m Smooth x=45':  (os.path.join(BASE_DIR, 'wss_x45_2000_hpc20_smooth.csv'),  '#4CAF50', '--', 1.8),
    '50m Rough x=45':   (os.path.join(BASE_DIR, 'wss_x45_2000_hpc20_rough.csv'),   '#795548', '--', 2.0),
}

# HPC x-section directories (for flow development plots)
HPC_SMOOTH_DIR = os.path.join(BASE_DIR, 'HPC 20 Smooth')
HPC_ROUGH_DIR  = os.path.join(BASE_DIR, 'HPC20Rough')

OUTPUT_DIR = '.'

# ================================================================
# PHYSICAL PARAMETERS
# ================================================================
rho = 1000.0;  g = 9.81;  S = 0.0016;  mu = 0.76
d50 = 0.060;   d90 = 0.080;  Rs = 1.65;  tau_cr_star = 0.03
delta_cr_star = 0.88

Dc = Rs * tau_cr_star * d50 / (S * delta_cr_star)  # 2.109 m
half_Bf = 0.94
tau_0 = rho * g * Dc * S       # 33.1 Pa
tau_cr = rho * Rs * g * tau_cr_star * d50  # 29.1 Pa
junct_star = half_Bf / Dc       # ~0.446

# Table 3 bank polynomial (mu=0.76, delta*_cr=0.88)
a0, a1, a2, a3 = 1.0014, -0.0531, -0.0610, -0.0101

# ================================================================
# HELPER FUNCTIONS
# ================================================================
def load_VD(filepath):
    """Load V&D analytical CSV → positive half: y*, D*, δ*."""
    if not os.path.exists(filepath):
        print(f"  WARNING: {filepath} not found"); return None
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1, filling_values=np.nan)
    y, D, DEL = data[:, 0], data[:, 1], data[:, 2]
    valid = ~np.isnan(D) & ~np.isnan(DEL)
    return {'y_star': y[valid], 'D_star': D[valid], 'delta_star': DEL[valid]}

def load_CFD(filepath):
    """Load CFD CSV → positive half: y*, δ*, τ_Pa."""
    if not os.path.exists(filepath):
        print(f"  WARNING: {filepath} not found"); return None
    with open(filepath, 'r') as f:
        header = f.readline().strip().replace('\r','').split(',')
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    if data.ndim == 1 or len(data) == 0: return None

    y_m = data[:, 0]
    if len(header) >= 4 and 'kinematic' in header[1].lower():
        tau_Pa = data[:, 2]
    else:
        tau_Pa = data[:, 1]

    y_star = y_m / Dc
    delta_star = tau_Pa / tau_0
    idx = np.argsort(y_star)
    y_star, delta_star, tau_Pa = y_star[idx], delta_star[idx], tau_Pa[idx]

    pos = y_star >= 0
    return {'y_star': y_star[pos], 'delta_star': delta_star[pos], 'tau_Pa': tau_Pa[pos]}

def load_all_cfd(case_dict):
    """Load all CFD cases from a dict."""
    result = {}
    for label, (fpath, color, ls, lw) in case_dict.items():
        d = load_CFD(fpath)
        if d is not None:
            d['color'], d['ls'], d['lw'] = color, ls, lw
            result[label] = d
    return result

def load_HPC_xsections(xdir, x_vals=None):
    """Load multiple x-section CSVs."""
    if x_vals is None: x_vals = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    result = {}
    for x in x_vals:
        d = load_CFD(os.path.join(xdir, f'wss_x{x}_2000.csv'))
        if d is not None: result[x] = d
    return result

def smooth(y, w=15):
    """Moving average smoothing."""
    if len(y) < w: return y
    k = np.ones(w)/w
    s = np.convolve(y, k, mode='same')
    h = w//2; s[:h] = y[:h]; s[-h:] = y[-h:]
    return s

def bed_center_delta(data, thr=0.3):
    """Average δ* near center (y* < threshold)."""
    if data is None: return np.nan
    m = data['y_star'] < thr
    return np.mean(data['delta_star'][m]) if np.any(m) else np.nan

def make_geometry_half():
    """Generate half-channel D*(y*) from Table 3 polynomial."""
    d = np.linspace(0, 3.5, 500)
    D_bank = np.maximum(a3*d**3 + a2*d**2 + a1*d + a0, 0)
    y_bank = junct_star + d
    # Trim after D* reaches 0
    last_pos = np.searchsorted(-D_bank, 0)  # first index where D=0
    y_flat = np.linspace(0, junct_star, 30)
    D_flat = np.ones(30)
    y_all = np.concatenate([y_flat, y_bank[1:last_pos+1]])
    D_all = np.concatenate([D_flat, D_bank[1:last_pos+1]])
    return y_all, D_all

# ================================================================
# PLOT STYLE
# ================================================================
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'figure.dpi': 150, 'lines.linewidth': 2,
})

# ================================================================
# LOAD DATA
# ================================================================
print("="*70)
print("V&D CFD Validation — Master Plot v3")
print("="*70)
print(f"  Dc={Dc:.3f}m  τ₀={tau_0:.1f}Pa  τ_cr={tau_cr:.1f}Pa  δ*_cr={delta_cr_star}")

vd = load_VD(VD_FILE)
cyc = load_all_cfd(CFD_CYCLIC)
hpc45 = load_all_cfd(CFD_HPC_X45)
hpc_sm_xs = load_HPC_xsections(HPC_SMOOTH_DIR)
hpc_rg_xs = load_HPC_xsections(HPC_ROUGH_DIR)
y_geom, D_geom = make_geometry_half()

print("\nBed center δ* values:")
if vd: print(f"  V&D analytical:  {vd['delta_star'][0]:.3f}")
for lab, d in {**cyc, **hpc45}.items():
    print(f"  {lab:30s}: {bed_center_delta(d):.3f}")


# ================================================================
# FIGURE 1: Master Comparison — D* and δ* (all cases vs V&D)
# ================================================================
print("\n--- Figure 1: Master Comparison ---")
fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(13, 11))

# (a) Geometry D*
if vd: ax1a.plot(vd['y_star'], vd['D_star'], 'k-', lw=3, label='V&D Analytical (MATLAB)', zorder=10)
ax1a.plot(y_geom, D_geom, 'b--', lw=2, label='CFD Geometry (Table 3)', zorder=5)
ax1a.axvline(x=junct_star, color='gray', ls=':', alpha=0.5, lw=1, label=f'Bed-bank junction (y*={junct_star:.2f})')
ax1a.set_xlabel('y*'); ax1a.set_ylabel('D*')
ax1a.set_title('(a) Channel Geometry D*(y*) — Half Channel')
ax1a.legend(loc='lower left'); ax1a.grid(True, alpha=0.2)
ax1a.set_xlim([0, 4]); ax1a.set_ylim([-0.05, 1.15])

# (b) Stress depth δ* — two panels: main + inset for HPC rough
if vd: ax1b.plot(vd['y_star'], vd['delta_star'], 'k-', lw=3, label='V&D Analytical', zorder=10)
for lab, d in cyc.items():
    ax1b.plot(d['y_star'], smooth(d['delta_star'], 11), color=d['color'], ls=d['ls'], lw=d['lw'], label=lab)
for lab, d in hpc45.items():
    if 'Rough' in lab:
        # Plot HPC rough on secondary axis annotation instead of clipping
        dc = bed_center_delta(d)
        ax1b.annotate(f'{lab}\nδ*_center={dc:.2f}\n(off scale)', xy=(0.5, 0.95),
                      fontsize=9, color=d['color'], ha='center',
                      xycoords='axes fraction', va='top',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    else:
        ax1b.plot(d['y_star'], smooth(d['delta_star'], 11), color=d['color'], ls=d['ls'], lw=d['lw'], label=lab)

ax1b.axhline(y=delta_cr_star, color='darkred', ls=':', lw=1.5, alpha=0.7, label=f'δ*_cr = {delta_cr_star}')
ax1b.axvline(x=junct_star, color='gray', ls=':', alpha=0.5, lw=1)
ax1b.set_xlabel('y*'); ax1b.set_ylabel('δ*')
ax1b.set_title('(b) Stress Depth δ*(y*) — All Cases')
ax1b.legend(loc='upper right', ncol=2); ax1b.grid(True, alpha=0.2)
ax1b.set_xlim([0, 4]); ax1b.set_ylim([-0.05, 1.05])

plt.suptitle('Figure 1: All CFD Cases vs V&D Analytical Solution (Half Channel)\n'
             f'μ={mu}, δ*_cr={delta_cr_star}, d₅₀={d50*1000:.0f}mm, S={S}',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Fig1_master_comparison.png'), dpi=200, bbox_inches='tight')
print("  Saved: Fig1_master_comparison.png"); plt.close()


# ================================================================
# FIGURE 2: Cyclic Cases — δ* comparison
# ================================================================
print("\n--- Figure 2: Cyclic Cases ---")
fig2, ax2 = plt.subplots(figsize=(13, 6))

if vd: ax2.plot(vd['y_star'], vd['delta_star'], 'k-', lw=3, label='V&D Analytical', zorder=10)
for lab, d in cyc.items():
    ax2.plot(d['y_star'], smooth(d['delta_star'], 11), color=d['color'], ls=d['ls'], lw=d['lw'], label=lab)

ax2.axhline(y=delta_cr_star, color='darkred', ls=':', lw=1.5, alpha=0.7, label=f'δ*_cr = {delta_cr_star}')
ax2.axvline(x=junct_star, color='gray', ls=':', alpha=0.4, lw=1, label='Bed-bank junction')
ax2.set_xlabel('y*'); ax2.set_ylabel('δ*')
ax2.set_title('Figure 2: Cyclic Channel Cases — δ* Comparison\n'
              'Note: CFD wall patches may not extend to water margin (D*=0)')
ax2.legend(loc='upper right'); ax2.grid(True, alpha=0.2)
ax2.set_xlim([0, 4]); ax2.set_ylim([-0.05, 1.05])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Fig2_cyclic_comparison.png'), dpi=200, bbox_inches='tight')
print("  Saved: Fig2_cyclic_comparison.png"); plt.close()


# ================================================================
# FIGURE 3: Flow Development — δ* and τ_Pa at bed center vs x
# ================================================================
print("\n--- Figure 3: Flow Development ---")
fig3, ((ax3a, ax3b), (ax3c, ax3d)) = plt.subplots(2, 2, figsize=(16, 10))

for xs_data, label, color, ax_delta, ax_tau in [
    (hpc_sm_xs, '50m Smooth', '#2196F3', ax3a, ax3c),
    (hpc_rg_xs, '50m Rough',  '#F44336', ax3b, ax3d),
]:
    if xs_data:
        x_vals = sorted(xs_data.keys())
        d_center = [bed_center_delta(xs_data[x]) for x in x_vals]
        t_center = [np.mean(xs_data[x]['tau_Pa'][xs_data[x]['y_star'] < 0.3]) for x in x_vals]

        ax_delta.plot(x_vals, d_center, 'o-', color=color, lw=2.5, ms=8, label=f'{label} δ* center')
        ax_delta.axhline(y=delta_cr_star, color='darkred', ls=':', lw=1.5, label=f'δ*_cr={delta_cr_star}')
        ax_delta.axhline(y=0.9, color='darkgreen', ls='--', lw=1.5, label='V&D center δ*=0.9')
        ax_delta.set_xlabel('x (m)'); ax_delta.set_ylabel('δ* at bed center')
        ax_delta.set_title(f'{label} — δ* Development'); ax_delta.legend(fontsize=8)
        ax_delta.grid(True, alpha=0.2); ax_delta.set_xlim([0, 50])

        ax_tau.plot(x_vals, t_center, 's-', color=color, lw=2.5, ms=8, label=f'{label} τ center')
        ax_tau.axhline(y=tau_0, color='darkgreen', ls='--', lw=1.5, label=f'τ₀={tau_0:.1f} Pa')
        ax_tau.axhline(y=tau_cr, color='darkred', ls=':', lw=1.5, label=f'τ_cr={tau_cr:.1f} Pa')
        ax_tau.set_xlabel('x (m)'); ax_tau.set_ylabel('τ at bed center (Pa)')
        ax_tau.set_title(f'{label} — τ Development'); ax_tau.legend(fontsize=8)
        ax_tau.grid(True, alpha=0.2); ax_tau.set_xlim([0, 50])
    else:
        for ax in [ax_delta, ax_tau]:
            ax.text(0.5, 0.5, f'{label}\nData not found', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='gray')

plt.suptitle('Figure 3: Flow Development Along 50m Channel', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Fig3_flow_development.png'), dpi=200, bbox_inches='tight')
print("  Saved: Fig3_flow_development.png"); plt.close()


# ================================================================
# FIGURE 4: 50m Cross-Sections — δ* at different x locations
# ================================================================
print("\n--- Figure 4: 50m Cross-Sections ---")
x_show = [0, 10, 25, 45]
colors_x = {0: '#F44336', 10: '#FF9800', 25: '#4CAF50', 45: '#2196F3'}

for smooth_flag, suffix, title_extra in [(True, 'smoothed', '(Smoothed)'), (False, 'raw', '(Raw/Unsmoothed)')]:
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 6))

    for xs_data, label, ax_panel in [
        (hpc_sm_xs, '50m Smooth Wall', ax4a),
        (hpc_rg_xs, '50m Rough Wall',  ax4b),
    ]:
        if vd: ax_panel.plot(vd['y_star'], vd['delta_star'], 'k-', lw=2, label='V&D δ*', zorder=10)

        if xs_data:
            all_delta_vals = []
            for x in x_show:
                if x in xs_data:
                    d = xs_data[x]
                    ds = smooth(d['delta_star'], 11) if smooth_flag else d['delta_star']
                    ax_panel.plot(d['y_star'], ds, color=colors_x[x], lw=1.8, label=f'x={x}m')
                    all_delta_vals.extend(ds.tolist())

            # Auto-scale ylim to show ALL data
            if all_delta_vals:
                ymax = max(all_delta_vals) * 1.1
                ax_panel.set_ylim([-0.05, max(ymax, 1.1)])
            else:
                ax_panel.set_ylim([-0.05, 1.05])
        else:
            ax_panel.set_ylim([-0.05, 1.05])

        ax_panel.axhline(y=delta_cr_star, color='darkred', ls=':', lw=1.5, alpha=0.5,
                        label=f'δ*_cr={delta_cr_star}')
        ax_panel.axvline(x=junct_star, color='gray', ls=':', alpha=0.3)
        ax_panel.set_xlabel('y*'); ax_panel.set_ylabel('δ*')
        ax_panel.set_title(f'{label} {title_extra}')
        ax_panel.legend(fontsize=9); ax_panel.grid(True, alpha=0.2)
        ax_panel.set_xlim([0, 4])

    plt.suptitle(f'Figure 4: Cross-Channel δ* at Different Streamwise Locations {title_extra}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = f'Fig4_hpc_xsections_{suffix}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200, bbox_inches='tight')
    print(f"  Saved: {fname}"); plt.close()


# ================================================================
# FIGURE 5: V&D Style — D* and δ* on same axes (key figure)
# ================================================================
print("\n--- Figure 5: V&D Style Combined ---")
fig5, ax5 = plt.subplots(figsize=(13, 7))

# Geometry
if vd: ax5.plot(vd['y_star'], vd['D_star'], 'k-', lw=3, label='V&D D* (geometry)')
ax5.plot(y_geom, D_geom, 'b-', lw=2, alpha=0.6, label='CFD D* (geometry)')

# V&D stress
if vd: ax5.plot(vd['y_star'], vd['delta_star'], 'k--', lw=3, label='V&D δ* (stress depth)')

# CFD stress — plot smooth, rough 2m, and v2312
for lab in ['Smooth 2m', 'Rough 2m (OF12)', 'Rough 2m (v2312 atmNutk)']:
    if lab in cyc:
        d = cyc[lab]
        ax5.plot(d['y_star'], smooth(d['delta_star'], 11),
                 color=d['color'], ls=d['ls'], lw=d['lw'], label=f'CFD δ* — {lab}')

ax5.axhline(y=delta_cr_star, color='darkred', ls=':', lw=1.5, alpha=0.7,
            label=f'δ*_cr = {delta_cr_star} (threshold)')
ax5.axvline(x=junct_star, color='gray', ls=':', alpha=0.4, label='Bed-bank junction')
ax5.set_xlabel('y*', fontsize=13); ax5.set_ylabel('D*, δ*', fontsize=13)
ax5.set_title('Figure 5: V&D Style — Geometry and Stress Depth on Same Axes\n'
              f'μ={mu}, δ*_cr={delta_cr_star}, Dc={Dc:.3f}m',
              fontsize=14, fontweight='bold')
ax5.legend(fontsize=10, loc='upper right'); ax5.grid(True, alpha=0.2)
ax5.set_xlim([0, 4]); ax5.set_ylim([-0.05, 1.15])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Fig5_VD_style_combined.png'), dpi=200, bbox_inches='tight')
print("  Saved: Fig5_VD_style_combined.png"); plt.close()


# ================================================================
# FIGURE 6: Summary Table
# ================================================================
print("\n--- Figure 6: Summary Table ---")
fig6, ax6 = plt.subplots(figsize=(18, 5))
ax6.axis('off')

headers = ['Case', 'Domain', 'Wall Function', 'dP/dx\n(m/s²)', 'dP/dx\n(% V&D)',
           'Bed δ*\n(center)', 'Recovery\nvs V&D (%)', 'Notes']
gS = g * S

rows = []
vd_center = vd['delta_star'][0] if vd else 0.9

rows.append(['V&D Analytical', '∞ (uniform)', 'Log-law (Keulegan)',
             f'{gS:.4f}', '100%', f'{vd_center:.3f}', '100%', 'Reference'])

# Pressure gradients from logs (manually entered)
dp_dx = {
    'Smooth 2m': 0.0043, 'Rough 2m (OF12)': 0.0154,
    'Rough 4m (OF12)': 0.0132, 'Rough 2m (v2312 atmNutk)': 0.0166,
}

for lab, d in {**cyc, **hpc45}.items():
    dc = bed_center_delta(d)
    dp = dp_dx.get(lab, None)
    dp_str = f'{dp:.4f}' if dp else '—'
    dp_pct = f'{dp/gS*100:.0f}%' if dp else '—'
    rec = f'{dc/vd_center*100:.0f}%' if not np.isnan(dc) else '—'
    notes = ''
    if 'Smooth 2m' in lab: notes = 'No roughness baseline'
    elif 'Rough 2m (OF12)' in lab: notes = 'Best global balance (98%)'
    elif '4m' in lab: notes = 'Mesh sensitivity'
    elif 'v2312' in lab: notes = 'Alt. wall function'
    elif '50m' in lab and 'Smooth' in lab: notes = 'Developing flow'
    elif '50m' in lab and 'Rough' in lab: notes = 'Inlet BC mismatch'
    rows.append([lab, 'cyclic' if 'Cyclic' not in lab and '50m' not in lab else
                 ('2m cyclic' if '2m' in lab else ('4m cyclic' if '4m' in lab else '50m')),
                 'See legend', dp_str, dp_pct, f'{dc:.3f}', rec, notes])

table = ax6.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center')
table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.0, 1.8)
for j in range(len(headers)):
    table[0, j].set_facecolor('#1565C0')
    table[0, j].set_text_props(color='white', fontweight='bold')
    table[1, j].set_facecolor('#FFF9C4')  # V&D row
for i in range(2, len(rows)+1):
    for j in range(len(headers)):
        table[i, j].set_facecolor('#F5F5F5' if i%2==0 else 'white')

ax6.set_title('Figure 6: Results Summary\n'
              f'τ₀={tau_0:.1f}Pa, τ_cr={tau_cr:.1f}Pa, V&D center δ*={vd_center:.3f}',
              fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Fig6_summary_table.png'), dpi=200, bbox_inches='tight')
print("  Saved: Fig6_summary_table.png"); plt.close()


# ================================================================
# DONE
# ================================================================
print(f"\n{'='*70}")
print("FIGURES SAVED:")
print("  Fig1_master_comparison.png     — All cases D* and δ* vs V&D")
print("  Fig2_cyclic_comparison.png     — Cyclic cases δ* overlay")
print("  Fig3_flow_development.png      — δ* and τ vs x (50m cases)")
print("  Fig4_hpc_xsections_smoothed.png — 50m cross-sections (smoothed)")
print("  Fig4_hpc_xsections_raw.png     — 50m cross-sections (raw)")
print("  Fig5_VD_style_combined.png     — D* + δ* on same axes (key figure)")
print("  Fig6_summary_table.png         — Results summary table")
print(f"{'='*70}")
