# analyze_polars.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# adjust path
csv_path = "polars.csv"

# find header row that starts with 'alpha'
with open(csv_path, 'r', encoding='utf8', errors='ignore') as f:
    lines = f.readlines()
hdr_idx = None
for i,l in enumerate(lines[:60]):
    if l.strip().lower().startswith('alpha'):
        hdr_idx = i
        break
if hdr_idx is None:
    raise RuntimeError("Header row with 'alpha' not found near top of file.")

df = pd.read_csv(csv_path, skiprows=hdr_idx)
df.columns = [c.strip() for c in df.columns]

# show summary
print(df.columns)
print(df[['alpha','CL','CDi','CDv','CD','Cm']].head())

# basic diagnostics
alpha = df['alpha'].to_numpy()
CL    = df['CL'].to_numpy()
CD    = df['CD'].to_numpy()
CDi   = df['CDi'].to_numpy() if 'CDi' in df.columns else np.zeros_like(CD)
CDv   = df['CDv'].to_numpy() if 'CDv' in df.columns else np.zeros_like(CD)

# detect if CD looks like just CDi
if np.allclose(CD, CDi, atol=1e-6):
    print("NOTICE: CD equals CDi everywhere -> viscous (profile) drag appears missing.")

# compute CL slope in linear region (-4..8 deg)
mask = (alpha >= -4) & (alpha <= 8)
p = np.polyfit(alpha[mask], CL[mask], 1)
CL_slope_per_deg = p[0]
CL_slope_per_rad = CL_slope_per_deg * (180.0/np.pi)
print(f"CL_slope ≈ {CL_slope_per_deg:.4f} per deg ≈ {CL_slope_per_rad:.3f} per rad")
print(f"CL_max = {np.nanmax(CL):.3f} at alpha = {alpha[np.nanargmax(CL)]:.1f}°")
LD = CL / np.where(CD<=0, np.nan, CD)
idx = np.nanargmax(LD)
print(f"Max L/D ~ {LD[idx]:.2f} at alpha {alpha[idx]} deg  (if CD missing viscous, L/D is optimistic)")

# plots
plt.figure(); plt.plot(alpha, CL, '-o'); plt.xlabel('alpha (deg)'); plt.ylabel('CL'); plt.grid()
plt.figure(); plt.plot(alpha, CD, '-o'); plt.xlabel('alpha (deg)'); plt.ylabel('CD'); plt.grid()
plt.figure(); plt.plot(CL, CD, '-o'); plt.xlabel('CL'); plt.ylabel('CD'); plt.grid()
plt.figure(); plt.plot(alpha, CL/CD, '-o'); plt.xlabel('alpha (deg)'); plt.ylabel('L/D'); plt.grid()
plt.show()

# Build interpolators (cubic for alpha->CL, alpha->CD). For CL->alpha use pre-stall branch only.
alpha_to_CL = interp1d(alpha, CL, kind='cubic', fill_value='extrapolate')
alpha_to_CD = interp1d(alpha, CD, kind='cubic', fill_value='extrapolate')

# Build CL->alpha on pre-stall branch (alpha <= alpha_at_CLmax)
alpha_CLmax = alpha[np.nanargmax(CL)]
mask_pre = alpha <= alpha_CLmax
CLpre = CL[mask_pre]; alphapre = alpha[mask_pre]
order = np.argsort(CLpre)
CLpre_sorted = CLpre[order]; alphapre_sorted = alphapre[order]
CL_to_alpha = interp1d(CLpre_sorted, alphapre_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')

# Save interpolated table to CSV for later SUAVE import
alpha_grid = np.linspace(alpha.min(), alpha.max(), 201)
out = pd.DataFrame({'alpha_deg': alpha_grid,
                    'CL': alpha_to_CL(alpha_grid),
                    'CD': alpha_to_CD(alpha_grid)})
out.to_csv('polars_interpolated.csv', index=False)
print("Wrote polars_interpolated.csv")
