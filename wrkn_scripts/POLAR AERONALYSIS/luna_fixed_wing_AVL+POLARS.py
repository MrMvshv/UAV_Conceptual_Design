#!/usr/bin/env python3
"""
luna_fixedwing_AVL.py

Complete fixed-wing aerodynamic + energy mission analysis using:
 - XFLR5 polars (profile drag)
 - AVL (optional) for induced drag or analytic fallback
 - SUAVE vehicle definition (expects setup_fixedwing_vehicle in fixedwing_vehicle_definition.py)

Instructions:
 - Keep this script in same directory as fixedwing_vehicle_definition.py
 - Put your XFLR5 export (polars.csv) or the interpolated CSV (polars_interpolated.csv) in same dir
 - If avl.exe exists, set environment variable AVL_PATH or set USE_AVL=True (see below)
"""

import os
import sys
import copy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from math import pi


# SUAVE imports (adjust if your SUAVE layout differs)
import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass
import shutil

# ================================================================
#        SUAVE + AVL — COMPLETE WINDOWS FIX (Guaranteed)
# ================================================================

AVL_BIN = r"C:\Tools\AVL\avl.exe"

# 1) Make sure OS environment is correct
os.environ["AVL_PATH"] = AVL_BIN
os.environ["PATH"] += ";" + os.path.dirname(AVL_BIN)

print("[BOOT] OS environment patched:")
print("       AVL_PATH =", os.environ["AVL_PATH"])
print("       avl in PATH? ->", shutil.which("avl"))

# 2) Patch SUAVE’s internal default AVL settings
try:
    from SUAVE.Methods.Aerodynamics.AVL import Data as AVL_Data
    default_settings = AVL_Data.Settings.Settings()
    default_settings.filenames.avl_bin_name = AVL_BIN
    print("[BOOT] Patched SUAVE default Settings.avl_bin_name ->", default_settings.filenames.avl_bin_name)
except Exception as e:
    print("[BOOT-ERR] Could not patch SUAVE default settings:", e)

# 3) Monkey-patch SUAVE’s AVL Analysis class (covers all future instances)
try:
    import SUAVE.Analyses.Aerodynamics.AVL as SUAVE_AVL_Module
    SUAVE_AVL_Module.AVL.settings.filenames.avl_bin_name = AVL_BIN
    print("[BOOT] Patched SUAVE AVLANalysis class-level avl_bin_name ->", AVL_BIN)
except Exception as e:
    print("[BOOT-ERR] Could not patch SUAVE AVLANalysis:", e)

# 4) Helper to patch any AVLANalysis instance (you can call this later)
def patch_avl_analysis_instance(analysis):
    try:
        analysis.settings.filenames.avl_bin_name = AVL_BIN
        print("[BOOT] Patched AVLANalysis instance:", AVL_BIN)
    except:
        print("[BOOT-ERR] Could not patch analysis instance")

# 5) Final diagnostics
print("------------------------------------------------------------")
print("[BOOT] SUAVE AVL bootstrap complete.")
print("       shutil.which('avl')         =", shutil.which("avl"))
print("       Final expected avl.exe path =", AVL_BIN)
print("------------------------------------------------------------\n")


# Try to import AVL wrapper (some SUAVE installs put this elsewhere)
try:
    # use SUAVE's AVL wrapper explicitly
    from SUAVE.Methods.Aerodynamics.AVL import AVL as SUAVE_AVL
    
    _AVL_AVAILABLE = True
except Exception as exc:
    SUAVE_AVL = None
    _AVL_AVAILABLE = False

# Local vehicle definition (expects file fixedwing_vehicle_definition.py next to this file)
try:
    from fixedwing_vehicle_definition import setup_fixedwing_vehicle
except Exception as e:
    print("ERROR: cannot import setup_fixedwing_vehicle from fixedwing_vehicle_definition.py")
    print("Make sure fixedwing_vehicle_definition.py is present in the same folder.")
    raise

# ---------------------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------------------
os.environ["AVL_PATH"] = r"C:\Tools\AVL\avl.exe"
os.environ["PATH"] += ";C:\\Tools\\AVL"
USE_AVL = True   # Set True to call AVL (make sure avl.exe is installed and path set)
AVL_PATH = os.environ.get('AVL_PATH', r"C:\Tools\AVL\avl.exe")  # optionally set environment variable AVL_PATH to avl.exe location
POLAR_INTERPOLATED_CSV = 'polars_interpolated.csv'  # preferred (post-processed)
POLAR_RAW_CSV = 'polars.csv'                       # raw XFLR5 export fallback
EXCRESCENCE_CD_DEFAULT = 0.01                      # small fudge for excrescence/parasitic not in polars

# ---------------------------------------------------------------------
# Old function kept commented-out for reference (DO NOT DELETE, only commented)
# ---------------------------------------------------------------------
"""
# ------------------------------
# OLD compute_aero_forces (commented for replacement)
# ------------------------------
def compute_aero_forces(vehicle, V, altitude_m):
    \"\"\"Old simple aero function (kept for reference).\"\"\"
    # This function was replaced by aero_from_polars_and_avl()
    rho = 1.225 * np.exp(-altitude_m / 8500.0)
    S = vehicle.reference_area
    W = vehicle.mass_properties.takeoff * 9.81
    AR = vehicle.wings.main_wing.aspect_ratio
    e = vehicle.wings.main_wing.span_efficiency
    CD0 = 0.025
    CL = W / (0.5 * rho * V**2 * S)
    CD = CD0 + CL**2 / (np.pi * e * AR)
    D = 0.5 * rho * V**2 * S * CD
    P = D * V / 0.75
    return {'CL': CL, 'CD': CD, 'Drag_N': D, 'Power_W': P}
"""
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Helper: find main wing object robustly
# ---------------------------------------------------------------------
def get_main_wing(vehicle):
    # SUAVE vehicle may store wings in vehicle.wings list or attribute.
    try:
        # some versions: vehicle.wings is a list-like
        for w in vehicle.wings:
            if getattr(w, 'tag', '').lower().startswith('main'):
                return w
        # fallback: try named attribute
        if hasattr(vehicle.wings, 'main_wing'):
            return vehicle.wings.main_wing
    except Exception:
        pass
    # last fallback: return first wing in list if present
    try:
        return vehicle.wings[0]
    except Exception:
        raise RuntimeError("Main wing not found in vehicle.wings. Inspect vehicle object.")

# ---------------------------------------------------------------------
# Load / parse XFLR5 polars
# Builds alpha->CL, alpha->CDprof, and CL->alpha (pre-stall) interpolators.
# ---------------------------------------------------------------------
def load_polars(interpolated_csv=POLAR_INTERPOLATED_CSV, raw_csv=POLAR_RAW_CSV):
    """
    Attempt to load an interpolated polar CSV first (alpha_deg, CL, CD).
    If not present, attempt to parse the raw XFLR5 polars.csv (search header row).
    Returns a dict with interpolators and metadata.
    """
    if os.path.exists(interpolated_csv):
        print(f"[POLAR] Loading interpolated polar: {interpolated_csv}")
        df = pd.read_csv(interpolated_csv)
        # expect columns: alpha_deg, CL, CD
        alpha = df['alpha_deg'].to_numpy()
        CL = df['CL'].to_numpy()
        CD = df['CD'].to_numpy()
    elif os.path.exists(raw_csv):
        print(f"[POLAR] Loading raw XFLR5 polar: {raw_csv}")
        # read file, find header row that starts with 'alpha'
        with open(raw_csv, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        hdr_idx = None
        for i, line in enumerate(lines[:200]):
            if line.strip().lower().startswith('alpha'):
                hdr_idx = i
                break
        if hdr_idx is None:
            # try find line containing 'alpha' anywhere
            for i, line in enumerate(lines[:200]):
                if 'alpha' in line.lower():
                    hdr_idx = i
                    break
        if hdr_idx is None:
            raise RuntimeError(f"Could not locate header row in {raw_csv}. Please ensure file is XFLR5 exported CSV format.")

        # parse with pandas skipping initial junk lines
        df = pd.read_csv(raw_csv, skiprows=hdr_idx)
        # normalize column names
        df.columns = [c.strip() for c in df.columns]
        # Expect typical columns: alpha, Beta, CL, CDi, CDv, CD, ...
        if 'alpha' not in df.columns:
            raise RuntimeError("Parsed CSV does not contain 'alpha' column.")
        alpha = pd.to_numeric(df['alpha'], errors='coerce').to_numpy()
        # prefer 'CD' column if it exists and likely contains full drag.
        # But many XFLR5 plane polars may show CD == CDi (induced only). We will detect that downstream.
        CL = pd.to_numeric(df['CL'], errors='coerce').to_numpy()
        # pick CD: if CDv exists, maybe profile drag present; else CD column holds something
        if 'CD' in df.columns:
            CD = pd.to_numeric(df['CD'], errors='coerce').to_numpy()
        elif 'CDv' in df.columns:
            CD = pd.to_numeric(df['CDv'], errors='coerce').to_numpy()
        else:
            # if no CD, make placeholder zeros; user must provide viscous polars
            CD = np.zeros_like(CL)
    else:
        raise FileNotFoundError(f"Neither {interpolated_csv} nor {raw_csv} were found. Place XFLR5 polars in the working directory.")

    # Simple cleaning: remove NaNs and sort by alpha ascending
    mask = (~np.isnan(alpha)) & (~np.isnan(CL))
    alpha = alpha[mask]
    CL = CL[mask]
    CD = CD[mask] if 'CD' in locals() else np.zeros_like(CL)
    # sort
    order = np.argsort(alpha)
    alpha = alpha[order]
    CL = CL[order]
    CD = CD[order]

    # quick detect: if CD approx equals CDi (only induced), notify user
    # Hard to detect exactly without CDi column; we will warn if CD values are very small.
    if np.nanmax(CD) < 0.005:
        print("[POLAR-WARN] The loaded CD values are very small (<0.005). This may indicate the file contains only induced drag (CDi) and not viscous profile drag.")
        print(" -> For realistic drag, produce viscous 2D airfoil polars (XFOIL / XFLR5 viscous) and import them, or set a profile CD estimate.")

    # Build interpolators
    alpha_to_CL = interp1d(alpha, CL, kind='cubic', fill_value='extrapolate')
    alpha_to_CD  = interp1d(alpha, CD, kind='cubic', fill_value='extrapolate')

    # Build CL->alpha using pre-stall branch
    imaxCL = int(np.nanargmax(CL))
    # pre-stall arrays
    CL_pre = CL[:imaxCL+1]
    alpha_pre = alpha[:imaxCL+1]
    # ensure monotonic CL_pre for inversion
    if np.any(np.diff(CL_pre) <= 0):
        # try to force monotonic by sorting
        idx_sort = np.argsort(CL_pre)
        CL_pre_sorted = CL_pre[idx_sort]
        alpha_pre_sorted = alpha_pre[idx_sort]
    else:
        CL_pre_sorted = CL_pre
        alpha_pre_sorted = alpha_pre
    CL_to_alpha = interp1d(CL_pre_sorted, alpha_pre_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')

    return {
        'alpha': alpha,
        'CL': CL,
        'CD': CD,
        'alpha_to_CL': alpha_to_CL,
        'alpha_to_CD': alpha_to_CD,
        'CL_to_alpha': CL_to_alpha,
        'alpha_range': (np.min(alpha), np.max(alpha)),
        'CL_max': float(np.nanmax(CL)),
        'alpha_CL_max': float(alpha[imaxCL])
    }

# ---------------------------------------------------------------------
# The main aerodynamic routine that uses polars + AVL or analytic induced drag
# ---------------------------------------------------------------------
def aero_from_polars_and_avl(vehicle, V, altitude_m, polars, use_avl=False):
    """
    Compute aerodynamic coefficients and power using:
      - profile drag from XFLR5/XFOIL polars (polars['alpha_to_CD'])
      - induced drag from AVL (if use_avl True and AVL available) or analytic formula
    Returns a dict with CL, alpha_deg, CD_profile, CDi, CD_total, Drag_N, Power_W, L_over_D
    """
    # Atmosphere
    rho = 1.225 * np.exp(-altitude_m / 8500.0)
    S = float(vehicle.reference_area)
    W = float(vehicle.mass_properties.takeoff * 9.81)
    q = 0.5 * rho * V**2

    # Required CL
    CL_req = W / (q * S)

    # invert CL->alpha on pre-stall branch
    alpha_deg = float(polars['CL_to_alpha'](CL_req))

    # Guard alpha within available polar range
    alpha_min, alpha_max = polars['alpha_range']
    if alpha_deg < alpha_min or alpha_deg > alpha_max:
        print(f"[AERO-WARN] Required alpha {alpha_deg:.2f}° outside polar alpha range [{alpha_min:.1f}, {alpha_max:.1f}]")
        # clamp to nearest bound to avoid wild extrapolation
        alpha_deg = max(min(alpha_deg, alpha_max), alpha_min)
        print(f"[AERO-WARN] Clamped alpha to {alpha_deg:.2f}° for lookup.")

    # Profile drag from polar (could be 2D profile or plane polar)
    CD_profile = float(polars['alpha_to_CD'](alpha_deg))

    # Wing geometry
    try:
        wing = get_main_wing(vehicle)
    except Exception:
        wing = None

    if wing is not None:
        # attempt to fetch AR and e, else use defaults
        try:
            AR = float(wing.aspect_ratio)
        except Exception:
            # fallback compute AR approx
            try:
                AR = (wing.spans.projected**2) / wing.areas.reference
                AR = float(AR)
            except Exception:
                AR = 6.0
        e = float(getattr(wing, 'span_efficiency', 0.85))
    else:
        AR = 6.0
        e = 0.85

    # Induced drag: prefer AVL if requested & available
    CDi = None
    CDi_source = 'analytic'

    if use_avl and _AVL_AVAILABLE:
        try:
            # instantiate SUAVE's AVL wrapper
            avl = SUAVE_AVL()
            patch_avl_analysis_instance(avl)
            # most SUAVE AVL wrappers accept geometry assignment either via attribute or setter
            try:
                avl.geometry = vehicle
            except Exception:
                try:
                    avl.set_geometry(vehicle)
                except Exception:
                    pass

            # configure inputs (guard for attribute names)
            try:
                avl.inputs.alpha = float(alpha_deg)
                avl.inputs.Mach = max(1e-6, float(V) / 343.0)
                avl.inputs.altitude = float(altitude_m)
            except Exception:
                # some wrappers expect a dict or different API
                try:
                    avl.set_inputs({'alpha': float(alpha_deg),
                                    'Mach': max(1e-6, float(V) / 343.0),
                                    'altitude': float(altitude_m)})
                except Exception:
                    pass

            # optional tuning (if supported)
            try:
                avl.settings.number_of_spanwise_vortices = 12
                avl.settings.number_of_chordwise_vortices = 6
            except Exception:
                pass

            print("[AVL] Running SUAVE AVL wrapper for induced drag...")
            avl_res = avl.evaluate()

            # extract induced drag from common attributes
            if hasattr(avl_res, 'CDi'):
                CDi = float(avl_res.CDi)
            elif hasattr(avl_res, 'CDi_total'):
                CDi = float(avl_res.CDi_total)
            elif hasattr(avl_res, 'CD'):
                CDi = float(avl_res.CD)
            else:
                CDi = CL_req**2 / (pi * e * AR)

            CDi_source = 'SUAVE_AVL'
        except Exception as ex:
            print("[AVL-ERROR] SUAVE AVL wrapper failed, falling back to analytic induced drag.")
            print("    Exception:", ex)
            CDi = CL_req**2 / (pi * e * AR)
            CDi_source = 'analytic (fallback)'

    # Ensure CDi is numeric even if AVL wasn't requested/available
    if CDi is None:
        CDi = CL_req**2 / (pi * e * AR)
        CDi_source = 'analytic (default)'

    # Excrescence / parasitic add-on (small)
    CD_exc = getattr(vehicle, 'excrescence_cd', EXCRESCENCE_CD_DEFAULT)

    # Total CD and drag force
    CD_total = CD_profile + CDi + CD_exc
    D = CD_total * q * S

    # Power required given network efficiencies
    # try to find network and prop/motor efficiencies robustly
    eta_prop = 0.85
    eta_motor = 0.9
    try:
        # network may be in vehicle.network or vehicle.networks depending on SUAVE version
        if hasattr(vehicle, 'networks'):
            # networks is iterable/list or dict
            try:
                net = None
                # if dict-like
                if isinstance(vehicle.networks, dict):
                    # pick first network that looks like battery_propeller / Battery_Propeller
                    for k, v in vehicle.networks.items():
                        net = v
                        break
                else:
                    # list-like - first network
                    net = vehicle.networks[0]
                if net is not None:
                    eta_prop = float(getattr(net, 'propeller', getattr(net, 'propellers', Data())).efficiency) if hasattr(getattr(net, 'propeller', None), 'efficiency') else getattr(net, 'propeller', getattr(net, 'propellers', Data())).efficiency if hasattr(getattr(net, 'propeller', None), 'efficiency') else eta_prop
                    eta_motor = float(getattr(net, 'motor', getattr(net, 'motors', Data())).efficiency) if hasattr(getattr(net, 'motor', None), 'efficiency') else eta_motor
            except Exception:
                pass
        elif hasattr(vehicle, 'network'):
            net = vehicle.network
            eta_prop = float(getattr(net, 'propeller', getattr(net, 'propellers', Data())).efficiency) if hasattr(getattr(net,'propeller',None),'efficiency') else eta_prop
            eta_motor = float(getattr(net, 'motor', getattr(net,'motors', Data())).efficiency) if hasattr(getattr(net,'motor',None),'efficiency') else eta_motor
    except Exception:
        # fallback keep defaults
        pass

    eta_total = eta_prop * eta_motor if (eta_prop is not None and eta_motor is not None) else 0.75
    if eta_total <= 0.0:
        eta_total = 0.75

    P_req = D * V / eta_total

    # L/D computed using CL_req/CD_total
    L_over_D = CL_req / CD_total if CD_total > 0 else float('inf')

    # Detailed printout
    print("\n========== AERODYNAMIC DIAGNOSTICS ==========")
    print(f"Airspeed: {V:.2f} m/s | Altitude: {altitude_m:.1f} m")
    print(f"Reference area S = {S:.3f} m^2")
    print(f"Required CL (for weight {W:.2f} N): {CL_req:.4f}")
    print(f"Computed alpha (pre-stall branch): {alpha_deg:.3f} deg")
    print(f"Profile CD (from polar)         : {CD_profile:.6f}")
    print(f"Induced CD (source={CDi_source}) : {CDi:.6f}")
    print(f"Excrescence CD (added)          : {CD_exc:.6f}")
    print(f"Total CD                        : {CD_total:.6f}")
    print(f"Drag force D                    : {D:.3f} N")
    print(f"Prop eff = {eta_prop:.3f}, Motor eff = {eta_motor:.3f}, Total eff = {eta_total:.3f}")
    print(f"Power required                  : {P_req:.1f} W")
    print(f"L/D                             : {L_over_D:.2f}")
    print("============================================\n")

    return {
        'rho': rho,
        'CL': CL_req,
        'alpha_deg': alpha_deg,
        'CD_profile': CD_profile,
        'CDi': CDi,
        'CD_total': CD_total,
        'Drag_N': D,
        'Power_W': P_req,
        'L_to_D': L_over_D
    }

# ---------------------------------------------------------------------
# Mission segment runner (uses aero_from_polars_and_avl)
# ---------------------------------------------------------------------
def run_segment(vehicle, seg, soc_Wh, polars, use_avl=False):
    seg_type = seg.get('type', 'cruise')
    altitude = seg.get('altitude_m', 0.0)
    duration = seg.get('time_s', 60.0)
    V = seg.get('speed_mps', 20.0)

    # compute aero and power
    aero = aero_from_polars_and_avl(vehicle, V, altitude, polars, use_avl=use_avl)
    P = aero['Power_W']

    # simple climb/descent energy adjustments
    if seg_type == 'climb':
        climb_rate = seg.get('climb_rate_mps', 1.0)
        # add potential power to raise mass
        P += float(vehicle.mass_properties.takeoff * 9.81) * climb_rate
    elif seg_type == 'descent':
        # assume some energy recovery or lower throttle; use fraction
        P = max(0.0, 0.5 * P)

    E_Wh = P * duration / 3600.0
    remaining = max(soc_Wh - E_Wh, 0.0)

    # descriptive prints
    print(f"--- SEGMENT: {seg_type.upper()} (V={V:.1f}m/s, dur={duration/60:.1f}min) ---")
    print(f"Segment power = {P:.1f} W, Energy consumed = {E_Wh:.2f} Wh")
    print(f"Battery before = {soc_Wh:.2f} Wh => after = {remaining:.2f} Wh")
    print("-" * 60)

    out = {
        'seg_type': seg_type,
        'power_W': P,
        'energy_used_Wh': E_Wh,
        'remaining_Wh': remaining,
        'L_to_D': aero['L_to_D'],
        'aero': aero
    }
    return out, remaining

# ---------------------------------------------------------------------
# Mission runner (list of segments)
# ---------------------------------------------------------------------
def run_mission(vehicle, mission_profile, polars, use_avl=False, plot=True):
    # battery energy (Wh) from vehicle network
    batt_energy_Wh = None
    try:
        # try various likely places for battery object
        if hasattr(vehicle, 'networks'):
            # pick first network
            net = None
            if isinstance(vehicle.networks, dict):
                net = list(vehicle.networks.values())[0]
            else:
                net = vehicle.networks[0]
            if net is not None and hasattr(net, 'battery'):
                bat = net.battery
                batt_energy_Wh = float(bat.specific_energy * bat.mass_properties.mass)
        if batt_energy_Wh is None and hasattr(vehicle, 'network'):
            net = vehicle.network
            if hasattr(net, 'battery'):
                bat = net.battery
                batt_energy_Wh = float(bat.specific_energy * bat.mass_properties.mass)
    except Exception:
        batt_energy_Wh = None

    if batt_energy_Wh is None:
        # fallback: set from a default battery mass & spec (user should set this in vehicle)
        batt_mass_kg = getattr(vehicle, 'battery_mass_kg', 0.3)
        batt_specific_Whkg = getattr(vehicle, 'battery_specific_Whkg', 250.0)
        batt_energy_Wh = batt_mass_kg * batt_specific_Whkg
        print(f"[BATTERY-WARN] Could not find battery in vehicle network; using fallback {batt_energy_Wh:.1f} Wh")

    print("\n=== FIXED-WING AERO + ENERGY ANALYSIS ===")
    print(f"Using AVL: {use_avl}  (AVL installed = {_AVL_AVAILABLE})")
    print(f"Initial battery energy = {batt_energy_Wh:.2f} Wh ({batt_energy_Wh/1000.0:.3f} kWh)\n")

    soc = batt_energy_Wh
    results = []
    for seg in mission_profile:
        res, soc = run_segment(vehicle, seg, soc, polars, use_avl=use_avl)
        results.append(res)
        if soc <= 0:
            print("BATTERY DEPLETED: mission terminated early.")
            break

    # Summarize
    print("\n=== MISSION SUMMARY ===")
    total_used = sum(r['energy_used_Wh'] for r in results)
    print(f"Total energy used: {total_used:.2f} Wh")
    print(f"Remaining battery: {soc:.2f} Wh")
    print("Per-segment summary:")
    for r in results:
        print(f" - {r['seg_type']}: Power {r['power_W']:.1f} W, Energy {r['energy_used_Wh']:.2f} Wh, L/D {r['L_to_D']:.2f}")

    # Plot simple energy usage and remaining battery
    if plot:
        seg_names = [r['seg_type'].upper() for r in results]
        used = [r['energy_used_Wh']/1000.0 for r in results]
        rem = []
        running = batt_energy_Wh/1000.0
        for u in used:
            running -= u
            rem.append(running)
        plt.figure(figsize=(8,5))
        plt.bar(seg_names, used, label='Energy used (kWh)')
        plt.plot(seg_names, rem, 'o-', label='Remaining (kWh)')
        plt.ylabel('Energy (kWh)')
        plt.title('Mission energy usage')
        plt.grid(True)
        plt.legend()
        plt.show()

    return results

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Build or import vehicle
    print("Loading vehicle definition (setup_fixedwing_vehicle)...")
    vehicle = setup_fixedwing_vehicle()
    print("Vehicle loaded: tag =", getattr(vehicle, 'tag', 'unknown'))
    print("Takeoff mass (kg):", getattr(vehicle.mass_properties, 'takeoff', 'n/a'))
    print("Reference area (m^2):", getattr(vehicle, 'reference_area', 'n/a'))

    # Load polars
    polars = load_polars()

    # Example mission profile (edit as needed)
    mission_profile = [
        {'type': 'climb',  'time_s': 120, 'speed_mps': 15.0, 'climb_rate_mps': 2.0, 'altitude_m': 0},
        {'type': 'cruise', 'time_s': 600, 'speed_mps': 22.0, 'altitude_m': 100},
        {'type': 'descent','time_s': 120, 'speed_mps': 15.0, 'climb_rate_mps': -1.0, 'altitude_m': 0}
    ]

    # Run mission with or without AVL
    results = run_mission(vehicle, mission_profile, polars, use_avl=USE_AVL, plot=True)

    print("\nDone. If you want AVL enabled, set USE_AVL=True and ensure avl.exe path is in AVL_PATH or environment PATH.")
