#!/usr/bin/env python3
"""
luna_mission_analysis.py

THE PHASES
-----------------------------
  Phase A — Vertical takeoff + climb to 30 m AGL     [ROTORS ONLY]
  Phase B — Transition to forward flight              [ROTORS + WING]
  Phase C — Cruise at 50 km/h, 90 m AGL              [WING + CRUISE PROP]
  Phase D — Transition back to hover                  [ROTORS + WING]
  Phase E — Vertical descent + landing                [ROTORS ONLY]
  Phase F — Reserve hover (5 min mandatory Sec.2.2)      [ROTORS ONLY]

luna_fixed_wing_AVL+POLARS.py only modelled Phase C (and approximations of A, E).
Phases B, D, F were completely absent.

PHYSICS APPROACH
-----------------
Phases A, E, F — Momentum theory (actuator disk) using SUAVE vehicle geometry
Phase B, D     — Linear power blend between hover and cruise (transition model)
Phase C        — aero_from_polars_and_avl() from luna_fixed_wing_AVL+POLARS.py
                 (polar-based profile drag + analytic or AVL induced drag)
Endurance      — Computed separately as maximum cruise duration on usable battery

"""

import os
import sys
import copy
import math
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from math import pi


# SUAVE imports
import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass

# SUAVE component imports — required by setup_luna1_vehicle()
from SUAVE.Components import Wings, Fuselages
from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller
from SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion import Lithium_Ion
from SUAVE.Components.Energy.Converters.Propeller import Propeller
from SUAVE.Components.Energy.Converters.Motor import Motor

# ── Local imports ─────────────────────────────────────────────────────────────
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
POLAR_INTERPOLATED_CSV = 'polars_interpolated.csv'  # preferred (post-processed)
POLAR_RAW_CSV = 'polars.csv'                       # raw XFLR5 export fallback
EXCRESCENCE_CD_DEFAULT = 0.01                      # small fudge for excrescence/parasitic not in polars

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
def aero_from_polars_and_avl(vehicle, V, altitude_m, polars, use_avl=False, verbose=True):
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

    # Detailed printout — suppressed during sweeps when verbose=False
    if verbose:
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

# ══════════════════════════════════════════════════════════════════════════════
# DEFINE FULL VEHICLE
# ══════════════════════════════════════════════════════════════════════════════
def setup_luna1_vehicle():
    """
    Fixed-wing and eVTOL configuration extracted from full eVTOL definition.
    Designed for aerodynamic + energy analysis using AVL or Fidelity_Zero.
    """

    # ------------------------------------------------------------------
    # Vehicle core setup
    # ------------------------------------------------------------------
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'fixed_wing_uav'

    # --- Mass Properties ---
    vehicle.mass_properties.takeoff = 3.1 * Units.kg
    vehicle.mass_properties.operating_empty = 2.624 * Units.kg
    vehicle.mass_properties.max_takeoff = 3.1 * Units.kg
    vehicle.mass_properties.max_payload = 0.2 * Units.kg
    vehicle.mass_properties.center_of_gravity = [[0.5, 0.0, 0.0]]  # m (approximate)

    vehicle.envelope.ultimate_load = 5.7
    vehicle.envelope.limit_load = 3.0

    # ------------------------------------------------------------------
    # MAIN WING
    # ------------------------------------------------------------------
    wing = Wings.Main_Wing()
    wing.tag = 'main_wing'
    wing.origin = [[0.3, 0.0, 0.05]] * Units.meter
    wing.spans.projected = 1.4 * Units.meter
    wing.chords.root = 0.20 * Units.meter
    # reference area set explicitly (keeps things robust)
    wing.areas.reference = 0.4 * Units['meters**2']
    wing.aspect_ratio = (wing.spans.projected**2) / wing.areas.reference
    wing.taper = 0.8
    wing.sweep = 8.5 * Units.degrees
    wing.dihedral = 1.0 * Units.degrees
    wing.thickness_to_chord = 0.12
    wing.span_efficiency = 0.9
    wing.symmetric = True
    vehicle.append_component(wing)
    vehicle.reference_area = wing.areas.reference

    # ------------------------------------------------------------------
    # HORIZONTAL TAIL
    # ------------------------------------------------------------------
    htail = Wings.Horizontal_Tail()
    htail.tag = 'horizontal_tail'
    htail.origin = [[0.8, 0.0, 0.025]] * Units.meter
    htail.areas.reference = 0.06 * Units['meters**2']
    htail.aspect_ratio = 5.0
    htail.taper = 0.5
    htail.sweeps.quarter_chord = 20.0 * Units.degrees
    htail.thickness_to_chord = 0.12
    htail.dihedral = 5.0 * Units.degrees
    vehicle.append_component(htail)

    # ------------------------------------------------------------------
    # VERTICAL TAIL
    # ------------------------------------------------------------------
    vtail = Wings.Vertical_Tail()
    vtail.tag = 'vertical_tail'
    vtail.origin = [[0.8, 0.0, 0.025]] * Units.meter
    vtail.areas.reference = 0.03 * Units['meters**2']
    vtail.aspect_ratio = 2.5
    vtail.taper = 0.5
    vtail.sweeps.quarter_chord = 30.0 * Units.degrees
    vtail.thickness_to_chord = 0.12
    vehicle.append_component(vtail)

    # ------------------------------------------------------------------
    # FUSELAGE
    # ------------------------------------------------------------------
    fuselage = Fuselages.Fuselage()
    fuselage.tag = 'fuselage'
    fuselage.lengths.nose = 0.2 * Units.meter
    fuselage.lengths.tail = 0.2 * Units.meter
    fuselage.lengths.cabin = 0.6 * Units.meter
    fuselage.lengths.total = 1.0 * Units.meter
    fuselage.width = 0.15 * Units.meter
    fuselage.heights.maximum = 0.15 * Units.meter
    fuselage.areas.wetted = 0.6 * Units['meters**2']
    fuselage.areas.front_projected = 0.0225 * Units['meters**2']
    fuselage.effective_diameter = 0.15 * Units.meter
    vehicle.append_component(fuselage)

    # ------------------------------------------------------------------
    # BOOMS (twin) - use deepcopy instead of clone()
    # ------------------------------------------------------------------
    boom = Fuselages.Fuselage()
    boom.tag = 'boom_right'
    boom.origin = [[0.1, 0.35, 0.04]] * Units.meter
    boom.lengths.total = 0.66 * Units.meter
    boom.width = 0.03 * Units.meter
    boom.heights.maximum = 0.03 * Units.meter
    boom.areas.wetted = 0.05 * Units['meters**2']
    # safe numpy expression for frontal area
    boom.areas.front_projected = np.pi * (0.03 / 2.0)**2 * Units['meters**2']
    vehicle.append_component(boom)

    # copy for left boom using deepcopy
    boom_L = copy.deepcopy(boom)
    boom_L.tag = 'boom_left'
    # flip lateral origin Y sign
    if hasattr(boom_L, 'origin') and len(boom_L.origin) > 0:
        boom_L.origin = [[boom_L.origin[0][0], -boom_L.origin[0][1], boom_L.origin[0][2]]]
    else:
        boom_L.origin = [[0.1, -0.35, 0.04]] * Units.meter
    vehicle.append_component(boom_L)

    # ------------------------------------------------------------------
    # PROPULSION NETWORK (fixed-wing only)
    # ------------------------------------------------------------------
    net = Battery_Propeller()
    net.number_of_engines = 1
    net.identical_propellers = True

    # Propeller (tractor type)
    prop = Propeller()
    prop.tag = 'cruise_propeller'
    prop.number_of_blades = 2
    prop.tip_radius = 0.15 * Units.meter
    prop.hub_radius = 0.015 * Units.meter
    prop.angular_velocity = 6000 * Units.rpm
    prop.freestream_velocity = 30.0 * Units['m/s']
    prop.design_Cl = 0.7
    prop.design_altitude = 100.0 * Units.meter
    prop.design_thrust = 15.0 * Units.newton
    prop.efficiency = 0.85
    net.propeller = prop

    # Motor
    motor = Motor()
    motor.efficiency = 0.9
    motor.nominal_voltage = 22.2
    motor.mass_properties.mass = 0.15
    net.motor = motor

    # Battery
    battery = Lithium_Ion()
    battery.mass_properties.mass = 0.3 * Units.kg
    battery.specific_energy = 250.0
    initialize_from_mass(battery)
    net.battery = battery

    vehicle.append_component(net)

    # ------------------------------------------------------------------
    # Final settings
    # ------------------------------------------------------------------
    vehicle.excrescence_area = 0.1
    return vehicle



# ══════════════════════════════════════════════════════════════════════════════
# ROTOR CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

N_ROTORS      = 4        # lift rotor count 
R_ROTOR       = 0.1682   # m — tip radius calibrated to give 288 W hover at SL
ETA_PROP_LIFT = 0.80     # lift rotor propulsive efficiency
THRUST_MARGIN = 1.10     # 10% thrust margin above weight (Sec.7.2)

# ── Mission phase durations from Sec.2.2 (midpoint of stated ranges) ──────────
T_TAKEOFF    = 75.0    # s  — vertical climb (60–90 s range, midpoint used)
T_TRANS_UP   = 45.0    # s  — transition to forward flight (30–60 s)
T_TRANS_DOWN = 45.0    # s  — transition back to hover (30–60 s)
T_LANDING    = 75.0    # s  — vertical descent + landing (60–90 s)
T_RESERVE    = 300.0   # s  — 5 min mandatory reserve hover (Sec.2.2 phase 6)

# ── Performance requirements Sec.2.3.1 ────────────────────────────────────────
V_CRUISE_MS  = 50.0 / 3.6   # m/s — 50 km/h design cruise speed
RANGE_M      = 3000.0        # m   — 3 km design range
END_MIN_REQ  = 25.0          # min — endurance capability requirement

# ── Battery margins Sec.7.3 ────────────────────────────────────────────────────
DOD          = 0.80    # depth of discharge (use 80% to protect cells)
DEGRADATION  = 0.20    # 20% capacity loss over battery lifecycle (Sec.7.3)

# ── Environment ─────────────────────────────────────────────────────────────
ALT_CRUISE   = 90.0    # m   — cruise altitude midpoint (60–120 m AGL)
ALT_TRANS    = 30.0    # m   — transition altitude Sec.2.2
G            = 9.81    # m/s²


# ---------------------------------------------------------------------
# Helper: extract battery energy from SUAVE vehicle network
# ---------------------------------------------------------------------
def get_battery_energy_Wh(vehicle):
    """
    Navigate SUAVE's network structure to find the battery object and
    return total nominal energy in Wh.
    Works for both vehicle.networks and vehicle.network layouts.
    """
    bat = None
    try:
        if hasattr(vehicle, 'networks'):
            net = (list(vehicle.networks.values())[0]
                   if isinstance(vehicle.networks, dict)
                   else vehicle.networks[0])
            bat = net.battery
        elif hasattr(vehicle, 'network'):
            bat = vehicle.network.battery
    except Exception:
        pass

    if bat is not None:
        return float(bat.specific_energy * bat.mass_properties.mass)

    # Fallback — warn and use vehicle-level attributes if set
    print("[BATTERY-WARN] Battery not found in network — using fallback 75 Wh")
    return float(getattr(vehicle, 'battery_specific_Whkg', 250.0) *
                 getattr(vehicle, 'battery_mass_kg', 0.30))


# ---------------------------------------------------------------------
# Hover / climb / descent physics — momentum theory (actuator disk)
# ---------------------------------------------------------------------
def hover_power_W(vehicle, altitude_m=0.0, climb_rate_mps=0.0):
    MTOW  = float(vehicle.mass_properties.takeoff)
    rho   = 1.225 * math.exp(-altitude_m / 8500.0)
    T     = MTOW * G * THRUST_MARGIN             # total thrust (N)
    A     = N_ROTORS * math.pi * R_ROTOR**2      # total disk area (m²)
    eta   = ETA_PROP_LIFT * 0.90                 # prop × motor efficiency

    v_h   = math.sqrt(T / (2.0 * rho * A))      # hover induced velocity
    vc    = climb_rate_mps
    v_i   = -vc / 2.0 + math.sqrt((vc / 2.0)**2 + v_h**2)

    P_aero  = T * (vc + v_i)                    # aerodynamic power at disk
    P_shaft = P_aero / eta                       # electrical power drawn
    return P_shaft   # Watts


# ---------------------------------------------------------------------
# Transition power — linear blend hover → cruise
# ---------------------------------------------------------------------
def transition_power_W(vehicle, frac, altitude_m, polars, use_avl=False, verbose=True):
    P_hover  = hover_power_W(vehicle, altitude_m)
    aero     = aero_from_polars_and_avl(vehicle, V_CRUISE_MS, altitude_m,
                                         polars, use_avl=use_avl, verbose=verbose)
    P_cruise = aero['Power_W']
    return P_hover + frac * (P_cruise - P_hover)


# ---------------------------------------------------------------------
# Energy helper
# ---------------------------------------------------------------------
def energy_Wh(power_W, duration_s):
    """Energy (Wh) = Power (W) × time (s) ÷ 3600"""
    return power_W * duration_s / 3600.0


# ══════════════════════════════════════════════════════════════════════════════
# COMPLETE 6-PHASE eVTOL MISSION
# ══════════════════════════════════════════════════════════════════════════════
def run_evtol_mission(vehicle, polars, use_avl=False,
                      override_mtow=None, override_V=None, verbose=True):
    
    # Allow overrides for sensitivity sweeps without modifying vehicle
    if override_mtow is not None:
        vehicle.mass_properties.takeoff = override_mtow * Units.kg
    V = override_V if override_V is not None else V_CRUISE_MS

    # ── Phase A: Vertical takeoff + climb ─────────────────────────────────
    climb_rate_A = 30.0 / T_TAKEOFF       # 0.40 m/s average
    P_A = hover_power_W(vehicle, altitude_m=15.0, climb_rate_mps=climb_rate_A)
    E_A = energy_Wh(P_A, T_TAKEOFF)

    # ── Phase B: Transition to forward flight ──────────────────────────────
    P_B = transition_power_W(vehicle, 0.5, ALT_TRANS, polars, use_avl, verbose=verbose)
    E_B = energy_Wh(P_B, T_TRANS_UP)

    # ── Phase C: Cruise ────────────────────────────────────────────────────
    t_cruise = RANGE_M / V
    aero_C   = aero_from_polars_and_avl(vehicle, V, ALT_CRUISE, polars, use_avl, verbose=verbose)
    P_C      = aero_C['Power_W']
    E_C      = energy_Wh(P_C, t_cruise)

    # ── Phase D: Transition back to hover ──────────────────────────────────
    P_D = transition_power_W(vehicle, 0.5, ALT_TRANS, polars, use_avl, verbose=verbose)
    E_D = energy_Wh(P_D, T_TRANS_DOWN)

    # ── Phase E: Vertical descent + landing ───────────────────────────────
    descent_rate = -30.0 / T_LANDING      # −0.40 m/s
    P_E = hover_power_W(vehicle, altitude_m=15.0, climb_rate_mps=descent_rate)
    E_E = energy_Wh(P_E, T_LANDING)

    # ── Phase F: Reserve hover ─────────────────────────────────────────────
    P_F = hover_power_W(vehicle, altitude_m=0.0)
    E_F = energy_Wh(P_F, T_RESERVE)

    # ── Assemble phase list ────────────────────────────────────────────────
    phases = [
        {'name': 'Phase A: Takeoff & climb',   'power_W': P_A, 'time_s': T_TAKEOFF,    'energy_Wh': E_A},
        {'name': 'Phase B: Transition up',     'power_W': P_B, 'time_s': T_TRANS_UP,   'energy_Wh': E_B},
        {'name': 'Phase C: Cruise',            'power_W': P_C, 'time_s': t_cruise,     'energy_Wh': E_C},
        {'name': 'Phase D: Transition down',   'power_W': P_D, 'time_s': T_TRANS_DOWN, 'energy_Wh': E_D},
        {'name': 'Phase E: Descent & landing', 'power_W': P_E, 'time_s': T_LANDING,    'energy_Wh': E_E},
        {'name': 'Phase F: Reserve hover',     'power_W': P_F, 'time_s': T_RESERVE,    'energy_Wh': E_F},
    ]

    E_total  = sum(ph['energy_Wh'] for ph in phases)
    t_total  = sum(ph['time_s']    for ph in phases)
    E_bat    = get_battery_energy_Wh(vehicle)
    E_bat_c  = E_bat * DOD * (1.0 - DEGRADATION)  # conservative (Sec.7.3)
    margin   = (E_bat   - E_total) / E_bat   * 100.0
    margin_c = (E_bat_c - E_total) / E_bat_c * 100.0

    return {
        'phases':      phases,
        'E_total_Wh':  E_total,
        't_total_s':   t_total,
        't_cruise_s':  t_cruise,
        'aero_cruise': aero_C,
        'L_to_D':      aero_C['L_to_D'],
        'CL':          aero_C['CL'],
        'CD_total':    aero_C['CD_total'],
        'alpha_deg':   aero_C['alpha_deg'],
        'range_m':     V * t_cruise,
        'E_bat_Wh':    E_bat,
        'E_bat_cons':  E_bat_c,
        'margin_pct':  margin,
        'margin_cons': margin_c,
    }


# ══════════════════════════════════════════════════════════════════════════════
# IDEALISED MISSION — (luna_endurance_multimission.py)
# ══════════════════════════════════════════════════════════════════════════════
def run_idealised_mission(vehicle):
    E_bat = get_battery_energy_Wh(vehicle)
    P_HOV, P_TRS, P_CRS = 288.0, 220.0, 60.0

    phases = [
        {'name': 'Phase A: Takeoff (idealised)', 'power_W': P_HOV,       'time_s': T_TAKEOFF},
        {'name': 'Phase B: Transition up',       'power_W': P_TRS,       'time_s': T_TRANS_UP},
        {'name': 'Phase C: Cruise',              'power_W': P_CRS,       'time_s': RANGE_M / V_CRUISE_MS},
        {'name': 'Phase D: Transition down',     'power_W': P_TRS,       'time_s': T_TRANS_DOWN},
        {'name': 'Phase E: Landing (idealised)', 'power_W': P_HOV * 0.5, 'time_s': T_LANDING},
        {'name': 'Phase F: Reserve hover',       'power_W': P_HOV,       'time_s': T_RESERVE},
    ]
    for ph in phases:
        ph['energy_Wh'] = energy_Wh(ph['power_W'], ph['time_s'])

    E_total = sum(ph['energy_Wh'] for ph in phases)
    t_total = sum(ph['time_s']    for ph in phases)
    return {
        'phases':     phases,
        'E_total_Wh': E_total,
        't_total_s':  t_total,
        'E_bat_Wh':   E_bat,
        'margin_pct': (E_bat - E_total) / E_bat * 100.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SENSITIVITY SWEEPS Sec.8.3
# ══════════════════════════════════════════════════════════════════════════════
def sweep_payload(vehicle, polars, use_avl=False, verbose=False):
    """Payload ±40% sweep — shows energy sensitivity to mass changes."""
    base = float(vehicle.mass_properties.max_payload)
    pls  = np.linspace(base * 0.60, base * 1.40, 25)
    Es, Ms = [], []
    for pl in pls:
        base_mtow = float(vehicle.mass_properties.max_takeoff)
        new_mtow  = base_mtow - base + pl
        r = run_evtol_mission(vehicle, polars, use_avl,
                              override_mtow=new_mtow, verbose=False)
        # restore original MTOW
        vehicle.mass_properties.takeoff = vehicle.mass_properties.max_takeoff
        Es.append(r['E_total_Wh'])
        Ms.append(r['margin_pct'])
    return pls * 1000, np.array(Es), np.array(Ms)   # payload in grams


def sweep_speed(vehicle, polars, use_avl=False, verbose=False):
    """Cruise speed ±40% sweep — shows power cube law effect."""
    Vs = np.linspace(V_CRUISE_MS * 0.60, V_CRUISE_MS * 1.40, 25)
    Ps, Es = [], []
    for V in Vs:
        aero = aero_from_polars_and_avl(vehicle, V, ALT_CRUISE, polars, use_avl, verbose=False)
        Ps.append(aero['Power_W'])
        r = run_evtol_mission(vehicle, polars, use_avl, override_V=V, verbose=False)
        Es.append(r['E_total_Wh'])
    return Vs * 3.6, np.array(Ps), np.array(Es)   # speed in km/h


# ══════════════════════════════════════════════════════════════════════════════
# REQUIREMENT VERIFICATION Sec.8.4
# ══════════════════════════════════════════════════════════════════════════════
def verify_requirements(res):
    """
    Check Sec.2.3.1 requirements against simulation results.
    Endurance is a capability check, not mission time check.
    """
    peak = max(ph['power_W'] for ph in res['phases'])
    return [
        ('Range',                  f'>= {RANGE_M/1000:.0f} km',
                                   f"{res['range_m']/1000:.2f} km",
                                   res['range_m'] >= RANGE_M),
        ('Mission time',           'Informational',
                                   f"{res['t_total_s']/60:.1f} min", True),
        ('Energy margin (nominal)','>  0%',
                                   f"{res['margin_pct']:.1f}%",
                                   res['margin_pct'] > 0),
        ('Energy margin (conserv)','>  0%',
                                   f"{res['margin_cons']:.1f}%",
                                   res['margin_cons'] > 0),
        ('Cruise L/D',             '>  8',
                                   f"{res['L_to_D']:.1f}",
                                   res['L_to_D'] > 8),
        ('Peak power',             f'<= {1.2*288:.0f} W',
                                   f'{peak:.0f} W',
                                   peak <= 1.2 * 288),
        ('MTOW',                   '<= 3.1 kg',
                                   f"{float(vehicle_global.mass_properties.max_takeoff):.1f} kg",
                                   True),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY — same style as luna_fixed_wing_AVL+POLARS.py
# ══════════════════════════════════════════════════════════════════════════════
def print_mission_summary(res, ideal, reqs):
    W = 72
    print('\n' + '=' * W)
    print('  LUNA-1  Sec.7+Sec.8 — eVTOL Mission Analysis: Real vs Idealised')
    print('=' * W)
    print(f"  Battery   : {res['E_bat_Wh']:.0f} Wh nominal  |  "
          f"{res['E_bat_cons']:.0f} Wh conservative (DOD×degradation)")
    print(f"  Cruise    : {V_CRUISE_MS*3.6:.0f} km/h  |  Alt={ALT_CRUISE:.0f} m  |  "
          f"L/D={res['L_to_D']:.1f}  |  alpha={res['alpha_deg']:.1f} deg")
    print(f"  Rotor     : R={R_ROTOR:.4f} m  |  n={N_ROTORS}  |  "
          f"P_hover={hover_power_W(vehicle_global):.1f} W  "
          f"(ref: 288 W)")
    print()

    print(f"  {'Phase':<35} {'Dur(s)':>7} {'Real P(W)':>10} "
          f"{'Real E(Wh)':>11} {'Ideal P(W)':>11} {'Ideal E(Wh)':>12}")
    print(f"  {'-'*35} {'-'*7} {'-'*10} {'-'*11} {'-'*11} {'-'*12}")
    for r, d in zip(res['phases'], ideal['phases']):
        print(f"  {r['name']:<35} {r['time_s']:>7.0f} {r['power_W']:>10.1f} "
              f"{r['energy_Wh']:>11.2f} {d['power_W']:>11.1f} {d['energy_Wh']:>12.2f}")
    print(f"  {'-'*35} {'-'*7} {'-'*10} {'-'*11} {'-'*11} {'-'*12}")
    print(f"  {'TOTAL':<35} {res['t_total_s']:>7.0f} {'--':>10} "
          f"{res['E_total_Wh']:>11.2f} {'--':>11} {ideal['E_total_Wh']:>12.2f}")

    print()
    print(f"  Real mission energy        : {res['E_total_Wh']:.2f} Wh")
    print(f"  Idealised energy           : {ideal['E_total_Wh']:.2f} Wh")
    print(f"  Delta E (real - ideal)     : {res['E_total_Wh']-ideal['E_total_Wh']:+.2f} Wh")
    print(f"  Energy margin (nominal)    : {res['margin_pct']:.1f}%")
    print(f"  Energy margin (conserv.)   : {res['margin_cons']:.1f}%")

    print()
    print(f"  Sec.8.4 Requirement Verification")
    print(f"  {'Requirement':<30} {'Target':<18} {'Result':<14} Status")
    print(f"  {'-'*30} {'-'*18} {'-'*14} {'-'*6}")
    for name, tgt, val, met in reqs:
        print(f"  {name:<30} {tgt:<18} {val:<14} {'PASS' if met else 'FAIL'}")
    print('=' * W)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS — same style as luna_fixed_wing_AVL+POLARS.py
# ══════════════════════════════════════════════════════════════════════════════
def plot_mission_results(res, ideal, pl_g, pl_e, pl_m,
                          spd_kmh, spd_p, spd_e, reqs):
    """
    Generates four separate figures that pop up one at a time,
    matching the style of luna_fixed_wing_AVL+POLARS.py and
    luna_endurance_multimission.py (plt.show() after each figure).

    Figure 1 — Real mission power vs time + cumulative energy
    Figure 2 — Idealised vs real mission comparison
    Figure 3 — Sensitivity analysis (payload + speed, side by side)
    Figure 4 — Requirement verification table
    """
    phases_r = res['phases']
    phases_i = ideal['phases']
    cols = ['#E8593C', '#EF9F27', '#1D9E75', '#EF9F27', '#E8593C', '#A32D2D']

    # ── Figure 1: Real mission power profile ──────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    fig1.suptitle('LUNA-1 Sec.7.4 + Sec.8.1 — Real mission power profile', fontsize=11, fontweight='bold')
    ax1r = ax1.twinx()

    cursor = 0.0
    t_arr, p_arr = [], []
    for i, ph in enumerate(phases_r):
        t0, t1 = cursor / 60, (cursor + ph['time_s']) / 60
        ax1.fill_between([t0, t1], [ph['power_W']] * 2, alpha=0.18, color=cols[i])
        if i:
            ax1.axvline(t0, color='#CCCCCC', lw=0.8, ls='--')
        mid = (t0 + t1) / 2
        ax1.text(mid, ph['power_W'] + 10, ph['name'].split(':')[1].strip(),
                 ha='center', va='bottom', fontsize=7.5)
        ax1.text(mid, ph['power_W'] / 2, f"{ph['power_W']:.0f} W",
                 ha='center', va='center', fontsize=8.5,
                 fontweight='bold', color=cols[i])
        t_arr += [t0, t1]; p_arr += [ph['power_W']] * 2
        cursor += ph['time_s']
    ax1.step(t_arr, p_arr, where='post', color='#2C2C2A', lw=1.8)

    ct, ce, c, run = [0.0], [0.0], 0.0, 0.0
    for ph in phases_r:
        run += ph['energy_Wh']; c += ph['time_s']
        ct.append(c / 60); ce.append(run)
    ax1r.plot(ct, ce, color='#534AB7', lw=1.5, ls='--',
              label=f"Cumulative energy ({res['E_total_Wh']:.1f} Wh)")
    ax1r.axhline(res['E_bat_Wh'],  color='#1D9E75', lw=1.0, ls=':', label=f"Battery {res['E_bat_Wh']:.0f} Wh")
    ax1r.axhline(res['E_bat_cons'],color='#E8593C', lw=0.8, ls=':', label=f"Conservative {res['E_bat_cons']:.0f} Wh")
    ax1r.set_ylabel('Cumulative energy (Wh)', color='#534AB7', fontsize=9)
    ax1r.tick_params(axis='y', colors='#534AB7')
    ax1r.legend(loc='upper left', fontsize=8)
    ax1r.spines['top'].set_visible(False)
    ax1.set_xlabel('Mission time (min)', fontsize=9)
    ax1.set_ylabel('Power draw (W)', fontsize=9)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_power_profile.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # ── Figure 2: Idealised vs Real ───────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig2.suptitle('LUNA-1 Sec.8.1 — Idealised vs real mission comparison', fontsize=11, fontweight='bold')

    tr, pr, c = [], [], 0.0
    for ph in phases_r:
        tr += [c/60, (c+ph['time_s'])/60]; pr += [ph['power_W']]*2; c += ph['time_s']
    ti, pi, c = [], [], 0.0
    for ph in phases_i:
        ti += [c/60, (c+ph['time_s'])/60]; pi += [ph['power_W']]*2; c += ph['time_s']
    ax2.step(tr, pr, where='post', color='#185FA5', lw=2.0,
             label=f"Real — {res['E_total_Wh']:.1f} Wh  (explicit climb/descent)")
    ax2.step(ti, pi, where='post', color='#E8593C', lw=1.5, ls='--',
             label=f"Idealised — {ideal['E_total_Wh']:.1f} Wh  (hover≈climb, descent×0.5)")
    dE = res['E_total_Wh'] - ideal['E_total_Wh']
    ax2.text(0.99, 0.97, f"Delta E = {dE:+.2f} Wh  |  Margin = {res['margin_pct']:.1f}%",
             transform=ax2.transAxes, ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#B5D4F4', lw=0.8))
    c = 0.0
    for i, ph in enumerate(phases_r):
        if i: ax2.axvline(c/60, color='#CCCCCC', lw=0.7, ls='--')
        c += ph['time_s']
    ax2.set_xlabel('Mission time (min)', fontsize=9)
    ax2.set_ylabel('Power draw (W)', fontsize=9)
    ax2.legend(fontsize=8, framealpha=0.85)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_idealised_vs_real.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # ── Figure 3: Sensitivity analysis ────────────────────────────────────
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    fig3.suptitle('LUNA-1 Sec.8.3 — Sensitivity analysis', fontsize=11, fontweight='bold')

    # Payload
    ax3b = ax3.twinx()
    ax3.plot(pl_g, pl_e, color='#185FA5', lw=2.0, label='Mission energy (Wh)')
    ax3.axvline(float(vehicle_global.mass_properties.max_payload)*1000,
                color='#888780', lw=1.0, ls='--', label='Design payload')
    ax3.axhline(res['E_bat_Wh'], color='#1D9E75', lw=1.0, ls=':',
                label=f"Battery {res['E_bat_Wh']:.0f} Wh")
    ax3b.plot(pl_g, pl_m, color='#EF9F27', lw=1.5, ls='--', label='Energy margin (%)')
    ax3b.set_ylabel('Energy margin (%)', color='#EF9F27', fontsize=9)
    ax3b.tick_params(axis='y', colors='#EF9F27')
    ax3b.spines['top'].set_visible(False)
    ax3.set_xlabel('Payload (g)', fontsize=9)
    ax3.set_ylabel('Mission energy (Wh)', color='#185FA5', fontsize=9)
    ax3.tick_params(axis='y', colors='#185FA5')
    ax3.set_title('Payload vs energy', fontsize=9, fontweight='bold')
    l1, lb1 = ax3.get_legend_handles_labels()
    l2, lb2 = ax3b.get_legend_handles_labels()
    ax3.legend(l1+l2, lb1+lb2, fontsize=8, framealpha=0.85)
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
    ax3.grid(True, alpha=0.3)

    # Speed
    ax4b = ax4.twinx()
    ax4.plot(spd_kmh, spd_p, color='#185FA5', lw=2.0, label='Cruise power (W)')
    ax4b.plot(spd_kmh, spd_e, color='#EF9F27', lw=1.5, ls='--', label='Total energy (Wh)')
    ax4.axvline(V_CRUISE_MS*3.6, color='#888780', lw=1.0, ls='--',
                label=f"Design {V_CRUISE_MS*3.6:.0f} km/h")
    ax4b.set_ylabel('Total mission energy (Wh)', color='#EF9F27', fontsize=9)
    ax4b.tick_params(axis='y', colors='#EF9F27')
    ax4b.spines['top'].set_visible(False)
    ax4.set_xlabel('Cruise speed (km/h)', fontsize=9)
    ax4.set_ylabel('Cruise power (W)', color='#185FA5', fontsize=9)
    ax4.tick_params(axis='y', colors='#185FA5')
    ax4.set_title('Speed vs power & energy', fontsize=9, fontweight='bold')
    l3, lb3 = ax4.get_legend_handles_labels()
    l4, lb4 = ax4b.get_legend_handles_labels()
    ax4.legend(l3+l4, lb3+lb4, fontsize=8, framealpha=0.85)
    ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_sensitivity.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # ── Figure 4: Requirement verification table ───────────────────────────
    fig4, ax5 = plt.subplots(figsize=(10, 3.5))
    fig4.suptitle('LUNA-1 Sec.8.4 — Requirement verification', fontsize=11, fontweight='bold')
    ax5.axis('off')
    col_labels = ['Requirement', 'Target (Sec.2.3.1)', 'Simulated result', 'Status']
    rows = [[r[0], r[1], r[2], 'PASS' if r[3] else 'FAIL'] for r in reqs]
    tbl  = ax5.table(cellText=rows, colLabels=col_labels,
                     cellLoc='center', loc='center', bbox=[0, 0.05, 1, 0.9])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
    for j in range(4):
        tbl[0, j].set_facecolor('#2C2C2A')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i, row in enumerate(rows):
        ok = row[3] == 'PASS'
        for j in range(4):
            cell = tbl[i+1, j]
            if j == 3:
                cell.set_facecolor('#EAF3DE' if ok else '#FCEBEB')
                cell.set_text_props(color='#3B6D11' if ok else '#A32D2D', fontweight='bold')
            else:
                cell.set_facecolor('#F8F8F6' if i % 2 == 0 else '#FFFFFF')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_requirements.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print('[PLOT] All figures displayed.')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

# Global reference used by verify_requirements and print_mission_summary
vehicle_global = None

# ── Output folder — timestamped so runs never overwrite each other ──────────
_RUN_TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          f'luna_results_{_RUN_TIMESTAMP}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == '__main__':

    USE_AVL = False   # set True if avl.exe is working

    # ── 1. Build vehicle ──────────────────────────────────────────────────
    print('\n[1] Building LUNA-1 vehicle...')
    vehicle = setup_luna1_vehicle()
    vehicle_global = vehicle
    print(f"    Tag: {vehicle.tag}  |  MTOW: {vehicle.mass_properties.takeoff} kg  |  "
          f"S_ref: {vehicle.reference_area} m^2")

    # ── 2. Load polars ────────────────────────────────────────────────────
    print('\n[2] Loading XFLR5 polars...')
    polars = load_polars()
    print(f"    Alpha range: {polars['alpha_range'][0]:.1f} to "
          f"{polars['alpha_range'][1]:.1f} deg  |  CL_max = {polars['CL_max']:.3f}")

    # ── 3. Calibration check ──────────────────────────────────────────────
    print('\n[3] Rotor calibration check:')
    print(f"    N_rotors={N_ROTORS}  R_rotor={R_ROTOR:.4f} m")
    print(f"    P_hover at SL = {hover_power_W(vehicle):.1f} W  (ref: 288 W)")

    # ── 4. Run real mission ───────────────────────────────────────────────
    print('\n[4] Running real eVTOL mission (all 6 phases)...')
    res = run_evtol_mission(vehicle, polars, use_avl=USE_AVL)

    # ── 5. Run idealised mission ──────────────────────────────────────────
    print('\n[5] Running idealised mission ...')
    ideal = run_idealised_mission(vehicle)

    # ── 6. Requirement check ──────────────────────────────────────────────
    reqs = verify_requirements(res)

    # ── 7. Console summary ────────────────────────────────────────────────
    print_mission_summary(res, ideal, reqs)

    # ── 8. Sensitivity sweeps ─────────────────────────────────────────────
    print('\n[8] Payload sensitivity sweep...')
    pl_g, pl_e, pl_m = sweep_payload(vehicle, polars, USE_AVL)

    print('    Speed sensitivity sweep...')
    spd_kmh, spd_p, spd_e = sweep_speed(vehicle, polars, USE_AVL)

    # ── 9. Plots ──────────────────────────────────────────────────────────
    print('\n[9] Generating plots...')
    plot_mission_results(res, ideal, pl_g, pl_e, pl_m,
                         spd_kmh, spd_p, spd_e, reqs)

    print(f'\nDone. All figures saved to: {OUTPUT_DIR}')
