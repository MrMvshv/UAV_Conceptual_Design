#!/usr/bin/env python3
"""
run_analysis.py

Fixed-wing UAV mission analysis using SUAVE 2.5.2:
 - Aerodynamics from XFLR5 polars
 - Optional AVL induced drag
 - Energy consumption from battery + power network
 - Mission with climb, cruise, descent segments

Place `fixedwing_vehicle_definition.py` in same folder.
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Analyses import Aerodynamics, Mission, Weights, Energy
from SUAVE.Analyses.Atmospheric import US_Standard_1976 as Atmosphere

atmo = Atmosphere()

# Optional AVL wrapper
try:
    from SUAVE.Methods.Aerodynamics.AVL import AVL as SUAVE_AVL
    _AVL_AVAILABLE = True
except Exception:
    SUAVE_AVL = None
    _AVL_AVAILABLE = False

# Local vehicle definition
from fixedwing_vehicle_definition import setup_fixedwing_vehicle

# ----------------------------
# Load XFLR5 polars
# ----------------------------
def load_polars(csv_file='polars_interpolated.csv'):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"{csv_file} not found")
    df = pd.read_csv(csv_file)
    alpha = df['alpha_deg'].to_numpy()
    CL = df['CL'].to_numpy()
    CD = df['CD'].to_numpy()

    # Ensure sorted
    idx_sort = np.argsort(alpha)
    alpha = alpha[idx_sort]
    CL = CL[idx_sort]
    CD = CD[idx_sort]

    # Interpolators
    alpha_to_CL = interp1d(alpha, CL, kind='cubic', fill_value='extrapolate')
    alpha_to_CD = interp1d(alpha, CD, kind='cubic', fill_value='extrapolate')
    
    # CL -> alpha (pre-stall)
    imaxCL = int(np.nanargmax(CL))
    CL_to_alpha = interp1d(CL[:imaxCL+1], alpha[:imaxCL+1],
                           kind='linear', bounds_error=False, fill_value='extrapolate')

    return {'alpha_to_CL': alpha_to_CL,
            'alpha_to_CD': alpha_to_CD,
            'CL_to_alpha': CL_to_alpha,
            'alpha_range': (alpha[0], alpha[-1])}

# ----------------------------
# Compute aerodynamics
# ----------------------------
def compute_aero(vehicle, V, altitude_m, polars, use_avl=False):
    # Atmospheric properties
    conditions = atmo.compute_values(altitude_m)
    rho = float(np.atleast_1d(conditions.density)[0])
    #rho = Atmosphere.density(altitude_m)
    S = vehicle.reference_area
    W = vehicle.mass_properties.takeoff * 9.81
    q = 0.5 * rho * V**2

    # Required lift coefficient
    CL_req = W / (q * S)
    alpha_deg = float(polars['CL_to_alpha'](CL_req))

    # Clamp to polar range
    alpha_min, alpha_max = polars['alpha_range']
    alpha_deg = max(min(alpha_deg, alpha_max), alpha_min)

    CD_profile = float(polars['alpha_to_CD'](alpha_deg))

    # Wing geometry for induced drag
    wing = None
    if len(vehicle.wings) > 0:
        wing = list(vehicle.wings.values())[0]
    print(f"{wing.tag if wing else 'No wing'} geometry used for induced drag")
    #print(f"{vehicle.wings[0].tag if len(vehicle.wings) > 0 else 'No wing'} geometry used for induced drag")
    #wing = vehicle.wings[0] if len(vehicle.wings) > 0 else None
    AR = wing.aspect_ratio if wing else 6.0
    e = getattr(wing, 'span_efficiency', 0.85)

    # Induced drag: AVL if available
    if use_avl and _AVL_AVAILABLE:
        try:
            avl = SUAVE_AVL()
            avl.geometry = vehicle
            avl.inputs.alpha = alpha_deg
            avl.inputs.altitude = altitude_m
            avl.inputs.Mach = max(V / 343.0, 1e-6)
            res = avl.evaluate()
            CDi = getattr(res, 'CDi', CL_req**2/(np.pi*AR*e))
            CD_source = 'AVL'
        except Exception:
            CDi = CL_req**2/(np.pi*AR*e)
            CD_source = 'analytic (fallback)'
    else:
        CDi = CL_req**2/(np.pi*AR*e)
        CD_source = 'analytic'

    CD_total = CD_profile + CDi + 0.01  # small excrescence
    D = CD_total * q * S

    # Power: assume prop/motor efficiencies
    eta_prop = 0.85
    eta_motor = 0.9
    P_req = D * V / (eta_prop * eta_motor)
    L_to_D = CL_req / CD_total

    return {'CL': CL_req, 'alpha_deg': alpha_deg, 'CD_profile': CD_profile,
            'CDi': CDi, 'CD_total': CD_total, 'Drag_N': D, 'Power_W': P_req,
            'L_to_D': L_to_D, 'CD_source': CD_source}


# ----------------------------
# COmpute hover aerodynamic
# ----------------------------
def compute_hover_power(vehicle, altitude_m, n_rotors=4, rotor_radius=0.15, FM=0.7):
    conditions = atmo.compute_values(altitude_m)
    rho = float(np.atleast_1d(conditions.density)[0])

    T = vehicle.mass_properties.takeoff * 9.81
    A_total = n_rotors * np.pi * rotor_radius**2

    vi = np.sqrt(T / (2 * rho * A_total))
    P_ideal = T * vi

    P_hover = P_ideal / FM  # include losses

    return P_hover
# ----------------------------
# Mission Segment
# ----------------------------
def run_segment(vehicle, seg, soc_Wh, polars, use_avl=False):
    V = seg.get('speed_mps', 20.0)
    altitude = seg.get('altitude_m', 0.0)
    duration = seg.get('time_s', 60.0)
    seg_type = seg.get('type', 'cruise')

    if seg_type == 'hover':
        aero = compute_aero(vehicle, V, altitude, polars, use_avl)
        P = compute_hover_power(vehicle, altitude)

    elif seg_type == 'transition':
        P_hover = compute_hover_power(vehicle, altitude)
        aero = compute_aero(vehicle, V, altitude, polars, use_avl)
        P_forward = aero['Power_W']

        blend = seg.get('blend', 0.5)
        P = blend * P_hover + (1 - blend) * P_forward

    else:
        aero = compute_aero(vehicle, V, altitude, polars, use_avl)
        P = aero['Power_W']

    # Simple climb/descent adjustment
    if seg_type == 'climb':
        climb_rate = seg.get('climb_rate_mps', 1.0)
        P += vehicle.mass_properties.takeoff * 9.81 * climb_rate
    elif seg_type == 'descent':
        P *= 0.5

    E_Wh = P * duration / 3600.0
    remaining = max(soc_Wh - E_Wh, 0.0)

    print(f"--- {seg_type.upper()} | V={V:.1f} m/s | Alt={altitude:.1f} m ---")
    print(f"Power={P:.1f} W | Energy used={E_Wh:.2f} Wh | Remaining={remaining:.2f} Wh")

    out = {'seg_type': seg_type, 'power_W': P, 'energy_Wh': E_Wh, 'remaining_Wh': remaining,
           'aero': aero}
    return out, remaining

# ----------------------------
# Run Mission
# ----------------------------
def run_mission(vehicle, mission_profile, polars, use_avl=False):
    # Battery energy
    batt_energy = getattr(vehicle, 'battery_mass_kg', 0.3) * getattr(vehicle, 'battery_specific_Whkg', 250)
    print(f"Initial battery energy: {batt_energy:.1f} Wh")

    soc = batt_energy
    results = []
    for seg in mission_profile:
        res, soc = run_segment(vehicle, seg, soc, polars, use_avl)
        results.append(res)
        if soc <= 0:
            print("Battery depleted. Mission terminated early.")
            break

    # Summary
    print("\nMission Summary:")
    for r in results:
        print(f"{r['seg_type']}: Energy={r['energy_Wh']:.2f} Wh, L/D={r['aero']['L_to_D']:.2f}, CDi source={r['aero']['CD_source']}")
    return results

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    vehicle = setup_fixedwing_vehicle()
    print(f"Vehicle loaded: tag={vehicle.tag}, Takeoff mass={vehicle.mass_properties.takeoff:.2f} kg")

    polars = load_polars()

    mission_profile = [
        {'type': 'hover', 'time_s': 60, 'altitude_m': 30},
        {'type': 'transition', 'time_s': 45, 'speed_mps': 10, 'altitude_m': 30, 'blend': 0.7},
        {'type': 'climb',  'time_s': 120, 'speed_mps': 15.0, 'climb_rate_mps': 2.0, 'altitude_m': 100},
        {'type': 'cruise', 'time_s': 600, 'speed_mps': 22.0, 'altitude_m': 100},
        {'type': 'descent','time_s': 120, 'speed_mps': 15.0, 'climb_rate_mps': -1.0, 'altitude_m': 30}
    ]

    results = run_mission(vehicle, mission_profile, polars, use_avl=True)

    #PLOTTING RESULTS

    output_dir = "output_graphs/"
    os.makedirs(output_dir, exist_ok=True)
    # ----------------------------
    # Extract data for plotting
    # ----------------------------
    segments = [r['seg_type'].capitalize() for r in results]
    power = [r['power_W'] for r in results]
    energy = [r['energy_Wh'] for r in results]
    ld = [r['aero']['L_to_D'] for r in results]


    # ----------------------------
    # 1. Power vs Segment
    # ----------------------------
    plt.figure()
    plt.bar(segments, power)
    plt.xlabel('Mission Segment')
    plt.ylabel('Power (W)')
    plt.title('Power Requirement by Mission Segment')
    plt.grid()
    plt.savefig(os.path.join(output_dir, "power_vs_segment.png"), dpi=300, bbox_inches='tight')
    # ----------------------------
    # 2. Energy Breakdown
    # ----------------------------
    plt.figure()
    plt.pie(energy, labels=segments, autopct='%1.1f%%')
    plt.title('Energy Distribution Across Mission')
    plt.savefig(os.path.join(output_dir, "energy_distribution.png"), dpi=300, bbox_inches='tight')
    # ----------------------------
    # 3. Lift-to-Drag Ratio
    # ----------------------------
    plt.figure()
    plt.plot(segments, ld, marker='o')
    plt.xlabel('Mission Segment')
    plt.ylabel('Lift-to-Drag Ratio (L/D)')
    plt.title('Aerodynamic Efficiency Across Mission')
    plt.grid()
    plt.savefig(os.path.join(output_dir, "ld_ratio.png"), dpi=300, bbox_inches='tight')
    # ----------------------------
    # 4. Cumulative Energy
    # ----------------------------
    cumulative_energy = np.cumsum(energy)

    plt.figure()
    plt.plot(segments, cumulative_energy, marker='o')
    plt.xlabel('Mission Segment')
    plt.ylabel('Cumulative Energy (Wh)')
    plt.title('Cumulative Energy Consumption')
    plt.grid()
    plt.savefig(os.path.join(output_dir, "cumulative_energy.png"), dpi=300, bbox_inches='tight')
    # ----------------------------
    # Show all plots
    # ----------------------------
    plt.show()
