#!/usr/bin/env python3
"""
fixedwing_energy_aero_analysis.py
---------------------------------
Standalone aerodynamic + energy analysis for a fixed-wing electric UAV
using SUAVE 2.5.2.

Key features:
 - Fidelity_Zero aerodynamic estimation (fast)
 - Per-segment power and energy calculation
 - SOC tracking and energy plots
 - Clear printed diagnostics for verification
"""

import numpy as np
import matplotlib.pyplot as plt
from SUAVE.Core import Units, Data
from SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion import Lithium_Ion
from SUAVE.Components.Energy.Converters.Motor import Motor
from SUAVE.Components.Energy.Converters.Propeller import Propeller
from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass


# ============================================================
# VEHICLE BUILDER
# ============================================================
def build_fixedwing():
    """Define a very simple fixed-wing UAV with electric propulsion."""
    vehicle = Data()
    vehicle.tag = 'fixed_wing_uav'

    # --- Mass and reference area ---
    vehicle.mass_properties = Data()
    vehicle.mass_properties.takeoff = 3.0  # kg
    vehicle.reference_area = 0.4           # m²

    # --- Wing geometry ---
    wing = Data()
    wing.aspect_ratio = 8.0
    wing.span_efficiency = 0.9
    wing.CD0 = 0.025
    wing.CL_max = 1.2
    vehicle.wing = wing

    # --- Powertrain ---
    net = Battery_Propeller()
    net.number_of_engines = 1

    prop = Propeller()
    prop.efficiency = 0.85
    net.propeller = prop

    motor = Motor()
    motor.efficiency = 0.9
    motor.nominal_voltage = 22.2
    net.motor = motor

    batt = Lithium_Ion()
    batt.mass_properties = Data()
    batt.mass_properties.mass = 0.8
    batt.specific_energy = 250.0  # Wh/kg
    initialize_from_mass(batt)
    net.battery = batt

    vehicle.network = net
    return vehicle


# ============================================================
# AERODYNAMIC MODEL
# ============================================================
def compute_aero_forces(vehicle, V, altitude_m):
    """Compute lift, drag, and power at given speed and altitude."""

    rho = 1.225 * np.exp(-altitude_m / 8500.0)
    S = vehicle.reference_area
    W = vehicle.mass_properties.takeoff * 9.81
    AR = vehicle.wing.aspect_ratio
    e = vehicle.wing.span_efficiency
    CD0 = vehicle.wing.CD0

    # Lift required for steady level flight
    CL = W / (0.5 * rho * V**2 * S)

    # Drag components
    CDi = CL**2 / (np.pi * e * AR)
    CD = CD0 + CDi
    D = 0.5 * rho * V**2 * S * CD

    # Power required (account for prop + motor efficiency)
    eta_total = vehicle.network.propeller.efficiency * vehicle.network.motor.efficiency
    P_req = D * V / eta_total

    return {
        'rho': rho,
        'CL': CL,
        'CD': CD,
        'Drag_N': D,
        'Power_W': P_req,
        'L_to_D': CL / CD,
    }


# ============================================================
# MISSION SEGMENT
# ============================================================
def run_segment(vehicle, seg, soc_Wh):
    """Run one mission segment and update SOC."""
    seg_type = seg.get('type', 'cruise')
    altitude = seg.get('altitude_m', 0)
    time_s = seg.get('time_s', 60)
    V = seg.get('speed_mps', 20)

    if seg_type == 'cruise':
        aero = compute_aero_forces(vehicle, V, altitude)
        P = aero['Power_W']

    elif seg_type == 'climb':
        climb_rate = seg.get('climb_rate_mps', 2.0)
        aero = compute_aero_forces(vehicle, V, altitude)
        P = aero['Power_W'] + vehicle.mass_properties.takeoff * 9.81 * climb_rate

    elif seg_type == 'descent':
        descent_rate = abs(seg.get('climb_rate_mps', -1.0))
        aero = compute_aero_forces(vehicle, V, altitude)
        P = aero['Power_W'] * 0.5  # half power assumed in descent

    else:
        raise ValueError(f"Unknown segment type: {seg_type}")

    E_Wh = P * time_s / 3600.0
    new_soc = soc_Wh - E_Wh
    if new_soc < 0: new_soc = 0.0

    print(f"\n--- {seg_type.upper()} SEGMENT ---")
    print(f"Airspeed: {V:.1f} m/s | Altitude: {altitude:.0f} m | Duration: {time_s/60:.1f} min")
    print(f"Air density: {aero['rho']:.3f} kg/m³")
    print(f"CL = {aero['CL']:.3f}, CD = {aero['CD']:.4f}, L/D = {aero['L_to_D']:.1f}")
    print(f"Drag = {aero['Drag_N']:.2f} N")
    print(f"Power required = {P:.1f} W | Energy used = {E_Wh:.2f} Wh")
    print(f"Battery remaining = {new_soc:.2f} Wh\n{'-'*55}")

    return {
        'seg_type': seg_type,
        'power_W': P,
        'energy_used_Wh': E_Wh,
        'remaining_Wh': new_soc,
        'L_to_D': aero['L_to_D']
    }, new_soc


# ============================================================
# MISSION RUNNER
# ============================================================
def run_mission(vehicle, mission_profile, plot=True):
    """Run all mission segments with SOC tracking."""
    batt = vehicle.network.battery
    soc_Wh = batt.specific_energy * batt.mass_properties.mass
    print(f"\n=== FIXED-WING ENERGY + AERODYNAMIC ANALYSIS ===")
    print(f"Initial battery energy: {soc_Wh:.1f} Wh ({soc_Wh/1000:.3f} kWh)")

    results = []
    for seg in mission_profile:
        res, soc_Wh = run_segment(vehicle, seg, soc_Wh)
        results.append(res)
        if soc_Wh <= 0:
            print("❌ Battery depleted — mission terminated early.")
            break

    if plot:
        plot_energy_profile(results)
    return results


# ============================================================
# PLOTTING
# ============================================================
def plot_energy_profile(results):
    """Plot remaining battery energy vs. segments."""
    segs = [r['seg_type'].upper() for r in results]
    used = [r['energy_used_Wh'] / 1000 for r in results]
    rem = [r['remaining_Wh'] / 1000 for r in results]

    plt.figure(figsize=(8, 5))
    plt.bar(segs, used, color='coral', label='Energy Used (kWh)')
    plt.plot(segs, rem, 'o-', color='green', label='Remaining (kWh)')
    plt.title('Fixed-Wing Mission Energy Profile')
    plt.xlabel('Segment')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    vehicle = build_fixedwing()

    mission_profile = [
        {'type': 'climb', 'time_s': 120, 'speed_mps': 15, 'climb_rate_mps': 2, 'altitude_m': 0},
        {'type': 'cruise', 'time_s': 900, 'speed_mps': 22, 'altitude_m': 100},
        {'type': 'descent', 'time_s': 120, 'speed_mps': 15, 'climb_rate_mps': -1, 'altitude_m': 0},
    ]

    results = run_mission(vehicle, mission_profile, plot=True)
