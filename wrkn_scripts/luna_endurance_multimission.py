#!/usr/bin/env python3
"""
luna_endurance_multimission_fixed.py
------------------------------------
Corrected and improved multi-segment mission analysis for the Luna UAV demo.

Key features:
 - Consistent battery energy unit handling (SUAVE reports specific_energy in Wh/kg)
 - SOC (state-of-charge) tracking across mission segments
 - Stops mission if battery is depleted
 - Cleaner and explicit energy/power calculations for hover, climb, cruise, descent
 - Per-segment diagnostics and a simple plot helper
"""

import numpy as np
import matplotlib.pyplot as plt
from SUAVE.Core import Data, Units
from SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion import Lithium_Ion
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass
from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller
from SUAVE.Components.Energy.Converters.Motor import Motor
from SUAVE.Components.Energy.Converters.Propeller import Propeller


# ============================================================
# VEHICLE BUILDER
# ============================================================
def build_vehicle():
    vehicle = Data()
    vehicle.tag = 'luna_uav'

    # --- Mass Properties ---
    vehicle.mass_properties = Data()
    vehicle.mass_properties.takeoff = 3.1  # kg (total takeoff mass)
    vehicle.mass_properties.payload = 0.2

    # --- Network Setup ---
    network = Battery_Propeller()
    network.number_of_engines = 4
    network.identical_propellers = True

    # --- Propeller ---
    prop = Propeller()
    prop.tip_radius = 0.15 * Units.meter
    prop.hub_radius = 0.015 * Units.meter
    prop.efficiency = 0.85
    prop.number_of_blades = 2
    network.propeller = prop

    # --- Motor ---
    motor = Motor()
    motor.efficiency = 0.9
    motor.nominal_voltage = 22.2  # 6S nominal voltage
    motor.mass_properties = Data()
    motor.mass_properties.mass = 0.15
    network.motor = motor

    # --- Battery ---
    battery = Lithium_Ion()
    battery.mass_properties = Data()
    battery.mass_properties.mass = 1.0  # more battery for mission
     # Set realistic specific energy BEFORE initialization
    battery.specific_energy = 250.0  # Wh/kg (typical Li-ion value)
    initialize_from_mass(battery)
    network.battery = battery

    vehicle.network = network
    return vehicle


# ============================================================
# HOVER ANALYSIS
# ============================================================
def run_hover(vehicle, altitude_m=0.0, hover_time_s=300.0, realistic=True, verbose=True):
    """
    Compute hover power and energy usage for a given hover duration at altitude.
    Returns a dict with detailed results.
    """
    rho = 1.225 * np.exp(-altitude_m / 8500.0)
    batt = vehicle.network.battery
    W = vehicle.mass_properties.takeoff * 9.81  # N (weight)
    R = vehicle.network.propeller.tip_radius
    A = np.pi * R**2

    # --- Efficiencies & margins ---
    eta_prop = 0.8 if realistic else 1.0
    eta_motor = 0.9 if realistic else 1.0
    eta_total = eta_prop * eta_motor
    thrust_margin = 1.1  # 10% reserve
    W_eff = W * thrust_margin

    # --- Induced velocity & power (momentum theory) ---
    v_induced = np.sqrt(W_eff / (2 * rho * A))
    P_ideal = W_eff * v_induced
    P_total = P_ideal / eta_total

    # --- Battery energy (Wh/kg → Wh) ---
    battery_energy_Wh = batt.specific_energy * batt.mass_properties.mass
    battery_energy_kWh = battery_energy_Wh / 1000.0

    # --- Energy used during hover ---
    E_hover_Wh = (P_total * hover_time_s) / 3600  # W·s → Wh
    E_hover_kWh = E_hover_Wh / 1000.0

    SOC_drop = E_hover_Wh / battery_energy_Wh if battery_energy_Wh > 0 else np.inf
    endurance_s = hover_time_s / SOC_drop if SOC_drop > 0 else np.inf
    endurance_min = endurance_s / 60.0

    # --- Electrical details ---
    V_nom = vehicle.network.motor.nominal_voltage
    I_draw_A = P_total / V_nom
    capacity_Ah = battery_energy_Wh / V_nom
    C_rate = I_draw_A / capacity_Ah if capacity_Ah > 0 else np.inf

    if verbose:
        print("\n[1] Hover segment initialized ...")
        print(f"Altitude: {altitude_m:.0f} m | Air density: {rho:.3f} kg/m³")

        print("\n--- Hover Performance Summary ---")
        print(f"Weight (N):               {W:.2f}")
        print(f"Rotor disk area (m²):     {A:.3f}")
        print(f"Induced velocity (m/s):   {v_induced:.2f}")
        print(f"Thrust-to-weight margin:  {100*(thrust_margin-1):.1f}%")
        print(f"Effective efficiencies:   Prop={eta_prop:.2f}, Motor={eta_motor:.2f}, Total={eta_total:.2f}")
        print(f"Total power (W):          {P_total:.1f}")
        print(f"Energy used (kWh):        {E_hover_kWh:.3f}")
        print(f"Battery energy (kWh):     {battery_energy_kWh:.3f}")
        print(f"SOC drop (fraction):      {SOC_drop:.3f}")
        print(f"Est. endurance (min):     {endurance_min:.1f}")

        print("\n--- Electrical Load Details ---")
        print(f"Nominal voltage (V):      {V_nom:.1f}")
        print(f"Battery capacity (Ah):    {capacity_Ah:.2f}")
        print(f"Current draw (A):         {I_draw_A:.1f}")
        print(f"Discharge rate (C):       {C_rate:.1f}")
        if C_rate > 20:
            print("⚠️  WARNING: Battery C-rate dangerously high! (risk of overheating)")
        elif C_rate > 10:
            print("⚠️  Note: High C-rate, pack stress likely in sustained hover.")

        print("\n--- Energy Balance Notes ---")
        print(f"Power-to-weight ratio:    {P_total/W:.3f} W/N")
        print(f"Energy-to-weight ratio:   {(battery_energy_kWh*3.6e6/W):.1f} J/N")
        print(f"Hover time limit (ideal): {endurance_min:.1f} min (at {altitude_m:.0f} m)")

    return {
        'P_total_W': P_total,
        'E_hover_kWh': E_hover_kWh,
        'E_hover_Wh': E_hover_Wh,
        'SOC_drop': SOC_drop,
        'endurance_min': endurance_min,
        'radius_m': R,
        'altitude_m': altitude_m,
        'rho': rho,
        'C_rate': C_rate,
        'I_draw_A': I_draw_A,
        'battery_energy_kWh': battery_energy_kWh,
        'battery_energy_Wh': battery_energy_Wh,
        'capacity_Ah': capacity_Ah
    }


# ============================================================
# SEGMENT RUNNER
# ============================================================
def run_segment(vehicle, seg, soc_Wh, verbose=True):
    seg_type = seg.get('type', 'hover')
    time_s = seg.get('time_s', 60)
    altitude = seg.get('altitude_m', 0)
    speed = seg.get('speed_mps', 0)
    climb_rate = seg.get('climb_rate_mps', 0)

    if seg_type == 'hover':
        res = run_hover(vehicle, altitude_m=altitude, hover_time_s=time_s, realistic=True, verbose=False)
        energy_used_Wh = res['E_hover_Wh']
        diag = res

    elif seg_type == 'climb':
        hover_res = run_hover(vehicle, altitude_m=altitude, hover_time_s=1, realistic=True, verbose=False)
        P_hover = hover_res['P_total_W']
        P_climb = P_hover + vehicle.mass_properties.takeoff * 9.81 * climb_rate
        energy_used_Wh = (P_climb * time_s) / 3600
        diag = {'P_climb_W': P_climb}

    elif seg_type == 'cruise':
        rho = 1.225 * np.exp(-altitude / 8500.0)
        CdA = seg.get('CdA', 0.05)
        V = speed
        D = 0.5 * rho * V**2 * CdA
        P_cruise = D * V / 0.75  # 75% total efficiency
        energy_used_Wh = (P_cruise * time_s) / 3600
        diag = {'P_cruise_W': P_cruise, 'drag_N': D}

    elif seg_type == 'descent':
        hover_res = run_hover(vehicle, altitude_m=altitude, hover_time_s=1, realistic=True, verbose=False)
        P_hover = hover_res['P_total_W']
        P_descent = 0.5 * P_hover
        energy_used_Wh = (P_descent * time_s) / 3600
        diag = {'P_descent_W': P_descent}

    else:
        energy_used_Wh = 0.0
        diag = {}

    new_soc_Wh = soc_Wh - energy_used_Wh
    return energy_used_Wh, new_soc_Wh, diag


# ============================================================
# MISSION RUNNER
# ============================================================
def run_mission(vehicle, mission_profile, verbose=True, plot=True):
    batt = vehicle.network.battery
    battery_energy_Wh = batt.specific_energy * batt.mass_properties.mass
    soc_Wh = battery_energy_Wh
    results = []

    if verbose:
        print("\n=== Multi-Segment Mission Analysis ===")
        print(f"Battery total energy: {battery_energy_Wh/1000.0:.3f} kWh ({battery_energy_Wh:.1f} Wh)")

    for seg in mission_profile:
        energy_used_Wh, soc_Wh, diag = run_segment(vehicle, seg, soc_Wh, verbose=False)
        results.append({
            'type': seg.get('type', 'unknown'),
            'time_s': seg.get('time_s', 0),
            'energy_used_Wh': energy_used_Wh,
            'remaining_Wh': soc_Wh,
            'diag': diag
        })

        if verbose:
            print(f"→ {seg.get('type','?').upper():8s} | Used: {energy_used_Wh/1000.0:.3f} kWh | Remaining: {max(soc_Wh,0)/1000.0:.3f} kWh")

        if soc_Wh <= 0:
            if verbose:
                print("❌ Battery depleted! Mission aborted.")
            break

    total_used_Wh = sum([r['energy_used_Wh'] for r in results])

    if verbose:
        print("\n=== Mission Summary ===")
        print(f"Total energy used: {total_used_Wh/1000.0:.3f} kWh")
        print(f"Battery remaining: {max(soc_Wh,0)/1000.0:.3f} kWh")
        print(f"Total segments:    {len(results)}")

    if plot:
        plot_energy_profile(results)

    return results, soc_Wh


# ============================================================
# PLOTTING UTILITY
# ============================================================
def plot_energy_profile(results):
    segs = [r['type'].upper() for r in results]
    used = [r['energy_used_Wh'] / 1000.0 for r in results]
    rem = [r['remaining_Wh'] / 1000.0 for r in results]

    plt.figure(figsize=(8, 5))
    plt.bar(segs, used, color='red', label='Energy Used (kWh)')
    plt.plot(segs, rem, 'o-', color='green', label='Remaining Energy (kWh)')
    plt.title('Mission Energy Profile')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    vehicle = build_vehicle()
    hover_result = run_hover(vehicle, altitude_m=1800.0, hover_time_s=300.0, realistic=True, verbose=True)

    mission_profile = [
        {'type': 'hover', 'time_s': 300, 'altitude_m': 0},
        {'type': 'climb', 'time_s': 120, 'climb_rate_mps': 2.0, 'altitude_m': 100},
        {'type': 'cruise', 'time_s': 600, 'speed_mps': 20.0, 'altitude_m': 100},
        {'type': 'descent', 'time_s': 120, 'altitude_m': 0},
        {'type': 'hover', 'time_s': 180, 'altitude_m': 0}
    ]

    run_mission(vehicle, mission_profile, verbose=True, plot=True)
