#!/usr/bin/env python3
"""
fixedwing_energy_aero_avl.py
---------------------------------
Fixed-wing eUAV aerodynamic + energy analysis using SUAVE 2.5.2
with optional AVL-based aerodynamic polars.

Usage:
    Set `USE_AVL = True` to compute real polars via AVL
    (requires AVL executable installed and on PATH).
"""

import numpy as np
import matplotlib.pyplot as plt
from SUAVE.Core import Units, Data
from SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion import Lithium_Ion
from SUAVE.Components.Energy.Converters.Motor import Motor
from SUAVE.Components.Energy.Converters.Propeller import Propeller
from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass
from SUAVE.Methods.Aerodynamics import AVL

# ============================================================
# USER OPTION: USE AVL
# ============================================================
USE_AVL = True   # Set to False to use simple parabolic model

# ============================================================
# VEHICLE BUILDER
# ============================================================
def build_fixedwing():
    """Define a fixed-wing UAV with realistic geometry."""
    from SUAVE.Components import Wings, Fuselages
    from SUAVE.Vehicle import Vehicle

    vehicle = Vehicle()
    vehicle.tag = 'fixed_wing_uav'

    # --- Mass ---
    vehicle.mass_properties.takeoff = 3.0  # kg
    vehicle.mass_properties.operating_empty = 2.8
    vehicle.mass_properties.max_takeoff = 3.0

    # --- Main Wing ---
    main_wing = Wings.Main_Wing()
    main_wing.tag = 'main_wing'
    main_wing.aspect_ratio = 8.0
    main_wing.sweep = 0.0 * Units.deg
    main_wing.taper = 0.8
    main_wing.thickness_to_chord = 0.12
    main_wing.span_efficiency = 0.9
    main_wing.symmetric = True
    main_wing.areas.reference = 0.4
    main_wing.chords.mean_aerodynamic = 0.2
    main_wing.spans.projected = np.sqrt(main_wing.aspect_ratio * main_wing.areas.reference)
    main_wing.twists.root = 0.0 * Units.deg
    main_wing.twists.tip  = 0.0 * Units.deg
    main_wing.Airfoil = 'NACA2412'
    vehicle.append_component(main_wing)

    # --- Fuselage (optional geometry for AVL completeness) ---
    fuselage = Fuselages.Fuselage()
    fuselage.areas.front_projected = 0.015
    fuselage.areas.side_projected = 0.1
    vehicle.append_component(fuselage)

    # --- Propulsion network ---
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
    batt.mass_properties.mass = 0.8
    batt.specific_energy = 250.0
    initialize_from_mass(batt)
    net.battery = batt

    vehicle.append_component(net)
    return vehicle


# ============================================================
# AERODYNAMICS: AVL or Parabolic
# ============================================================
def compute_aero(vehicle, V, altitude_m, use_avl=False):
    """Compute CL, CD, and Power required using AVL or analytic model."""
    rho = 1.225 * np.exp(-altitude_m / 8500.0)
    S = vehicle.wings.main_wing.areas.reference
    W = vehicle.mass_properties.takeoff * 9.81
    eta_total = vehicle.networks.battery_propeller.propeller.efficiency * \
                vehicle.networks.battery_propeller.motor.efficiency

    if use_avl:
        print("\n[AVL] Running vortex lattice solver for aerodynamic coefficients...")
        avl = AVL()
        avl.geometry = vehicle
        avl.settings.number_of_points = 8
        avl.settings.number_of_spanwise_vortices = 8
        results = avl.evaluate()

        # Interpolate CL and CD for current AoA / speed
        CL = float(results.CL)
        CD = float(results.CD)
        print(f"[AVL] Results → CL={CL:.3f}, CD={CD:.4f}")
    else:
        # --- Analytic model (Fidelity_Zero approximation) ---
        AR = vehicle.wings.main_wing.aspect_ratio
        e = vehicle.wings.main_wing.span_efficiency
        CD0 = 0.025
        CL = W / (0.5 * rho * V**2 * S)
        CD = CD0 + CL**2 / (np.pi * e * AR)

    D = 0.5 * rho * V**2 * S * CD
    P_req = D * V / eta_total
    L_to_D = CL / CD

    return {
        'rho': rho,
        'CL': CL,
        'CD': CD,
        'Drag_N': D,
        'Power_W': P_req,
        'L_to_D': L_to_D
    }


# ============================================================
# SEGMENT ANALYSIS
# ============================================================
def run_segment(vehicle, seg, soc_Wh, use_avl=False):
    seg_type = seg.get('type', 'cruise')
    altitude = seg.get('altitude_m', 0)
    time_s = seg.get('time_s', 60)
    V = seg.get('speed_mps', 20)

    aero = compute_aero(vehicle, V, altitude, use_avl=use_avl)

    if seg_type == 'climb':
        climb_rate = seg.get('climb_rate_mps', 2.0)
        P = aero['Power_W'] + vehicle.mass_properties.takeoff * 9.81 * climb_rate
    elif seg_type == 'descent':
        P = aero['Power_W'] * 0.5
    else:
        P = aero['Power_W']

    E_Wh = P * time_s / 3600.0
    new_soc = max(soc_Wh - E_Wh, 0.0)

    print(f"\n--- {seg_type.upper()} SEGMENT ---")
    print(f"Airspeed: {V:.1f} m/s | Altitude: {altitude:.0f} m | Duration: {time_s/60:.1f} min")
    print(f"CL = {aero['CL']:.3f}, CD = {aero['CD']:.4f}, L/D = {aero['L_to_D']:.1f}")
    print(f"Drag = {aero['Drag_N']:.2f} N | Power = {P:.1f} W | Energy = {E_Wh:.2f} Wh")
    print(f"Battery remaining: {new_soc:.2f} Wh\n{'-'*55}")

    return {'seg_type': seg_type, 'power_W': P, 'energy_used_Wh': E_Wh,
            'remaining_Wh': new_soc, 'L_to_D': aero['L_to_D']}, new_soc


# ============================================================
# MISSION EXECUTION
# ============================================================
def run_mission(vehicle, mission_profile, use_avl=False, plot=True):
    batt = vehicle.networks.battery_propeller.battery
    soc_Wh = batt.specific_energy * batt.mass_properties.mass

    print("\n=== FIXED-WING AERO + ENERGY ANALYSIS ===")
    print(f"Using AVL: {use_avl}")
    print(f"Initial battery energy: {soc_Wh:.1f} Wh ({soc_Wh/1000:.3f} kWh)")

    results = []
    for seg in mission_profile:
        res, soc_Wh = run_segment(vehicle, seg, soc_Wh, use_avl=use_avl)
        results.append(res)
        if soc_Wh <= 0:
            print("❌ Battery depleted — mission ended early.")
            break

    if plot:
        plot_energy_profile(results)
    return results


# ============================================================
# PLOTTING
# ============================================================
def plot_energy_profile(results):
    segs = [r['seg_type'].upper() for r in results]
    used = [r['energy_used_Wh'] / 1000 for r in results]
    rem = [r['remaining_Wh'] / 1000 for r in results]

    plt.figure(figsize=(8, 5))
    plt.bar(segs, used, color='orange', label='Energy Used (kWh)')
    plt.plot(segs, rem, 'o-', color='green', label='Remaining Energy (kWh)')
    plt.grid(True)
    plt.title('Fixed-Wing Mission Energy Profile')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.show()

def compute_wing_lift(vehicle, V, altitude_m, use_avl=False):
    """
    Returns wing lift (N) and a small diagnostic dict.
    Uses compute_aero_forces (or AVL if enabled) to get CL.
    """
    aero = compute_aero(vehicle, V, altitude_m) if not use_avl else compute_aero(vehicle, V, altitude_m, use_avl=True)
    rho = aero['rho']
    CL = aero['CL']
    S = vehicle.reference_area
    L = 0.5 * rho * V**2 * S * CL
    W = vehicle.mass_properties.takeoff * 9.81

    return {
        'Lift_N': L,
        'Weight_N': W,
        'Lift_minus_weight_N': L - W,
        'CL': CL,
        'S_ref_m2': S,
        'rho': rho
    }

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    #use module's vehicle definition
    #vehicle = build_fixedwing()

    #use external vehicle definition
    from fixedwing_vehicle_definition import setup_fixedwing_vehicle
    vehicle = setup_fixedwing_vehicle()

    print("TAG:", vehicle.tag)
    print("Takeoff mass (kg):", vehicle.mass_properties.takeoff)
    print("Reference area (m^2):", vehicle.reference_area)
    # check VEHICLE exists
    print("\n--- COMPONENT CHECK ---")
    for name, group in [('Wings', vehicle.wings),
                        ('Fuselages', vehicle.fuselages),
                        ('Networks', vehicle.networks)]:
        print(f"\n{name}:")
        for comp in group:
            print(" •", comp.tag)


    mission = [
        {'type': 'climb', 'time_s': 120, 'speed_mps': 15, 'climb_rate_mps': 2, 'altitude_m': 0},
        {'type': 'cruise', 'time_s': 600, 'speed_mps': 22, 'altitude_m': 100},
        {'type': 'descent', 'time_s': 120, 'speed_mps': 15, 'climb_rate_mps': -1, 'altitude_m': 0},
    ]

    #run mission using AVL generated polars
    #run_mission(vehicle, mission, use_avl=USE_AVL, plot=True)

    #use empirical polars(inaccurate)
    results = run_mission(vehicle, mission, use_avl=False, plot=True)

    res = compute_wing_lift(vehicle, V=22.0, altitude_m=100.0, use_avl=False)
    print(f"Wing Lift = {res['Lift_N']:.2f} N | Weight = {res['Weight_N']:.2f} N | Delta = {res['Lift_minus_weight_N']:.2f} N")

    print(f"RESULTS:\n{results}")
