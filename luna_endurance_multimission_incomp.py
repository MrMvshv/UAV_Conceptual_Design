# ==========================================================
#  LUNA Mission Analysis — Multi-Segment Energy Study
#  Compatible with SUAVE 2.5.2
#  Author: GPT-5 🧠 | Modified for multi-phase analysis
# ==========================================================

import SUAVE
import numpy as np
from SUAVE.Core import Units, Data
from SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion import Lithium_Ion
from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller
from SUAVE.Components.Energy.Converters import Motor
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Vehicle setup
# ----------------------------------------------------------
def build_vehicle(rpm=4000):
    vehicle = Data()
    vehicle.tag = 'quad_hover_demo'

    # --- Mass properties ---
    vehicle.mass_properties = Data()
    vehicle.mass_properties.takeoff = 3.1  # kg
    vehicle.mass_properties.operating_empty = 2.9
    vehicle.mass_properties.payload = 0.2

    # --- Energy network ---
    network = Battery_Propeller()
    network.number_of_engines = 4
    network.identical_propellers = True

    # --- Propeller setup ---
    prop = Data()
    prop.tag = 'hover_prop'
    prop.tip_radius = 0.15 * Units.meter
    prop.hub_radius = 0.015 * Units.meter
    prop.efficiency = 0.85
    prop.number_of_blades = 2
    prop.angular_velocity = rpm * 2 * np.pi / 60
    prop.design_rpm = rpm
    network.propeller = prop

    # --- Motor setup ---
    motor = Motor()
    motor.efficiency = 0.9
    motor.nominal_voltage = 22.2  # 6S battery voltage
    motor.mass_properties = Data()
    motor.mass_properties.mass = 0.15
    network.motor = motor

    # --- Battery setup (6S config) ---
    battery = Lithium_Ion()
    battery.mass_properties = Data()
    battery.mass_properties.mass = 0.3  # 300 g
    battery.nominal_voltage = 22.2
    initialize_from_mass(battery)
    network.battery = battery

    vehicle.network = network
    return vehicle


# ----------------------------------------------------------
# Utility function — reusable plotting tool
# ----------------------------------------------------------
def plot_mission_profile(time_vec, power_vec, soc_vec, segment_labels):
    plt.figure(figsize=(10,5))
    plt.subplot(2,1,1)
    plt.plot(time_vec, np.array(power_vec)/1000, label="Power (kW)")
    plt.ylabel("Power [kW]")
    plt.title("Mission Power Profile")
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(time_vec, np.array(soc_vec)*100, label="SOC [%]", color="orange")
    plt.ylabel("Battery SOC [%]")
    plt.xlabel("Time [s]")
    plt.grid(True)

    # Annotate segments
    for label, t in segment_labels:
        plt.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
        plt.text(t, 50, label, rotation=90, va='center', ha='right')

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# Core energy calculation reused by all segments
# ----------------------------------------------------------
def power_required(vehicle, rho, thrust_factor=1.0):
    W = vehicle.mass_properties.takeoff * 9.81
    R = vehicle.network.propeller.tip_radius
    A = np.pi * R**2
    v_induced = np.sqrt(W / (2 * rho * A))
    P_hover = (W * v_induced) / (vehicle.network.propeller.efficiency *
                                 vehicle.network.motor.efficiency)
    return P_hover * thrust_factor


# ----------------------------------------------------------
# Hover segment
# ----------------------------------------------------------
def run_hover(vehicle, altitude_m=1800.0, duration_s=300.0):
    rho = 1.225 * np.exp(-altitude_m / 8500)
    P_hover = power_required(vehicle, rho)

    battery = vehicle.network.battery
    E_batt_kWh = battery.specific_energy * battery.mass_properties.mass / 3600.0

    E_used = P_hover * duration_s / 3.6e6
    soc_drop = E_used / E_batt_kWh
    endurance_min = duration_s / 60.0 / soc_drop if soc_drop > 0 else np.inf

    print(f"\n[1] Hover segment initialized ...")
    print(f"Altitude: {altitude_m:.0f} m | Air density: {rho:.3f} kg/m³")
    print("\n--- Hover Performance Summary ---")
    print(f"Weight (N):               {vehicle.mass_properties.takeoff * 9.81:.2f}")
    print(f"Rotor disk area (m²):     {np.pi * vehicle.network.propeller.tip_radius**2:.3f}")
    print(f"Total power (W):          {P_hover:.1f}")
    print(f"Energy used (kWh):        {E_used:.3f}")
    print(f"Battery energy (kWh):     {E_batt_kWh:.3f}")
    print(f"SOC drop (fraction):      {soc_drop:.3f}")
    print(f"Estimated endurance (min):{endurance_min:.1f}")

    return duration_s, P_hover, soc_drop


# ----------------------------------------------------------
# Climb segment
# ----------------------------------------------------------
def run_climb(vehicle, climb_rate=3.0, duration_s=60.0, altitude_m=1800.0):
    rho = 1.225 * np.exp(-altitude_m / 8500)
    P_hover = power_required(vehicle, rho)
    W = vehicle.mass_properties.takeoff * 9.81
    P_climb = P_hover + W * climb_rate  # climb power = hover + potential rate
    E_used = P_climb * duration_s / 3.6e6
    print(f"\n[2] Climb segment: +{climb_rate} m/s for {duration_s}s")
    print(f"Total climb power (W): {P_climb:.1f}")
    return duration_s, P_climb, E_used


# ----------------------------------------------------------
# Cruise segment
# ----------------------------------------------------------
def run_cruise(vehicle, airspeed=20.0, duration_s=600.0, altitude_m=1800.0):
    rho = 1.225 * np.exp(-altitude_m / 8500)
    drag_coefficient = 0.7  # typical drag estimate for multirotor frame
    frontal_area = 0.05     # m²
    drag = 0.5 * rho * airspeed**2 * drag_coefficient * frontal_area
    P_cruise = drag * airspeed / (vehicle.network.motor.efficiency)
    E_used = P_cruise * duration_s / 3.6e6
    print(f"\n[3] Cruise segment: {airspeed} m/s for {duration_s}s")
    print(f"Total cruise power (W): {P_cruise:.1f}")
    return duration_s, P_cruise, E_used


# ----------------------------------------------------------
# Descent segment
# ----------------------------------------------------------
def run_descent(vehicle, descent_rate=2.0, duration_s=60.0, altitude_m=1800.0):
    rho = 1.225 * np.exp(-altitude_m / 8500)
    P_hover = power_required(vehicle, rho)
    W = vehicle.mass_properties.takeoff * 9.81
    P_descent = max(P_hover - 0.6 * W * descent_rate, 0.1 * P_hover)  # save energy
    E_used = P_descent * duration_s / 3.6e6
    print(f"\n[4] Descent segment: -{descent_rate} m/s for {duration_s}s")
    print(f"Total descent power (W): {P_descent:.1f}")
    return duration_s, P_descent, E_used


# ----------------------------------------------------------
# Mission runner
# ----------------------------------------------------------
def run_full_mission(vehicle):
    battery = vehicle.network.battery
    E_batt_kWh = battery.specific_energy * battery.mass_properties.mass / 3600.0

    time_vec = [0]
    power_vec = []
    soc_vec = [1.0]
    t = 0
    E_used_total = 0
    labels = []

    for name, seg in [
        ("Hover", lambda: run_hover(vehicle, duration_s=180)),
        ("Climb", lambda: run_climb(vehicle, duration_s=60)),
        ("Cruise", lambda: run_cruise(vehicle, duration_s=600)),
        ("Descent", lambda: run_descent(vehicle, duration_s=60))
    ]:
        dur, P, E_used = seg()
        E_used_total += E_used
        soc_drop = E_used / E_batt_kWh
        t += dur
        time_vec.append(t)
        power_vec.append(P)
        soc_vec.append(max(1.0 - (E_used_total / E_batt_kWh), 0))
        labels.append((name, t))

    print("\n=== Mission Summary ===")
    print(f"Total energy used (kWh): {E_used_total:.3f}")
    print(f"Battery remaining (kWh): {max(E_batt_kWh - E_used_total, 0):.3f}")
    print(f"Total flight time (min): {t/60:.1f}")
    print(f"Final SOC (%): {max(100*(1 - E_used_total/E_batt_kWh), 0):.1f}")

    plot_mission_profile(time_vec, power_vec, soc_vec, labels)


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
if __name__ == "__main__":
    vehicle = build_vehicle()
