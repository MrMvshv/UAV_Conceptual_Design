
import SUAVE
assert SUAVE.__version__ == '2.5.2', 'Requires SUAVE 2.5.2'
import numpy as np
import matplotlib.pyplot as plt

# Core + physical attributes
from SUAVE.Core import Data, Units
from SUAVE.Attributes.Gases import Air

# Components
from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller
from SUAVE.Components.Energy.Converters.Motor import Motor
from SUAVE.Components.Energy.Converters.Propeller import Propeller
from SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion import Lithium_Ion


# Methods
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass

# Analyses
from SUAVE.Analyses import Energy
from SUAVE.Analyses import Atmospheric

# Hover mission segment
from SUAVE.Analyses.Mission.Segments.Hover.Hover import Hover


def build_vehicle(rpm=None):
    vehicle = Data()
    vehicle.tag = 'quad_hover_demo'

    # --- Mass properties ---
    vehicle.mass_properties = Data()
    vehicle.mass_properties.takeoff = 3.1  # kg
    vehicle.mass_properties.operating_empty = 2.9
    vehicle.mass_properties.payload = 0.2

    # --- Energy Network ---
    network = Battery_Propeller()
    network.number_of_engines = 4
    network.identical_propellers = True

    # --- Propeller setup ---
    prop = Propeller()
    prop.tag = 'hover_prop'
    prop.tip_radius = 0.15 * Units.meter
    prop.hub_radius = 0.015 * Units.meter
    prop.efficiency = 0.85
    prop.design_thrust = vehicle.mass_properties.takeoff * 9.81 / 4
    prop.number_of_blades = 2
    prop.angular_velocity = 4500 * 2 * np.pi / 60  # rad/s
    if rpm is not None:
        prop.tip_speed = 2 * np.pi * rpm * prop.tip_radius / 60.0
        prop.design_rpm = rpm



    network.propeller = prop

    # --- Motor setup ---
    motor = Motor()
    motor.efficiency = 0.9
    motor.mass_properties = Data()
    motor.mass_properties.mass = 0.15
    motor.nominal_voltage = 22.0
    network.motor = motor

    # --- Battery setup ---
    battery = Lithium_Ion()
    battery.mass_properties = Data()
    battery.mass_properties.mass = 0.3  # kg
    initialize_from_mass(battery)
    print(f"[DEBUG] SUAVE battery specs:")
    print(f"  Specific energy: {battery.specific_energy/3600:.1f} Wh/kg")
    print(f"  Specific power:  {battery.specific_power/3600:.1f} W/kg")
    total_energy_J = getattr(battery, "max_energy", None)
    if total_energy_J is None or total_energy_J == 0:
        total_energy_J = battery.specific_energy * battery.mass_properties.mass

    print(f"  Energy total:    {total_energy_J/3.6e6:.3f} kWh")

    print(f"  Battery mass:    {battery.mass_properties.mass:.3f} kg")

    network.battery = battery

    # --- Assign network to vehicle ---
    vehicle.network = network
    return vehicle


def build_analyses(vehicle):
    analyses = Data()

    # Atmosphere
    atmosphere = Atmospheric.US_Standard_1976()
    analyses.atmosphere = atmosphere

    # Energy (powertrain)
    energy = Energy.Energy()
    energy.network = vehicle.network
    analyses.energy = energy

    return analyses


def run_hover(vehicle, analyses, altitude_m=1800.0, hover_time_s=300.0, realistic=True, verbose=True):
    """
    Runs a hover simulation for the given vehicle.
    Adds realistic efficiency factors, margins, and battery drain tracking.
    """

    # --- [1] Atmospheric conditions ---
    rho_val = 1.225 * np.exp(-altitude_m / 8500.0)  # simple exponential model
    if verbose:
        print("\n[1] Creating hover segment ...")
        print(f"Air density: {rho_val:.3f} kg/m³")

    # --- [2] Vehicle + prop info ---
    W = vehicle.mass_properties.takeoff * 9.81  # N
    R = vehicle.network.propeller.tip_radius
    A = np.pi * R ** 2
    batt = vehicle.network.battery

    # Detect whether SUAVE stored energy in J/kg or Wh/kg
    spec_E = getattr(batt, 'specific_energy', 0.0)
    if spec_E > 1e4:  # probably J/kg
        spec_E_Wh_per_kg = spec_E / 3600.0
    else:
        spec_E_Wh_per_kg = spec_E

    battery_energy_kWh = spec_E_Wh_per_kg * batt.mass_properties.mass / 1000.0

    if verbose:
        print(f"Battery specific energy: {spec_E_Wh_per_kg/1000:.3f} kWh/kg (≈ {spec_E_Wh_per_kg:.0f} Wh/kg)")
        print(f"Battery energy total:    {battery_energy_kWh:.3f} kWh")
        print(f"Battery mass:            {batt.mass_properties.mass:.3f} kg")

    # --- [3] Ideal induced velocity ---
    v_induced = np.sqrt(W / (2 * rho_val * A))
    P_ideal = W * v_induced

    # --- [4] Realism modifiers ---
    if realistic:
        eta_prop = 0.8          # prop efficiency (80%)
        eta_motor = 0.9         # motor+ESC (90%)
        eta_total = eta_prop * eta_motor
        interference_factor = 0.9   # 10% loss from rotor interference
        control_margin = 1.1        # 10% hover thrust reserve
        structure_factor = 1.05     # 5% extra structural weight

        W_eff = W * control_margin * structure_factor
        P_total = (W_eff * v_induced / eta_total) / interference_factor
    else:
        P_total = P_ideal

    # --- [5] Battery energy & endurance ---
    E_hover = (P_total * hover_time_s) / 3.6e6  # J → kWh
    SOC_drop = E_hover / battery_energy_kWh
    endurance_s = hover_time_s / SOC_drop if SOC_drop > 0 else np.inf
    endurance_min = endurance_s / 60.0

    if verbose:
        print("\n--- Hover Performance ---")
        print(f"Weight (N):               {W:.2f}")
        print(f"Disk area (m²):           {A:.3f}")
        print(f"Induced velocity (m/s):   {v_induced:.2f}")
        print(f"Total power (W):          {P_total:.1f}")
        print(f"Energy used (kWh):        {E_hover:.3f}")
        print(f"Battery energy (kWh):     {battery_energy_kWh:.3f}")
        print(f"SOC drop (fraction):      {SOC_drop:.3f}")
        print(f"Estimated endurance (min):{endurance_min:.1f}")

    return {
        'P_total_W': P_total,
        'E_hover_kWh': E_hover,
        'SOC_drop': SOC_drop,
        'endurance_min': endurance_min,
        'radius_m': R,
        'altitude_m': altitude_m,
        'rho': rho_val
    }


def run_hover_sweep(param_name, param_values, vehicle_builder, altitude_m=1800.0, hover_time_s=300.0, rpm=None, realistic=True):
    """
    Sweeps a given vehicle parameter and records hover performance.
    Optionally sets propeller RPM.
    """
    results = []
    print(f"\n[🚀] Running hover sweep for: {param_name}\n")

    for val in param_values:
        print(f"Testing {param_name} = {val}")

        # --- Build a fresh vehicle for each iteration ---
        vehicle = vehicle_builder()
        target = vehicle
        for key in param_name.split('.')[:-1]:
            target = getattr(target, key)
        setattr(target, param_name.split('.')[-1], val)

        # --- Apply RPM if provided ---
        if rpm is not None:
            prop = vehicle.network.propeller
            prop.design_rpm = rpm
            prop.tip_speed = 2 * np.pi * rpm * prop.tip_radius / 60.0

        # --- Run the hover simulation ---
        res = run_hover(vehicle, None, altitude_m=altitude_m, hover_time_s=hover_time_s, realistic=realistic, verbose=False)
        res['param_value'] = val
        if rpm is not None:
            res['rpm'] = rpm
        results.append(res)

    return results

def plot_hover_results(results_list, x_label='Parameter', title='Hover Performance Sweep'):
    """
    Plots power and endurance vs parameter.
    Supports multiple RPM runs.
    """
    # Group results by RPM if multiple
    rpms = sorted(list(set([r.get('rpm', None) for r in results_list])))

    plt.figure(figsize=(8,6))
    for rpm in rpms:
        subset = [r for r in results_list if r.get('rpm', None) == rpm]
        x = [r['param_value'] for r in subset]
        y1 = [r['P_total_W'] for r in subset]
        plt.plot(x, y1, 'o-', label=f'Power @ {rpm} RPM')

    plt.xlabel(x_label)
    plt.ylabel("Total Hover Power (W)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8,6))
    for rpm in rpms:
        subset = [r for r in results_list if r.get('rpm', None) == rpm]
        x = [r['param_value'] for r in subset]
        y2 = [r['endurance_min'] for r in subset]
        plt.plot(x, y2, 's--', label=f'Endurance @ {rpm} RPM')

    plt.xlabel(x_label)
    plt.ylabel("Endurance (min)")
    plt.title(title + " — Endurance")
    plt.legend()
    plt.grid(True)
    plt.show()

def estimate_battery_mass_for_endurance(vehicle, desired_minutes, hover_power_W):
    """
    Estimate how much battery mass is needed to sustain a target hover endurance.
    Handles both Wh/kg and J/kg units automatically.
    """
    battery = vehicle.network.battery
    spec_E = getattr(battery, 'specific_energy', 0.0)

    # Detect if SUAVE stored it in J/kg (large values like 500k–1e6)
    if spec_E > 10000:
        specific_energy_Whkg = spec_E / 3600.0
    else:
        specific_energy_Whkg = spec_E  # already in Wh/kg

    target_energy_Wh = hover_power_W * (desired_minutes / 60.0)  # W × h = Wh
    required_mass_kg = target_energy_Wh / specific_energy_Whkg

    print(f"\n🔋 To hover {desired_minutes:.1f} minutes at {hover_power_W:.1f} W:")
    print(f"    Required battery energy: {target_energy_Wh/1000:.3f} kWh")
    print(f"    Required battery mass:   {required_mass_kg:.2f} kg\n")

    return required_mass_kg

if __name__ == "__main__":
    vehicle = build_vehicle(rpm=4000)
    analyses = build_analyses(vehicle)

    hover_result = run_hover(vehicle, analyses, altitude_m=1800.0, hover_time_s=300.0)
    estimate_battery_mass_for_endurance(vehicle, desired_minutes=30, hover_power_W=hover_result['P_total_W'])
    #plot_hover_results(hover_result, 'Propeller Tip Radius (m)')

    radii = np.linspace(0.3, 0.8, 6)
    for rpm in [2500, 4000, 6000]:
        print(f"\n=== Running Sweep for RPM = {rpm} ===")
        results = run_hover_sweep(
            param_name='network.propeller.tip_radius',
            param_values=radii,
            vehicle_builder=build_vehicle,
            altitude_m=1800.0,
            hover_time_s=300.0,
            rpm=rpm,
            realistic=True
        )
        plot_hover_results(results, x_label='Propeller Tip Radius (m)', title=f'Hover Performance vs Radius @ {rpm} RPM')
