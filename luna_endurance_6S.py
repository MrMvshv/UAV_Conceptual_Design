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
    motor.nominal_voltage = 22.2  # 6S Li-ion nominal voltage
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
    Runs a detailed hover simulation for the given vehicle.
    Adds descriptive breakdowns of power components, battery use, and performance tradeoffs.
    """

    # --- Atmospheric setup ---
    rho_val = 1.225 * np.exp(-altitude_m / 8500.0)
    if verbose:
        print("\n[1] Hover segment initialized ...")
        print(f"Altitude: {altitude_m:.0f} m | Air density: {rho_val:.3f} kg/m³")

    # --- Basic geometry and weight ---
    W = vehicle.mass_properties.takeoff * 9.81  # N
    R = vehicle.network.propeller.tip_radius
    A = np.pi * R ** 2
    batt = vehicle.network.battery

    # --- Battery parameters ---
    spec_E = getattr(batt, 'specific_energy', 0.0)
    spec_E_Whkg = spec_E / 3600.0 if spec_E > 1e4 else spec_E
    battery_energy_kWh = spec_E_Whkg * batt.mass_properties.mass / 1000.0

    # --- Induced velocity (momentum theory) ---
    v_induced = np.sqrt(W / (2 * rho_val * A))
    P_ideal = W * v_induced

    # --- Efficiency and realism factors ---
    if realistic:
        eta_prop, eta_motor = 0.8, 0.9
        interference_factor, control_margin, structure_factor = 0.9, 1.1, 1.05
        eta_total = eta_prop * eta_motor
        W_eff = W * control_margin * structure_factor
        P_total = (W_eff * v_induced / eta_total) / interference_factor
    else:
        eta_total = 1.0
        P_total = P_ideal

    # --- Energy and endurance calculations ---
    E_hover = (P_total * hover_time_s) / 3.6e6
    SOC_drop = E_hover / battery_energy_kWh
    endurance_min = (hover_time_s / SOC_drop) / 60 if SOC_drop > 0 else np.inf

    # --- Electrical performance ---
    V_nom = vehicle.network.motor.nominal_voltage
    I_draw = P_total / (V_nom * eta_total)
    batt_energy_Wh = battery_energy_kWh * 1000
    capacity_Ah = batt_energy_Wh / V_nom
    C_rate = I_draw / capacity_Ah

    if verbose:
        print("\n--- Hover Performance Summary ---")
        print(f"Weight (N):               {W:.2f}")
        print(f"Rotor disk area (m²):     {A:.3f}")
        print(f"Induced velocity (m/s):   {v_induced:.2f}")
        print(f"Thrust-to-weight margin:  {(control_margin - 1) * 100:.1f}%")
        print(f"Effective efficiencies:   Prop={eta_prop:.2f}, Motor={eta_motor:.2f}, Total={eta_total:.2f}")
        print(f"Total power (W):          {P_total:.1f}")
        print(f"Energy used (kWh):        {E_hover:.3f}")
        print(f"Battery energy (kWh):     {battery_energy_kWh:.3f}")
        print(f"SOC drop (fraction):      {SOC_drop:.3f}")
        print(f"Est. endurance (min):     {endurance_min:.1f}")

        print("\n--- Electrical Load Details ---")
        print(f"Nominal voltage (V):      {V_nom:.1f}")
        print(f"Battery capacity (Ah):    {capacity_Ah:.2f}")
        print(f"Current draw (A):         {I_draw:.1f}")
        print(f"Discharge rate (C):       {C_rate:.1f}")
        if C_rate > 20:
            print("⚠️  C-rate critical (>20C) — high thermal risk!")
        elif C_rate > 10:
            print("⚠️  High C-rate — heavy pack stress expected.")
        else:
            print("✅ C-rate within safe limits.")

        print("\n--- Energy Balance Notes ---")
        print(f"Power-to-weight ratio:    {P_total/W:.3f} W/N (efficiency metric)")
        print(f"Energy-to-weight ratio:   {battery_energy_kWh*3.6e6/W:.1f} J/N (battery sizing metric)")
        print(f"Hover time limit (ideal): {endurance_min:.1f} min (at {altitude_m:.0f} m)")

    return {
        'P_total_W': P_total,
        'E_hover_kWh': E_hover,
        'SOC_drop': SOC_drop,
        'endurance_min': endurance_min,
        'radius_m': R,
        'rho': rho_val,
        'C_rate': C_rate,
        'current_A': I_draw,
        'capacity_Ah': capacity_Ah,
        'v_induced': v_induced,
        'eta_total': eta_total
    }

def plot_trade(parameter_values, results, x_label, y_label, title, log_scale=False):
    """
    Generic trade study plotter for performance parameters.
    Args:
        parameter_values: list/array of independent variable values
        results: list/array of corresponding results
        x_label, y_label, title: plot labels
        log_scale: set True for log-log plotting
    """
    plt.figure(figsize=(7,5))
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    if log_scale:
        plt.xscale('log'); plt.yscale('log')
    plt.plot(parameter_values, results, 'o-', lw=2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.show()



def estimate_battery_mass_for_endurance(vehicle, desired_minutes, hover_power_W):
    """
    Estimate how much battery mass is needed to sustain a target hover endurance.
    Handles both Wh/kg and J/kg units automatically.
    """
    battery = vehicle.network.battery
    spec_E = getattr(battery, 'specific_energy', 0.0)
    if spec_E > 10000:
        specific_energy_Whkg = spec_E / 3600.0
    else:
        specific_energy_Whkg = spec_E

    target_energy_Wh = hover_power_W * (desired_minutes / 60.0)
    required_mass_kg = target_energy_Wh / specific_energy_Whkg

    print(f"\n🔋 To hover {desired_minutes:.1f} minutes at {hover_power_W:.1f} W:")
    print(f"    Required battery energy: {target_energy_Wh/1000:.3f} kWh")
    print(f"    Required battery mass:   {required_mass_kg:.2f} kg\n")

    return required_mass_kg


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    vehicle = build_vehicle(rpm=4000)
    analyses = build_analyses(vehicle)

    hover_result = run_hover(vehicle, analyses, altitude_m=1800.0, hover_time_s=300.0)
    estimate_battery_mass_for_endurance(vehicle, desired_minutes=30, hover_power_W=hover_result['P_total_W'])

    # Example: plot endurance vs battery mass
    battery_masses = np.linspace(0.2, 1.0, 5)
    endurances = []
    for m in battery_masses:
        vehicle.network.battery.mass_properties.mass = m
        res = run_hover(vehicle, analyses, verbose=False)
        endurances.append(res['endurance_min'])

    plot_trade(battery_masses, endurances,
            x_label='Battery Mass (kg)',
            y_label='Endurance (min)',
            title='Endurance vs Battery Mass Tradeoff')