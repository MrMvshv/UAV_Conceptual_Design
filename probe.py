
import SUAVE
assert SUAVE.__version__ == '2.5.2', 'Requires SUAVE 2.5.2'
import numpy as np

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


def build_vehicle():
    vehicle = Data()
    vehicle.tag = 'quad_hover_demo'

    # --- Mass properties ---
    vehicle.mass_properties = Data()
    vehicle.mass_properties.takeoff = 10.0  # kg
    vehicle.mass_properties.operating_empty = 7.0
    vehicle.mass_properties.payload = 2.0

    # --- Energy Network ---
    network = Battery_Propeller()
    network.number_of_engines = 4
    network.identical_propellers = True

    # --- Propeller setup ---
    prop = Propeller()
    prop.tag = 'hover_prop'
    prop.tip_radius = 0.6 * Units.meter
    prop.hub_radius = 0.05 * Units.meter
    prop.efficiency = 0.85
    prop.design_thrust = vehicle.mass_properties.takeoff * 9.81 / 4
    prop.number_of_blades = 2
    prop.angular_velocity = 2500. * 2 * np.pi / 60  # rad/s
    network.propeller = prop

    # --- Motor setup ---
    motor = Motor()
    motor.efficiency = 0.9
    motor.mass_properties = Data()
    motor.mass_properties.mass = 0.3
    motor.nominal_voltage = 48.0
    network.motor = motor

    # --- Battery setup ---
    battery = Lithium_Ion()
    battery.mass_properties = Data()
    battery.mass_properties.mass = 3.0  # kg
    initialize_from_mass(battery)
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


def run_hover(vehicle, analyses, altitude_m=1800.0, hover_time_s=300.0):
    print("\n[1] Creating hover segment ...")
    segment = Hover()
    segment.tag = 'hover_segment'
    segment.analyses = analyses
    segment.state = Data()
    segment.state.conditions = Data()
    segment.state.conditions.altitude = altitude_m

    weight_N = vehicle.mass_properties.takeoff * 9.81
    segment.state.unknowns = Data()
    segment.state.unknowns.thrust = weight_N
    segment.time = hover_time_s

    # --- Atmosphere ---
    rho = analyses.atmosphere.compute_values(altitude_m, 0.0).density
    rho_val = float(np.atleast_1d(rho)[0])
    print(f"Air density: {rho_val:.3f} kg/m³")


    # --- Propeller power ---
    network = vehicle.network
    A = network.number_of_engines * np.pi * (network.propeller.tip_radius**2)
    v_induced = np.sqrt(weight_N / (2 * rho_val * A))
    P_induced = weight_N * v_induced
    P_hover = P_induced / (network.propeller.efficiency * network.motor.efficiency)
    E_hover = P_hover * hover_time_s

   # --- Battery metrics ---
    battery = vehicle.network.battery
    E_battery = battery.specific_energy * battery.mass_properties.mass
    print(f"Battery specific energy: {battery.specific_energy/3.6e6:.3f} kWh/kg")
    SOC_drop = E_hover / E_battery

    results = Data()
    results.air_density = rho
    results.weight = weight_N
    results.induced_velocity = v_induced
    results.total_power = P_hover
    results.energy_used = E_hover
    results.battery_energy = E_battery
    results.soc_drop = SOC_drop
    results.hover_time = hover_time_s
    results.endurance_est = hover_time_s / SOC_drop if SOC_drop > 0 else None

    print("\n--- Hover Performance ---")
    print(f"Weight (N):               {weight_N:.2f}")
    print(f"Disk area (m²):           {A:.3f}")
    print(f"Induced velocity (m/s):   {v_induced:.2f}")
    print(f"Total power (W):          {P_hover:.1f}")
    print(f"Energy used (kWh):        {E_hover/3.6e6:.3f}")
    print(f"Battery energy (kWh):     {E_battery/3.6e6:.3f}")
    print(f"SOC drop (fraction):      {SOC_drop:.3f}")
    print(f"Estimated endurance (s):  {results.endurance_est:.1f}")
    print(f"Estimated endurance (min):{results.endurance_est/60:.1f}")

    return results


if __name__ == "__main__":
    vehicle = build_vehicle()
    analyses = build_analyses(vehicle)
    run_hover(vehicle, analyses, altitude_m=1800.0, hover_time_s=300.0)
