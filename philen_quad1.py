# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 09:45:00 2025

@author: user
"""
# ----------------------------------------------------------------------
#   Minimal Quadcopter Setup for 1 kg MTOW (SUAVE-compatible / Spyder-ready)
#   - Uses real SUAVE if available
#   - Otherwise falls back to simple shim classes so it can run in Spyder
# ----------------------------------------------------------------------

from types import SimpleNamespace

# --- Optional SUAVE import with graceful fallback ---------------------
import SUAVE
from SUAVE.Core import Units
from SUAVE.Components.Energy.Networks import Network
from SUAVE.Components.Energy.Converters import Propeller, Motor
from SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion import Lithium_Ion



# ----------------------------------------------------------------------
#   Vehicle Setup
# ----------------------------------------------------------------------

def setup_vehicle():
    # Initialize Vehicle
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Quadcopter_1kg'

    # Mass Properties
    vehicle.mass_properties.takeoff = 1.0 * Units.kg
    vehicle.mass_properties.operating_empty = 0.8 * Units.kg
    vehicle.mass_properties.payload = 0.2 * Units.kg
    vehicle.reference_area = 0.05   # small body reference

    # Energy Network (Quadcopter)
    net = Network()
    net.tag = 'Quad_Network'

    # Battery
    battery = Lithium_Ion()
    battery.mass_properties.mass = 0.20 * Units.kg
    battery.specific_energy = 200.0 * Units.Wh / Units.kg
    battery.resistance = 0.01
    # ensure battery has an outputs.power field for linking (shim already has)
    if not hasattr(battery, 'outputs'):
        battery.outputs = SimpleNamespace(power=None)
    net.battery = battery

    # Motors + Propellers (4 identical)
    motor_mass = 0.05 * Units.kg
    prop_radius = 0.07 * Units.m   # ~7 cm radius (≈ 5.5" prop)

    net.propellers = []
    for i in range(4):
        prop = Propeller()
        # for some SUAVE versions, .tag may be read-only; guard accordingly
        try:
            prop.tag = f'prop_{i+1}'
        except Exception:
            pass
        # set properties that exist in both real and shim classes
        if hasattr(prop, 'number_blades'):
            prop.number_blades = 2
        if hasattr(prop, 'radius'):
            prop.radius = prop_radius
        if hasattr(prop, 'design_Cl'):
            prop.design_Cl = 0.5
        if hasattr(prop, 'design_power'):
            prop.design_power = 100.0  # W

        motor = Motor()
        if hasattr(motor, 'mass_properties'):
            motor.mass_properties.mass = motor_mass
        if hasattr(motor, 'efficiency'):
            motor.efficiency = 0.85

        # Link motor to prop, battery to motor (no-op for shims)
        if not hasattr(prop, 'inputs'):
            prop.inputs = SimpleNamespace()
        if not hasattr(motor, 'inputs'):
            motor.inputs = SimpleNamespace()
        if not hasattr(motor, 'outputs'):
            motor.outputs = SimpleNamespace()

        # Create the attributes if missing
        if not hasattr(prop.inputs, 'power'):
            prop.inputs.power = None
        if not hasattr(motor.inputs, 'power_in'):
            motor.inputs.power_in = None
        if not hasattr(motor.outputs, 'power'):
            motor.outputs.power = None
        if not hasattr(battery, 'outputs'):
            battery.outputs = SimpleNamespace(power=None)

        prop.inputs.power = getattr(motor.outputs, 'power', None)
        motor.inputs.power_in = getattr(battery.outputs, 'power', None)

        net.propellers.append(prop)

    # attach network (keep both real SUAVE and shim happy)
    if hasattr(vehicle, 'append_component'):
        vehicle.append_component(net)
    else:
        # some SUAVE versions use vehicle.networks list
        if not hasattr(vehicle, 'networks'):
            vehicle.networks = []
        vehicle.networks.append(net)

    return vehicle

# ----------------------------------------------------------------------
#   Convergence Check
# ----------------------------------------------------------------------

def convergence(vehicle, tol=1e-2, max_iter=20):
    prev_mass = 0.0
    for i in range(max_iter):
        # In a real case you’d call SUAVE analyses here (sizing, performance, etc.)
        current_mass = float(vehicle.mass_properties.takeoff)
        if abs(current_mass - prev_mass) < tol:
            print(f"Converged at iteration {i+1}: {current_mass:.3f} kg")
            break
        prev_mass = current_mass
    else:
        print("Did not converge within max iterations")

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

if __name__ == '__main__':
    vehicle = setup_vehicle()
    convergence(vehicle)