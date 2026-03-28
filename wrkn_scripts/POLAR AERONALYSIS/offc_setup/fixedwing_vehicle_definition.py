# fixedwing_vehicle_definition.py
import numpy as np
import copy            # for deepcopy
import SUAVE
from SUAVE.Core import Units
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass

from SUAVE.Components import Wings, Fuselages
from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller
from SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion import Lithium_Ion
from SUAVE.Components.Energy.Converters.Propeller import Propeller
from SUAVE.Components.Energy.Converters.Motor import Motor


def setup_fixedwing_vehicle():
    """
    Fixed-wing configuration extracted from full eVTOL definition.
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
