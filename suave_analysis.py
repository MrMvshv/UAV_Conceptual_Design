import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Components.Energy.Networks import Lift_Cruise
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_energy

#vehicle definition
def base_vehicle():
    # Vehicle initialization
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Lift_Cruise_UAV'
    
    # Mass properties (initial estimate - will be refined)
    vehicle.mass_properties.takeoff = 15.0 * Units.kg
    vehicle.mass_properties.operating_empty = 10.0 * Units.kg
    vehicle.mass_properties.max_takeoff = 15.0 * Units.kg
    
    # Basic parameters
    vehicle.reference_area = 1.5 * Units['meters**2']
    vehicle.envelope.ultimate_load = 3.0
    vehicle.envelope.limit_load = 1.5
    
    return vehicle

# aerodynamics configuration
# This function configures the aerodynamics of the vehicle
# It includes the main wing and fuselage
def configure_aerodynamics(vehicle):
    # Main wing
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
    wing.areas.reference = 1.5 * Units['meters**2']
    wing.spans.projected = 3.0 * Units.meters
    wing.aspect_ratio = wing.spans.projected**2 / wing.areas.reference
    wing.sweeps.quarter_chord = 0.0 * Units.deg
    wing.thickness_to_chord = 0.12
    wing.taper = 0.8
    wing.dihedral = 5.0 * Units.degrees
    wing.vertical = False
    wing.symmetric = True
    
    # Add to vehicle
    vehicle.append_component(wing)
    
    # Fuselage (simplified)
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'
    fuselage.lengths.total = 2.0 * Units.meters
    fuselage.width = 0.3 * Units.meters
    fuselage.heights.maximum = 0.4 * Units.meters
    vehicle.append_component(fuselage)
    
    return vehicle

# propulsion configuration
# This function configures the propulsion system of the vehicle
# It includes the lift and cruise propellers, motors, and battery
def configure_propulsion(vehicle):
    # Initialize network
    net = Lift_Cruise()
    net.tag = 'Lift_Cruise_Network'
    
    # Lift propellers (4 for quad configuration)
    for i in range(4):
        lift_prop = SUAVE.Components.Energy.Converters.Propeller()
        lift_prop.tag = f'lift_propeller_{i+1}'
        lift_prop.number_of_blades = 2
        lift_prop.tip_radius = 0.25 * Units.meters
        lift_prop.hub_radius = 0.05 * Units.meters
        lift_prop.angular_velocity = 4000. * Units.rpm
        net.lift_propellers.append(lift_prop)
        
        lift_motor = SUAVE.Components.Energy.Converters.Motor()
        lift_motor.tag = f'lift_motor_{i+1}'
        lift_motor.efficiency = 0.85
        lift_motor.nominal_voltage = 24.0 * Units.volt
        net.lift_motors.append(lift_motor)
    
    # Cruise propeller
    cruise_prop = SUAVE.Components.Energy.Converters.Propeller()
    cruise_prop.tag = 'cruise_propeller'
    cruise_prop.number_of_blades = 3
    cruise_prop.tip_radius = 0.3 * Units.meters
    cruise_prop.hub_radius = 0.075 * Units.meters
    cruise_prop.angular_velocity = 2500. * Units.rpm
    net.cruise_propeller = cruise_prop
    
    cruise_motor = SUAVE.Components.Energy.Converters.Motor()
    cruise_motor.tag = 'cruise_motor'
    cruise_motor.efficiency = 0.9
    cruise_motor.nominal_voltage = 24.0 * Units.volt
    net.cruise_motor = cruise_motor
    
    # Battery
    bat = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    bat.tag = 'battery'
    bat.mass_properties.mass = 4.0 * Units.kg
    bat.energy = 2000.0 * Units.Wh
    bat.voltage = 24.0
    net.battery = bat
    
    # Add to vehicle
    vehicle.append_component(net)
    
    return vehicle

#mission configuration
# This function sets up the mission segments for the UAV
# It includes hover, transition, and cruise segments
def setup_mission(vehicle):
    # Mission initialization
    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'Lift_Cruise_Mission'
    
    # Atmospheric conditions
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    planet = SUAVE.Analyses.Planets.Planet()
    
    # Segment 1: Hover (vertical climb)
    hover = SUAVE.Analyses.Mission.Segments.Hover.Climb()
    hover.tag = 'hover_climb'
    hover.altitude_start = 0.0 * Units.meters
    hover.altitude_end = 50.0 * Units.meters
    hover.climb_rate = 2.0 * Units.meters/Units.seconds
    hover.battery_energy = vehicle.networks.battery.energy
    hover.analyses.extend([atmosphere, planet])
    
    # Segment 2: Transition (tilt or accelerate)
    transition = SUAVE.Analyses.Mission.Segments.Transition.Constant_Acceleration_Constant_Angle()
    transition.tag = 'transition'
    transition.altitude = 50.0 * Units.meters
    transition.air_speed_start = 0.0 * Units.meters/Units.seconds
    transition.air_speed_end = 15.0 * Units.meters/Units.seconds
    transition.acceleration = 1.5 * Units.meters/Units.seconds**2
    transition.pitch_initial = 0.0 * Units.degrees
    transition.pitch_final = 5.0 * Units.degrees
    
    # Segment 3: Cruise
    cruise = SUAVE.Analyses.Mission.Segments.Cruise.Constant_Speed_Constant_Altitude()
    cruise.tag = 'cruise'
    cruise.altitude = 100.0 * Units.meters
    cruise.air_speed = 20.0 * Units.meters/Units.seconds
    cruise.distance = 10.0 * Units.km
    
    # Add segments to mission
    mission.append_segment(hover)
    mission.append_segment(transition)
    mission.append_segment(cruise)
    
    return mission

# Full analysis function
# This function integrates all components and runs the analysis
# It builds the vehicle, configures aerodynamics and propulsion, sets up the mission, and runs the analysis
def full_analysis():
    # Build the vehicle
    vehicle = base_vehicle()
    vehicle = configure_aerodynamics(vehicle)
    vehicle = configure_propulsion(vehicle)
    
    # Setup analyses
    analyses = SUAVE.Analyses.Vehicle()
    
    # Aerodynamics analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)
    
    # Stability analysis (basic)
    stability = SUAVE.Analyses.Stability.Fidelity_Zero()
    analyses.append(stability)
    
    # Energy analysis
    energy = SUAVE.Analyses.Energy.Energy()
    analyses.append(energy)
    
    # Mission analysis
    mission = setup_mission(vehicle)
    analyses.append(mission)
    
    # Finalize analyses
    vehicle.finalize()
    analyses.finalize()
    
    # Run mission
    results = mission.evaluate()
    
    return vehicle, analyses, results


def print_results(results):
    print("\nMission Summary:")
    print("Hover time:", results.segments.hover_climb.conditions.frames.inertial.time[-1,0], "s")
    print("Transition time:", results.segments.transition.conditions.frames.inertial.time[-1,0], "s")
    print("Cruise time:", results.segments.cruise.conditions.frames.inertial.time[-1,0], "s")
    
    print("\nEnergy Consumption:")
    hover_energy = results.segments.hover_climb.conditions.propulsion.battery_energy[:,0]
    print("Hover energy used:", hover_energy[0] - hover_energy[-1], "Wh")
    
    cruise_energy = results.segments.cruise.conditions.propulsion.battery_energy[:,0]
    print("Cruise energy used:", cruise_energy[0] - cruise_energy[-1], "Wh")

    print("Full results:  ", results)
    print("Full Analysis: ", analyses)
    print("Full Vehicle Description: ", vehicle)

if __name__ == "__main__":
    vehicle, analyses, results = full_analysis()
    print_results(results)