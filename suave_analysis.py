import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Components.Energy.Networks import Lift_Cruise
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass

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
    vehicle.reference_area = wing.areas.reference
    
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
    
    # Initialize lists
    net.lift_rotors = []
    net.lift_motors = []
    
    # Add 4 lift rotors
    for i in range(4):
        rotor = SUAVE.Components.Energy.Converters.Rotor()
        rotor.tag = f'rotor_{i+1}'
        rotor.tip_radius = 0.12 * Units.meters
        rotor.hub_radius = 0.02 * Units.meters
        rotor.number_of_blades = 2
        net.lift_rotors.append(rotor)
        
        motor = SUAVE.Components.Energy.Converters.Motor()
        motor.tag = f'motor_{i+1}'
        motor.efficiency = 0.85
        net.lift_motors.append(motor)
    
    # Cruise propeller
    net.propeller = SUAVE.Components.Energy.Converters.Propeller()
    net.propeller.tip_radius = 0.15 * Units.meters
    
    # Cruise motor
    net.motor = SUAVE.Components.Energy.Converters.Motor()
    net.motor.efficiency = 0.9
    
        # Battery configuration
    battery = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    battery.mass_properties.mass = 4.0 * Units.kg
    battery.energy = 2000.0 * Units.Wh
    battery.voltage = 22.0 * Units.volt
    
    # Assign battery to network
    net.battery = battery
    # Battery
    #net.battery = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    #net.battery.energy = 2000 * Units.Wh
    
    vehicle.append_component(net)
    return vehicle
    

#mission configuration
# This function sets up the mission segments for the UAV
# It includes hover, transition, and cruise segments
def setup_mission(vehicle):
    # Mission initialization
    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'Lift_Cruise_Mission'
    
    #Battery
    battery = vehicle.networks.lift_cruise_network.battery

    # Atmospheric conditions
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    planet = SUAVE.Analyses.Planets.Planet()
    
    # Segment 1: Hover (vertical climb)
    hover = SUAVE.Analyses.Mission.Segments.Hover.Climb()
    hover.tag = 'hover_climb'
    hover.altitude_start = 0.0 * Units.meters
    hover.altitude_end = 50.0 * Units.meters
    hover.climb_rate = 2.0 * Units.meters/Units.seconds
    hover.battery_energy = battery.energy
    hover.analyses.extend([atmosphere, planet])
    
    # Segment 2: Transition (tilt or accelerate)
    transition = SUAVE.Analyses.Mission.Segments.Transition.Constant_Acceleration_Constant_Angle_Linear_Climb()
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

def create_hover_segment(vehicle):
    hover = SUAVE.Analyses.Mission.Segments.Hover.Climb()
    hover.tag = 'hover'
    hover.altitude_start = 0.0 * Units.meters
    hover.altitude_end = 30.0 * Units.meters
    hover.battery_energy = vehicle.networks.lift_cruise_network.battery.energy
    return hover

def create_transition_segment():
    transition = SUAVE.Analyses.Mission.Segments.Transition.Constant_Acceleration_Constant_Angle_Linear_Climb()
    transition.tag = 'transition'
    transition.acceleration = 1.5 * Units.m/Units.s**2
    return transition

def create_cruise_segment():
    cruise = SUAVE.Analyses.Mission.Segments.Cruise.Constant_Speed_Constant_Altitude()
    cruise.tag = 'cruise'
    cruise.air_speed = 15.0 * Units.m/Units.s
    return cruise
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
    
  # 3. Add standard analyses
    analyses.aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    analyses.stability = SUAVE.Analyses.Stability.Fidelity_Zero()
    analyses.energy = SUAVE.Analyses.Energy.Energy()
    
    # 4. Create mission analysis PROPERLY
    mission = SUAVE.Analyses.Mission.Mission()
    mission.tag = 'base_mission'
    
    # 5. Create and populate segments
    segments = SUAVE.Analyses.Mission.Segments.Segment.Container()
    
    # Hover segment
    segment = SUAVE.Analyses.Mission.Segments.Hover.Climb()
    segment.tag = "hover_climb"
    segment.altitude_start = 0.0 * Units.ft
    segment.altitude_end = 100.0 * Units.ft
    segments.append(segment)
    
    # Transition segment
    segment = SUAVE.Analyses.Mission.Segments.Transition.Constant_Acceleration_Constant_Angle_Linear_Climb()
    segment.tag = "transition"
    segment.altitude = 100.0 * Units.ft
    segments.append(segment)
    
    # Cruise segment
    segment = SUAVE.Analyses.Mission.Segments.Cruise.Constant_Speed_Constant_Altitude()
    segment.tag = "cruise"
    segment.distance = 10.0 * Units.nautical_mile
    segments.append(segment)
    
    # 6. Assign segments to mission
    mission.segments = segments
    
    # 7. Add mission to analyses
    analyses.mission = mission
    
    # 8. Finalize everything
    analyses.finalize()
    vehicle.finalize()
    
    # 9. Run mission
    results = mission.evaluate()

    print("Mission type:", mission.typestring())  # Should return "Mission"
    print("Segment container type:", segments.typestring())  # Should return "Segment.Container"
    
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