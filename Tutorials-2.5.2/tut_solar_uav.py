# tut_solar_UAV.py
# 
# Created:  Jul 2014, E. Botero
# Modified: Aug 2017, E. Botero

#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
assert SUAVE.__version__=='2.5.2', 'These tutorials only work with the SUAVE 2.5.2 release'
from SUAVE.Core import Units

import numpy as np
import pylab as plt
import time

from SUAVE.Plots.Performance.Mission_Plots import *
from SUAVE.Components.Energy.Networks.Solar import Solar
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass

#save plots
def save_plots(results):
    plot_solar_flux(results)
    plt.savefig('solar_flux.png')
    plt.close()
    
    plot_aerodynamic_forces(results)
    plt.savefig('aero_forces.png') 
    plt.close()
    
    plot_aerodynamic_coefficients(results)
    plt.savefig('aero_coefficients.png')
    plt.close()

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    
    # build the vehicle, configs, and analyses
    print("Building the vehicle, configs, and analyses...")
    configs, analyses = full_setup()
    print("Finished building the vehicle, configs, and analyses")
    configs.finalize()
    analyses.finalize()    
    print("Finished finalizing the vehicle, configs, and analyses")
    # weight analysis
    weights = analyses.configs.base.weights
    breakdown = weights.evaluate()          
    print("Finished weight analysis")
    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()
    print("Finished mission analysis")
    # plot results    
    plot_mission(results)
    print("Finished plotting results")
    print("results = ", results)
    # save plots
    save_plots(results)
    print("Finished saving plots")
    print("done main!")
    return

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():
    
    # vehicle data
    vehicle  = vehicle_setup()
    configs  = configs_setup(vehicle)
    
    # vehicle analyses
    configs_analyses = analyses_setup(configs)
    
    # mission analyses
    mission  = mission_setup(configs_analyses,vehicle)
    missions_analyses = missions_setup(mission)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses
    
    return configs, analyses

# ----------------------------------------------------------------------
#   Build the Vehicle
# ----------------------------------------------------------------------

def vehicle_setup():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Solar'
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    # mass properties
    vehicle.mass_properties.takeoff         = 250. * Units.kg
    vehicle.mass_properties.operating_empty = 250. * Units.kg
    vehicle.mass_properties.max_takeoff     = 250. * Units.kg 
    
    # basic parameters
    vehicle.reference_area                    = 80.       
    vehicle.envelope.ultimate_load            = 2.0
    vehicle.envelope.limit_load               = 1.5
    vehicle.envelope.maximum_dynamic_pressure = 0.5*1.225*(40.**2.) #Max q

    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------   

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
    
    wing.areas.reference         = vehicle.reference_area
    wing.spans.projected         = 40.0 * Units.meter
    wing.aspect_ratio            = (wing.spans.projected**2)/wing.areas.reference 
    wing.sweeps.quarter_chord    = 0.0 * Units.deg
    wing.symmetric               = True
    wing.thickness_to_chord      = 0.12
    wing.taper                   = 1.0
    wing.vertical                = False
    wing.high_lift               = True 
    wing.dynamic_pressure_ratio  = 1.0
    wing.chords.mean_aerodynamic = wing.areas.reference/wing.spans.projected
    wing.chords.root             = wing.areas.reference/wing.spans.projected
    wing.chords.tip              = wing.areas.reference/wing.spans.projected
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees
    wing.highlift                = False  
    wing.vertical                = False 
    wing.number_ribs             = 26.
    wing.number_end_ribs         = 2.
    wing.transition_x_upper      = 0.6
    wing.transition_x_lower      = 1.0
    wing.origin                  = [[3.0,0.0,0.0]] # meters
    wing.aerodynamic_center      = [1.0,0.0,0.0] # meters
    
    # add to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Horizontal_Tail()
    wing.tag = 'horizontal_stabilizer'
    
    wing.aspect_ratio         = 20. 
    wing.sweeps.quarter_chord = 0 * Units.deg
    wing.thickness_to_chord   = 0.12
    wing.taper                = 1.0
    wing.areas.reference      = vehicle.reference_area * .15
    wing.areas.wetted         = 2.0 * wing.areas.reference
    wing.areas.exposed        = 0.8 * wing.areas.wetted
    wing.areas.affected       = 0.6 * wing.areas.wetted       
    wing.spans.projected      = np.sqrt(wing.aspect_ratio*wing.areas.reference)
    wing.twists.root          = 0.0 * Units.degrees
    wing.twists.tip           = 0.0 * Units.degrees      
    
    wing.vertical                = False 
    wing.symmetric               = True
    wing.dynamic_pressure_ratio  = 0.9      
    wing.number_ribs             = 5.0
    wing.chords.root             = wing.areas.reference/wing.spans.projected
    wing.chords.tip              = wing.areas.reference/wing.spans.projected
    wing.chords.mean_aerodynamic = wing.areas.reference/wing.spans.projected  
    wing.origin                  = [[10.,0.0,0.0]] # meters
    wing.aerodynamic_center      = [0.5,0.0,0.0] # meters
  
    # add to vehicle
    vehicle.append_component(wing)    
    
    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Vertical_Tail()
    wing.tag = 'vertical_stabilizer'    
    
    
    wing.aspect_ratio         = 20.       
    wing.sweeps.quarter_chord = 0 * Units.deg
    wing.thickness_to_chord   = 0.12
    wing.taper                = 1.0
    wing.areas.reference      = vehicle.reference_area * 0.1
    wing.spans.projected      = np.sqrt(wing.aspect_ratio*wing.areas.reference)

    wing.chords.root             = wing.areas.reference/wing.spans.projected
    wing.chords.tip              = wing.areas.reference/wing.spans.projected
    wing.chords.mean_aerodynamic = wing.areas.reference/wing.spans.projected 
    wing.areas.wetted            = 2.0 * wing.areas.reference
    wing.areas.exposed           = 0.8 * wing.areas.wetted
    wing.areas.affected          = 0.6 * wing.areas.wetted    
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees  
    wing.origin                  = [[10.,0.0,0.0]] # meters
    wing.aerodynamic_center      = [0.5,0.0,0.0] # meters
    wing.symmetric               = True          
    wing.vertical                = True 
    wing.t_tail                  = False
    wing.dynamic_pressure_ratio  = 1.0
    wing.number_ribs             = 5.
  
    # add to vehicle
    vehicle.append_component(wing)  
    
    
    # ------------------------------------------------------------------
    #   Nacelle  
    # ------------------------------------------------------------------
    nacelle              = SUAVE.Components.Nacelles.Nacelle()
    nacelle.diameter     = 0.2 * Units.meters
    nacelle.length       = 0.01 * Units.meters
    nacelle.tag          = 'nacelle' 
    nacelle.areas.wetted =  nacelle.length *(2*np.pi*nacelle.diameter/2.)
    vehicle.append_component(nacelle) 
        
    
    #------------------------------------------------------------------
    # Propulsor
    #------------------------------------------------------------------
    
    # build network
    net = Solar()
    net.number_of_engines = 1.

    # Component 1 the Sun?
    sun = SUAVE.Components.Energy.Processes.Solar_Radiation()
    net.solar_flux = sun
    
    # Component 2 the solar panels
    panel = SUAVE.Components.Energy.Converters.Solar_Panel()
    panel.area                 = vehicle.reference_area * 0.9
    panel.efficiency           = 0.25
    panel.mass_properties.mass = panel.area*(0.60 * Units.kg)
    net.solar_panel            = panel
    
    # Component 3 the ESC
    esc = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    esc.efficiency = 0.95 # Gundlach for brushless motors
    net.esc        = esc
    
    # Component 5 the Propeller
    # Design the Propeller
    prop = SUAVE.Components.Energy.Converters.Propeller()
    prop.number_of_blades    = 2.0
    prop.freestream_velocity = 40.0 * Units['m/s']# freestream
    prop.angular_velocity    = 150. * Units['rpm']
    prop.tip_radius          = 4.25 * Units.meters
    prop.hub_radius          = 0.05 * Units.meters
    prop.design_Cl           = 0.7
    prop.design_altitude     = 15.0 * Units.km
    prop.design_power        = None
    prop.design_thrust       = 120.
    prop                     = propeller_design(prop)
    
    net.propellers.append(prop)

    # Component 4 the Motor
    motor = SUAVE.Components.Energy.Converters.Motor()
    motor.resistance           = 0.006
    motor.no_load_current      = 2.5  * Units.ampere
    motor.speed_constant       = 30. * Units['rpm'] # RPM/volt converted to (rad/s)/volt    
    motor.propeller_radius     = prop.tip_radius
    motor.propeller_Cp         = prop.design_power_coefficient
    motor.gear_ratio           = 12. # Gear ratio
    motor.gearbox_efficiency   = .98 # Gear box efficiency
    motor.expected_current     = 60. # Expected current
    motor.mass_properties.mass = 2.0  * Units.kg
    net.motors.append(motor)
    
    # Component 6 the Payload
    payload = SUAVE.Components.Energy.Peripherals.Payload()
    payload.power_draw           = 50. * Units.watts 
    payload.mass_properties.mass = 5.0 * Units.kg
    net.payload                  = payload
    
    # Component 7 the Avionics
    avionics = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.power_draw = 50. * Units.watts
    net.avionics        = avionics      

    # Component 8 the Battery
    bat = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    bat.mass_properties.mass = 95.0 * Units.kg
    bat.specific_energy      = 800. * Units.Wh/Units.kg
    bat.max_voltage          = 130.0
    initialize_from_mass(bat)
    net.battery              = bat
   
    #Component 9 the system logic controller and MPPT
    logic = SUAVE.Components.Energy.Distributors.Solar_Logic()
    logic.system_voltage  = 120.0
    logic.MPPT_efficiency = 0.95
    net.solar_logic       = logic
    
    # add the solar network to the vehicle
    vehicle.append_component(net)  

    return vehicle

# ----------------------------------------------------------------------
#   Define the Configurations
# ---------------------------------------------------------------------

def configs_setup(vehicle):
    
    # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------
    
    configs = SUAVE.Components.Configs.Config.Container()
    
    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    configs.append(base_config)
    
    # ------------------------------------------------------------------
    #   Cruise Configuration
    # ------------------------------------------------------------------
    
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'cruise'
    
    configs.append(config)
    
    return configs


# ----------------------------------------------------------------------
#   Define the Vehicle Analyses
# ----------------------------------------------------------------------

def analyses_setup(configs):
    
    analyses = SUAVE.Analyses.Analysis.Container()
    
    # build a base analysis for each config
    for tag,config in configs.items():
        analysis = base_analysis(config)
        analyses[tag] = analysis
    
    return analyses

def base_analysis(vehicle):

    # ------------------------------------------------------------------
    #   Initialize the Analyses
    # ------------------------------------------------------------------     
    analyses = SUAVE.Analyses.Vehicle()
    
    # ------------------------------------------------------------------
    #  Basic Geometry Relations
    sizing = SUAVE.Analyses.Sizing.Sizing()
    sizing.features.vehicle = vehicle
    analyses.append(sizing)
    
    # ------------------------------------------------------------------
    #  Weights
    weights = SUAVE.Analyses.Weights.Weights_UAV()
    weights.settings.empty = \
        SUAVE.Methods.Weights.Correlations.Human_Powered.empty
    weights.vehicle = vehicle
    analyses.append(weights)
    
    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)
    
    # ------------------------------------------------------------------
    #  Energy
    energy = SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.networks #what is called throughout the mission (at every time step))
    analyses.append(energy)
    
    # ------------------------------------------------------------------
    #  Planet Analysis
    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)
    
    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)   
    
    # done!
    return analyses    


# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------
def mission_setup(analyses,vehicle):
    
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'The Test Mission'

    mission.atmosphere  = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    mission.planet      = SUAVE.Attributes.Planets.Earth()
    
    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments
    
    # base segment
    base_segment = Segments.Segment()   
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    
    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Analyses.Mission.Segments.Cruise.Constant_Mach_Constant_Altitude(base_segment)
    segment.tag = "cruise1"
    
    # connect vehicle configuration
    segment.analyses.extend(analyses.cruise)
    
    # segment attributes     
    segment.state.numerics.number_control_points = 64
    segment.start_time     = time.strptime("Tue, Jun 21 11:30:00  2022", "%a, %b %d %H:%M:%S %Y",)
    segment.altitude       = 15.0  * Units.km 
    segment.mach           = 0.12
    segment.distance       = 3050.0 * Units.km
    segment.battery_energy = vehicle.networks.solar.battery.max_energy*0.3 #Charge the battery to start
    segment.latitude       = 37.4300   # this defaults to degrees (do not use Units.degrees)
    segment.longitude      = -122.1700 # this defaults to degrees
    
    segment = vehicle.networks.solar.add_unknowns_and_residuals_to_segment(segment,initial_power_coefficient = 0.05)   
    
    mission.append_segment(segment)    

    # ------------------------------------------------------------------    
    #   Mission definition complete    
    # ------------------------------------------------------------------
    
    return mission

def missions_setup(base_mission):

    # the mission container
    missions = SUAVE.Analyses.Mission.Mission.Container()
    
    # ------------------------------------------------------------------
    #   Base Mission
    # ------------------------------------------------------------------
    
    missions.base = base_mission
    
    # done!
    return missions  

# ----------------------------------------------------------------------
#   Plot Results
# ----------------------------------------------------------------------
def plot_mission(results):

    # Plot Flight Conditions 
    plot_flight_conditions(results) 
    
    # Plot Solar Conditions 
    plot_solar_flux(results)

    # Plot Aerodynamic Coefficients
    plot_aerodynamic_coefficients(results)  
    
    # Drag Components
    plot_drag_components(results)    

    # Plot Aircraft Flight Speed
    plot_aircraft_velocities(results)

    # Plot Aircraft Electronics
    plot_battery_pack_conditions(results)

    # Plot Propeller Conditions 
    plot_propeller_conditions(results) 

    # Plot Electric Motor and Propeller Efficiencies 
    plot_eMotor_Prop_efficiencies(results)

    
    return 

if __name__ == '__main__':
    print("Running the Solar UAV Tutorial")
    print("This may take a few minutes...") 
    main()
    print("Finished running the Solar UAV Tutorial")
    print("Now plotting results...")
    plt.show()
    print("Finished plotting results")
    print("Done!")  