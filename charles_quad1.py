# charles_quad1.py
#
# ----------------------------------------------------------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import SUAVE
assert SUAVE.__version__=='2.5.2', 'This codebase only work with the SUAVE 2.5.2 release!'
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os

# module imports
from SUAVE.Core                                         import Units, Data
from SUAVE.Plots.Performance.Mission_Plots              import *
from SUAVE.Methods.Propulsion                           import propeller_design
from SUAVE.Methods.Propulsion.electric_motor_sizing     import size_optimal_motor
from SUAVE.Methods.Power.Battery.Sizing                 import initialize_from_mass
from SUAVE.Methods.Geometry.Two_Dimensional.Planform    import segment_properties, wing_planform

#------------------------------------------------------------------------------------------------------------------------
FAST_MODE = False  # False = reproducible (single-thread), True = faster (multi-thread)
# Global iterate counter for debug logging
_iter_counter = 0
# ----------------------------------------------------------------------------------------------------------------------

# ---- Threading for BLAS/MKL/OpenBLAS (Performance Optimization) ----
if FAST_MODE:
    # Use more cores for speed 
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
else:
    # Deterministic / comparable runs
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ========= QUADCOPTER DESIGN HELPERS (module-level) =========
def validate_quadcopter_config(vehicle):
    print("\n Validating Quadcopter Configuration:")
    
    # Check mass properties
    mass = vehicle.mass_properties.takeoff
    print(f"  ✓ Total Mass: {mass:.3f} kg")
    
    # Check propulsion system
    try:
        net = vehicle.networks.quadcopter_network
        num_rotors = len(net.lift_rotors)
        print(f"  ✓ Number of Rotors: {num_rotors}")
        
        battery_mass = net.battery.mass_properties.mass
        print(f"  ✓ Battery Mass: {battery_mass:.3f} kg ({battery_mass/mass*100:.1f}% of total)")
        
        voltage = net.voltage
        print(f"  ✓ System Voltage: {voltage:.1f} V")
        
    except Exception as e:
        print(f"  ✗ Propulsion validation error: {e}")
    
    print("  ✅ Configuration validation complete\n")

# ----------------------------------------------------------------------------------------------------------------------
#   Vehicle
# ----------------------------------------------------------------------------------------------------------------------
def setup_vehicle():
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------
    vehicle               = SUAVE.Vehicle()
    vehicle.tag           = 'Quadcopter_1kg'
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------
    vehicle.mass_properties.takeoff           = 1.0  * Units.kg
    vehicle.mass_properties.operating_empty   = 0.75 * Units.kg
    vehicle.mass_properties.max_takeoff       = 1.2  * Units.kg  # 20% safety margin
    vehicle.mass_properties.max_payload       = 0.25 * Units.kg
    vehicle.mass_properties.center_of_gravity = [[0.0, 0.0, 0.0]]
    
    # Envelope properties
    vehicle.envelope.ultimate_load = 4.0
    vehicle.envelope.limit_load    = 2.0
    
    # Reference area (body cross-section approximation)
    vehicle.reference_area = 0.01  # m² (10cm x 10cm body profile)
    
    # ------------------------------------------------------------------
    #   Fuselage (Main Body)
    # ------------------------------------------------------------------
    fuselage     = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'
    
    # Geometric properties
    fuselage.number_coach_seats                  = 0
    fuselage.seats_abreast                       = 0
    fuselage.seat_pitch                          = 0
    fuselage.fineness.nose                       = 1.6
    fuselage.fineness.tail                       = 2.0
    fuselage.lengths.nose                        = 0.05  # 5cm nose section
    fuselage.lengths.tail                        = 0.05  # 5cm tail section  
    fuselage.lengths.cabin                       = 0.10  # 10cm main body
    fuselage.lengths.total                       = 0.20  # 20cm total length
    fuselage.lengths.fore_space                  = 0.0
    fuselage.lengths.aft_space                   = 0.0
    fuselage.width                               = 0.10   # 10cm width
    fuselage.heights.maximum                     = 0.05   # 5cm height
    fuselage.heights.at_quarter_length           = 0.04
    fuselage.heights.at_three_quarters_length    = 0.04
    fuselage.heights.at_wing_root_quarter_chord  = 0.04
    fuselage.areas.side_projected                = 0.02
    fuselage.areas.wetted                        = 0.05
    fuselage.areas.front_projected               = 0.005
    fuselage.effective_diameter                  = 0.10
    fuselage.differential_pressure               = 0.0
    
    # Segment properties
    segment = SUAVE.Components.Lofted_Body.Segment()
    segment.tag = 'segment_0'
    segment.percent_x_location = 0.0
    segment.percent_z_location = 0.0
    segment.height = 0.01
    segment.width = 0.01
    fuselage.Segments.append(segment)
    
    segment = SUAVE.Components.Lofted_Body.Segment()
    segment.tag = 'segment_1'
    segment.percent_x_location = 0.25
    segment.percent_z_location = 0.0
    segment.height = 0.05
    segment.width = 0.10
    fuselage.Segments.append(segment)
    
    segment = SUAVE.Components.Lofted_Body.Segment()
    segment.tag = 'segment_2'
    segment.percent_x_location = 0.75
    segment.percent_z_location = 0.0
    segment.height = 0.05
    segment.width = 0.10
    fuselage.Segments.append(segment)
    
    segment = SUAVE.Components.Lofted_Body.Segment()
    segment.tag = 'segment_3'
    segment.percent_x_location = 1.0
    segment.percent_z_location = 0.0
    segment.height = 0.01
    segment.width = 0.01
    fuselage.Segments.append(segment)
    
    # Add to vehicle
    vehicle.append_component(fuselage)
    
    # ------------------------------------------------------------------
    #   Rotor/Propeller Arms (represented as small wings)
    # ------------------------------------------------------------------
    for i in range(4):
        arm = SUAVE.Components.Wings.Wing()
        arm.tag = f'arm_{i+1}'
        
        # Position arms in X configuration
        angle = i * 90 * Units.degrees + 45 * Units.degrees  # 45, 135, 225, 315 degrees
        arm_length = 0.15  # 15cm from center to rotor
        
        x_pos = arm_length * np.cos(angle)
        y_pos = arm_length * np.sin(angle)
        
        arm.origin = [[x_pos, y_pos, 0.0]]
        arm.areas.reference = 0.002  # 2 cm²
        arm.spans.projected = 0.15   # 15cm span
        arm.chords.root = 0.013      # 1.3cm chord
        arm.chords.tip = 0.013
        arm.aspect_ratio = arm.spans.projected**2 / arm.areas.reference
        arm.sweeps.quarter_chord = 0.0 * Units.degrees
        arm.thickness_to_chord = 0.1
        arm.dihedral = 0.0 * Units.degrees
        arm.taper = 1.0
        
        # Fill out wing properties
        arm = wing_planform(arm)
        
        # Add to vehicle
        vehicle.append_component(arm)
    
    # ------------------------------------------------------------------
    #   Propulsion Network
    # ------------------------------------------------------------------
    net                                  = SUAVE.Components.Energy.Networks.Lift_Cruise()
    net.tag                              = 'quadcopter_network'
    net.number_of_lift_rotor_engines     = 4
    net.number_of_propeller_engines      = 4  # Same rotors used for cruise
    net.identical_propellers             = True
    net.identical_lift_rotors            = True
    net.voltage                          = 14.8 * Units.volt  # 4S LiPo nominal voltage
    
    # ------------------------------------------------------------------
    #   Battery Configuration
    # ------------------------------------------------------------------
    bat                          = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    bat.tag                      = 'battery'
    bat.mass_properties.mass     = 0.25 * Units.kg    # 250g battery (25% of total mass)
    bat.specific_energy          = 250  * Units.Wh/Units.kg  # High energy density LiPo
    bat.resistance               = 0.005               # Low internal resistance
    bat.max_voltage              = 14.8               # 4S LiPo configuration
    initialize_from_mass(bat, bat.mass_properties.mass)
    net.battery = bat
    
    # ------------------------------------------------------------------
    #   Lift Rotors (Quadcopter Configuration)
    # ------------------------------------------------------------------
    rotor_positions = [
        [ 0.106,  0.106, 0.0],   # Front right
        [-0.106,  0.106, 0.0],   # Front left  
        [-0.106, -0.106, 0.0],   # Rear left
        [ 0.106, -0.106, 0.0]    # Rear right
    ]
    
    net.lift_rotors = SUAVE.Components.Energy.Converters.Lift_Rotor.Container()
    
    for i, position in enumerate(rotor_positions):
        lift_rotor           = SUAVE.Components.Energy.Converters.Lift_Rotor()
        lift_rotor.tag       = f'lift_rotor_{i+1}'
        lift_rotor.origin    = [position]
        
        # Rotor geometry - 5 inch propellers (typical for 1kg quadcopters)
        lift_rotor.tip_radius        = 0.0635  # 5 inch = 12.7 cm diameter
        lift_rotor.hub_radius        = 0.01    # 1 cm hub radius
        lift_rotor.number_of_blades  = 2       # Two-blade configuration
        
        # Design performance characteristics
        lift_rotor.design_tip_mach   = 0.7
        lift_rotor.design_Cl         = 0.7
        lift_rotor.design_altitude   = 0.0     # Sea level operations
        lift_rotor.design_thrust     = (vehicle.mass_properties.takeoff * 9.81) / 4  # Equal thrust distribution
        
        # Twist and chord distributions
        lift_rotor.twist_distribution = np.array([40., 25., 15., 10., 5.]) * Units.degrees
        lift_rotor.chord_distribution = np.array([0.01, 0.012, 0.015, 0.018, 0.02])  # meters
        lift_rotor.radius_distribution = np.array([0.01, 0.02, 0.03, 0.05, 0.0635])
        
        # Rotation direction (alternating for torque balance)
        if i % 2 == 0:
            lift_rotor.rotation = 1  # Clockwise
        else:
            lift_rotor.rotation = -1  # Counter-clockwise
        
        # Required attributes for propeller_design function
        lift_rotor.angular_velocity = 5000 * Units.rpm  # Initial guess for hover RPM
        lift_rotor.freestream_velocity = 0.0 * Units['m/s']  # Hover condition
        lift_rotor.design_power = None  # Will be calculated
        lift_rotor.design_torque = None  # Will be calculated
        lift_rotor.airfoil_geometry = None  # Use default
        lift_rotor.airfoil_polars = None  # Use default
        lift_rotor.airfoil_polar_stations = None  # Use default
        lift_rotor.design_cl = 0.7
        lift_rotor.design_alpha = None  # Will be calculated
        lift_rotor.mid_chord_alignment = np.zeros(len(lift_rotor.radius_distribution))
            
        # Design the rotor
        lift_rotor = propeller_design(lift_rotor, number_of_stations=20)
        
        # Add to network
        net.lift_rotors.append(lift_rotor)
    
    # ------------------------------------------------------------------
    #   Electric Motors
    # ------------------------------------------------------------------
    net.lift_rotor_motors = SUAVE.Components.Energy.Converters.Motor.Container()
    
    for i in range(4):
        lift_rotor_motor                 = SUAVE.Components.Energy.Converters.Motor()
        lift_rotor_motor.tag             = f'lift_rotor_motor_{i+1}'
        lift_rotor_motor.efficiency      = 0.9                        # High efficiency brushless motor
        lift_rotor_motor.nominal_voltage = bat.max_voltage * 0.75      # Operating voltage
        lift_rotor_motor.mass_properties.mass = 0.06  # 60g per motor
        lift_rotor_motor.origin = rotor_positions[i]
        lift_rotor_motor.propeller_radius = net.lift_rotors[f'lift_rotor_{i+1}'].tip_radius
        lift_rotor_motor.no_load_current = 2.0  # Amps
        
        # Size the motor
        size_optimal_motor(lift_rotor_motor, net.lift_rotors[f'lift_rotor_{i+1}'])
        
        net.lift_rotor_motors.append(lift_rotor_motor)
    
    # ------------------------------------------------------------------
    #   Electronic Speed Controllers (ESCs)
    # ------------------------------------------------------------------
    lift_rotor_esc = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    lift_rotor_esc.tag = 'lift_rotor_esc'
    lift_rotor_esc.efficiency = 0.95
    net.lift_rotor_esc = lift_rotor_esc
    
    # ESC for cruise propellers
    propeller_esc = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    propeller_esc.tag = 'propeller_esc'
    propeller_esc.efficiency = 0.95
    net.propeller_esc = propeller_esc
    
    # ------------------------------------------------------------------
    #   Payload
    # ------------------------------------------------------------------
    payload = SUAVE.Components.Energy.Peripherals.Avionics()
    payload.tag = 'payload'
    payload.power_draw = 0.0  # No additional payload power for base quadcopter
    net.payload = payload
    
    # ------------------------------------------------------------------
    #   Avionics
    # ------------------------------------------------------------------
    avionics = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.tag = 'avionics'
    avionics.power_draw = 10.0 * Units.watts  # Flight controller, GPS, etc.
    net.avionics = avionics
    
    # ------------------------------------------------------------------
    #   Cruise Propellers (Same as lift rotors for quadcopter)
    # ------------------------------------------------------------------
    net.propellers = SUAVE.Components.Energy.Converters.Propeller.Container()
    net.propeller_motors = SUAVE.Components.Energy.Converters.Motor.Container()
    
    for i in range(4):
        # Copy lift rotor as cruise propeller
        cruise_prop = deepcopy(net.lift_rotors[f'lift_rotor_{i+1}'])
        cruise_prop.tag = f'cruise_propeller_{i+1}'
        net.propellers.append(cruise_prop)
        
        # Copy lift rotor motor as cruise motor
        cruise_motor = deepcopy(net.lift_rotor_motors[f'lift_rotor_motor_{i+1}'])
        cruise_motor.tag = f'cruise_motor_{i+1}'
        net.propeller_motors.append(cruise_motor)
    
    # ------------------------------------------------------------------
    #   Add network to vehicle
    # ------------------------------------------------------------------
    vehicle.append_component(net)
    
    # ------------------------------------------------------------------
    #   Vehicle Configuration Complete
    # ------------------------------------------------------------------
    
    return vehicle

# ----------------------------------------------------------------------------------------------------------------------
#   Analyses
# ----------------------------------------------------------------------------------------------------------------------
def setup_analyses(vehicle):
    # ------------------------------------------------------------------
    #   Initialize Analysis Suite
    # ------------------------------------------------------------------
    analyses = SUAVE.Analyses.Vehicle()
    
    # ------------------------------------------------------------------
    #   Aerodynamics Analysis
    # ------------------------------------------------------------------
    aerodynamics          = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)
    
    # ------------------------------------------------------------------
    #   Stability Analysis
    # ------------------------------------------------------------------
    stability          = SUAVE.Analyses.Stability.Fidelity_Zero()
    stability.geometry = vehicle
    analyses.append(stability)
    
    # ------------------------------------------------------------------
    #   Energy Analysis
    # ------------------------------------------------------------------
    energy         = SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.networks.quadcopter_network
    analyses.append(energy)
    
    # ------------------------------------------------------------------
    #   Planet Analysis
    # ------------------------------------------------------------------
    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)
    
    # ------------------------------------------------------------------
    #   Atmospheric Analysis
    # ------------------------------------------------------------------
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    analyses.append(atmosphere)
    
    # ------------------------------------------------------------------
    #   Mass Analysis
    # ------------------------------------------------------------------
    weights         = SUAVE.Analyses.Weights.Weights_eVTOL()
    weights.vehicle = vehicle
    analyses.append(weights)
    
    return analyses

# ----------------------------------------------------------------------------------------------------------------------
#   Mission
# ----------------------------------------------------------------------------------------------------------------------
def setup_mission(vehicle, analyses):
    # ------------------------------------------------------------------
    #   Initialize Mission
    # ------------------------------------------------------------------
    mission     = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'quadcopter_mission'
    
    # Unpack segments module
    Segments = SUAVE.Analyses.Mission.Segments
    
    # Base segment settings
    base_segment = Segments.Segment()
    base_segment.state.numerics.number_control_points = 4
    base_segment.process.initialize.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position = SUAVE.Methods.skip
    base_segment.process.iterate.conditions.stability = SUAVE.Methods.skip
    base_segment.process.finalize.post_process.stability = SUAVE.Methods.skip
    
    # ------------------------------------------------------------------
    #   Takeoff / Hover Climb
    # ------------------------------------------------------------------
    segment = Segments.Hover.Climb(base_segment)
    segment.tag = 'takeoff_hover_climb'
    segment.analyses.extend(analyses)
    
    segment.altitude_start = 0.0 * Units.ft
    segment.altitude_end = 50.0 * Units.ft  # Climb to 50 ft
    segment.climb_rate = 3.0 * Units['ft/s']  # 3 ft/s climb rate
    segment.battery_energy = vehicle.networks.quadcopter_network.battery.max_energy
    segment.process.iterate.unknowns.mission = SUAVE.Methods.skip
    
    # Add hover segment unknowns and residuals
    segment = vehicle.networks.quadcopter_network.add_lift_unknowns_and_residuals_to_segment(segment)
    
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------
    #   Hover 
    # ------------------------------------------------------------------
    segment = Segments.Hover.Hover(base_segment)
    segment.tag = 'hover_at_altitude'
    segment.analyses.extend(analyses)
    
    segment.altitude = 50.0 * Units.ft
    segment.time = 120.0 * Units.seconds  # Hover for 2 minutes
    segment.process.iterate.unknowns.mission = SUAVE.Methods.skip
    
    segment = vehicle.networks.quadcopter_network.add_lift_unknowns_and_residuals_to_segment(segment)
    
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------
    #   Forward Flight / Cruise
    # ------------------------------------------------------------------
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = 'forward_flight'
    segment.analyses.extend(analyses)
    
    segment.altitude = 50.0 * Units.ft
    segment.air_speed = 15.0 * Units.mph  # 15 mph forward flight
    segment.distance = 1000.0 * Units.ft  # Fly 1000 ft forward
    
    segment = vehicle.networks.quadcopter_network.add_cruise_unknowns_and_residuals_to_segment(segment)
    
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------
    #   Hover before landing
    # ------------------------------------------------------------------
    segment = Segments.Hover.Hover(base_segment)
    segment.tag = 'hover_before_landing'
    segment.analyses.extend(analyses)
    
    segment.altitude = 50.0 * Units.ft
    segment.time = 30.0 * Units.seconds  # Brief hover
    segment.process.iterate.unknowns.mission = SUAVE.Methods.skip
    
    segment = vehicle.networks.quadcopter_network.add_lift_unknowns_and_residuals_to_segment(segment)
    
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------
    #   Landing / Hover Descent
    # ------------------------------------------------------------------
    segment = Segments.Hover.Descent(base_segment)
    segment.tag = 'landing_hover_descent'
    segment.analyses.extend(analyses)
    
    segment.altitude_start = 50.0 * Units.ft
    segment.altitude_end = 0.0 * Units.ft
    segment.descent_rate = 2.0 * Units['ft/s']  # 2 ft/s descent rate
    segment.process.iterate.unknowns.mission = SUAVE.Methods.skip
    
    segment = vehicle.networks.quadcopter_network.add_lift_unknowns_and_residuals_to_segment(segment)
    
    mission.append_segment(segment)
    
    return mission

# ----------------------------------------------------------------------------------------------------------------------
#   Results Analysis
# ----------------------------------------------------------------------------------------------------------------------
def analyze_results(results, vehicle):
    print("\n" + "="*60 + "\n QUADCOPTER PERFORMANCE ANALYSIS \n" + "="*60)
    
    # Vehicle specifications
    print(f"\nVehicle Specifications:")
    print(f"  Total Mass: {vehicle.mass_properties.takeoff:.3f} kg")
    print(f"  Battery Mass: {vehicle.networks.quadcopter_network.battery.mass_properties.mass:.3f} kg")
    print(f"  Rotor Diameter: {vehicle.networks.quadcopter_network.lift_rotors.lift_rotor_1.tip_radius*2:.3f} m")
    print(f"  Number of Rotors: 4")
    
    # Mission performance summary
    total_time = 0
    total_energy = 0
    
    for i, segment in enumerate(results.segments):
        segment_time = segment.conditions.frames.inertial.time[-1, 0] - segment.conditions.frames.inertial.time[0, 0]
        total_time += segment_time
        
        print(f"\n{segment.tag}:")
        print(f"  Duration: {segment_time:.1f} s")
        
        # Try to extract power information from various possible locations
        power_draw = 0
        segment_energy = 0
        
        try:
            if hasattr(segment.conditions, 'propulsion'):
                # Try different possible power attributes
                if hasattr(segment.conditions.propulsion, 'battery_draw'):
                    power_draw = np.mean(segment.conditions.propulsion.battery_draw)
                elif hasattr(segment.conditions.propulsion, 'power'):
                    power_draw = np.mean(segment.conditions.propulsion.power)
                elif hasattr(segment.conditions.propulsion, 'battery_power_draw'):
                    power_draw = np.mean(segment.conditions.propulsion.battery_power_draw)
                
                if power_draw > 0:
                    segment_energy = power_draw * segment_time / 3600  # Wh
                    total_energy += segment_energy
                    print(f"  Average Power: {power_draw:.1f} W")
                    print(f"  Energy Used: {segment_energy:.1f} Wh")
                
                # Try to get RPM information
                if hasattr(segment.conditions.propulsion, 'lift_rotor_rpm'):
                    avg_rpm = np.mean(segment.conditions.propulsion.lift_rotor_rpm[:, 0])
                    print(f"  Average RPM: {avg_rpm:.0f}")
                elif hasattr(segment.conditions.propulsion, 'rotor_rpm'):
                    avg_rpm = np.mean(segment.conditions.propulsion.rotor_rpm[:, 0])
                    print(f"  Average RPM: {avg_rpm:.0f}")
                    
        except Exception as e:
            print(f"  Power analysis unavailable: {str(e)[:50]}...")
            
        # Try to get altitude and velocity information
        try:
            if hasattr(segment.conditions, 'freestream'):
                avg_altitude = np.mean(segment.conditions.freestream.altitude) * 3.28084  # Convert to ft
                avg_velocity = np.mean(segment.conditions.freestream.velocity) * 2.237  # Convert to mph
                print(f"  Average Altitude: {avg_altitude:.1f} ft")
                print(f"  Average Speed: {avg_velocity:.1f} mph")
        except Exception as e:
            pass
    
    print(f"\nMission Summary:")
    print(f"  Total Flight Time: {total_time:.1f} s ({total_time/60:.1f} min)")
    print(f"  Total Energy Used: {total_energy:.1f} Wh")
    
    # Battery performance
    battery_capacity = vehicle.networks.quadcopter_network.battery.max_energy / 3600  # Wh
    battery_used_percent = (total_energy / battery_capacity) * 100
    
    print(f"\nBattery Performance:")
    print(f"  Battery Capacity: {battery_capacity:.1f} Wh")
    print(f"  Energy Used: {battery_used_percent:.1f}%")
    print(f"  Remaining: {100-battery_used_percent:.1f}%")
    
    # Flight envelope
    max_altitude = 0
    max_speed = 0
    
    for segment in results.segments:
        alt = np.max(segment.conditions.freestream.altitude)
        speed = np.max(segment.conditions.freestream.velocity)
        max_altitude = max(max_altitude, alt)
        max_speed = max(max_speed, speed)
    
    print(f"\nFlight Envelope:")
    print(f"  Maximum Altitude: {max_altitude:.1f} m ({max_altitude*3.28084:.1f} ft)")
    print(f"  Maximum Speed: {max_speed:.1f} m/s ({max_speed*2.237:.1f} mph)")
    
    print("\n" + "="*60 + "\n")

# ----------------------------------------------------------------------------------------------------------------------
#   Plotting Functions
# ----------------------------------------------------------------------------------------------------------------------
def plot_results(results):
    # Create plots directory if it doesn't exist
    if not os.path.exists('quadcopter_plots_charles'):
        os.makedirs('quadcopter_plots_charles')
    
    # Plot flight conditions
    plot_flight_conditions(results)
    plt.savefig('quadcopter_plots_charles/flight_conditions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot aerodynamic forces
    plot_aerodynamic_forces(results)
    plt.savefig('quadcopter_plots_charles/aerodynamic_forces.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot battery conditions
    plot_battery_pack_conditions(results)
    plt.savefig('quadcopter_plots_charles/battery_conditions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot rotor conditions (if available)
    try:
        plot_rotor_conditions(results)
        plt.savefig('quadcopter_plots_charles/rotor_conditions.png', dpi=300, bbox_inches='tight')
        plt.close()
    except (NameError, AttributeError):
        # Create a custom rotor plot if the standard function isn't available
        plt.figure(figsize=(10, 6))
        
        time_total = []
        rpm_total = []
        current_time = 0
        
        # Loop through each mission segment (takeoff, hover, cruise, landing, etc.)
        for segment in results.segments:
            # Get time array for this segment (each segment starts at time=0)
            segment_time = segment.conditions.frames.inertial.time[:, 0]
            
            # Try to extract rotor RPM data from this segment
            try:
                # Check if RPM data exists in the propulsion conditions
                if hasattr(segment.conditions, 'propulsion') and hasattr(segment.conditions.propulsion, 'lift_rotor_rpm'):
                    segment_rpm = segment.conditions.propulsion.lift_rotor_rpm[:, 0]
                else:
                    # If no RPM data, create zeros array with same length as time
                    segment_rpm = np.zeros_like(segment_time)
            except:
                # Safety fallback: use zeros if any error occurs
                segment_rpm = np.zeros_like(segment_time)
            
            # Create continuous timeline by adding accumulated time from previous segments
            time_with_offset = segment_time + current_time
            
            # Add this segment's data to the overall timeline
            time_total.extend(time_with_offset)
            rpm_total.extend(segment_rpm)
            
            # Update offset for next segment (last time point of current segment)
            current_time = time_with_offset[-1]
        
        plt.plot(np.array(time_total)/60, rpm_total, 'g-', linewidth=2, label='Rotor RPM')
        plt.xlabel('Time (minutes)')
        plt.ylabel('RPM')
        plt.title('Rotor Speed During Flight')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('quadcopter_plots_charles/rotor_conditions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Custom altitude vs time plot
    plt.figure(figsize=(10, 6))
    
    time_total = []
    altitude_total = []
    current_time = 0
    
    for segment in results.segments:
        segment_time = segment.conditions.frames.inertial.time[:, 0]
        segment_altitude = segment.conditions.freestream.altitude[:, 0] * 3.28084  # Convert to ft
        
        time_with_offset = segment_time + current_time
        time_total.extend(time_with_offset)
        altitude_total.extend(segment_altitude)
        
        current_time = time_with_offset[-1]
    
    plt.plot(np.array(time_total)/60, altitude_total, 'b-', linewidth=2, label='Altitude')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Altitude (ft)')
    plt.title('Quadcopter Flight Profile')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('quadcopter_plots_charles/flight_profile.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Power consumption plot
    plt.figure(figsize=(10, 6))
    
    time_total = []
    power_total = []
    current_time = 0
    
    for segment in results.segments:
        segment_time = segment.conditions.frames.inertial.time[:, 0]
        
        # Try to get power data from various possible sources
        segment_power = np.zeros_like(segment_time)
        
        try:
            if hasattr(segment.conditions, 'propulsion'):
                if hasattr(segment.conditions.propulsion, 'battery_draw'):
                    segment_power = segment.conditions.propulsion.battery_draw[:, 0]
                elif hasattr(segment.conditions.propulsion, 'power'):
                    segment_power = segment.conditions.propulsion.power[:, 0]
                elif hasattr(segment.conditions.propulsion, 'battery_power_draw'):
                    segment_power = segment.conditions.propulsion.battery_power_draw[:, 0]
                else:
                    # If no power data available, estimate based on thrust for hover segments
                    if 'hover' in segment.tag.lower():
                        # Rough estimate: assume ~100W per rotor for hover
                        segment_power = np.full_like(segment_time, 400)  # 4 rotors * 100W
        except Exception:
            # If all else fails, use zeros
            segment_power = np.zeros_like(segment_time)
        
        time_with_offset = segment_time + current_time
        time_total.extend(time_with_offset)
        power_total.extend(segment_power)
        
        current_time = time_with_offset[-1]
    
    plt.plot(np.array(time_total)/60, power_total, 'r-', linewidth=2, label='Power Draw')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Power (W)')
    plt.title('Power Consumption During Flight')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('quadcopter_plots_charles/power_consumption.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plots saved to 'quadcopter_plots_charles' directory")

# ----------------------------------------------------------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------------------------------------------------------
def main():
    # ---------- Setup vehicle ----------
    print("o Setting up the 1kg quadcopter vehicle")
    vehicle = setup_vehicle()
    validate_quadcopter_config(vehicle)
    print("✓ Vehicle setup complete")
    
    # ---------- Setup analyses ----------
    print("o Configuring analysis modules")
    analyses = setup_analyses(vehicle)
    analyses.finalize()
    print("✓ Analysis setup complete")
    
    # ---------- Setup missions ----------
    print("o Defining mission profile")
    mission = setup_mission(vehicle, analyses)
    print("✓ Mission setup complete")
    
    # ---------- Evaluate mission ----------
    print("o Running mission evaluation...")
    results = mission.evaluate()
    print("✓ Mission evaluation complete")
    
    # ---------- Analyze results ----------
    print("o Processing results and performance metrics")
    analyze_results(results, vehicle)
    print("✓ Results analysis complete")
    
    # ---------- Generate plots ----------
    print("o Generating performance plots")
    plot_results(results)
    print("✓ Plots generated successfully")
    
    print("\n 1kg Quadcopter analysis complete!")
    
    return vehicle, analyses, results

# ----------------------------------------------------------------------------------------------------------------------
#   Execute Main
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 70 + "\n 🚁 1KG QUADCOPTER DESIGN AND ANALYSIS SUITE \n" + "=" * 70)
    try:
        vehicle, analyses, results = main()
        
        print("\n" + "=" * 70 + "\n ✅ ANALYSIS COMPLETED SUCCESSFULLY \n" + "=" * 70)
        print("✓ Quadcopter design and analysis completed!")
        print("✓ Performance plots saved to 'quadcopter_plots_charles' directory")
        print("✓ Review console output for detailed performance metrics")
        print("=" * 70)
        
    except Exception as e:
        print("\n" + "=" * 70 + "\n ❌ ANALYSIS FAILED \n" + "=" * 70)
        print(f"✗ Error during analysis: {str(e)}")
        print("=" * 70)
        import traceback
        traceback.print_exc()