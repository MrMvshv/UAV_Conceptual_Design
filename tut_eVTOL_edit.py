# eVTOL_tutorial.py
#
# Created: Nov 2021, E. Botero
#

# ----------------------------------------------------------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------------------------------------------------------

# pacakgee imports
import SUAVE
assert SUAVE.__version__=='2.5.2', 'These tutorials only work with the SUAVE 2.5.2 release'
import numpy as np
import os

# module imports
from SUAVE.Core                                          import Units
from SUAVE.Attributes.Gases                              import Air
from SUAVE.Plots.Performance.Mission_Plots               import *
from SUAVE.Input_Output.OpenVSP                          import write
from SUAVE.Methods.Geometry.Two_Dimensional.Planform     import segment_properties, wing_segmented_planform, wing_planform
from SUAVE.Methods.Propulsion                            import propeller_design
from SUAVE.Methods.Propulsion.electric_motor_sizing      import size_optimal_motor
from SUAVE.Methods.Power.Battery.Sizing                  import initialize_from_mass


from copy import deepcopy




#exporting to vsp
def export_to_vsp(vehicle, base_filename='eVTOL'):
    """Export SUAVE vehicle to OpenVSP with auto-numbering"""
    try:
        # Auto-numbering logic
        counter = 0
        filename = base_filename
        while os.path.exists(f"{filename}.vsp3"):
            counter += 1
            filename = f"{base_filename}_{counter}"
        
        # Add extension (SUAVE's writer will handle this)
        full_path = os.path.abspath(filename)
        
        # Perform export
        write(vehicle, filename)  # Note: Don't include .vsp3 here
        
        # Verify
        if os.path.exists(f"{filename}.vsp3"):
            print(f"✓ Successfully created: {full_path}.vsp3")
            return True
        raise RuntimeError("File was not created")

    except Exception as e:
        print(f"✗ Export failed: {str(e)}")
        return False

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
# ----------------------------------------------------------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    # Setup a vehicle
    print("Setting up the vehicle")
    vehicle = setup_vehicle()
    print(" Vehicle setup complete")
    #print("Vehicle networks:", vehicle.networks.keys())
    #print("Lift Cruise Network:", vehicle.networks.lift_cruise)

    # success = export_to_vsp(vehicle)
    # if success:
    #     print("\nNext steps:")
    #     print("1. Open the exported file in OpenVSP")
    #     print("2. Verify the geometry and parameters")
    #     print("3. Proceed with further analysis or modifications")
    # #print("Wings:", vehicle.wings)
    #print("Fuselages:", vehicle.Fuselages)
    #print("Propulsors:", vehicle.propulsors)
    #print("Vehicle contents:")
    #print(vehicle.__dict__)
    #print("exporting vehicle")
    # export the vehicle
 
    # Setup analyses
    print("Setting up the analyses")
    analyses = setup_analyses(vehicle)
    print("finalizing...")
    analyses.finalize() #<- this builds surrogate models!
    
    # Setup a mission
    print("Setting up the mission")
    mission  = setup_mission(vehicle, analyses)


    print(f"Number of lift rotors: {len(vehicle.networks.lift_cruise.lift_rotors)}")

    for i, rotor in enumerate(vehicle.networks.lift_cruise.lift_rotors):
        print(f"Lift rotor #{i}:")
        print(f"  Radius: {rotor}")




    # Run the mission    
    results = mission.evaluate()
    print("Mission evaluation complete")
    for segment in results.segments:
        print(f"\n--- Residuals for segment: {segment.tag} ---")
        try:
            residuals = segment.conditions.residuals
            unknowns  = segment.conditions.unknowns
            for key, val in residuals.items():
                print(f"  Residual: {key:<20} {val}")
            for key, val in unknowns.items():
                print(f"  Unknown:  {key:<20} {val}")
        except Exception as e:
            print(f"  Could not read residuals: {e}")
    
    #print results
    #print("Running eVTOL tutorial")
    print("results: ")
    print(results)
    # plot the mission
    print("making plots")
    make_plots(results)
    print("saving plots")
    save_plots(results)
    print("done")
    
    return
    
    
# ----------------------------------------------------------------------------------------------------------------------
#   Vehicle
# ----------------------------------------------------------------------------------------------------------------------

def setup_vehicle():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    # Create a vehicle and set level properties
    vehicle               = SUAVE.Vehicle()
    vehicle.tag           = 'eVTOL'
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------
    # mass properties
    vehicle.mass_properties.takeoff           = 3.1 * Units.kg
    vehicle.mass_properties.operating_empty   = 2.624 * Units.kg       
    vehicle.mass_properties.max_takeoff       = 3.1 * Units.kg 
    vehicle.mass_properties.max_payload       =  0.2 * Units.kg
    vehicle.mass_properties.center_of_gravity = [[0.5,   0.  ,  0. ]] # I made this up
    
    # basic parameters
    vehicle.envelope.ultimate_load = 5.7
    vehicle.envelope.limit_load    = 3.    
    
    # ------------------------------------------------------------------
    # WINGS
    # ------------------------------------------------------------------
    # WING PROPERTIES
    wing                          = SUAVE.Components.Wings.Main_Wing()
    wing.tag                      = 'main_wing'
    wing.origin = [[0.3, 0., 0.05]] * Units.meter  # Adjusted to match fuselage
    wing.spans.projected = 1.4 * Units.meter  # Original: 35ft (~10.67m) → Scaled to ~1.5m
    wing.chords.root = 0.2 * Units.meter  # Original: 3.25ft (~0.99m) → Scaled to 0.3m


    # Root Segment
    segment = SUAVE.Components.Wings.Segment()
    segment.tag = 'Root'
    segment.percent_span_location = 0.
    segment.twist = 0. * Units.degrees
    segment.root_chord_percent = 1.5  # Root chord extension
    segment.dihedral_outboard = 1.0 * Units.degrees
    segment.sweeps.quarter_chord = 8.5 * Units.degrees
    segment.thickness_to_chord = 0.18  # Airfoil thickness (e.g., NACA 2418)
    wing.Segments.append(segment)

    # Mid Segment (22.7% span) - 10%
    segment = SUAVE.Components.Wings.Segment()
    segment.tag = 'Section_2'
    segment.percent_span_location = 0.1
    segment.twist = 0. * Units.degrees
    segment.root_chord_percent = 1.0  # Transition to uniform chord
    segment.dihedral_outboard = 1.0 * Units.degrees
    segment.sweeps.quarter_chord = 0.0 * Units.degrees
    segment.thickness_to_chord = 0.12  # Thinner airfoil outboard
    wing.Segments.append(segment)

    # Tip Segment
    segment = SUAVE.Components.Wings.Segment()
    segment.tag = 'Tip'
    segment.percent_span_location = 1.0
    segment.twist = 0. * Units.degrees
    segment.root_chord_percent = 0.8
    segment.dihedral_outboard = 0.0 * Units.degrees
    segment.sweeps.quarter_chord = 0.0 * Units.degrees
    segment.thickness_to_chord = 0.12
    wing.Segments.append(segment)

    
    # Fill out more segment properties automatically
    wing = segment_properties(wing)
    wing =  wing_segmented_planform(wing)
    
    
    ## ALSO SET THE VEHICLE REFERENCE AREA
    vehicle.reference_area         = wing.areas.reference

    # add to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------
    # HORIZONTAL TAIL (Scaled to ~0.3m span)
    # ------------------------------------------------------------------
    htail = SUAVE.Components.Wings.Horizontal_Tail()
    htail.tag = 'horizontal_tail'
    htail.areas.reference = 0.06 * Units['meters**2']  # Original: 2.0 ft² → ~0.186m² → Scaled to 0.06m²
    htail.taper = 0.5
    htail.sweeps.quarter_chord = 20. * Units.degrees
    htail.aspect_ratio = 5.0
    htail.thickness_to_chord = 0.12
    htail.dihedral = 5. * Units.degrees
    htail.origin = [[0.8, 0.0, 0.025]] * Units.meter  # Adjusted to tail position

    # Auto-compute planform
    htail = wing_planform(htail)
    vehicle.append_component(htail)

    
    # ------------------------------------------------------------------
    # VERTICAL TAIL (Scaled to ~0.2m height)
    # ------------------------------------------------------------------
    vtail = SUAVE.Components.Wings.Vertical_Tail()
    vtail.tag = 'vertical_tail'
    vtail.areas.reference = 0.03 * Units['meters**2']  # Original: 1.0 ft² → ~0.093m² → Scaled to 0.03m²
    vtail.taper = 0.5
    vtail.sweeps.quarter_chord = 30. * Units.degrees
    vtail.aspect_ratio = 2.5
    vtail.thickness_to_chord = 0.12
    vtail.origin = [[0.8, 0.0, 0.025]] * Units.meter  # Co-located with htail

    # Auto-compute planform
    vtail = wing_planform(vtail)
    vehicle.append_component(vtail)
    # Add a fuseelage
    
    # ---------------------------------------------------------------
    # FUSELAGE
    # ---------------------------------------------------------------
    # FUSELAGE PROPERTIES
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'eVTOL_Fuselage' # Updated tag

    # Basic parameters scaled for 1m vehicle
    fuselage.fineness.nose = 1.33 # (nose_length / width) = 0.2 / 0.15
    fuselage.fineness.tail = 1.33 # (tail_length / width) = 0.2 / 0.15
    fuselage.lengths.nose = 0.2 * Units.meter # 20 cm nose
    fuselage.lengths.tail = 0.2 * Units.meter # 20 cm tail
    fuselage.lengths.cabin = 0.6 * Units.meter # 60 cm cabin = 1.0m total - nose - tail
    fuselage.lengths.total = fuselage.lengths.nose + fuselage.lengths.cabin + fuselage.lengths.tail # Calculated total: 1.0 m
    fuselage.width = 0.15 * Units.meter # Max width 15 cm
    fuselage.heights.maximum = 0.15 * Units.meter # Max height 15 cm

    # Specific heights (can be refined, starting similar to max height)
    # Scaling based on the ratio of new max_height (0.15m) to old max_height (4.65ft ~= 1.417m)
    # Scale factor = 0.15 / 1.417 ~= 0.106
    # However, simpler approach for initial design: base it on the new max height
    fuselage.heights.at_quarter_length = 0.14 * Units.meter # Slightly less than max if tapered
    fuselage.heights.at_wing_root_quarter_chord = 0.15 * Units.meter # Assume wing at max height point
    fuselage.heights.at_three_quarters_length = 0.14 * Units.meter # Slightly less than max if tapered

    # Estimated areas for the 1m fuselage
    fuselage.areas.wetted = 0.6 * Units['meters**2'] # Previous estimate
    fuselage.areas.front_projected = (0.15 * 0.15) * Units['meters**2'] # Approx 0.0225 m^2

    fuselage.effective_diameter = 0.15 * Units.meter # Based on max width/height

    # Unpressurized
    fuselage.differential_pressure = 0.0 * Units.pascal # Use units

    # Initialize Segments list if it doesn't exist (Good practice)
    if not hasattr(fuselage, 'Segments'):
        fuselage.Segments = [] # Use list for append

    # Segment
    segment                           = SUAVE.Components.Lofted_Body_Segment.Segment()
    segment.tag                       = 'segment_0'
    segment.percent_x_location        = 0.
    segment.percent_z_location        = 0.
    segment.height = 0.01 * Units.meter # Very small at the tip
    segment.width = 0.01 * Units.meter # Very small at the tip
    fuselage.Segments.append(segment)

    # Segment
    segment                           = SUAVE.Components.Lofted_Body_Segment.Segment()
    segment.tag                       = 'segment_1'
    segment.percent_x_location        = 0.06
    segment.percent_z_location = 0. # Keep centerline for simplicity now
    segment.height = 0.08 * Units.meter # Scale: 0.52 * 0.107 = 0.055 -> adjusted up
    segment.width = 0.10 * Units.meter # Scale: 0.75 * 0.107 = 0.08 -> adjusted up
    fuselage.Segments.append(segment)

    # Segment
    segment                           = SUAVE.Components.Lofted_Body_Segment.Segment()
    segment.tag                       = 'segment_2'
    segment.percent_x_location = 0.20 # Adjusted location (was 0.25)
    segment.percent_z_location = 0.
    segment.height = 0.15 * Units.meter # Reaching max height
    segment.width = 0.15 * Units.meter # Reaching max width
    fuselage.Segments.append(segment)

    # Segment
    segment                           = SUAVE.Components.Lofted_Body_Segment.Segment()
    segment.tag                       = 'segment_3'
    segment.percent_x_location = 0.5 # Adjusted location (was 0.475) - Mid point approx
    segment.percent_z_location = 0.
    segment.height = 0.15 * Units.meter # Max Height
    segment.width = 0.15 * Units.meter # Max Width
    fuselage.Segments.append(segment)

    # Segment
    segment                           = SUAVE.Components.Lofted_Body_Segment.Segment()
    segment.tag                       = 'segment_4'
    segment.percent_x_location = 0.8 # Adjusted location (was 0.75) - Start of tail
    segment.percent_z_location = 0.01 # Slight raise maybe
    segment.height = 0.10 * Units.meter # Tapering down (Scale: 0.6 * 0.107 = 0.06 -> adjusted)
    segment.width = 0.08 * Units.meter # Tapering down (Scale: 0.4 * 0.107 = 0.04 -> adjusted)
    fuselage.Segments.append(segment)

    # Segment
    segment                           = SUAVE.Components.Lofted_Body_Segment.Segment()
    segment.tag                       = 'segment_5'
    segment.percent_x_location = 1. # End of tail
    segment.percent_z_location = 0.02 # Maybe slightly higher tail tip
    segment.height = 0.01 * Units.meter # Very small at the tip
    segment.width = 0.01 * Units.meter # Very small at the tip
    fuselage.Segments.append(segment)

    # add to vehicle
    vehicle.append_component(fuselage)    
    

    #-------------------------------------------------------------------
    # Booms
    #-------------------------------------------------------------------
    # Add booms for the motors
    boom                                   = SUAVE.Components.Fuselages.Fuselage()
    boom.tag                                = 'boom_R'

    # Placement: Needs careful consideration based on wing/rotor placement.
    # Example: Place start of boom slightly behind nose, outboard, slightly below fuselage centerline.
    # Assume wing starts at x=0.25m, boom under wing. Place boom start at x=0.3m?
    # Place outboard Y=0.5m? Place Z=-0.1m? (Relative to vehicle origin 0,0,0)
    boom.origin = [[0.1, 0.35, 0.04]] * Units.meter # Units are important!

    # Boom Dimensions (scaled down significantly)
    boom.lengths.total = 0.66 * Units.meter # Example: Boom slightly longer than fuselage? Or shorter? Depends on layout. Adjust as needed.
    boom.lengths.nose = 0.1 * Units.meter # Short nose cone
    boom.lengths.tail = 0.1 * Units.meter # Short tail cone
    boom.lengths.cabin = boom.lengths.total - boom.lengths.nose - boom.lengths.tail # Should define cabin length if using segments, otherwise less critical.

    boom.width = 0.03 * Units.meter # e.g., 8 cm diameter boom
    boom.heights.maximum = 0.03 * Units.meter
    boom.heights.at_quarter_length = 0.03 * Units.meter
    boom.heights.at_three_quarters_length = 0.03 * Units.meter
    boom.heights.at_wing_root_quarter_chord = 0.03 * Units.meter # Less relevant for boom unless wing attaches to it

    boom.effective_diameter = 0.03 * Units.meter

    # Areas (Recalculate for new dimensions)
    # Wetted Area: Approx surface area. Cylinder approx: pi * D * L_cyl = pi * 0.08 * (1.2-0.1-0.1) = pi * 0.08 * 1.0 = ~0.25 m^2. Add ends. Let's estimate higher.
    boom.areas.wetted = 0.05 * Units['meters**2'] # Rough estimate
    # Frontal Area: Area of cross-section. pi * (D/2)^2 = pi * (0.08/2)^2 = pi * 0.04^2 = ~0.005 m^2
    boom.areas.front_projected = np.pi * (0.03/2)**2 * Units['meters**2']

    # Fineness (Length / Diameter)
    boom.fineness.nose = boom.lengths.nose / boom.width # 0.1 / 0.08 = 1.25
    boom.fineness.tail = boom.lengths.tail / boom.width # 0.1 / 0.08 = 1.25
        
    vehicle.append_component(boom)
    
    # Now attach the mirrored boom
    other_boom              = deepcopy(boom)
    other_boom.origin[0][1] = -boom.origin[0][1]
    other_boom.tag          = 'boom_L'
    vehicle.append_component(other_boom)    

    
    #------------------------------------------------------------------
    # Network
    #------------------------------------------------------------------
    net                              = SUAVE.Components.Energy.Networks.Lift_Cruise()
    net.number_of_lift_rotor_engines = 4
    net.number_of_propeller_engines  = 1
    net.identical_propellers         = True
    net.identical_lift_rotors        = True    
    net.voltage                      = 14.8 * Units.volt  # Typical for small eVTOLs, e.g., 4S LiPo battery    
    
    #------------------------------------------------------------------
    # Electronic Speed Controller
    #------------------------------------------------------------------
    lift_rotor_esc              = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    lift_rotor_esc.efficiency   = 0.95
    net.lift_rotor_esc          = lift_rotor_esc

    propeller_esc            = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    propeller_esc.efficiency = 0.95
    net.propeller_esc        = propeller_esc
    
    #------------------------------------------------------------------
    # Payload
    #------------------------------------------------------------------
    payload                      = SUAVE.Components.Energy.Peripherals.Avionics()
    payload.power_draw           = 0.    
    net.payload                  = payload
    
    #------------------------------------------------------------------
    # Avionics
    #------------------------------------------------------------------
    avionics            = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.power_draw = 50. * Units.watts
    net.avionics        = avionics    
    
    #------------------------------------------------------------------
    # Design Battery
    #------------------------------------------------------------------
    bat                      = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiNiMnCoO2_18650() 
    bat.mass_properties.mass = 0.3 * Units.kg        
    bat.max_voltage          = net.voltage   
    initialize_from_mass(bat)    
    net.battery              = bat      
    
    #------------------------------------------------------------------
    # Design Rotors and Propellers
    #------------------------------------------------------------------    
    
    # The tractor propeller
    propeller                        = SUAVE.Components.Energy.Converters.Propeller()
    propeller.origin                 = [[0.0, 0.0, 0.0]]  # Positioned at rear of fuselage
    propeller.number_of_blades       = 2                   # Reduced for small scale
    propeller.tip_radius             = 0.15 * Units.meter  # 30cm diameter (~1/6 of large example)
    propeller.hub_radius             = 0.015 * Units.meter # 3cm hub
    propeller.angular_velocity       = 6000 * Units.rpm    # Higher RPM for smaller prop
    propeller.freestream_velocity    = 30. * Units['m/s']    # ~58 knots cruise
    propeller.design_Cl              = 0.7                 # Same airfoil loading
    propeller.design_altitude        = 100. * Units.meter  # Low altitude operation
    propeller.design_thrust          = 15. * Units.newton  # ~3.4 lbf cruise thrust
    propeller.airfoil_geometry       = ['./Airfoils/NACA_4412.txt']
    propeller.airfoil_polars         = [['./Airfoils/Polars/NACA_4412_polar_Re_50000.txt' ,
                                         './Airfoils/Polars/NACA_4412_polar_Re_100000.txt' ,
                                         './Airfoils/Polars/NACA_4412_polar_Re_200000.txt' ,
                                         './Airfoils/Polars/NACA_4412_polar_Re_500000.txt' ,
                                         './Airfoils/Polars/NACA_4412_polar_Re_1000000.txt' ]]    
    propeller.airfoil_polar_stations = np.zeros((20),dtype=np.int8).tolist() 
    propeller                        = propeller_design(propeller)
    net.propellers.append(propeller)
    
  # The lift rotors
    #lift_rotor                            = SUAVE.Components.Energy.Converters.Lift_Rotor()
    lift_rotor                            = SUAVE.Components.Energy.Converters.Propeller()
    lift_rotor.blade_solidity = 0.12  # Moderate value, more market-like 
    lift_rotor.tip_radius                 = 0.18 * Units.meter  # 13cm diameter
    lift_rotor.hub_radius                 = 0.0075 * Units.meter # 4cm hub
    lift_rotor.number_of_blades           = 2                  # Balance efficiency/weight
    lift_rotor.design_tip_mach            = 0.25               # Lower tip Mach for noise
    lift_rotor.freestream_velocity        = 2.5 * Units['m/s']    # ~500 ft/min descent
    #lift_rotor.angular_velocity           = lift_rotor.design_tip_mach*Air().compute_speed_of_sound()/lift_rotor.tip_radius
    lift_rotor.angular_velocity           = 5700 * Units.rpm    # Higher RPM for small rotors
    lift_rotor.design_Cl                  = 0.6              # Higher for hover
    lift_rotor.design_altitude            = 500. * Units.meter # Ground effect considered
    lift_rotor.design_thrust              = 9.6 * Units.newton # ~lbf per rotor
    lift_rotor.variable_pitch             = False              # Important for small craft
    lift_rotor.airfoil_geometry       = ['./Airfoils/NACA_4412.txt']
    lift_rotor.airfoil_polars         = [['./Airfoils/Polars/NACA_4412_polar_Re_50000.txt' ,
                                         './Airfoils/Polars/NACA_4412_polar_Re_100000.txt' ,
                                         './Airfoils/Polars/NACA_4412_polar_Re_200000.txt' ,
                                         './Airfoils/Polars/NACA_4412_polar_Re_500000.txt' ,
                                         './Airfoils/Polars/NACA_4412_polar_Re_1000000.txt' ]]    
    lift_rotor.airfoil_polar_stations = np.zeros((20),dtype=np.int8).tolist()

    # Design the rotor
    number_of_stations = 20
    lift_rotor.twist_distribution = np.linspace(25, 5, number_of_stations)

    lift_rotor                            = propeller_design(lift_rotor)

    R = lift_rotor.tip_radius
    Rh = lift_rotor.hub_radius
    chi0 = Rh / R
    chi = np.linspace(chi0, 1.0, number_of_stations)

    lift_rotor.n_stations = number_of_stations
    lift_rotor.radius_distribution = chi * R

    # Set parameters
    c_min = 0.02  # chord at root and tip
    c_max = 0.08   # peak chord at 0.4*R
    chi_peak = 0.33  # location of peak chord (normalized)

    # Use a Gaussian centered at chi_peak
    sigma = 0.15  # controls spread (smaller = sharper peak)
    gaussian_peak = np.exp(-((chi - chi_peak)**2) / (2 * sigma**2))

    # Normalize to peak at 1.0
    gaussian_peak /= np.max(gaussian_peak)

    # Apply chord profile
    chord_guess = c_min + (c_max - c_min) * gaussian_peak

    lift_rotor.chord_distribution = chord_guess
    lift_rotor.propeller_radius = R
    lift_rotor.override_geometry = False



    #print("\n\n\n\n\Lift Rotor Design:")
    #print(lift_rotor.chord_distribution)
    #print(lift_rotor.twist_distribution)
    print("====================================================")   
  #  print(lift_rotor.geometry)

   

    # Appending rotors - positions based on your boom locations
    rotations = [1,-1,1,-1]  # Alternating rotation directions
    origins   = [[0.2,  0.35, 0.09],  # Right front boom
                [0.2, -0.35, 0.09],  # Left front boom
                [0.7,  0.35, 0.09],  # Right rear boom
                [0.7, -0.35, 0.09]]  # Left rear boom

    for ii in range(4):
        lr          = deepcopy(lift_rotor)
        lr.tag      = 'lift_rotor_' + str(ii+1)
        lr.rotation = rotations[ii]
        lr.origin   = [origins[ii]]
        net.lift_rotors.append(lr)   
    

    
    #------------------------------------------------------------------
    # Design Motors
    #------------------------------------------------------------------
   # Propeller (Cruise) Motor
    propeller_motor                      = SUAVE.Components.Energy.Converters.Motor()
    propeller_motor.efficiency           = 0.90          # Lower for small motors
    propeller_motor.nominal_voltage      = bat.max_voltage
    propeller_motor.mass_properties.mass = 0.15 * Units.kg
    propeller_motor.origin               = propeller.origin
    propeller_motor.propeller_radius     = propeller.tip_radius
    propeller_motor.no_load_current      = 0.5          # Amps
    propeller_motor                      = size_optimal_motor(propeller_motor,propeller)
    net.propeller_motors.append(propeller_motor)    

    # Lift Rotor Motors
    lift_rotor_motor                         = SUAVE.Components.Energy.Converters.Motor()
    lift_rotor_motor.efficiency              = 0.82
    lift_rotor_motor.nominal_voltage         = bat.max_voltage
    lift_rotor_motor.mass_properties.mass    = 0.2 * Units.kg
    lift_rotor_motor.origin                  = lift_rotor.origin
    lift_rotor_motor.propeller_radius        = lift_rotor.tip_radius
    lift_rotor_motor.gearbox_efficiency      = 1.0      # Direct drive common at small scale
    lift_rotor_motor.no_load_current         = 0.8      # Amps
    lift_rotor_motor                         = size_optimal_motor(lift_rotor_motor,lift_rotor) 

    for _ in range(4):
        lrm = deepcopy(lift_rotor_motor)
        lrm.tag = 'lift_motor_' + str(_+1)
        net.lift_rotor_motors.append(lrm)    
        
    
    vehicle.append_component(net)
    
    
    # Now account for things that have been overlooked for now:
    vehicle.excrescence_area = 0.1
    
    return vehicle
    
    
# ----------------------------------------------------------------------------------------------------------------------
#   Analyses
# ----------------------------------------------------------------------------------------------------------------------
   
def setup_analyses(vehicle):
    # ------------------------------------------------------------------
    #   Initialize the Analyses
    # ------------------------------------------------------------------
    analyses = SUAVE.Analyses.Vehicle()

    # ------------------------------------------------------------------
    #  Weights
    weights = SUAVE.Analyses.Weights.Weights_eVTOL()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.4 * vehicle.excrescence_area/vehicle.reference_area
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Energy
    energy= SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.networks
    analyses.append(energy)
    
    # ------------------------------------------------------------------
    #  Noise Analysis
    noise = SUAVE.Analyses.Noise.Fidelity_One()
    noise.geometry = vehicle
    analyses.append(noise)

    # ------------------------------------------------------------------
    #  Planet Analysis
    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)

    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)

    return analyses


# ----------------------------------------------------------------------------------------------------------------------
#   Mission
# ----------------------------------------------------------------------------------------------------------------------

def setup_mission(vehicle,analyses):
    
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    mission            = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag        = 'the_mission'   
    
    # unpack Segments module
    Segments                                                 = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment                                             = Segments.Segment()
    base_segment.state.numerics.number_control_points        = 8

    #tweaking solver settings
    base_segment.state.numerics.iterations = 50
    base_segment.state.numerics.tolerance  = 1e-4

    base_segment.process.initialize.initialize_battery       = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip   
    base_segment.process.iterate.conditions.stability        = SUAVE.Methods.skip
    base_segment.process.finalize.post_process.stability     = SUAVE.Methods.skip      
    ones_row                                                 = base_segment.state.ones_row
    
    # ------------------------------------------------------------------
    #   Constant Altitude Hover Segment
    # ------------------------------------------------------------------
    segment     = Segments.Hover.Hover(base_segment)
    segment.tag = "hover_segment"
    segment.analyses.extend(analyses)
    segment.altitude         = 100.0 * Units.ft         # constant altitude
    segment.duration         = 30.0 * Units.seconds     # hover duration
    segment.battery_energy   = vehicle.networks.lift_cruise.battery.max_energy * 0.95
    segment.process.iterate.unknowns.mission = SUAVE.Methods.skip

    # Set initial guesses if needed
    segment.initial_throttle = 0.95

    # Add lift network unknowns and residuals
    segment = vehicle.networks.lift_cruise.add_lift_unknowns_and_residuals_to_segment(segment)

    # Add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Hover Climb Segment
    # ------------------------------------------------------------------
    # segment     = Segments.Hover.Climb(base_segment)
    # segment.tag = "hover_climb"
    # segment.analyses.extend(analyses)
    # segment.altitude_start                                   = 0.0   * Units.ft
    # segment.altitude_end                                     = 100.  * Units.ft
    # segment.climb_rate                                       = 200.  * Units['ft/min']
    # segment.battery_energy                                   = vehicle.networks.lift_cruise.battery.max_energy*0.95
    # segment.process.iterate.unknowns.mission                 = SUAVE.Methods.skip
    # segment = vehicle.networks.lift_cruise.add_lift_unknowns_and_residuals_to_segment(segment)
    
    # # add to misison
    # mission.append_segment(segment)
    
    # # ------------------------------------------------------------------
    # #   Second Climb Segment: Constant Speed, Constant Rate
    # # ------------------------------------------------------------------
    # segment                                            = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    # segment.tag                                        = 'wing_climb'
    # segment.analyses.extend(analyses)
    # segment.air_speed                                  = 70. * Units.knots
    # segment.altitude_end                               = 3000. * Units.ft
    # segment.climb_rate                                 = 500. * Units['ft/min'] 
    # segment = vehicle.networks.lift_cruise.add_cruise_unknowns_and_residuals_to_segment(segment)
    
    # # add to misison
    # mission.append_segment(segment)        
    
    # # ------------------------------------------------------------------
    # #   Cruise
    # # ------------------------------------------------------------------
    # segment                                            = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    # segment.tag                                        = "Cruise"
    # segment.analyses.extend(analyses)
    # segment.distance                                   = 50.   * Units.nautical_miles
    # segment.air_speed                                  = 100.  * Units.knots
    # segment = vehicle.networks.lift_cruise.add_cruise_unknowns_and_residuals_to_segment(segment)

    # # add to misison
    # mission.append_segment(segment)    
    
    # # ------------------------------------------------------------------
    # #  Descent
    # # ------------------------------------------------------------------
    # segment                                            = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    # segment.tag                                        = "wing_descent"
    # segment.analyses.extend(analyses)
    # segment.air_speed                                  = 100. * Units.knots
    # segment.altitude_end                               = 100 * Units.ft
    # segment.descent_rate                               = 300. * Units['ft/min'] 
    # segment = vehicle.networks.lift_cruise.add_cruise_unknowns_and_residuals_to_segment(segment)

    # # add to misison
    # mission.append_segment(segment)       
    
    # # ------------------------------------------------------------------
    # #  Hover Descent
    # # ------------------------------------------------------------------
    # segment                                            = Segments.Hover.Descent(base_segment)
    # segment.tag                                        = "hover_descent"
    # segment.analyses.extend(analyses)
    # segment.altitude_end                              = 0.
    # segment.descent_rate                              = 100 * Units['ft/min'] 
    # segment.process.iterate.unknowns.mission          = SUAVE.Methods.skip
    # segment = vehicle.networks.lift_cruise.add_lift_unknowns_and_residuals_to_segment(segment)

    # # add to misison
    # mission.append_segment(segment)          

    return mission

# ----------------------------------------------------------------------------------------------------------------------
#   Plots
# ----------------------------------------------------------------------------------------------------------------------

def make_plots(results):
    
    plot_flight_conditions(results)

    plot_aerodynamic_coefficients(results)
    
    plot_battery_pack_conditions(results)
    
    plot_lift_cruise_network(results)

if __name__ == '__main__':
    print("Running eVTOL tutorial")
    main()
    print("Done")
    plt.show()
    print('show graph done')