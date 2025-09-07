# LunaDesign1.py
#
# Created: Nov 2021, E. Botero
# Extensively edited for Luna EVTOL Design By:
# Luna Design Team:
# Irungu Macharia
# Brian Vuyiya
#

# TO-DO
#1. find all TO-DOs in the code and do them
#2. Uncomment/remove commented blocks as appropriate(advanced)
#3. Do analyses and iterate till completely satisfactory results...

# ----------------------------------------------------------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------------------------------------------------------

# package imports
import SUAVE
assert SUAVE.__version__=='2.5.2', 'This codebase only work with the SUAVE 2.5.2 release!'
import numpy as np
import os
from datetime import datetime
import glob
import matplotlib.pyplot as plt
import SUAVE.Methods.Missions.Segments.Common.Noise as Noise

# module imports
from SUAVE.Core                                          import Units
from SUAVE.Attributes.Gases                              import Air
from SUAVE.Plots.Performance.Mission_Plots               import *
from SUAVE.Input_Output.OpenVSP                          import write
from SUAVE.Methods.Geometry.Two_Dimensional.Planform     import segment_properties, wing_segmented_planform, wing_planform
from SUAVE.Methods.Propulsion                            import propeller_design
from SUAVE.Methods.Propulsion.electric_motor_sizing      import size_optimal_motor
from SUAVE.Methods.Power.Battery.Sizing                  import initialize_from_mass
from SUAVE.Analyses import Process

from copy import deepcopy
from scipy.interpolate import RectBivariateSpline

# ----------------------------------------------------------------------------------------------------------------------
#   VSP GENERATION
# ----------------------------------------------------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------------------------------------------
#   Plots
# ----------------------------------------------------------------------------------------------------------------------

#TO-DO: Add more plots to see more data - done, verify...
def save_plots(results, plot_all=False):
    """
    Save SUAVE plots to a sequential dated folder
    
    Parameters:
    results: Mission results object
    plot_all: If True, plot all available graphs. If False, plot only main analysis graphs
    """
    # Create sequential folder with date
    base_dir = os.getcwd()
    date_str = datetime.now().strftime("%Y%m%d")
    
    # Find existing folders with the same date pattern
    existing_folders = glob.glob(os.path.join(base_dir, f"graphs_*_{date_str}"))
    
    # Determine the next sequential number
    if existing_folders:
        numbers = []
        for folder in existing_folders:
            try:
                folder_base = os.path.basename(folder)
                number = int(folder_base.split('_')[1])
                numbers.append(number)
            except (ValueError, IndexError):
                continue
        next_number = max(numbers) + 1 if numbers else 1
    else:
        next_number = 1
    
    # Create folder name
    folder_name = f"graphs_{next_number:02d}_{date_str}"
    folder_path = os.path.join(base_dir, folder_name)
    
    # Create the directory
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created folder: {folder_path}")
    
    # Dictionary of available plot functions with descriptions
    available_plots = {
        # Main analysis plots (always plotted)
        'main': {
            'plot_aerodynamic_forces': 'Aerodynamic forces over mission',
            'plot_aerodynamic_coefficients': 'Aerodynamic coefficients (CL, CD)',
            'plot_mission': 'Mission profile overview',
            'plot_battery_pack_conditions': 'Battery pack conditions',
            'plot_altitude_sweep': 'Altitude profile',
            'plot_flight_profile': 'Flight profile summary'
        },
        
        # Additional plots (plotted only if plot_all=True)
        'additional': {
            'plot_battery_cell_conditions': 'Battery cell-level conditions',
            'plot_disc_loading': 'Disk loading analysis',
            'plot_electric_motor_efficiency': 'Motor efficiency',
            'plot_lift_distribution': 'Lift distribution',
            'plot_propeller_conditions': 'Propeller performance',
            'plot_rotor_conditions': 'Rotor performance',
            'plot_solar_flux': 'Solar flux analysis',
            'plot_stability': 'Stability analysis',
            'plot_vehicle': 'Vehicle geometry',
            'plot_vehicle_vlm_panels': 'VLM panels visualization',
            'plot_drag_breakdown': 'Drag breakdown analysis'
        }
    }
    
    # Combine plots based on mode
    if plot_all:
        plots_to_generate = {**available_plots['main'], **available_plots['additional']}
        print("Generating ALL available plots...")
    else:
        plots_to_generate = available_plots['main']
        print("Generating main analysis plots...")
    
    successful_plots = 0
    failed_plots = []
    
    # Generate and save plots
    for plot_func_name, description in plots_to_generate.items():
        try:
            # Get the plot function from globals
            plot_func = globals().get(plot_func_name)
            
            if plot_func is None:
                print(f"✗ Function not found: {plot_func_name}")
                failed_plots.append(plot_func_name)
                continue
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Call the plot function
            plot_func(results)
            
            # Format title
            title_name = plot_func_name.replace('plot_', '').replace('_', ' ').title()
            plt.title(f"{title_name}\n{description}", fontsize=14, pad=20)
            plt.tight_layout()
            
            # Save plot
            filename = f"{plot_func_name}.png"
            filepath = os.path.join(folder_path, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved: {filename}")
            successful_plots += 1
            
        except Exception as e:
            print(f"✗ Failed {plot_func_name}: {str(e)}")
            failed_plots.append(plot_func_name)
            plt.close()  # Ensure figure is closed even if error occurs
    
    # Print summary
    print(f"\n=== PLOT GENERATION SUMMARY ===")
    print(f"Folder: {folder_path}")
    print(f"Successful: {successful_plots}")
    print(f"Failed: {len(failed_plots)}")
    
    if failed_plots:
        print("Failed plots:")
        for failed in failed_plots:
            print(f"  - {failed}")
    
    return folder_path, successful_plots, failed_plots



    
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
    net.identical_lift_rotors        = False    
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
    #TO-DO: replace tractor prop params with real world prop
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
    
  # The lift rotors - vsp generation
    # #print("\n\n\n\n\Lift Rotor Design(for openvsp visualization)")
    # #lift_rotor                            = SUAVE.Components.Energy.Converters.Lift_Rotor()
    # lift_rotor                            = SUAVE.Components.Energy.Converters.Propeller()
    # lift_rotor.blade_solidity = 0.12  # Moderate value, more market-like 
    # lift_rotor.tip_radius                 = 0.18 * Units.meter  # 13cm diameter
    # lift_rotor.hub_radius                 = 0.0075 * Units.meter # 4cm hub
    # lift_rotor.number_of_blades           = 2                  # Balance efficiency/weight
    # lift_rotor.design_tip_mach            = 0.25               # Lower tip Mach for noise
    # lift_rotor.freestream_velocity        = 2.5 * Units['m/s']    # ~500 ft/min descent
    # lift_rotor.angular_velocity           = 5700 * Units.rpm    # Higher RPM for small rotors
    # lift_rotor.design_Cl                  = 0.6              # Higher for hover
    # lift_rotor.design_altitude            = 500. * Units.meter # Ground effect considered
    # lift_rotor.design_thrust              = 9.6 * Units.newton # ~lbf per rotor
    # lift_rotor.variable_pitch             = False              # Important for small craft
    # lift_rotor.airfoil_geometry       = ['./Airfoils/NACA_4412.txt']
    # lift_rotor.airfoil_polars         = [['./Airfoils/Polars/NACA_4412_polar_Re_50000.txt' ,
    #                                      './Airfoils/Polars/NACA_4412_polar_Re_100000.txt' ,
    #                                      './Airfoils/Polars/NACA_4412_polar_Re_200000.txt' ,
    #                                      './Airfoils/Polars/NACA_4412_polar_Re_500000.txt' ,
    #                                      './Airfoils/Polars/NACA_4412_polar_Re_1000000.txt' ]]    
    # lift_rotor.airfoil_polar_stations = np.zeros((20),dtype=np.int8).tolist()

    # # Design the rotor
    # number_of_stations = 20
    # lift_rotor.twist_distribution = np.linspace(25, 5, number_of_stations)

    # lift_rotor                            = propeller_design(lift_rotor)

    # R = lift_rotor.tip_radius
    # Rh = lift_rotor.hub_radius
    # chi0 = Rh / R
    # chi = np.linspace(chi0, 1.0, number_of_stations)

    # lift_rotor.n_stations = number_of_stations
    # lift_rotor.radius_distribution = chi * R

    # # Set parameters
    # c_min = 0.02  # chord at root and tip
    # c_max = 0.08   # peak chord at 0.4*R
    # chi_peak = 0.33  # location of peak chord (normalized)

    # # Use a Gaussian centered at chi_peak
    # sigma = 0.15  # controls spread (smaller = sharper peak)
    # gaussian_peak = np.exp(-((chi - chi_peak)**2) / (2 * sigma**2))

    # # Normalize to peak at 1.0
    # gaussian_peak /= np.max(gaussian_peak)

    # # Apply chord profile
    # chord_guess = c_min + (c_max - c_min) * gaussian_peak

    # lift_rotor.chord_distribution = chord_guess
    # lift_rotor.propeller_radius = R
    # lift_rotor.override_geometry = False

    # print("====================================================")   
    # print(lift_rotor.geometry)



    ###LIFT ROTOR DESIGN (blade-element) - Gemfan 1045###
    # TO-DO - Replace with real world prop - composite 1038 prop

    #lift rotors, analytical generation using market prop - Gemfan 1045
    #fixed the fixed thrust bug by using a propeller component instead of lift rotor component
    print("\n\n\n\n Lift Rotor Design (blade-element) — Gemfan 10x4.5")

    # --- Build a physically-meaningful Gemfan 10x4.5 model ---
    R_tip      = 0.127          # 10 in dia = 0.254 m → radius 0.127 m
    R_hub      = 0.020
    B          = 2              # blades
    pitch_m    = 0.1143         # 4.5 in → 0.1143 m geometric pitch
    rpm_nom    = 7500.0
    omega_nom  = 2.0*np.pi*rpm_nom/60.0   # ~785 rad/s

    # Station layout (root → tip). Use 8–10 stations for stability.
    n_stations = 8
    r_dist     = np.linspace(R_hub, R_tip, n_stations)

    # Chord distribution (meters): slightly tapered
    # Target solidity ~0.10–0.12; for R=0.127, B=2 this implies avg chord ≈ 0.02–0.025 m
    c_root = 0.032
    c_tip  = 0.016
    chord  = np.linspace(c_root, c_tip, n_stations)

    # Twist distribution (deg): set from geometric pitch at 75%R, washout to the tip
    # beta_geo = arctan(pitch/(2*pi*r))
    beta_75R = np.degrees(np.arctan(pitch_m/(2.0*np.pi*0.75*R_tip)))  # ~10.8 deg
    beta_root = beta_75R + 3.0   # a bit more at root
    beta_tip  = beta_75R - 3.0   # a bit less at tip
    twist_deg = np.linspace(beta_root, beta_tip, n_stations)

    # --- Analytic airfoil model (surrogate CL/CD over Re, alpha) ---
    # Alpha range in radians and Re range typical for ~10" props
    alpha_deg = np.linspace(-12.0, 18.0, 31)     # dense and smooth
    alpha     = np.radians(alpha_deg)
    Re_vals   = np.array([8e4, 1.2e5, 1.6e5, 2.0e5, 2.5e5])  # benign coverage

    alpha0   = np.radians(-2.0)  # zero-lift angle
    Cl_max   = 1.2               # soft cap
    Cd0      = 0.012
    k_ind    = 0.010

    # Build CL(Re, alpha) and CD(Re, alpha) as 2D arrays with smooth stall limiting
    CL_table = []
    CD_table = []
    for Re in Re_vals:
        Cl_line = 2.0*np.pi*(alpha - alpha0)
        # Soft stall limiting (tanh smooths the slope near +/- stall)
        Cl_line = Cl_max*np.tanh(Cl_line/Cl_max)
        Cd_line = Cd0 + k_ind*(Cl_line**2)
        CL_table.append(Cl_line)
        CD_table.append(Cd_line)
    CL_table = np.vstack(CL_table)   # shape (nRe, nAlpha)
    CD_table = np.vstack(CD_table)

    # Build surrogates (C2 is overkill; linear (kx=1,ky=1) is numerically safe)
    cl_spline = RectBivariateSpline(Re_vals, alpha, CL_table, kx=1, ky=1)
    cd_spline = RectBivariateSpline(Re_vals, alpha, CD_table, kx=1, ky=1)

    # --- SUAVE propeller object configured for blade-element ---
    lift_proto = SUAVE.Components.Energy.Converters.Propeller()
    lift_proto.tag                 = "Gemfan_10x4.5"
    lift_proto.propeller_radius    = R_tip
    lift_proto.hub_radius          = R_hub
    lift_proto.number_of_blades    = B

    # Enable blade-element analysis
    lift_proto.override_geometry   = False
    lift_proto.airfoil_flag        = True
    lift_proto.use_2d_analysis     = True
    lift_proto.use_blade_element   = True

    # Provide geometry distributions
    lift_proto.n_stations          = n_stations
    lift_proto.radius_distribution = r_dist        # meters
    lift_proto.chord_distribution  = chord         # meters
    lift_proto.twist_distribution  = np.radians(twist_deg) # radians
    lift_proto.thickness_to_chord  = (0.12*np.ones(n_stations))

    # Airfoil polars (use one "default" polar everywhere)
    lift_proto.airfoil_geometry        = ['default']
    lift_proto.airfoil_polars          = [['default']]
    lift_proto.airfoil_polar_stations  = [0]
    lift_proto.airfoil_cl_surrogates   = {'default': cl_spline}
    lift_proto.airfoil_cd_surrogates   = {'default': cd_spline}

    # Operating guesses (not enforced if network solves throttle/RPM)
    lift_proto.angular_velocity    = omega_nom          # rad/s initial guess
    lift_proto.variable_pitch      = False
    lift_proto.beta_0 = np.radians(twist_deg).tolist()


    # Optional: a rough solidity number for reporting only
    lift_proto.blade_solidity      = 2.0*np.trapz(chord, r_dist)/(np.pi*R_tip)

    # --- Instantiate the four rotors with positions and rotation sense ---
    rotations = [ 1, -1,  1, -1]   # alternate
    origins   = [[0.2,  0.35, 0.09],
                [0.2, -0.35, 0.09],
                [0.7,  0.35, 0.09],
                [0.7, -0.35, 0.09]]
    
    # --- Design-point placeholders (for motor sizing)
    #  These act as “targets” for motor sizing not as the actual thrust calculation during the mission 
    # (that comes from the blade-element model).---
    lift_proto.design_thrust = 12.0 * Units.newton       # per rotor target at hover
    lift_proto.design_power  = 180.0 * Units.watt        # ballpark
    lift_proto.design_torque = lift_proto.design_power / lift_proto.angular_velocity

    net.lift_rotors = {}   # use dict, not list

    for ii in range(4):
        lr          = deepcopy(lift_proto)
        lr.tag      = f'lift_rotor_{ii+1}'
        lr.rotation = rotations[ii]
        lr.origin   = origins[ii]
        lr.inputs.pitch_command   = 0.0
        lr.inputs.y_axis_rotation = 0.0
        net.lift_rotors[lr.tag]   = lr   # assign to dict


    
    #------------------------------------------------------------------
    # Design Motors
    #------------------------------------------------------------------
   # Propeller (Cruise) Motor
   # TO-DO: replace with real world cruise motor
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
    # default config
    # lift_rotor_motor                         = SUAVE.Components.Energy.Converters.Motor()
    # lift_rotor_motor.efficiency              = 0.82
    # lift_rotor_motor.nominal_voltage         = bat.max_voltage
    # lift_rotor_motor.mass_properties.mass    = 0.2 * Units.kg
    # lift_rotor_motor.origin                  = lift_rotor.origin
    # lift_rotor_motor.propeller_radius        = lift_rotor.tip_radius
    # lift_rotor_motor.gearbox_efficiency      = 1.0      # Direct drive common at small scale
    # lift_rotor_motor.no_load_current         = 0.8      # Amps

    #market config -  Sunnysky X2212 980KV 
    #TO-DO: Validate this section
    lift_rotor_motor = SUAVE.Components.Energy.Converters.Motor()
    lift_rotor_motor.tag = "X2212_980KV"

    # Real-world specs
    lift_rotor_motor.efficiency = 0.85
    lift_rotor_motor.nominal_voltage = 14.8 * Units.volt  # 4S
    lift_rotor_motor.mass_properties.mass = 0.072 * Units.kg  # X2212 weight
    lift_rotor_motor.no_load_current = 0.5  # Typical for this class
    lift_rotor_motor.origin = lift_proto.origin
    lift_rotor_motor.propeller_radius = lift_proto.propeller_radius
    lift_rotor_motor.gearbox_efficiency = 1.0  # Direct drive
    lift_rotor_motor.resistance = 0.12  # Ohms, estimated
    lift_rotor_motor.speed_constant = 980 * Units['rpm/volt']  # KV rating
    lift_rotor_motor.max_power_output = 250.0 * Units.watt  # To reflect ESC/motor pairing


    # Optional: skip optimal sizing and assign manually for more realistic conditions
    lift_rotor_motor                         = size_optimal_motor(lift_rotor_motor,lift_proto) 

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

    # tell SUAVE to skip the noise computation step
    base_segment.process.iterate.noise  = SUAVE.Methods.skip
    base_segment.process.finalize.noise = SUAVE.Methods.skip

    base_segment.process.initialize.initialize_battery       = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip   
    base_segment.process.iterate.conditions.stability        = SUAVE.Methods.skip
    base_segment.process.finalize.post_process.stability     = SUAVE.Methods.skip      
    ones_row                                                 = base_segment.state.ones_row
    
    # ------------------------------------------------------------------
    #   Constant Altitude Hover Segment
    # ------------------------------------------------------------------
    print("Performing k alt hover segment mission analysis")
    segment     = Segments.Hover.Hover(base_segment)
    segment.tag = "hover_segment"
    segment.analyses.extend(analyses)
    segment.altitude         = 100.0 * Units.ft         # constant altitude
    segment.duration         = 300.0 * Units.seconds     # hover duration
    segment.battery_energy   = vehicle.networks.lift_cruise.battery.max_energy * 0.95
    segment.process.iterate.unknowns.mission = SUAVE.Methods.skip

    # Set initial guesses if needed
    segment.initial_throttle = 0.95

    # Add lift network unknowns and residuals
    segment = vehicle.networks.lift_cruise.add_lift_unknowns_and_residuals_to_segment(segment)

    segment.process.iterate.noise  = SUAVE.Methods.skip
    segment.process.finalize.noise = SUAVE.Methods.skip

    # Set control points and throttle profile (PART OF THE BUGFIX)
    segment.state.numerics.number_control_points = 8
    segment.conditions.propulsion.throttle = np.ones((8,1))*0.6

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
#   Print Results
# ----------------------------------------------------------------------------------------------------------------------


def print_segment_results(results):
    """Print detailed results for each mission segment"""
    for segment in results.segments:
        print(f"\n=== RESIDUALS FOR: {segment.tag.upper()} ===")
        try:
            residuals = segment.conditions.residuals
            unknowns = segment.conditions.unknowns
            for key, val in residuals.items():
                print(f"  Residual: {key:<20} {val}")
            for key, val in unknowns.items():
                print(f"  Unknown:  {key:<20} {val}")
        except Exception as e:
            print(f"  Could not read residuals: {e}")
        
        print_segment_performance(segment)


def print_segment_performance(segment):
    """Print performance metrics for a single segment"""
    print(f"\n=== RESULTS FOR: {segment.tag.upper()} ===")
    
    # Time
    time = segment.conditions.frames.inertial.time[:, 0]
    dt = time[-1] - time[0]
    print(f"  Duration       = {dt:.2f} s")
    
    # Altitude
    alt = -segment.conditions.frames.inertial.position_vector[:, 2]
    dz = alt[-1] - alt[0]
    print(f"  Final altitude = {alt[-1]:.2f} m")
    
    # Velocity
    vel = segment.conditions.freestream.velocity[:, 0]  # m/s
    climb_desc_rate = dz / dt
    
    if "hover" in segment.tag.lower():
        print(f"  Vertical rate  = {climb_desc_rate:.2f} m/s")
    
    elif "climb" in segment.tag.lower():
        print(f"  Mean airspeed  = {vel.mean():.2f} m/s")
        print(f"  Climb rate = {climb_desc_rate:.2f} m/s")
    
    elif "descent" in segment.tag.lower():
        print(f"  Mean airspeed  = {vel.mean():.2f} m/s")
        print(f"  Descent rate   = {climb_desc_rate:.2f} m/s")
    
    elif "cruise" in segment.tag.lower():
        print(f"  Mean cruise speed = {vel.mean():.2f} m/s")
    
    # Aerodynamics
    CL = segment.conditions.aerodynamics.lift_coefficient[:, 0]
    CD = segment.conditions.aerodynamics.drag_coefficient[:, 0]
    LD_ratio = CL.mean() / CD.mean() if CD.mean() > 0 else 0.0
    print(f"  Mean CL  = {CL.mean():.3f}")
    print(f"  Mean CD  = {CD.mean():.3f}")
    print(f"  Mean L/D = {LD_ratio:.2f}")
            
    # Lift Rotor Thrust
    lift_arr = segment.conditions.propulsion.lift_rotor_thrust
    per_rotor = lift_arr.copy()
    per_rotor[:, 1:] = lift_arr[:, 1:] - lift_arr[:, :-1]
    lift_total = per_rotor.sum(axis=1) 
    print("  Avg lift rotor thrust:")
    print(f"    Total   = {lift_total.mean():.2f} N") 
    
    # Per rotor thrust
    for i in range(per_rotor.shape[1]):
        print(f"    Rotor {i+1} = {per_rotor[:, i].mean():.2f} N")
    
    # Propeller Thrust
    prop_thrust = segment.conditions.propulsion.propeller_thrust[:, 0]
    print(f"  Avg total propeller thrust: {prop_thrust.mean():.2f} N")
    
    # Energy consumption
    batt = segment.conditions.propulsion.battery_energy[:, 0]
    print("  Battery Energy:")
    print(f"    Used = {(batt[0] - batt[-1]):.2f} J")
    print(f"    End  = {batt[-1] / 1000:.2f} KJ")
    
    # Battery state of charge
    soc = segment.conditions.propulsion.battery_state_of_charge[:, 0]
    print("  Battery SOC:")
    print(f"    Used = {(soc[0] - soc[-1]) * 100:.3f} %")
    print(f"    End  = {soc[-1] * 100:.3f} %")
    
    # Power draw
    power_draw = segment.conditions.propulsion.battery_power_draw[:, 0]
    print("  Battery Power draw:")
    print(f"    Avg  = {abs(power_draw.mean()):.2f} W")
    print(f"    Peak = {abs(power_draw.max()):.2f} W")


def print_mission_summary(results):
    """Print overall mission summary"""
    print("\n===== MISSION SUMMARY =====")
    
    # Calculate mission range
    mission_range = 0.0
    for segment in results.segments:
        time = segment.conditions.frames.inertial.time[:, 0]
        vel = segment.conditions.freestream.velocity[:, 0]
        mission_range += np.trapz(vel, time)
    
    # Total time
    total_time = results.segments[-1].conditions.frames.inertial.time[-1, 0]
    print(f"  Total mission time   = {total_time:.2f} s")
    
    # Total range 
    print(f"  Total mission range  = {mission_range:.2f} m")
    
    # Average velocity
    avg_velocity = mission_range / total_time if total_time > 0 else 0.0
    print(f"  Average mission vel  = {avg_velocity:.2f} m/s")
    
    # Total Energy used
    E_0 = results.segments[0].conditions.propulsion.battery_energy[0, 0]
    E_end = results.segments[-1].conditions.propulsion.battery_energy[-1, 0]
    E_diff = E_0 - E_end
    print(f"  Total energy used    = {E_diff / 1000:.3f} kJ")
    
    # Battery state of charge
    soc_end = results.segments[-1].conditions.propulsion.battery_state_of_charge[-1, 0]
    print(f"  Remainig Battery SOC = {soc_end * 100:.3f} %")


# ----------------------------------------------------------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    # Setup a vehicle
    print("Setting up the vehicle")
    vehicle = setup_vehicle()
    print("✓ Vehicle setup complete")


    #Open VSP generation
    #success = export_to_vsp(vehicle)
    #if success:
    #    print("\n✓ Successfully exported vehicle to OpenVSP format.\n")
 
    # Setup analyses
    print("Setting up the analyses")
    analyses = setup_analyses(vehicle)
    print("finalizing analysis...")
    analyses.finalize() #<- this builds surrogate models!
    print("✓ Analyzed.")


    # override the compute_noise function to be a no-op
    Noise.compute_noise = lambda segment: None
    # Setup a mission
    print("Setting up the mission")
    mission  = setup_mission(vehicle, analyses)

    print(f"Number of lift rotors: {len(vehicle.networks.lift_cruise.lift_rotors)}")

    # Run the mission    
    print("Commenced mission evaluation...Please wait...")
    results = mission.evaluate()
    print("✓ Mission evaluation complete, displaying results:")
    
    # Print Results
    print_segment_results(results)
    print_mission_summary(results)

    # plot the mission
    print("making and saving plots...")
    #main plots only
    save_plots(results)
    # Option 2: Save all available plots
    # save_plots(results, plot_all=True)
    print("✓ done plotting")
    
    return
    
if __name__ == '__main__':
    print("Running eVTOL tutorial")
    main()
    print("Done main")
    plt.show()
    print('show graph done')