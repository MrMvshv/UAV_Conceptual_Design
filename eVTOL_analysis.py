import os
from SUAVE.Core import Data, Units
from SUAVE.Components import Wings, Fuselages
from SUAVE.Input_Output.OpenVSP import write

def create_valid_aircraft():
    """Creates a SUAVE vehicle with all required OpenVSP fields"""
    vehicle = Data()
    vehicle.tag = 'TestAircraft'

    # Required base parameters
    vehicle.reference_area = 15 * Units['meters**2']
    vehicle.envelope = Data()
    vehicle.envelope.ultimate_load = 3.0
    vehicle.envelope.limit_load = 1.5

    # Main Wing (must use Main_Wing class)
    wing = Wings.Main_Wing()
    wing.tag = 'MainWing'
    wing.areas = Data()
    wing.areas.reference = 15 * Units['meters**2']
    wing.spans = Data()
    wing.spans.projected = 10 * Units.meters
    wing.aspect_ratio = 6.67
    wing.sweeps = Data()
    wing.sweeps.quarter_chord = 0.0 * Units.deg
    wing.taper = 0.6
    wing.thickness_to_chord = 0.12
    wing.chords = Data()
    wing.chords.root = wing.areas.reference * 2.0 / (wing.spans.projected * (1 + wing.taper))
    wing.chords.tip = wing.chords.root * wing.taper
    wing.twists = Data()
    wing.twists.root = 0.0 * Units.deg
    wing.twists.tip = 0.0 * Units.deg

     # Critical wing positioning parameters
    wing.origin = [[0.0, 0.0, 0.0]]  # X,Y,Z position
    wing.vertical = False  # Is this a vertical wing?
    wing.symmetric = True  # Is the wing symmetric?

    # Fuselage (with all required OpenVSP parameters)
    fuselage = Fuselages.Fuselage()
    fuselage.tag = 'eVTOL_Fuselage'

    fuselage.lengths = Data()
    fuselage.lengths.total = 1 * Units.meters
    fuselage.lengths.nose = 0.2 * Units.meters
    fuselage.lengths.tail = 0.2 * Units.meters
    # Estimate width based on typical fuselage aspect ratios (Length/Width ~ 5-10)
    # For 1m length, a width of 0.1m to 0.2m seems reasonable. Let's choose 0.15m.
    fuselage.width = 0.15 * Units.meters

    fuselage.heights = Data()
    fuselage.heights.maximum = 0.15 * Units.meters
    fuselage.heights.at_quarter_length = 0.15 * Units.meters
    fuselage.heights.at_wing_root_quarter_chord = 0.15 * Units.meters
    fuselage.heights.at_three_quarters_length = 0.15 * Units.meters
    fuselage.effective_diameter = 0.15 * Units.meters  # Critical for OpenVSP!

    fuselage.areas = Data()
    #NOTE: this is approximate, use advanced analysis tools to calculate
    fuselage.areas.wetted = 0.6 * Units['meters**2']
    #NOTE: this is approximate, use advanced analysis tools to calculate(CIRCULAR)
    fuselage.areas.front_projected = 0.018 * Units['meters**2']

    fuselage.fineness = Data()
    # Nose Fineness = nose_length / width = 0.2 / 0.15
    fuselage.fineness.nose = 0.2 / 0.15
    fuselage.fineness.tail = 0.2 / 0.15

    # Critical addition for vertical position
    fuselage.origin = [[0.0, 0.0, 0.0]]  # X,Y,Z position

    

    #fuselage.percent_z_location = 0.5  # Centerline placement
    fuselage.vertical = False  # Important for OpenVSP interpretation

    # Add components to vehicle
    vehicle.wings = Data()
    vehicle.wings['main_wing'] = wing

    #vehicle.fuselages = Data()
    #vehicle.fuselages['fuselage'] = fuselage
    vehicle.append_component(fuselage)

    # Initialize empty required containers
    vehicle.networks = Data()
    vehicle.nacelles = Data()
    vehicle.payload = Data()

    # Add basic mass properties (recommended)
    vehicle.mass_properties = Data()
    vehicle.mass_properties.max_takeoff = 1000 * Units.kg
    vehicle.mass_properties.takeoff = 1000 * Units.kg

    print("✓ Created valid aircraft with all required OpenVSP fields")
   #print(Data(vehicle))
    print("✓ Aircraft geometry is valid for OpenVSP export")
    return vehicle

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
    

if __name__ == '__main__':
    aircraft = create_valid_aircraft()
    success = export_to_vsp(aircraft)
    if success:
        print("\nNext steps:")
        print("1. Open the exported file in OpenVSP")
        print("2. Verify the geometry and parameters")
        print("3. Proceed with further analysis or modifications")