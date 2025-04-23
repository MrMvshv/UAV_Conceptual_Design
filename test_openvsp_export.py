import os
from SUAVE.Core import Data, Units
from SUAVE.Components import Wings, Fuselages
from SUAVE.Input_Output.OpenVSP import write

def create_valid_aircraft():
    """Creates a SUAVE vehicle with all required OpenVSP fields (SUAVE 2.5.2 compatible - Attempt 2)"""
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

    # Manually create the planform data
    wing.planform = Data()
    wing.planform.points = [
        [0.0, 0.0, 0.0],  # Leading edge root
        [wing.chords.root, 0.0, 0.0],  # Trailing edge root
        [wing.chords.tip + wing.spans.projected * 0.0 * Units.deg, wing.spans.projected / 2.0, 0.0], # Leading edge tip (assuming no sweep at LE)
        [wing.spans.projected * 0.0 * Units.deg, wing.spans.projected / 2.0, 0.0], # Placeholder - adjust based on sweep
        [wing.chords.tip + wing.spans.projected * 0.0 * Units.deg - (wing.chords.tip - wing.chords.root)*0.25, wing.spans.projected / 2.0, 0.0], # Quarter chord tip (placeholder)
        [(wing.chords.root - (wing.chords.root - wing.chords.tip)*0.25), 0.0, 0.0], # Quarter chord root
    ]
    wing.planform.normals = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]
    wing.planform.chord = [wing.chords.root, wing.chords.tip]
    wing.planform.span = [0.0, wing.spans.projected / 2.0] # Assuming symmetry

    # Fuselage (required for OpenVSP export)
    fuselage = Fuselages.Fuselage()
    fuselage.tag = 'Fuselage'
    fuselage.lengths = Data()
    fuselage.lengths.total = 5 * Units.meters
    fuselage.width = 1 * Units.meters
    fuselage.heights = Data()
    fuselage.heights.maximum = 1.2 * Units.meters
    fuselage.heights.at_quarter_length          = 1 * Units.meters
    fuselage.heights.at_wing_root_quarter_chord = 1 * Units.meters
    fuselage.heights.at_three_quarters_length   =  1 * Units.meters

    # Proper component containers
    vehicle.wings = Data()
    vehicle.wings['main_wing'] = wing
    vehicle.fuselages = Data()
    vehicle.fuselages['fuselage'] = fuselage

    # Initialize empty required containers
    vehicle.networks = Data()
    vehicle.nacelles = Data()
    vehicle.payload = Data()

    return vehicle

def export_to_vsp(vehicle, filename='test_export'):
    """Basic OpenVSP export without vsp_vehicle dependency"""
    try:
        # Simple write operation
        write(vehicle, filename)

        # Verify export
        if os.path.exists(filename):
            print(f"✓ Successfully created {os.path.abspath(filename)}")
            print("Open this file in OpenVSP GUI to view geometry")
            return True
        else:
            raise RuntimeError("File was not created")

    except Exception as e:
        print(f"✗ Export failed: {str(e)}")
        return False

if __name__ == '__main__':
    # Create and export
    aircraft = create_valid_aircraft()
    success = export_to_vsp(aircraft)

    if success:
        print("\nNext steps:")
        print("1. Open test_export.vsp3 in OpenVSP")
        print("2. Run Tools > Degenerate Geometry Export")
        print("3. Check component placement")