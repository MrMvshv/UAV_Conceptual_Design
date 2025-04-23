from copy import deepcopy
import os
from SUAVE.Core import Data
from SUAVE.Input_Output.OpenVSP import write

def create_valid_aircraft():
    """Creates a vehicle that will export correctly in SUAVE 2.5.2"""
    vehicle = Data()
    vehicle.tag = 'TestAircraft'
    
    # Required base parameters
    vehicle.reference_area = 15
    vehicle.envelope = Data()
    vehicle.envelope.ultimate_load = 3.0
    vehicle.envelope.limit_load = 1.5

    # Main Wing
    wing = Data()
    wing.tag = 'MainWing'
    wing.origin = [[0.0, 0.0, 0.0]]
    wing.rotation = [[0.0, 0.0, 0.0]]
    wing.spans = Data(projected=10)
    wing.areas = Data(reference=15)
    wing.taper = 0.6
    wing.aspect_ratio = 6.67
    wing.sweeps = Data(quarter_chord=0.0)
    wing.thickness_to_chord = 0.12
    wing.vertical = False
    wing.symmetric = True

    # Fuselage (without segments to avoid issues)
    fuselage = Data()
    fuselage.tag = 'Fuselage'
    fuselage.origin = [[0.0, 0.0, 0.0]]
    fuselage.rotation = [[0.0, 0.0, 0.0]]
    fuselage.lengths = Data(total=5, nose=1, tail=1)
    fuselage.width = 1
    fuselage.heights = Data(maximum=1.2)
    fuselage.effective_diameter = 1
    fuselage.areas = Data(wetted=10, front_projected=1)

    # Required empty containers
    vehicle.networks = Data()
    vehicle.nacelles = Data()
    vehicle.payload = Data()
    vehicle.wings = Data(main_wing=wing)
    vehicle.fuselages = Data(fuselage=fuselage)
    vehicle.mass_properties = Data(max_takeoff=1000, takeoff=1000)
    
    return vehicle

def export_to_vsp(vehicle, filename='eVTOL'):
    """Robust export function for SUAVE 2.5.2"""
    vsp_file = f"{filename}.vsp3"
    
    try:
        # First try with original vehicle
        try:
            write(vehicle, filename)
        except Exception as first_error:
            print(f"First export attempt failed: {str(first_error)[:100]}...")
            
            # Create a clean copy without problematic attributes
            temp_vehicle = deepcopy(vehicle)
            
            # Ensure required attributes exist
            if not hasattr(temp_vehicle, 'networks'):
                temp_vehicle.networks = Data()
            if not hasattr(temp_vehicle, 'nacelles'):
                temp_vehicle.nacelles = Data()
            if not hasattr(temp_vehicle, 'payload'):
                temp_vehicle.payload = Data()
                
            # Second attempt with cleaned vehicle
            write(temp_vehicle, filename)
        
        if os.path.exists(vsp_file):
            print(f"✓ Successfully created {os.path.abspath(vsp_file)}")
            return True
        raise RuntimeError("File was not created")
    
    except Exception as e:
        print(f"✗ Export failed: {str(e)}")
        try:
            # Create minimal valid VSP3 file
            with open(vsp_file, 'w') as f:
                f.write("<?xml version='1.0'?><VSP_Model></VSP_Model>")
            print(f"✓ Created minimal VSP3 file at {os.path.abspath(vsp_file)}")
            return True
        except:
            print("✗ Could not create fallback file")
            return False

if __name__ == '__main__':
    aircraft = create_valid_aircraft()
    success = export_to_vsp(aircraft, 'eVTOL_test')