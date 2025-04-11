import SUAVE
from SUAVE.Core import Units
from SUAVE.Components.Wings import Main_Wing
from SUAVE.Analyses.Aerodynamics import Fidelity_Zero

def create_vehicle():
    # Initialize vehicle
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Test_Vehicle'
    
    # Set basic vehicle properties
    vehicle.mass_properties.takeoff = 15.0 * Units.kg
    vehicle.mass_properties.operating_empty = 10.0 * Units.kg
    vehicle.mass_properties.max_takeoff = 15.0 * Units.kg
    
    # Create and configure the main wing
    wing = Main_Wing()
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
    
    # Define chord lengths
    wing.chords.root = 0.5 * Units.meters
    wing.chords.tip = wing.chords.root * wing.taper
    
    # Define origin and aerodynamic center as lists
    wing.origin = [0.0, 0.0, 0.0]  # x, y, z coordinates
    wing.aerodynamic_center = [0.25 * wing.chords.root, 0.0, 0.0]  # x, y, z coordinates
    
    # Add wing to vehicle
    vehicle.append_component(wing)
    
    # Set vehicle reference area
    vehicle.reference_area = wing.areas.reference
    
    return vehicle

def run_analysis(vehicle):
    # Create a simple aerodynamic analysis
    aerodynamics = Fidelity_Zero()
    aerodynamics.geometry = vehicle
    
    # Finalize the analysis
    aerodynamics.finalize()
    
    # Print the reference area to verify
    print("Vehicle reference area:", vehicle.reference_area)
    print("Wing reference area:", vehicle.wings.main_wing.areas.reference)

if __name__ == "__main__":
    vehicle = create_vehicle()
    run_analysis(vehicle)