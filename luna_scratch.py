# hover_quadcopter_minimal.py  (SUAVE 2.5.x)

from SUAVE.Core import Units, Data
import SUAVE

# --- Vehicle ---
vehicle = SUAVE.Vehicle()
vehicle.tag = '1kg_quad'
vehicle.mass_properties.takeoff = 1.0 * Units.kg          # total mass
vehicle.mass_properties.operating_empty = 0.8 * Units.kg  # rough split
vehicle.mass_properties.payload = 0.2 * Units.kg

# --- Energy network: Battery -> ESC -> Motor -> Prop (x4) ---
from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller
from SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion import Lithium_Ion
from SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller import Electronic_Speed_Controller
from SUAVE.Components.Energy.Converters.Motor_Lo_Fid import Motor_Lo_Fid
from SUAVE.Components.Energy.Converters.Propeller_Lo_Fid import Propeller_Lo_Fid
from SUAVE.Components.Energy.Peripherals.Avionics import Avionics

net = Battery_Propeller()
net.tag = 'e-quad'
net.identical_propellers = True
net.number_of_propeller_engines = 4

# Battery (simple constant-mass Li-Ion)
battery = Lithium_Ion()
battery.pack_config.series = 3      # 3S pack (example)
battery.pack_config.parallel = 1
battery.mass_properties.mass = 0.20 * Units.kg
battery.specific_energy = 180.0 * Units.Wh/Units.kg  # coarse; tune for your cells
net.battery = battery

# ESC (single efficiency number is common at this fidelity)
esc = Electronic_Speed_Controller()
esc.efficiency = 0.95
net.esc = esc

# Motor (low-fid: give it a rating & efficiency)
motor = Motor_Lo_Fid()
motor.efficiency = 0.88
motor.nominal_voltage = 11.1       # ~3S
motor.maximum_power = 120.0        # W (total per motor), safe above hover ~15–30 W
net.propeller_motors = [motor]

# Prop / Rotor (low-fid; minimal geometry for hover sizing)
prop = Propeller_Lo_Fid()
prop.number_of_blades = 2
prop.tip_radius = 0.12 * Units.m    # ~9.5" prop radius
prop.hub_radius = 0.012 * Units.m
prop.design_thrust = (vehicle.mass_properties.takeoff * 9.81 * Units.newton) / 4.0  # per rotor
prop.design_power = 25.0 * Units.W  # a reasonable hover/buffer per rotor
net.propellers = [prop]

# Optional peripheral load
avionics = Avionics()
avionics.power = 3.0 * Units.W
net.avionics = avionics

# Install network on the vehicle
vehicle.append_component(net)

# --- Analyses (minimum set for hover) ---
from SUAVE.Analyses.Atmospheric import Atmospheric
from SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976 import US_Standard_1976
from SUAVE.Analyses.Energy import Energy
from SUAVE.Analyses.Mission import Mission

analyses = Data()
atm = Atmospheric()
atm.atmosphere = US_Standard_1976()
analyses.atmosphere = atm
analyses.energy = Energy()          # lets the network evaluate

# --- Mission with one Hover segment ---
from SUAVE.Analyses.Mission.Segments.Hover.Hover import Hover

mission = Mission()
mission.tag = 'hover_test'

hover = Hover()                     # stationary hover segment
hover.tag = 'hover_10s'
hover.altitude = 0.0   * Units.m    # sea level
hover.time     = 10.0  * Units.s    # duration
hover.analyses = analyses

# Provide solver unknowns (either throttle OR prop power coefficient depending on setup)
# Start with a modest throttle guess:
hover.state.unknowns.throttle = 0.6

# IMPORTANT: let the network append its unknowns & residuals (battery voltage, torque match, etc.)
# (initial guesses are optional but can help)
net.add_unknowns_and_residuals_to_segment(
    hover,
    initial_battery_state_of_charge   = 0.95,
    initial_battery_cell_temperature  = 298.0,
    initial_voltage                   = 11.1,
    initial_battery_cell_current      = 0.0
)

mission.append_segment(hover)

# --- Evaluate ---
results = mission.evaluate(vehicle)

# --- Report (last time-step) ---
seg = results.segments['hover_10s']
i = -1
T_total   = seg.conditions.frames.inertial.total_force_vector[i,2] * -1.0  # +up
P_elect   = seg.conditions.propulsion.power[i,0]
SOC       = seg.conditions.propulsion.battery_state_of_charge[i,0]
omega     = seg.conditions.propulsion.propeller_angular_velocity[i,0] if 'propeller_angular_velocity' in seg.conditions.propulsion else float('nan')

print(f"Hover results @ t = {seg.conditions.frames.inertial.time[i,0]:.2f} s")
print(f" Total thrust (up): {T_total:.2f} N   (Weight = {vehicle.mass_properties.takeoff*Units.g0:.2f} N)")
print(f" Electrical power:  {P_elect:.1f} W")
print(f" Battery SOC:       {SOC*100:.1f} %")
print(f" Rotor speed (if available): {omega:.1f} rad/s")
