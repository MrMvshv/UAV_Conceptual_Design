#test fixed wing
# created  sep 2025----------------------------------------------------------------------

import sys, os, importlib, inspect, pkgutil, math
from types import SimpleNamespace


CANDIDATE_SUAVE_PATHS = [
    r"C:\Envs\SUAVE",
    r"C:\Envs\SUAVE\SUAVE",
]
for p in CANDIDATE_SUAVE_PATHS:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# --- SUAVE Imports ---
import SUAVE
from SUAVE.Core import Units
from SUAVE.Components.Energy.Networks import Network
from SUAVE.Components.Energy.Converters import Propeller, Motor

print(f"[INFO] SUAVE imported from: {os.path.dirname(inspect.getfile(SUAVE))}")

# ---------- Auto-discover a usable battery ----------
def find_battery_class():
    base_pkg_name = "SUAVE.Components.Energy.Storages.Batteries"
    Batteries = importlib.import_module(base_pkg_name)
    candidates = []

    def consider_module(mod, modname):
        for name, obj in vars(mod).items():
            if not isinstance(obj, type):
                continue
            try:
                sig = inspect.signature(obj)
                req = [p for p in sig.parameters.values()
                       if p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
                if len(req) > 1:
                    continue
            except Exception:
                continue
            lname = name.lower()
            score = 0
            if "lith" in lname: score += 10
            if "ion" in lname:  score += 5
            if "battery" in lname: score += 3
            if score > 0:
                candidates.append((score, obj, f"{modname}.{name}"))

    consider_module(Batteries, base_pkg_name)
    if hasattr(Batteries, "__path__"):
        for _, modname, _ in pkgutil.iter_modules(Batteries.__path__, Batteries.__name__ + "."):
            try:
                mod = importlib.import_module(modname)
                consider_module(mod, modname)
            except Exception:
                continue

    if not candidates:
        raise ImportError("No battery class found in SUAVE!")
    candidates.sort(key=lambda t: (-t[0], len(t[2])))
    cls = candidates[0][1]
    print(f"[INFO] Battery class selected: {candidates[0][2]}")
    return cls

BatteryClass = find_battery_class()

# ----------------- Constants -----------------
g = 9.80665
rho = 1.225
CL_MAX = 1.2
ETA_PROP  = 0.70
ETA_MOTOR = 0.85
ETA_ELEC  = 0.95
ETA_TOTAL = ETA_PROP * ETA_MOTOR * ETA_ELEC

# ----------------- Vehicle Setup -----------------
def setup_vehicle():
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'FixedWing_UAV'

    # Mass properties
    vehicle.mass_properties.takeoff = 1.5 * Units.kg
    vehicle.mass_properties.operating_empty = 1.2 * Units.kg
    vehicle.mass_properties.payload = 0.3 * Units.kg

    # Reference geometry (simple rectangular wing assumption)
    span = 1.2 * Units.m
    area = 0.25 * Units.m**2
    chord = area / span
    vehicle.reference_area = area

    # Network (single pusher)
    net = Network()
    net.tag = 'FW_Network'

    # Battery
    battery = BatteryClass()
    battery.tag = 'Main_Battery'
    if hasattr(battery, "mass_properties"):
        battery.mass_properties.mass = 0.30 * Units.kg
    if hasattr(battery, "specific_energy"):
        battery.specific_energy = 200.0 * Units.Wh / Units.kg
    battery.nominal_voltage = 14.8
    battery.max_discharge_frac = 0.8
    net.battery = battery

    # Motor + Prop
    prop = Propeller()
    prop.tag = 'pusher_prop'
    prop.number_blades = 2
    prop.radius = 0.15  # ~12” diameter
    if hasattr(prop, "design_Cl"): prop.design_Cl = 0.5

    motor = Motor()
    motor.tag = 'pusher_motor'
    if hasattr(motor, "mass_properties"):
        motor.mass_properties.mass = 0.08 * Units.kg
    if hasattr(motor, "efficiency"):
        motor.efficiency = ETA_MOTOR
    motor.max_power = 250.0  # W peak

    prop.motor = motor
    net.propellers = [prop]
    vehicle.append_component(net)

    vehicle.fw = SimpleNamespace(
        span=span,
        area=area,
        chord=chord
    )
    return vehicle


def compute_fixedwing_basics(vehicle, V_cruise=20.0, target_endurance_min=30.0):
    mtow = float(vehicle.mass_properties.takeoff)
    W = mtow * g
    S = float(vehicle.fw.area)
    b = float(vehicle.fw.span)

    # Wing loading
    wing_loading = W / S

    # Stall speed
    V_stall = math.sqrt(2 * W / (rho * S * CL_MAX))

    # Cruise power (induced + parasite, first-order)
    CD0 = 0.04
    AR = b**2 / S
    k = 1.0 / (math.pi * 0.8 * AR)
    q = 0.5 * rho * V_cruise**2
    CL = W / (q * S)
    CD = CD0 + k * CL**2
    D = q * S * CD
    P_req = D * V_cruise
    P_elec = P_req / ETA_TOTAL

    # Endurance
    net = next((c for c in getattr(vehicle, 'components', []) if isinstance(c, Network)), None)
    battery = getattr(net, 'battery', None)
    V = getattr(battery, 'nominal_voltage', 14.8)
    usable = getattr(battery, 'max_discharge_frac', 0.8)
    t_hr = target_endurance_min / 60.0
    energy_Wh = (P_elec * t_hr) / usable
    capacity_Ah = energy_Wh / V

    return {
        'MTOW_kg': mtow,
        'weight_N': W,
        'span_m': b,
        'area_m2': S,
        'wing_loading_N_m2': wing_loading,
        'V_stall_mps': V_stall,
        'V_cruise_mps': V_cruise,
        'CL_cruise': CL,
        'CD_cruise': CD,
        'drag_cruise_N': D,
        'P_cruise_req_W': P_req,
        'P_elec_cruise_W': P_elec,
        'battery_voltage_V': V,
        'usable_capacity_frac': usable,
        'target_endurance_min': target_endurance_min,
        'energy_required_Wh': energy_Wh,
        'capacity_required_Ah': capacity_Ah,
    }

# ----------------- Reporting -----------------
def print_summary(vehicle, basics):
    print("\n=============== FIXED-WING UAV SUMMARY ===============")
    print(f"Vehicle tag: {vehicle.tag}")
    print(f"MTOW: {basics['MTOW_kg']:.2f} kg (Weight: {basics['weight_N']:.1f} N)")
    print(f"Wing span: {basics['span_m']:.2f} m | Wing area: {basics['area_m2']:.3f} m²")
    print(f"Wing loading: {basics['wing_loading_N_m2']:.1f} N/m²")
    print(f"Stall speed: {basics['V_stall_mps']:.1f} m/s | Cruise speed: {basics['V_cruise_mps']:.1f} m/s")
    print(f"CL at cruise: {basics['CL_cruise']:.2f} | CD: {basics['CD_cruise']:.3f}")
    print(f"Drag at cruise: {basics['drag_cruise_N']:.2f} N")
    print(f"Power required (aero): {basics['P_cruise_req_W']:.1f} W | Electrical: {basics['P_elec_cruise_W']:.1f} W")

    print("\n-- Battery Sizing (target endurance) --")
    print(f"Voltage: {basics['battery_voltage_V']:.1f} V | Usable fraction: {basics['usable_capacity_frac']:.2f}")
    print(f"Target endurance: {basics['target_endurance_min']:.1f} min")
    print(f"Energy required: {basics['energy_required_Wh']:.1f} Wh | Capacity: {basics['capacity_required_Ah']:.2f} Ah")
    print("======================================================\n")

# ----------------- Main -----------------
if __name__ == "__main__":
    vehicle = setup_vehicle()
    basics = compute_fixedwing_basics(vehicle, V_cruise=20.0, target_endurance_min=30.0)
    print_summary(vehicle, basics)