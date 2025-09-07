# Custom plot functions for battery energy and thrust (if the built-in ones don't exist)
def plot_battery_energy_custom(results):
    """Custom battery energy plot if the built-in one fails"""
    time = []
    energy = []
    
    for segment in results.segments:
        seg_time = segment.conditions.frames.inertial.time[:, 0]
        seg_energy = segment.conditions.propulsion.battery_energy[:, 0]
        time.extend(seg_time)
        energy.extend(seg_energy)
    
    plt.plot(time, energy)
    plt.xlabel('Time (s)')
    plt.ylabel('Battery Energy (J)')
    plt.grid(True)
    plt.title('Battery Energy Consumption')


def plot_thrust_summary(results):
    """Custom thrust summary plot"""
    time = []
    lift_thrust = []
    prop_thrust = []
    
    for segment in results.segments:
        seg_time = segment.conditions.frames.inertial.time[:, 0]
        
        # Lift rotor thrust (sum of all lift rotors)
        if hasattr(segment.conditions.propulsion, 'lift_rotor_thrust'):
            lift_arr = segment.conditions.propulsion.lift_rotor_thrust
            lift_total = lift_arr.sum(axis=1) if lift_arr.ndim > 1 else lift_arr
            lift_thrust.extend(lift_total)
        
        # Propeller thrust
        if hasattr(segment.conditions.propulsion, 'propeller_thrust'):
            prop_thrust_seg = segment.conditions.propulsion.propeller_thrust[:, 0]
            prop_thrust.extend(prop_thrust_seg)
        
        time.extend(seg_time)
    
    plt.figure(figsize=(12, 8))
    if lift_thrust:
        plt.plot(time[:len(lift_thrust)], lift_thrust, label='Total Lift Thrust', linewidth=2)
    if prop_thrust:
        plt.plot(time[:len(prop_thrust)], prop_thrust, label='Propeller Thrust', linewidth=2)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Thrust (N)')
    plt.legend()
    plt.grid(True)
    plt.title('Thrust Summary')


# Fallback function if plot_battery_pack_conditions doesn't exist
def plot_battery_pack_conditions(results):
    """Fallback battery plot if the SUAVE built-in doesn't work"""
    try:
        # Try to use SUAVE's built-in function first
        from SUAVE.Plots.Performance.Mission_Plots import plot_battery_pack_conditions as suave_plot
        suave_plot(results)
    except:
        # Fallback to custom implementation
        plot_battery_energy_custom(results)