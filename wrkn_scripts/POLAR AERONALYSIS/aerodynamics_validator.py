import math

# =============================================================
# HELPER FUNCTIONS
# =============================================================

def status(value, low, high, unit=""):
    if low <= value <= high:
        return f"OK ({value:.3f}{unit})"
    elif value < low:
        return f"LOW ({value:.3f}{unit})  [Below typical range]"
    else:
        return f"HIGH ({value:.3f}{unit}) [Above typical range]"

def flag(value, low, high):
    if low <= value <= high:
        return "NORMAL"
    elif value < low:
        return "LOW"
    else:
        return "HIGH"


# =============================================================
# === MAIN VALIDATOR FUNCTION =================================
# =============================================================

def validate_fixed_wing(
        mass_kg=3.1,
        wing_area=0.40,
        span=1.8,
        chord_root=0.32,
        chord_tip=0.22,
        sweep_deg=5.0,
        tail_arm=0.45,
        htail_area=0.08,
        vtail_area=0.05,
        prop_eff=0.85,
        cruise_speed=22.0,
        stall_CLmax=1.2,
):
    print("\n====================================================")
    print(" FIXED-WING UAV AERODYNAMIC VALIDATION REPORT")
    print("====================================================\n")

    g = 9.81
    weight_N = mass_kg * g
    AR = span**2 / wing_area
    taper = chord_tip / chord_root

    print("=== INPUT GEOMETRY SUMMARY ===")
    print(f"Mass:                    {mass_kg:.2f} kg")
    print(f"Weight:                  {weight_N:.2f} N")
    print(f"Wing area S:             {wing_area:.3f} m²")
    print(f"Span b:                  {span:.3f} m")
    print(f"Aspect ratio AR:         {AR:.2f}")
    print(f"Taper ratio λ:           {taper:.2f}")
    print(f"Sweep (25%):             {sweep_deg:.1f}°")
    print(f"Horizontal tail area Sh: {htail_area:.3f} m²")
    print(f"Vertical tail area Sv:   {vtail_area:.3f} m²\n")

    # =============================================================
    # WING LOADING VALIDATION
    # =============================================================
    WL = weight_N / wing_area

    print("=== WING LOADING VALIDATION ===")
    print(f"Wing loading W/S = {WL:.1f} N/m²")

    print("Typical values:")
    print(" - Small UAV:                30–80 N/m²")
    print(" - Fast/high-performance:    80–120 N/m²")
    print(" - Heavy lift / endurance:   20–40 N/m²")

    WL_flag = flag(WL, 30, 120)
    print(f"→ Wing loading status: {WL_flag}\n")

    # =============================================================
    # CL REQUIRED IN CRUISE
    # =============================================================
    rho = 1.225
    q = 0.5 * rho * cruise_speed**2
    CL_cruise = weight_N / (q * wing_area)

    print("=== CRUISE LIFT COEFFICIENT ===")
    print(f"Cruise CL_required = {CL_cruise:.3f}")
    print("Typical small UAV cruise CL: 0.2 – 0.5")

    CL_flag = flag(CL_cruise, 0.15, 0.55)
    print(f"→ Cruise CL status: {CL_flag}\n")

    # =============================================================
    # STALL SPEED CHECK
    # =============================================================
    V_stall = math.sqrt(2 * weight_N / (rho * wing_area * stall_CLmax))

    print("=== STALL SPEED CHECK ===")
    print(f"Stall CLmax assumed = {stall_CLmax:.2f}")
    print(f"Estimated stall speed Vstall = {V_stall:.2f} m/s")
    print(f"Cruise speed = {cruise_speed} m/s")

    if cruise_speed < 1.5 * V_stall:
        print("⚠ WARNING: Cruise speed is <1.5× stall → too close.")
    else:
        print("✓ Cruise >1.5× stall → acceptable margin.")

    print()

    # =============================================================
    # TAIL SIZING & STABILITY INDICES
    # =============================================================
    MAC = (2/3)*chord_root*((1+taper+taper**2)/(1+taper))

    # Horizontal tail volume coefficient
    Vh = htail_area * tail_arm / (wing_area * MAC)

    # Vertical tail volume coefficient
    Vv = vtail_area * tail_arm / (wing_area * span)

    print("=== TAIL VOLUME COEFFICIENTS ===")
    print(f"MAC (mean aero chord): {MAC:.3f} m")
    print(f"Tail arm lt:           {tail_arm:.3f} m")

    print(f"Horizontal tail volume Vh = {Vh:.3f}")
    print("Typical Vh: 0.45 – 0.8")
    print(f"→ Status: {status(Vh, 0.45, 0.80)}\n")

    print(f"Vertical tail volume Vv = {Vv:.3f}")
    print("Typical Vv: 0.02 – 0.06")
    print(f"→ Status: {status(Vv, 0.02, 0.06)}\n")

    # =============================================================
    # REYNOLDS NUMBER CHECK
    # =============================================================
    Re = rho * cruise_speed * MAC / 1.8e-5

    print("=== REYNOLDS NUMBER ANALYSIS ===")
    print(f"Re_MAC at cruise = {Re/1e5:.2f} × 10⁵")

    if Re < 1e5:
        print("⚠ Very low Reynolds → high drag, poor lift, expect very low L/D")
    elif Re < 2e5:
        print("⚠ Low Reynolds → use low-Re airfoils (Selig, Drela, AG series)")
    else:
        print("✓ Reynolds is typical for small UAVs")

    print()

    # =============================================================
    # EXPECTED DRAG & L/D RANGE
    # =============================================================
    e = 0.75  # typical efficiency for small UAVs
    CDi = CL_cruise**2 / (math.pi * AR * e)

    print("=== DRAG & L/D ESTIMATION ===")
    print(f"Induced drag coefficient CDi = {CDi:.4f}")
    print("Typical small UAV induced drag at cruise: 0.003–0.01")

    CD0_guess = 0.015 + 0.002 * (span)  # heuristic
    CD_total = CDi + CD0_guess
    LD = CL_cruise / CD_total

    print(f"Estimated CD0 ≈ {CD0_guess:.4f}")
    print(f"Estimated total CD ≈ {CD_total:.4f}")
    print(f"Estimated L/D ≈ {LD:.1f}")

    if LD < 10:
        print("⚠ L/D likely low → geometry or aero inefficiencies present")
    elif LD < 14:
        print("✓ Typical small UAV L/D (10–14)")
    else:
        print("✓ High-performance L/D (14+)")

    print("\n====================================================")
    print(" AERODYNAMIC VALIDATION COMPLETE")
    print("====================================================\n")


# =============================================================
# Example run for your aircraft
# =============================================================

if __name__ == "__main__":
    validate_fixed_wing(
        mass_kg=3.1,
        wing_area=0.40,
        span=1.8,
        chord_root=0.32,
        chord_tip=0.22,
        sweep_deg=5,
        htail_area=0.08,
        vtail_area=0.05,
        tail_arm=0.45,
        cruise_speed=22.0,
        stall_CLmax=1.2,
    )
