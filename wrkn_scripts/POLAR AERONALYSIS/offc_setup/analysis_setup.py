# analysis_setup.py

import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Analyses import Aerodynamics
from SUAVE.Analyses.Mission import Mission
from SUAVE.Analyses.Weights import Weights
from SUAVE.Analyses.Energy import Energy
from SUAVE.Analyses.Atmospheric import Atmosphere


from SUAVE.Methods.Aerodynamics.AVL.run_avl import run_avl
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def load_polars(filename="polars_interpolated.csv"):
    df = pd.read_csv(filename)
    alpha = df["alpha_deg"].values
    CL = df["CL"].values
    CD = df["CD"].values

    return {
        "alpha": alpha,
        "CL": CL,
        "CD": CD,
        "alpha_to_CL": interp1d(alpha, CL, fill_value="extrapolate"),
        "alpha_to_CD": interp1d(alpha, CD, fill_value="extrapolate"),
    }


class XFLR5_Aero(Aerodynamics.Aerodynamics):
    """Custom aerodynamic model using interpolated XFLR5 polars."""

    def __init__(self, vehicle, polars):
        super().__init__()
        self.vehicle = vehicle
        self.polars = polars

    def evaluate(self, state, settings=None):
        results = Data()

        alpha = state.conditions.aerodynamics.angle_of_attack
        alpha_deg = float(alpha / Units.deg)

        CL = float(self.polars["alpha_to_CL"](alpha_deg))
        CD = float(self.polars["alpha_to_CD"](alpha_deg))

        # store in SUAVE container
        results.lift_coefficient = CL
        results.drag_coefficient = CD
        results.pitch_moment_coefficient = 0.0

        return results


class AVL_Aero(Aerodynamics.Aerodynamics):
    """Procedural AVL call wrapped inside an Analysis object."""

    def __init__(self, vehicle):
        super().__init__()
        self.vehicle = vehicle

    def evaluate(self, state, settings=None):

        geometry = self.vehicle  # SUAVE 2.5.2 expects full vehicle
        alpha = float(state.conditions.aerodynamics.angle_of_attack / Units.deg)

        # build AVL input dictionary
        avl_data = Data()
        avl_data.alpha = alpha
        avl_data.run_case_name = "case_1"

        # call AVL
        avl_results = run_avl(geometry, [avl_data])

        # extract coefficients
        CL = avl_results["case_1"].conditions.aerodynamics.lift_coefficient
        CD = avl_results["case_1"].conditions.aerodynamics.drag_coefficient
        Cm = avl_results["case_1"].conditions.aerodynamics.pitch_moment_coefficient

        out = Data()
        out.lift_coefficient = float(CL)
        out.drag_coefficient = float(CD)
        out.pitch_moment_coefficient = float(Cm)

        return out


def setup_analyses(vehicle, use_avl=False):

    analyses = SUAVE.Analyses.Analyses()

    # atmosphere
    atm = Atmosphere.Atmosphere()
    atm.model = Atmosphere.US_Standard_1976()
    analyses.append(atm)

    # weights
    weights = Weights.Weights()
    weights.vehicle = vehicle
    analyses.append(weights)

    # energy
    energy = Energy.Energy()
    energy.vehicle = vehicle
    analyses.append(energy)

    # aerodynamics
    polars = load_polars()
    if use_avl:
        aero = AVL_Aero(vehicle)
    else:
        aero = XFLR5_Aero(vehicle, polars)

    analyses.append(aero)

    # mission will be set later in main
    mission = Mission.Mission()
    analyses.append(mission)

    return analyses
