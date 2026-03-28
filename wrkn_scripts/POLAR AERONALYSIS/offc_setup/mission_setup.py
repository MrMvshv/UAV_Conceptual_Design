# mission_setup.py

from SUAVE.Analyses.Mission.Segments import Climb, Cruise, Descent
from SUAVE.Core import Units


def build_mission(analyses, vehicle):

    mission = SUAVE.Analyses.Mission.Mission()
    segs = mission.segments

    # CLIMB
    seg = Climb.Constant_Speed_Constant_Rate()
    seg.analyses.extend(analyses)
    seg.climb_rate = 2 * Units.m / Units.s
    seg.air_speed = 15 * Units.m / Units.s
    seg.altitude_start = 0
    seg.altitude_end = 100 * Units.m
    segs.append(seg)

    # CRUISE
    seg = Cruise.Constant_Speed_Constant_Altitude()
    seg.analyses.extend(analyses)
    seg.air_speed = 22 * Units.m / Units.s
    seg.altitude = 100 * Units.m
    seg.distance = 600 * Units.m
    segs.append(seg)

    # DESCENT
    seg = Descent.Constant_Speed_Constant_Rate()
    seg.analyses.extend(analyses)
    seg.descent_rate = 1 * Units.m / Units.s
    seg.air_speed = 15 * Units.m / Units.s
    seg.altitude_start = 100 * Units.m
    seg.altitude_end = 0
    segs.append(seg)

    return mission
