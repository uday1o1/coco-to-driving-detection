"""Solar elevation from a UTC timestamp and a location.

Used to label each nuImages frame as day, twilight or night, since the dataset
has no lighting attribute but does record the capture time and the city.
Formulas follow the NOAA solar calculator (Meeus, Astronomical Algorithms);
accuracy is well under a degree, far finer than the bins used here.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

# Approximate centres of the nuImages capture areas (degrees).
CITY_LATLON = {
    "boston-seaport": (42.351, -71.045),
    "singapore-onenorth": (1.299, 103.787),
    "singapore-queenstown": (1.294, 103.785),
    "singapore-hollandvillage": (1.311, 103.796),
}


def solar_elevation(ts_utc: float, lat: float, lon: float) -> float:
    """Elevation of the sun's centre above the horizon in degrees."""
    t = datetime.fromtimestamp(ts_utc, tz=timezone.utc)
    jd = t.timestamp() / 86400.0 + 2440587.5
    jc = (jd - 2451545.0) / 36525.0
    l0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    mr = math.radians(m)
    c = (math.sin(mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
         + math.sin(2 * mr) * (0.019993 - 0.000101 * jc)
         + math.sin(3 * mr) * 0.000289)
    app = l0 + c - 0.00569 - 0.00478 * math.sin(math.radians(125.04 - 1934.136 * jc))
    eps0 = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    eps = eps0 + 0.00256 * math.cos(math.radians(125.04 - 1934.136 * jc))
    decl = math.degrees(math.asin(math.sin(math.radians(eps)) * math.sin(math.radians(app))))
    y = math.tan(math.radians(eps / 2)) ** 2
    l0r = math.radians(l0)
    eqt = 4 * math.degrees(
        y * math.sin(2 * l0r) - 2 * e * math.sin(mr) + 4 * e * y * math.sin(mr) * math.cos(2 * l0r)
        - 0.5 * y * y * math.sin(4 * l0r) - 1.25 * e * e * math.sin(2 * mr))
    minutes = t.hour * 60 + t.minute + t.second / 60
    tst = (minutes + eqt + 4 * lon) % 1440
    ha = tst / 4 - 180 if tst / 4 >= 0 else tst / 4 + 180
    zen = math.degrees(math.acos(
        math.sin(math.radians(lat)) * math.sin(math.radians(decl))
        + math.cos(math.radians(lat)) * math.cos(math.radians(decl)) * math.cos(math.radians(ha))))
    return 90.0 - zen


def lighting(elev: float) -> str:
    """Civil-twilight bins: day above 0 deg, twilight to -6 deg, night below."""
    if elev > 0:
        return "day"
    return "twilight" if elev > -6 else "night"


if __name__ == "__main__":
    # Boston, 2018-06-21 16:45 UTC (12:45 EDT, close to solar noon): sun near 71 deg.
    noon = datetime(2018, 6, 21, 16, 45, tzinfo=timezone.utc).timestamp()
    assert 68 < solar_elevation(noon, *CITY_LATLON["boston-seaport"]) < 73
    # Singapore, 2018-08-01 14:00 UTC (22:00 local): night.
    late = datetime(2018, 8, 1, 14, 0, tzinfo=timezone.utc).timestamp()
    assert lighting(solar_elevation(late, *CITY_LATLON["singapore-onenorth"])) == "night"
    print("solar checks pass")
