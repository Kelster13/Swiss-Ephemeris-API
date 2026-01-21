from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from zoneinfo import ZoneInfo
from timezonefinder import TimezoneFinder
from geopy.geocoders import Nominatim
import swisseph as swe
import math
import requests
import os
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EPHE_PATH = os.path.join(BASE_DIR, "ephe")

swe.set_ephe_path(EPHE_PATH)

print("EPHE_PATH =", EPHE_PATH)
print("EPHE exists?", os.path.exists(EPHE_PATH))
print("seas exists?", os.path.exists(os.path.join(EPHE_PATH, "seas_18.se1")))

app = FastAPI(title="Birth Chart API (Swiss Ephemeris)", version="1.0.0")

# --- Health check ---
@app.get("/")
def health():
    return {"status": "ok"}

# --- LOCKED SETTINGS (Astro-Seek parity choices) ---
swe.set_ephe_path("./ephe")   # folder containing .se1 files
HOUSE_SYSTEM = b'P'           # Placidus
FLAGS = swe.FLG_SWIEPH | swe.FLG_SPEED  # Swiss ephemeris + speed
NODE_BODY = swe.MEAN_NODE     # Mean Node ("Node (M)")

# --- Tools ---
tf = TimezoneFinder()
geolocator = Nominatim(user_agent="birthchart_app_v1", timeout=10)

SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
         "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]

PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mercury": swe.MERCURY,
    "Venus": swe.VENUS,
    "Mars": swe.MARS,
    "Jupiter": swe.JUPITER,
    "Saturn": swe.SATURN,
    "Uranus": swe.URANUS,
    "Neptune": swe.NEPTUNE,
    "Pluto": swe.PLUTO,
    "Node(M)": NODE_BODY,
    
}

class ChartRequest(BaseModel):
    year: int = Field(..., ge=1, le=3000)
    month: int = Field(..., ge=1, le=12)
    day: int = Field(..., ge=1, le=31)
    hour: int = Field(..., ge=0, le=23)
    minute: int = Field(..., ge=0, le=59)
    location: str = Field(..., min_length=2, description="e.g. 'Huntington, WV, USA'")

    # Optional: if user wants to override geocoder ambiguity
    country: Optional[str] = Field(None, description="Optional country hint (e.g. 'US')")


def normalize_deg(deg: float) -> float:
    return deg % 360.0


def zodiac_sign_and_pos(ecl_lon_deg: float):
    lon = normalize_deg(ecl_lon_deg)
    sign_index = int(lon // 30)
    sign = SIGNS[sign_index]
    pos_in_sign = lon - sign_index * 30.0  # 0..30
    return sign, pos_in_sign


def round_to_arcminute(pos_in_sign: float):
    # Returns (deg, min) rounded to nearest arcminute with rollover handling.
    d = int(pos_in_sign)
    m = int(round((pos_in_sign - d) * 60.0))
    if m == 60:
        d += 1
        m = 0
        if d == 30:
            d = 29
            m = 59
    return d, m


def geocode_location(query: str):
    # 1) Try Nominatim first
    try:
        loc = geolocator.geocode(query, addressdetails=True)
        if loc:
            return loc.latitude, loc.longitude, loc.raw
    except Exception:
        pass  # fall through to fallback

    # 2) Fallback: Open-Meteo (no API key)
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": query,
                "count": 1,
                "language": "en",
                "format": "json"
            },
            timeout=10
        )
        r.raise_for_status()
        data = r.json()

        results = data.get("results") or []
        if not results:
            raise HTTPException(
                status_code=404,
                detail="Location not found. Try 'City, State, Country'."
            )

        best = results[0]
        lat = float(best["latitude"])
        lon = float(best["longitude"])

        return lat, lon, {"source": "open-meteo", "raw": best}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Geocoding error: {str(e)}"
        )




def resolve_timezone_name(lat: float, lon: float):
    tz_name = tf.timezone_at(lat=lat, lng=lon)
    if not tz_name:
        tz_name = tf.closest_timezone_at(lat=lat, lng=lon)
    if not tz_name:
        raise HTTPException(status_code=422, detail="Could not resolve timezone for this location.")
    return tz_name


def local_to_utc(year: int, month: int, day: int, hour: int, minute: int, tz_name: str):
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        # Windows + Python 3.13 fallback
        import tzdata
        tz = ZoneInfo(tz_name)

    local_dt = datetime(year, month, day, hour, minute, tzinfo=tz)
    utc_dt = local_dt.astimezone(ZoneInfo("UTC"))
    return local_dt, utc_dt


def julian_day_ut(utc_dt: datetime) -> float:
    ut_hours = utc_dt.hour + (utc_dt.minute / 60.0) + (utc_dt.second / 3600.0)
    return swe.julday(utc_dt.year, utc_dt.month, utc_dt.day, ut_hours)


def house_for_longitude(jd_ut: float, lat: float, lon: float, ecl_lon: float, houses: list, ascmc: list) -> int:
    """
    Determine house number (1..12) for a given ecliptic longitude.
    Swiss Ephemeris provides swe.house_pos(), but it needs obliquity.
    We'll use ascmc[9] = obliquity if available, else compute it.
    """
    # ascmc indices: 0=Asc, 1=MC, 2=ARMC, 3=Vertex, 4=EquAsc, 5=coAsc1, 6=coAsc2, 7=polAsc
    # In newer swe, obliquity is often in ascmc[9], but not always guaranteed.
    obliq = None
    if len(ascmc) > 9:
        obliq = ascmc[9]
    if obliq is None:
        # Compute obliquity of ecliptic (mean obliquity) in degrees
        obliq = swe.calc_ut(jd_ut, swe.ECL_NUT, FLAGS)[0][0]

    # house_pos expects: armc, geolat, eps, xpin, hsys
    armc = ascmc[2]
    hpos = swe.house_pos(armc, lat, obliq, [ecl_lon, 0.0], HOUSE_SYSTEM)
    # hpos is float house position like 5.83 -> house 5
    h = int(math.floor(hpos))
    if h < 1:
        h = 1
    if h > 12:
        h = 12
    return h


@app.post("/chart")
def chart(req: ChartRequest) -> Dict[str, Any]:
    # 1) Geocode: City/State/Country -> lat/lon
    lat, lon, raw = geocode_location(req.location)

    # 2) Resolve timezone from coordinates (global + historical DST correctness comes from tz database)
    tz_name = resolve_timezone_name(lat, lon)

    # 3) Convert local -> UTC
    local_dt, utc_dt = local_to_utc(req.year, req.month, req.day, req.hour, req.minute, tz_name)

    # 4) Julian Day (UT)
    jd_ut = julian_day_ut(utc_dt)

    # 5) Houses & angles (Placidus)
    houses, ascmc = swe.houses_ex(jd_ut, lat, lon, HOUSE_SYSTEM)
    asc = ascmc[0]
    mc = ascmc[1]

    # 6) Planets
    planets_out = {}
    for name, body in PLANETS.items():
        res = swe.calc_ut(jd_ut, body, FLAGS)[0]
        ecl_lon = normalize_deg(res[0])
        speed_lon = res[3]  # deg/day
        retro = speed_lon < 0

        sign, pos_in_sign = zodiac_sign_and_pos(ecl_lon)
        d, m = round_to_arcminute(pos_in_sign)

        # house assignment
        house_num = house_for_longitude(jd_ut, lat, lon, ecl_lon, houses, ascmc)

        planets_out[name] = {
            "sign": sign,
            "deg": d,
            "min": m,
            "lon_deg": ecl_lon,       # keep precise value for internal use
            "retrograde": retro,
            "house": house_num
        }

    # 7) Angles formatted
    asc_sign, asc_pos = zodiac_sign_and_pos(asc)
    asc_d, asc_m = round_to_arcminute(asc_pos)

    mc_sign, mc_pos = zodiac_sign_and_pos(mc)
    mc_d, mc_m = round_to_arcminute(mc_pos)

    # 8) House cusps formatted (1..12)
    cusps_out = []
    # swe.houses_ex returns list/tuple 13 length where index 0 is unused in some bindings; pyswisseph returns 12 cusps
    # We'll handle both.
    if len(houses) == 12:
        cusp_list = houses
        start_index = 1
    else:
        # assume 13, ignore index 0
        cusp_list = houses[1:]
        start_index = 1

    for i, cusp_lon in enumerate(cusp_list, start=1):
        cusp_lon = normalize_deg(cusp_lon)
        hs, hp = zodiac_sign_and_pos(cusp_lon)
        hd, hm = round_to_arcminute(hp)
        cusps_out.append({
            "house": i,
            "sign": hs,
            "deg": hd,
            "min": hm,
            "lon_deg": cusp_lon
        })

    return {
        "settings_locked": {
            "zodiac": "tropical",
            "houses": "placidus",
            "node": "mean",
            "ephemeris": "swiss_ephemeris"
        },
        "input": {
            "location_query": req.location,
            "geocoded": {
                "lat": lat,
                "lon": lon,
                "raw": raw
            },
            "timezone": tz_name,
            "local_datetime": local_dt.isoformat(),
            "utc_datetime": utc_dt.isoformat(),
            "julian_day_ut": jd_ut
        },
        "angles": {
            "ASC": {"sign": asc_sign, "deg": asc_d, "min": asc_m, "lon_deg": normalize_deg(asc)},
            "MC":  {"sign": mc_sign, "deg": mc_d, "min": mc_m, "lon_deg": normalize_deg(mc)}
        },
        "house_cusps": cusps_out,
        "planets": planets_out
    }
