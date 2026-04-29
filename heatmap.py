import pandas as pd
import numpy as np
import folium
import requests
import json
import os

# ─────────────────────────────────────────────
# STEP 1: LOAD AND PROCESS DATA
# ─────────────────────────────────────────────

def load_and_process_data():
    df = pd.read_csv("ncrb_data.csv")

    # Total cases across all years
    df["total_cases"] = df[["cases_2018", "cases_2019", "cases_2020", "cases_2021", "cases_2022"]].sum(axis=1)

    # Risk Index = (cases / population) × severity weight
    df["raw_risk"] = (df["total_cases"] / df["population_millions"]) * df["severity_weight"]

   
   # Normalize to 0–100
    min_r = df["raw_risk"].min()
    max_r = df["raw_risk"].max()
    df["risk_index"] = ((df["raw_risk"] - min_r) / (max_r - min_r)) * 100
    df["risk_index"] = df["risk_index"].round(1)

    # Safety floor for known high-risk states
    HIGH_RISK_FLOOR = {
        "Delhi": 65,
        "West Bengal": 70,
        "Rajasthan": 60,
        "Uttar Pradesh": 62,
        "Maharashtra": 55,
        "Assam": 58,
    }
    for state, floor in HIGH_RISK_FLOOR.items():
        mask = df["state"] == state
        df.loc[mask & (df["risk_index"] < floor), "risk_index"] = float(floor)

    df["risk_index"] = df["risk_index"].round(1)

    # Assign risk level
    def get_level(score):
        if score >= 80:
            return "EXTREME"
        elif score >= 60:
            return "HIGH"
        elif score >= 35:
            return "MEDIUM"
        else:
            return "LOW"

    df["level"] = df["risk_index"].apply(get_level)
    

    # Assign risk level
    def get_level(score):
        if score >= 80:
            return "EXTREME"
        elif score >= 60:
            return "HIGH"
        elif score >= 35:
            return "MEDIUM"
        else:
            return "LOW"

    df["level"] = df["risk_index"].apply(get_level)

    return df

# ─────────────────────────────────────────────
# STEP 2: DOWNLOAD INDIA GEOJSON
# ─────────────────────────────────────────────

def get_geojson():
    geojson_path = "india_states.geojson"

    if not os.path.exists(geojson_path):
        print("Downloading India GeoJSON...")
        url = "https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson"
        r = requests.get(url)
        with open(geojson_path, "w") as f:
            f.write(r.text)
        print("Downloaded successfully.")
    else:
        print("GeoJSON already exists, skipping download.")

    with open(geojson_path) as f:
        return json.load(f)

# ─────────────────────────────────────────────
# STEP 3: BUILD THE HEATMAP
# ─────────────────────────────────────────────

def build_heatmap(df, geojson_data):
    # Create base map centered on India
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB positron")

    # Color scale
    def risk_color(score):
        if score >= 80:
            return "#8B0000"   # Dark Red – Extreme
        elif score >= 60:
            return "#FF6600"   # Orange – High
        elif score >= 35:
            return "#FFD700"   # Yellow – Medium
        else:
            return "#90EE90"   # Light Green – Low

    # State name mapping (GeoJSON name → CSV name)
    name_map = {
        "West Bengal"       : "West Bengal",
        "Rajasthan"         : "Rajasthan",
        "Maharashtra"       : "Maharashtra",
        "Tamil Nadu"        : "Tamil Nadu",
        "Andhra Pradesh"    : "Andhra Pradesh",
        "Odisha"            : "Odisha",
        "Uttar Pradesh"     : "Uttar Pradesh",
        "Bihar"             : "Bihar",
        "Delhi"             : "Delhi",
        "Karnataka"         : "Karnataka",
        "Gujarat"           : "Gujarat",
        "Madhya Pradesh"    : "Madhya Pradesh",
        "Jharkhand"         : "Jharkhand",
        "Assam"             : "Assam",
        "Kerala"            : "Kerala",
        "Punjab"            : "Punjab",
        "Haryana"           : "Haryana",
        "Chhattisgarh"      : "Chhattisgarh",
        "Uttarakhand"       : "Uttarakhand",
        "Himachal Pradesh"  : "Himachal Pradesh",
        "Goa"               : "Goa",
        "Manipur"           : "Manipur",
        "Meghalaya"         : "Meghalaya",
        "Tripura"           : "Tripura",
        "Nagaland"          : "Nagaland",
        "Mizoram"           : "Mizoram",
        "Arunachal Pradesh" : "Arunachal Pradesh",
        "Sikkim"            : "Sikkim",
        "Jammu & Kashmir"   : "Jammu & Kashmir",
        "Telangana"         : "Telangana",
    }

    # Build a lookup dict from df
    data_lookup = {}
    for _, row in df.iterrows():
        data_lookup[row["state"]] = {
            "risk_index" : row["risk_index"],
            "cases_2022" : row["cases_2022"],
            "level"      : row["level"],
            "total_cases": row["total_cases"],
        }

    # Add each state as a GeoJSON layer with tooltip
    for feature in geojson_data["features"]:
        geo_name  = feature["properties"].get("NAME_1") or feature["properties"].get("name") or ""
        csv_name  = name_map.get(geo_name, geo_name)
        info      = data_lookup.get(csv_name)

        if info:
            color   = risk_color(info["risk_index"])
            tooltip = (
                f"<b>{csv_name}</b><br>"
                f"Risk Level : {info['level']}<br>"
                f"Risk Index : {info['risk_index']}/100<br>"
                f"Cases 2022 : {info['cases_2022']}<br>"
                f"Total (18-22): {info['total_cases']}"
            )
        else:
            color   = "#CCCCCC"
            tooltip = f"<b>{geo_name}</b><br>Data not available"

        folium.GeoJson(
            feature,
            style_function=lambda x, c=color: {
                "fillColor"   : c,
                "color"       : "white",
                "weight"      : 1.5,
                "fillOpacity" : 0.75,
            },
            tooltip=folium.Tooltip(tooltip),
        ).add_to(m)

    # ── Legend ──────────────────────────────────
    legend_html = """
    <div style="
        position: fixed; bottom: 40px; left: 40px; z-index: 1000;
        background: white; padding: 15px; border-radius: 10px;
        border: 2px solid grey; font-size: 14px; font-family: Arial;">
      <b>🚨 Trafficking Risk Level</b><br><br>
      <i style="background:#8B0000;width:18px;height:18px;display:inline-block;margin-right:8px;"></i> Extreme Risk (80–100)<br>
      <i style="background:#FF6600;width:18px;height:18px;display:inline-block;margin-right:8px;"></i> High Risk (60–79)<br>
      <i style="background:#FFD700;width:18px;height:18px;display:inline-block;margin-right:8px;"></i> Medium Risk (35–59)<br>
      <i style="background:#90EE90;width:18px;height:18px;display:inline-block;margin-right:8px;"></i> Low Risk (0–34)<br>
      <i style="background:#CCCCCC;width:18px;height:18px;display:inline-block;margin-right:8px;"></i> No Data
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save
    m.save("india_heatmap.html")
    print("✅ india_heatmap.html saved successfully!")

# ─────────────────────────────────────────────
# STEP 4: CITY LOOKUP FUNCTION (for Person 4)
# ─────────────────────────────────────────────

# Load data once at module level so Person 4 can import and use directly
_df = load_and_process_data()

def get_city_risk(city_name):
    """
    Given a city or state name, returns its risk score and case count.
    Example: get_city_risk("Delhi")
    Returns: {'risk_index': 74.0, 'cases_2022': 412, 'level': 'HIGH'}
    """
    # Normalize input
    city_name = city_name.strip().title()

    # City → State mapping for major Indian cities
    city_to_state = {
        "Mumbai"       : "Maharashtra",
        "Pune"         : "Maharashtra",
        "Delhi"        : "Delhi",
        "New Delhi"    : "Delhi",
        "Kolkata"      : "West Bengal",
        "Chennai"      : "Tamil Nadu",
        "Bangalore"    : "Karnataka",
        "Bengaluru"    : "Karnataka",
        "Hyderabad"    : "Telangana",
        "Jaipur"       : "Rajasthan",
        "Jodhpur"      : "Rajasthan",
        "Lucknow"      : "Uttar Pradesh",
        "Kanpur"       : "Uttar Pradesh",
        "Varanasi"     : "Uttar Pradesh",
        "Patna"        : "Bihar",
        "Bhopal"       : "Madhya Pradesh",
        "Indore"       : "Madhya Pradesh",
        "Bhubaneswar"  : "Odisha",
        "Ranchi"       : "Jharkhand",
        "Guwahati"     : "Assam",
        "Thiruvananthapuram": "Kerala",
        "Kochi"        : "Kerala",
        "Chandigarh"   : "Punjab",
        "Amritsar"     : "Punjab",
        "Surat"        : "Gujarat",
        "Ahmedabad"    : "Gujarat",
        "Raipur"       : "Chhattisgarh",
        "Dehradun"     : "Uttarakhand",
        "Shimla"       : "Himachal Pradesh",
        "Panaji"       : "Goa",
        "Imphal"       : "Manipur",
        "Shillong"     : "Meghalaya",
        "Agartala"     : "Tripura",
        "Kohima"       : "Nagaland",
        "Aizawl"       : "Mizoram",
        "Itanagar"     : "Arunachal Pradesh",
        "Gangtok"      : "Sikkim",
        "Srinagar"     : "Jammu & Kashmir",
    }

    # Resolve city to state if needed
    state_name = city_to_state.get(city_name, city_name)

    # Look up in dataframe
    match = _df[_df["state"].str.lower() == state_name.lower()]

    if match.empty:
        return {
            "error"      : f"'{city_name}' not found in database.",
            "tip"        : "Try using the state name e.g. 'West Bengal', 'Rajasthan'"
        }

    row = match.iloc[0]
    return {
        "city_queried" : city_name,
        "state"        : row["state"],
        "risk_index"   : float(row["risk_index"]),
        "cases_2022"   : int(row["cases_2022"]),
        "total_cases"  : int(row["total_cases"]),
        "level"        : row["level"],
    }


# ─────────────────────────────────────────────
# MAIN — Run everything
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("🔄 Loading and processing NCRB data...")
    df = load_and_process_data()
    print(df[["state", "risk_index", "level"]].to_string(index=False))

    print("\n🗺️  Building heatmap...")
    geojson = get_geojson()
    build_heatmap(df, geojson)

    print("\n🧪 Testing get_city_risk()...")
    print(get_city_risk("Delhi"))
    print(get_city_risk("Kolkata"))
    print(get_city_risk("Mumbai"))
    print(get_city_risk("West Bengal"))