import streamlit as st
import pandas as pd
import numpy as np
import folium

from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CASEFILE AI",
    page_icon="📍",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📍 CASEFILE")
st.subheader(
    "AI-Powered Missing Person Investigation and "
    "Probable Location Prediction System"
)

st.info(
    "Academic simulation only. This application uses synthetic "
    "case information. Predictions are probabilistic and must not "
    "be treated as proof of a person's real location."
)


# ============================================================
# LOAD DATA
# ============================================================

DATA_FILE = "data/processed/movement_features.csv"
CASE_FILE = "data/synthetic/missing_person_cases.csv"

try:
    df = pd.read_csv(DATA_FILE)
    cases = pd.read_csv(CASE_FILE)

except FileNotFoundError:
    st.error(
        "Dataset files not found.\n\n"
        "Make sure these files exist:\n"
        "data/processed/movement_features.csv\n"
        "data/synthetic/missing_person_cases.csv"
    )
    st.stop()


# ============================================================
# AREA COORDINATES
# ============================================================

AREA_COORDS = {
    "Area_A": (23.2599, 77.4126),
    "Area_B": (23.2480, 77.4340),
    "Area_C": (23.2710, 77.3980),
    "Area_D": (23.2350, 77.4210),
    "Area_E": (23.2850, 77.4400)
}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ CASEFILE Controls")

case_id = st.sidebar.selectbox(
    "Select Fictional Case",
    cases["Case_ID"].tolist()
)

case = cases[cases["Case_ID"] == case_id].iloc[0]


# ============================================================
# CASE INFORMATION
# ============================================================

st.header("📋 Case Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Case ID", case["Case_ID"])

with col2:
    st.metric("Person ID", case["Person_ID"])

with col3:
    st.metric("Last Known Area", case["Last_Area"])

with col4:
    st.metric("Age Group", case["Age_Group"])


col5, col6, col7, col8 = st.columns(4)

with col5:
    st.write("**Last Seen:**")
    st.write(case["Last_Seen_Time"])

with col6:
    st.write("**Day:**")
    st.write(case["Day"])

with col7:
    st.write("**Weather:**")
    st.write(case["Weather"])

with col8:
    st.write("**Usual Area:**")
    st.write(case["Usual_Area"])


# ============================================================
# DATA PREPARATION
# ============================================================

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)

df = df.dropna()

df["Hour"] = df["Timestamp"].dt.hour

df["DayOfWeek"] = df["Timestamp"].dt.dayofweek

df["Weekend"] = df["DayOfWeek"].isin(
    [5, 6]
).astype(int)


# ============================================================
# MODULE 1
# K-MEANS MOVEMENT CLUSTERING
# ============================================================

st.divider()

st.header("🔵 1. Movement Pattern Clustering")

cluster_features = [
    "Latitude",
    "Longitude",
    "Speed_kmh",
    "Distance_km"
]

scaler = StandardScaler()

X_cluster = scaler.fit_transform(
    df[cluster_features]
)

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(
    X_cluster
)

st.success(
    "K-Means clustering completed successfully."
)

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

st.bar_chart(cluster_counts)


# ============================================================
# MODULE 2
# ISOLATION FOREST
# ============================================================

st.header("⚠️ 2. Anomaly Detection")

anomaly_features = [
    "Latitude",
    "Longitude",
    "Speed_kmh",
    "Distance_km",
    "Hour"
]

isolation_forest = IsolationForest(
    n_estimators=150,
    contamination=0.05,
    random_state=42
)

df["Anomaly"] = isolation_forest.fit_predict(
    df[anomaly_features]
)

df["Anomaly_Label"] = np.where(
    df["Anomaly"] == -1,
    "Anomaly",
    "Normal"
)

total_records = len(df)

anomaly_count = int(
    (df["Anomaly"] == -1).sum()
)

normal_count = total_records - anomaly_count

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Total Movements",
        total_records
    )

with c2:
    st.metric(
        "Normal Movements",
        normal_count
    )

with c3:
    st.metric(
        "Anomalies",
        anomaly_count
    )


st.write("### Detected Anomalies")

anomaly_table = df[
    df["Anomaly"] == -1
][
    [
        "Person_ID",
        "Timestamp",
        "Latitude",
        "Longitude",
        "Speed_kmh",
        "Distance_km",
        "Area"
    ]
]

st.dataframe(
    anomaly_table.head(50),
    use_container_width=True
)


# ============================================================
# MODULE 3
# RANDOM FOREST LOCATION PREDICTION
# ============================================================

st.divider()

st.header("🎯 3. Probable Location Prediction")

area_codes = {
    area: index
    for index, area
    in enumerate(
        sorted(df["Area"].unique())
    )
}


prediction_features = [
    "Hour",
    "DayOfWeek",
    "Weekend",
    "Average_Speed",
    "Average_Distance",
    "Last_Area_Code",
    "Previous_Area_Code",
    "Time_Since_Last_Seen"
]


# Create training dataset

training_data = df[
    df["Person_ID"] == case["Person_ID"]
].copy()


# If selected person has insufficient data,
# use complete synthetic dataset.

if len(training_data) < 30:
    training_data = df.copy()


training_data["Average_Speed"] = (
    case["Average_Speed"]
)

training_data["Average_Distance"] = (
    case["Average_Distance"]
)

training_data["Last_Area_Code"] = (
    area_codes.get(
        case["Last_Area"],
        0
    )
)

training_data["Previous_Area_Code"] = (
    area_codes.get(
        case["Previous_Area"],
        0
    )
)

training_data["Time_Since_Last_Seen"] = (
    case["Time_Since_Last_Seen"]
)


# Random Forest

random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    class_weight="balanced"
)

random_forest.fit(
    training_data[prediction_features],
    training_data["Area"]
)


# ============================================================
# CASE INPUT
# ============================================================

last_seen_time = pd.to_datetime(
    case["Last_Seen_Time"]
)

case_input = pd.DataFrame(
    [
        {
            "Hour": last_seen_time.hour,

            "DayOfWeek":
                last_seen_time.dayofweek,

            "Weekend":
                int(
                    last_seen_time.dayofweek
                    in [5, 6]
                ),

            "Average_Speed":
                case["Average_Speed"],

            "Average_Distance":
                case["Average_Distance"],

            "Last_Area_Code":
                area_codes.get(
                    case["Last_Area"],
                    0
                ),

            "Previous_Area_Code":
                area_codes.get(
                    case["Previous_Area"],
                    0
                ),

            "Time_Since_Last_Seen":
                case["Time_Since_Last_Seen"]
        }
    ]
)


# ============================================================
# PREDICTION
# ============================================================

probabilities = (
    random_forest
    .predict_proba(case_input)[0]
)

classes = (
    random_forest.classes_
)

predictions = list(
    zip(
        classes,
        probabilities
    )
)

predictions.sort(
    key=lambda x: x[1],
    reverse=True
)


prediction_df = pd.DataFrame(
    predictions,
    columns=[
        "Area",
        "Probability"
    ]
)

prediction_df["Probability"] = (
    prediction_df["Probability"] * 100
).round(2)


st.subheader(
    "📊 Predicted Probable Areas"
)

st.dataframe(
    prediction_df,
    use_container_width=True
)

st.bar_chart(
    prediction_df.set_index(
        "Area"
    )["Probability"]
)


# ============================================================
# MODULE 4
# MARKOV ROUTE PREDICTION
# ============================================================

st.divider()

st.header("🛣️ 4. Probable Route Prediction")

st.write(
    "The route is estimated using historical "
    "area-to-area transition probabilities."
)


# Create transitions

transition_table = pd.crosstab(
    df["Area"],
    df["Area"].shift(-1),
    normalize="index"
).fillna(0)


current_area = case["Last_Area"]

route = [current_area]


for i in range(3):

    if current_area in transition_table.index:

        next_area = (
            transition_table
            .loc[current_area]
            .sort_values(
                ascending=False
            )
            .index[0]
        )

        route.append(
            next_area
        )

        current_area = next_area


st.success(
    " → ".join(route)
)


# ============================================================
# MODULE 5
# SEARCH PRIORITY SCORE
# ============================================================

st.divider()

st.header("📊 5. Search Priority Score")


visit_frequency = (
    df["Area"]
    .value_counts(
        normalize=True
    )
)


priority_results = []


for area, probability in predictions:

    prediction_score = (
        probability * 100
    )

    historical_score = (
        visit_frequency.get(
            area,
            0
        ) * 100
    )

    route_score = (
        100
        if area in route
        else 30
    )

    distance_score = (
        80
        if area == case["Last_Area"]
        else 55
    )

    time_score = (
        80
        if area == case["Usual_Area"]
        else 50
    )

    area_data = df[
        df["Area"] == area
    ]

    if len(area_data) > 0:

        anomaly_percentage = (
            area_data["Anomaly"]
            .eq(-1)
            .mean()
        )

    else:

        anomaly_percentage = 0


    anomaly_score = (
        70
        if anomaly_percentage > 0.05
        else 30
    )


    # Required weighting

    final_score = (

        0.30 * prediction_score

        + 0.20 * historical_score

        + 0.15 * route_score

        + 0.15 * distance_score

        + 0.10 * time_score

        + 0.10 * anomaly_score

    )


    priority_results.append(
        [
            area,
            round(
                final_score,
                2
            )
        ]
    )


priority_df = pd.DataFrame(
    priority_results,
    columns=[
        "Area",
        "Priority Score"
    ]
)


priority_df = (
    priority_df
    .sort_values(
        "Priority Score",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# Priority category

def get_priority(score):

    if score <= 30:
        return "Low"

    elif score <= 60:
        return "Medium"

    elif score <= 80:
        return "High"

    else:
        return "Very High"


priority_df["Priority"] = (
    priority_df[
        "Priority Score"
    ].apply(
        get_priority
    )
)


st.dataframe(
    priority_df,
    use_container_width=True
)


# ============================================================
# MODULE 6
# EXPLAINABLE AI
# ============================================================

st.divider()

st.header("🔍 6. Explainable AI")

top_area = (
    prediction_df
    .iloc[0]["Area"]
)


st.subheader(
    f"Why was {top_area} highly ranked?"
)


reasons = []


if (
    visit_frequency.get(
        top_area,
        0
    )
    >
    visit_frequency.mean()
):

    reasons.append(
        "High historical visit frequency."
    )


if top_area in route:

    reasons.append(
        "Area appears in the probable route."
    )


if (
    top_area
    ==
    case["Usual_Area"]
):

    reasons.append(
        "Area matches the synthetic usual area."
    )


if (
    top_area
    ==
    case["Last_Area"]
):

    reasons.append(
        "Area matches the last known area."
    )


reasons.append(
    "Random Forest prediction probability "
    "contributes to the priority score."
)


for reason in reasons:

    st.success(
        "✓ " + reason
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader(
    "Random Forest Feature Importance"
)


importance_df = pd.DataFrame(
    {
        "Feature":
            prediction_features,

        "Importance":
            random_forest
            .feature_importances_
    }
)


importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
)


st.bar_chart(
    importance_df.set_index(
        "Feature"
    )
)


# ============================================================
# MODULE 7
# INTERACTIVE MAP
# ============================================================

st.divider()

st.header("🗺️ 7. Interactive Investigation Map")


map_object = folium.Map(
    location=[
        23.26,
        77.42
    ],
    zoom_start=12
)


# Last known location

last_area = case["Last_Area"]

folium.Marker(
    AREA_COORDS[last_area],
    popup=(
        "Last Known Area: "
        + last_area
    ),
    tooltip="Last Known Location",
    icon=folium.Icon(
        icon="info-sign"
    )
).add_to(
    map_object
)


# Predicted locations

for area, probability in predictions:

    folium.CircleMarker(

        location=
            AREA_COORDS[area],

        radius=9,

        popup=(
            f"{area} - "
            f"{probability * 100:.2f}%"
        ),

        tooltip=(
            f"Predicted: {area}"
        ),

        fill=True

    ).add_to(
        map_object
    )


# Route line

route_points = [
    AREA_COORDS[area]
    for area in route
]


if len(route_points) > 1:

    folium.PolyLine(

        route_points,

        weight=5,

        popup=(
            "Synthetic probable route"
        )

    ).add_to(
        map_object
    )


# Show map

st_folium(
    map_object,
    width=1100,
    height=600
)


# ============================================================
# FINAL SUMMARY
# ============================================================

st.divider()

st.header("📌 Investigation Summary")


summary_col1, summary_col2 = st.columns(2)


with summary_col1:

    st.write(
        "**Top Probable Area:**"
    )

    st.success(
        prediction_df.iloc[0]["Area"]
    )

    st.write(
        "**Probable Route:**"
    )

    st.info(
        " → ".join(route)
    )


with summary_col2:

    top_priority = (
        priority_df.iloc[0]
    )

    st.write(
        "**Highest Priority Area:**"
    )

    st.success(
        top_priority["Area"]
    )

    st.write(
        "**Priority Score:**"
    )

    st.metric(
        "Score",
        top_priority[
            "Priority Score"
        ]
    )


# ============================================================
# ETHICAL DISCLAIMER
# ============================================================

st.divider()

st.warning(
    "⚠️ ETHICAL DISCLAIMER\n\n"
    "This is an academic simulation using synthetic "
    "information. ML predictions are probabilistic and "
    "can contain false positives and false negatives. "
    "An anomaly does not indicate criminal behavior, "
    "and a predicted area does not prove a person's location."
)
