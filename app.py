import streamlit as st
import pandas as pd
import numpy as np
import folium

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from streamlit_folium import st_folium


# ==========================================================
# CASEFILE - COLLEGE PROJECT
# ==========================================================

st.set_page_config(
    page_title="CASEFILE - AI Investigation System",
    page_icon="🔎",
    layout="wide"
)


# ==========================================================
# HEADER
# ==========================================================

st.title("🔎 CASEFILE")

st.header(
    "AI-Powered Missing Person Investigation and "
    "Probable Location Prediction System"
)

st.write(
    "An Advanced Machine Learning based academic "
    "investigation-support system."
)

st.info(
    "🎓 Academic Project Simulation\n\n"
    "This application uses synthetic data only. "
    "Predictions are probabilistic and are intended "
    "only for educational demonstration."
)


# ==========================================================
# SYNTHETIC DATA GENERATION
# ==========================================================

np.random.seed(42)

locations = {
    "Area A": (23.2599, 77.4126),
    "Area B": (23.2480, 77.4340),
    "Area C": (23.2710, 77.3980),
    "Area D": (23.2350, 77.4210),
    "Area E": (23.2850, 77.4400)
}

area_names = list(locations.keys())

records = []

for person in range(1, 11):

    person_id = f"P{person:03d}"

    for i in range(100):

        area = np.random.choice(area_names)

        latitude = (
            locations[area][0]
            + np.random.normal(0, 0.002)
        )

        longitude = (
            locations[area][1]
            + np.random.normal(0, 0.002)
        )

        hour = np.random.randint(6, 23)

        speed = max(
            2,
            np.random.normal(20, 5)
        )

        distance = max(
            0.2,
            np.random.normal(5, 2)
        )

        records.append([
            person_id,
            latitude,
            longitude,
            hour,
            speed,
            distance,
            area
        ])


df = pd.DataFrame(
    records,
    columns=[
        "Person_ID",
        "Latitude",
        "Longitude",
        "Hour",
        "Speed",
        "Distance",
        "Area"
    ]
)


# ==========================================================
# FICTIONAL CASE
# ==========================================================

case_id = "CASE-001"
person_id = "P001"

last_area = "Area A"
previous_area = "Area B"

last_seen = "18:45"
age_group = "18-25"


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("📋 Case Details")

st.sidebar.write(
    f"**Case ID:** {case_id}"
)

st.sidebar.write(
    f"**Person ID:** {person_id}"
)

st.sidebar.write(
    f"**Age Group:** {age_group}"
)

st.sidebar.write(
    f"**Last Known Area:** {last_area}"
)

st.sidebar.write(
    f"**Last Seen:** {last_seen}"
)

st.sidebar.write(
    "**Case Type:** Academic Simulation"
)


# ==========================================================
# DASHBOARD SUMMARY
# ==========================================================

st.subheader("📊 Case Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Case ID",
        case_id
    )

with c2:
    st.metric(
        "Person",
        person_id
    )

with c3:
    st.metric(
        "Last Area",
        last_area
    )

with c4:
    st.metric(
        "Movement Records",
        len(df)
    )


# ==========================================================
# 1. MOVEMENT PATTERN CLUSTERING
# ==========================================================

st.divider()

st.header("1️⃣ Movement Pattern Clustering")

st.write(
    "K-Means clustering groups movement records "
    "based on geographical position, speed and distance."
)

cluster_features = [
    "Latitude",
    "Longitude",
    "Speed",
    "Distance"
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

cluster_data = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

st.bar_chart(cluster_data)


# ==========================================================
# 2. ANOMALY DETECTION
# ==========================================================

st.divider()

st.header("2️⃣ Anomaly Detection")

st.write(
    "Isolation Forest identifies unusual movement "
    "patterns in the synthetic dataset."
)

anomaly_features = [
    "Latitude",
    "Longitude",
    "Speed",
    "Distance",
    "Hour"
]

isolation_model = IsolationForest(
    n_estimators=150,
    contamination=0.05,
    random_state=42
)

df["Anomaly"] = isolation_model.fit_predict(
    df[anomaly_features]
)

df["Movement Status"] = np.where(
    df["Anomaly"] == -1,
    "Anomaly",
    "Normal"
)

normal_count = (
    df["Movement Status"] == "Normal"
).sum()

anomaly_count = (
    df["Movement Status"] == "Anomaly"
).sum()

a1, a2 = st.columns(2)

with a1:
    st.metric(
        "Normal Movements",
        normal_count
    )

with a2:
    st.metric(
        "Anomalous Movements",
        anomaly_count
    )

st.subheader("Detected Unusual Movements")

st.dataframe(
    df[
        df["Movement Status"] == "Anomaly"
    ].head(20),
    use_container_width=True
)


# ==========================================================
# 3. LOCATION PREDICTION
# ==========================================================

st.divider()

st.header("3️⃣ Probable Location Prediction")

st.write(
    "Random Forest is used to estimate the probable "
    "area from historical movement characteristics."
)


# Encode areas

area_codes = {
    area: number
    for number, area
    in enumerate(area_names)
}


training_data = df.copy()

training_data["Last_Area_Code"] = (
    area_codes[last_area]
)

training_data["Previous_Area_Code"] = (
    area_codes[previous_area]
)

training_data["Average_Speed"] = 18

training_data["Average_Distance"] = 7


features = [
    "Hour",
    "Speed",
    "Distance",
    "Last_Area_Code",
    "Previous_Area_Code",
    "Average_Speed",
    "Average_Distance"
]


X_train = training_data[features]

y_train = training_data["Area"]


random_forest = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42
)

random_forest.fit(
    X_train,
    y_train
)


# Input for fictional case

case_input = pd.DataFrame(
    [{
        "Hour": 18,
        "Speed": 18,
        "Distance": 7,
        "Last_Area_Code": area_codes[last_area],
        "Previous_Area_Code": area_codes[previous_area],
        "Average_Speed": 18,
        "Average_Distance": 7
    }]
)


probabilities = (
    random_forest
    .predict_proba(case_input)[0]
)

classes = random_forest.classes_


results = list(
    zip(
        classes,
        probabilities
    )
)

results.sort(
    key=lambda x: x[1],
    reverse=True
)


prediction_df = pd.DataFrame(
    results,
    columns=[
        "Area",
        "Probability"
    ]
)

prediction_df["Probability (%)"] = (
    prediction_df["Probability"] * 100
).round(2)

prediction_df = prediction_df[
    [
        "Area",
        "Probability (%)"
    ]
]


st.dataframe(
    prediction_df,
    use_container_width=True
)

st.bar_chart(
    prediction_df.set_index(
        "Area"
    )["Probability (%)"]
)


# ==========================================================
# 4. ROUTE PREDICTION
# ==========================================================

st.divider()

st.header("4️⃣ Probable Route Prediction")

st.write(
    "A simple transition-probability approach "
    "is used to estimate a possible sequence of areas."
)


transition_table = pd.crosstab(
    df["Area"],
    df["Area"].shift(-1),
    normalize="index"
).fillna(0)


current_area = last_area

route = [
    current_area
]


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


# ==========================================================
# 5. SEARCH PRIORITY SCORE
# ==========================================================

st.divider()

st.header("5️⃣ Search Priority Scoring")

st.write(
    "The priority score combines predicted probability, "
    "historical movement frequency, route relevance, "
    "last-known area and anomaly information."
)


frequency = (
    df["Area"]
    .value_counts(
        normalize=True
    )
)


priority_records = []


for area, probability in results:

    prediction_score = (
        probability * 100
    )

    historical_score = (
        frequency.get(
            area,
            0
        ) * 100
    )

    route_score = (
        100
        if area in route
        else 30
    )

    last_area_score = (
        100
        if area == last_area
        else 50
    )

    usual_area_score = (
        100
        if area == "Area A"
        else 50
    )

    area_rows = df[
        df["Area"] == area
    ]

    if len(area_rows) > 0:

        anomaly_rate = (
            area_rows["Anomaly"] == -1
        ).mean()

    else:

        anomaly_rate = 0


    anomaly_score = (
        80
        if anomaly_rate > 0.05
        else 30
    )


    final_score = (

        prediction_score * 0.30

        + historical_score * 0.20

        + route_score * 0.15

        + last_area_score * 0.15

        + usual_area_score * 0.10

        + anomaly_score * 0.10

    )


    priority_records.append(
        [
            area,
            round(
                final_score,
                2
            )
        ]
    )


priority_df = pd.DataFrame(
    priority_records,
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


def get_priority(score):

    if score <= 30:
        return "Low"

    if score <= 60:
        return "Medium"

    if score <= 80:
        return "High"

    return "Very High"


priority_df["Priority Level"] = (
    priority_df["Priority Score"]
    .apply(get_priority)
)


st.dataframe(
    priority_df,
    use_container_width=True
)


# ==========================================================
# 6. EXPLAINABLE AI
# ==========================================================

st.divider()

st.header("6️⃣ Explainable AI")

top_area = prediction_df.iloc[0]["Area"]

st.write(
    f"### Top predicted area: **{top_area}**"
)

st.write(
    "The system considers the following factors:"
)

explanations = [
    "Historical movement frequency",
    "Movement speed and distance",
    "Time of movement",
    "Last known area",
    "Previous area",
    "Route transition pattern",
    "Random Forest prediction probability"
]

for item in explanations:

    st.write(
        "✓ " + item
    )


# Feature importance

importance_df = pd.DataFrame(
    {
        "Feature": features,
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

st.subheader(
    "Random Forest Feature Importance"
)

st.bar_chart(
    importance_df.set_index(
        "Feature"
    )
)


# ==========================================================
# 7. INTERACTIVE MAP
# ==========================================================

st.divider()

st.header("7️⃣ Interactive Investigation Map")

m = folium.Map(
    location=[
        23.26,
        77.42
    ],
    zoom_start=12
)


# Last known location

folium.Marker(
    locations[last_area],
    popup=(
        "Last Known Area: "
        + last_area
    ),
    tooltip="Last Known Location"
).add_to(m)


# Predicted areas

for area, probability in results:

    folium.CircleMarker(
        location=locations[area],
        radius=8,
        popup=(
            f"{area}<br>"
            f"Probability: "
            f"{probability * 100:.2f}%"
        ),
        tooltip=area,
        fill=True
    ).add_to(m)


# Route line

route_coordinates = [
    locations[area]
    for area in route
]


folium.PolyLine(
    route_coordinates,
    weight=5,
    tooltip="Probable Route"
).add_to(m)


st_folium(
    m,
    width=1100,
    height=600
)


# ==========================================================
# 8. FINAL RESULT
# ==========================================================

st.divider()

st.header("📌 Final Case Summary")

f1, f2, f3 = st.columns(3)

with f1:

    st.metric(
        "Top Predicted Area",
        prediction_df.iloc[0]["Area"]
    )

with f2:

    st.metric(
        "Priority Area",
        priority_df.iloc[0]["Area"]
    )

with f3:

    st.metric(
        "Priority Score",
        priority_df.iloc[0]["Priority Score"]
    )


st.success(
    "CASEFILE analysis completed successfully."
)


# ==========================================================
# ETHICAL DISCLAIMER
# ==========================================================

st.divider()

st.warning(
    "⚠️ Ethical Disclaimer: This application is an "
    "academic simulation using synthetic data. "
    "Its predictions are probabilistic and may contain "
    "errors. A predicted area is not proof of a person's "
    "actual location, and an anomaly does not indicate "
    "criminal activity."
)


st.caption(
    "CASEFILE | Advanced Machine Learning Individual Project"
)
