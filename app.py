import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates

# Set page config
st.set_page_config(
    page_title="Students Habits VS. Performance", 
    page_icon="🎓", 
    layout="wide"
)

# Load data function with caching
@st.cache_data
def load_data():
    # Load data from CSV file
    # Replace 'your_data.csv' with your actual file path
    df = pd.read_csv(r"C:\Users\USER\Desktop\Portfolio Data Analyst\Assignment Data\Dashboard\student_habits_performance.csv")
    df.head()
    
    # Data cleaning and preprocessing (if needed)
    # Remove Unwanted value from column
    df_cleaned = df[df['gender'] != 'Other']
    df_cleaned.to_csv('cleaned_file.csv', index=False)

    # Rename columns for consistency
    df.columns = [
    "Student_ID", "Age", "Gender", "Study_Hours_Per_Day", "Social_Media_Hours",
    "Netflix_Hours", "Part_Time_Job", "Attendance_Percentage", "Sleep_Hours",
    "Diet_Quality", "Exercise_Frequency_Per_Week", "Parental_Education_Level",
    "Internet_Quality", "Mental_Health_Rating", "Extracurricular_Participation",
    "Exam_Score"
    ]

    # Ensure hour-based columns are float
    df["Study_Hours_Per_Day"] = df["Study_Hours_Per_Day"].astype(float)
    df["Social_Media_Hours"] = df["Social_Media_Hours"].astype(float)
    df["Netflix_Hours"] = df["Netflix_Hours"].astype(float)
    df["Sleep_Hours"] = df["Sleep_Hours"].astype(float)
    df["Exercise_Frequency_Per_Week"] = df["Exercise_Frequency_Per_Week"].astype(float)
    
    return df

df = load_data()

#Title
st.title("🎓 Students Habits VS. performance")
st.markdown("Analyzing the correlation of students habits and their academic performance in recent examination.")


# KPI cards - Based on selected filters
st.subheader("Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📈 Average Exam Score", f"{df['Exam_Score'].mean():.2f}")

    st.metric("⏳ Avg. Study Hours/Day", f"{df['Study_Hours_Per_Day'].mean():.2f} hrs")

with col2:
    st.metric("📅 Avg. Attendance", f"{df['Attendance_Percentage'].mean():.2f}%")

    st.metric("❤️ Avg. Mental Health", f"{df['Mental_Health_Rating'].mean():.2f} / 10")

with col3:
    diet_mode = df['Diet_Quality'].mode()
    st.metric("🍽️ Most Common Diet", diet_mode[0] if not diet_mode.empty else "N/A")

    parent_mode = df['Parental_Education_Level'].mode()
    st.metric("👨‍👩‍🎓 Common Parental Education", parent_mode[0]
    if not parent_mode.empty else "N/A")

    part_time_count = df[df['Part_Time_Job'] == 'Yes'].shape[0]
    part_time_pct = (part_time_count / df.shape[0]) * 100 
    
with col4:
    st.metric("💤 Avg. Sleep Hours", f"{df['Sleep_Hours'].mean():.2f} hrs")
    
    exercise_count = df[df['Exercise_Frequency_Per_Week'] > 0].shape[0]
    percent_exercising = (exercise_count / df.shape[0]) * 100 if df.shape[0] else 0
    st.metric("💪 % Regular Exercise", f"{percent_exercising:.1f}%")


with col5:
    part_time_count = df[df['Part_Time_Job'] == 'Yes'].shape[0]
    part_time_pct = (part_time_count / df.shape[0]) * 100 
    if not df.empty:
        st.metric("📈 Average Exam Score", f"{df['Exam_Score'].mean():.2f}")
    else:
        st.metric("📈 Average Exam Score", "N/A")

    internet_mode = df['Internet_Quality'].mode()
    st.metric("🌐 Most Common Internet Quality", internet_mode[0] if not internet_mode.empty else "N/A")

# Sidebar filters
st.sidebar.header("Filter Data")

# Dynamic field selection based on available data
# Text Search
search_id = st.sidebar.text_input("Search Student ID (S1000-S1999)\nDisclaimer: If the data for the ID is not available, it may have been removed because they are identified as 'Other'. We respect the fact that Islam only recognize two genders only.")

# Gender Filter
if "Gender" in df.columns:
    gender_options = [g for g in sorted(df["Gender"].dropna().unique().tolist()) if g != "Other"]
    gender_options = ["All"] + gender_options
    gender = st.sidebar.radio("Gender", options=gender_options)

# Age Filter
if "Age" in df.columns:
    age_min, age_max = float(df["Age"].min()), float(df["Age"].max())
    age_range = st.sidebar.slider("Age Range", min_value=age_min, max_value=age_max,
                                   value=(age_min, age_max))

# Study & Lifestyle Hours
hour_columns = [
    "Study_Hours_Per_Day", "Social_Media_Hours", "Netflix_Hours",
    "Sleep_Hours", "Exercise_Frequency_Per_Week"
]

hour_filters = {}
for col in hour_columns:
    if col in df.columns:
        min_val, max_val = float(df[col].min()), float(df[col].max())
        hour_filters[col] = st.sidebar.slider(col.replace("_", " "), min_value=min_val, max_value=max_val,
                                              value=(min_val, max_val))

# Attendance
if "Attendance_Percentage" in df.columns:
    attend_range = st.sidebar.slider("Attendance Percentage", 0.0, 100.0,
                                     value=(df["Attendance_Percentage"].min(), df["Attendance_Percentage"].max()))

# Exam Score
if "Exam_Score" in df.columns:
    score_range = st.sidebar.slider("Exam Score", 0.0, 100.0,
                                    value=(df["Exam_Score"].min(), df["Exam_Score"].max()))

# Categorical Filters
categorical_filters = {
    "Part_Time_Job": "Part-Time Job",
    "Diet_Quality": "Diet Quality",
    "Parental_Education_Level": "Parental Education",
    "Internet_Quality": "Internet Quality",
    "Mental_Health_Rating": "Mental Health Rating",
    "Extracurricular_Participation": "Extracurricular Participation"
}

cat_selected = {}
for col, label in categorical_filters.items():
    if col in df.columns:
        options = ["All"] + sorted(df[col].dropna().unique().tolist())
        cat_selected[col] = st.sidebar.radio(label, options=options)

# --- Apply Filters ---
filtered_df = df.copy()

if search_id:
    filtered_df = filtered_df[filtered_df["Student_ID"].astype(str).str.contains(search_id, case=False)]

if "Gender" in df.columns and gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == gender]

if "Age" in df.columns:
    filtered_df = filtered_df[(filtered_df["Age"] >= age_range[0]) & (filtered_df["Age"] <= age_range[1])]

if "Attendance_Percentage" in df.columns:
    filtered_df = filtered_df[(filtered_df["Attendance_Percentage"] >= attend_range[0]) &
                              (filtered_df["Attendance_Percentage"] <= attend_range[1])]

if "Exam_Score" in df.columns:
    filtered_df = filtered_df[(filtered_df["Exam_Score"] >= score_range[0]) &
                              (filtered_df["Exam_Score"] <= score_range[1])]

for col, (min_val, max_val) in hour_filters.items():
    filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]

for col, selected in cat_selected.items():
    if selected != "All":
        filtered_df = filtered_df[filtered_df[col] == selected]

# --- Insert Image with Embedded Filters Info ---
st.subheader("📌 Click a Branch to Open Its Filter")

# Load image and draw labels
image_path = r"C:\Users\USER\Desktop\Portfolio Data Analyst\Assignment Data\Dashboard\istockphoto-1201688581-612x612.jpg"
img = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(img)

# Optional: Load a better font if available
try:
    font = ImageFont.truetype("arial.ttf", 18)
except:
    font = ImageFont.load_default()

# Define filter regions and labels (x1, y1, x2, y2, label)
image_areas = [
    ((50, 100, 200, 250), "Study_Hours_Per_Day"),
    ((250, 100, 400, 250), "Social_Media_Hours"),
    ((420, 100, 600, 250), "Netflix_Hours"),
    ((50, 260, 200, 400), "Sleep_Hours"),
    ((250, 260, 400, 400), "Exercise_Frequency_Per_Week"),
    ((450, 260, 600, 400), "Attendance_Percentage"),
    ((100, 420, 300, 540), "Diet_Quality"),
    ((320, 420, 500, 540), "Parental_Education_Level"),
    ((100, 560, 300, 670), "Internet_Quality"),
    ((320, 560, 500, 670), "Mental_Health_Rating"),
    ((220, 700, 420, 820), "Extracurricular_Participation")
]

# Draw labels
label_map = {
    "Study_Hours_Per_Day": "Study Hours",
    "Social_Media_Hours": "Social Media",
    "Netflix_Hours": "Netflix",
    "Sleep_Hours": "Sleep",
    "Exercise_Frequency_Per_Week": "Exercise",
    "Attendance_Percentage": "Attendance",
    "Diet_Quality": "Diet Quality",
    "Parental_Education_Level": "Parental Education",
    "Internet_Quality": "Internet Quality",
    "Mental_Health_Rating": "Mental Health",
    "Extracurricular_Participation": "Extracurriculars"
}

for coords, label in image_areas:
    x, y = coords[0], coords[1]
    draw.text((x + 5, y + 5), label_map[label], fill="white", font=font)

# Display image
value = streamlit_image_coordinates(img, key="filter_image_general")

# Placeholder for dynamic filter section
clicked_filter = None

if value:
    x, y = value["x"], value["y"]
    st.info(f"🖱️ Clicked coordinates: (x={x}, y={y})")

    # Detect clicked region
    for (x1, y1, x2, y2), label in image_areas:
        if x1 <= x <= x2 and y1 <= y <= y2:
            clicked_filter = label
            st.success(f"🧠 Showing filter for: {label_map[label]}")
            break

# Show the appropriate filter control dynamically
if clicked_filter in hour_columns:
    min_val = float(df[clicked_filter].min())
    max_val = float(df[clicked_filter].max())
    hour_filters[clicked_filter] = st.slider(f"{label_map[clicked_filter]}", min_val, max_val, (min_val, max_val))

elif clicked_filter in categorical_filters:
    options = ["All"] + sorted(df[clicked_filter].dropna().unique().tolist())
    cat_selected[clicked_filter] = st.radio(label_map[clicked_filter], options)

elif clicked_filter == "Attendance_Percentage":
    min_val = float(df["Attendance_Percentage"].min())
    max_val = float(df["Attendance_Percentage"].max())
    attend_range = st.slider("Attendance Percentage", min_val, max_val, (min_val, max_val))

# Main content

# --- Correlation & Performance Visuals ---
st.header("Correlation between Student Habits and Exam Performance")

# Scatter plots
scatter_pairs = [
    ("Study_Hours_Per_Day", "Exam_Score"),
    ("Social_Media_Hours", "Exam_Score"),
    ("Netflix_Hours", "Exam_Score"),
    ("Sleep_Hours", "Exam_Score"),
    ("Exercise_Frequency_Per_Week", "Exam_Score"),
    ("Attendance_Percentage", "Exam_Score")
]

for x, y in scatter_pairs:
    if x in filtered_df.columns and y in filtered_df.columns:
        fig, ax = plt.subplots()
        sns.scatterplot(data=filtered_df, x=x, y=y, ax=ax)
        ax.set_title(f"{x.replace('_', ' ')} vs. {y.replace('_', ' ')}")
        st.pyplot(fig)

# Box plots
box_plots = [
    "Mental_Health_Rating", "Diet_Quality", "Internet_Quality",
    "Part_Time_Job", "Extracurricular_Participation"
]

for cat in box_plots:
    if cat in filtered_df.columns:
        fig, ax = plt.subplots()
        sns.boxplot(data=filtered_df, x=cat, y="Exam_Score", ax=ax)
        ax.set_title(f"Exam Score by {cat.replace('_', ' ')}")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Correlation Heatmap
numeric_cols = [
    "Study_Hours_Per_Day", "Social_Media_Hours", "Netflix_Hours",
    "Sleep_Hours", "Exercise_Frequency_Per_Week", "Attendance_Percentage", "Exam_Score"
]

heatmap_data = filtered_df[numeric_cols].dropna()
if not heatmap_data.empty:
    fig, ax = plt.subplots()
    corr_matrix = heatmap_data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix of Habits and Exam Score")
    st.pyplot(fig)

# Raw data view
st.subheader("Filtered Data Preview")
st.dataframe(filtered_df.head(100), height=300)

# Add some explanatory text
st.markdown("""
### Insights:
- Explore how different fields of study compare in terms of employment outcomes
- Filter data using the sidebar to focus on specific student groups
- Hover over charts for detailed information
- Missing visualizations indicate required columns not found in the data
""")

# Add download button for filtered data
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download filtered data as CSV",
    data=csv,
    file_name="filtered_employability_data.csv",
    mime="text/csv"
)