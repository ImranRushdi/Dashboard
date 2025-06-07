import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.tree import DecisionTreeClassifier, plot_tree


# Set page config
st.set_page_config(
    page_title="Students Habits VS. Performance", 
    page_icon="🎓", 
    layout="wide"
)
# Inject custom CSS for digital-style background and glowing KPI text
st.markdown("""
    <style>
    /* Digital background */
    .stApp {
        background-color: #0f1117;
        background-image: linear-gradient(to bottom, #0f1117, #1a1f2e);
        color: #f5f5f5;
    }

    /* Glowing white KPI text */
    .element-container:has(.stMetric) {
        background-color: #1a1f2e;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
    }

    .stMetric label, .stMetric div {
        color: #ffffff !important;
        text-shadow: 0 0 0px #ffffff, 0 0 5px #ffffff;
    }

    /* Buttons and sliders */
    .stButton>button, .stSlider .st-cq {
        background-color: #1e2a38 !important;
        color: #00ffe7 !important;
        border: 1px solid #00ffe7 !important;
        border-radius: 8px;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #00ffe7 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Set dark theme for matplotlib
plt.style.use('dark_background')
sns.set_style("darkgrid", {
    "axes.facecolor": "#0f1117",
    "figure.facecolor": "#0f1117",
    "grid.color": "#1a1f2e"
})

# Customize plot colors to match your theme
plot_accent_color = "#00ffe7"  # Matches your accent color
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[plot_accent_color, "#ff6b6b", "#5f27cd"])
plt.rcParams['axes.edgecolor'] = "#f5f5f5"
plt.rcParams['axes.labelcolor'] = "#f5f5f5"
plt.rcParams['text.color'] = "#f5f5f5"
plt.rcParams['xtick.color'] = "#f5f5f5"
plt.rcParams['ytick.color'] = "#f5f5f5"

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
st.title("🎓 Students Habits VS. Performance")
st.markdown("Analyzing the correlation of students habits and their academic performance in recent examination.")

# Initialize session state for selected filters
if "selected_filters" not in st.session_state:
    st.session_state.selected_filters = []

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
        st.metric("📱 Avg. Social Media Hours", f"{df['Social_Media_Hours'].mean():.2f} hrs")
    else:
        st.metric("📱 Avg. Social Media Hours", "N/A")
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
# Remove the image coordinates import at top
# from streamlit_image_coordinates import streamlit_image_coordinates  # Remove this line

# --- Replace the image filter section with this branch-like slicer ---
st.subheader("🌿 Branch Selector - Choose Two Variables to Explore")

# Create two columns for branch selection
col1, col2 = st.columns(2)

# First branch selector
with col1:
    st.markdown("**Main Branch**")
    branch1 = st.selectbox(
        "Select first variable:",
        options=[
            "Study_Hours_Per_Day", "Social_Media_Hours", "Netflix_Hours",
            "Sleep_Hours", "Exercise_Frequency_Per_Week", "Attendance_Percentage",
            "Diet_Quality", "Parental_Education_Level", "Internet_Quality",
            "Mental_Health_Rating", "Extracurricular_Participation"
        ],
        format_func=lambda x: x.replace("_", " ").title(),
        key="branch1"
    )

# Second branch selector
with col2:
    st.markdown("**Secondary Branch**")
    branch2 = st.selectbox(
        "Select second variable:",
        options=[
            "Study_Hours_Per_Day", "Social_Media_Hours", "Netflix_Hours",
            "Sleep_Hours", "Exercise_Frequency_Per_Week", "Attendance_Percentage",
            "Diet_Quality", "Parental_Education_Level", "Internet_Quality",
            "Mental_Health_Rating", "Extracurricular_Participation", "Exam_Score"
        ],
        format_func=lambda x: x.replace("_", " ").title(),
        key="branch2"
    )

# Add some branch-like visual styling
st.markdown("""
<style>
    /* Branch connector visualization */
    .branch-connector {
        display: flex;
        justify-content: center;
        margin: 10px 0;
    }
    .branch-node {
        width: 15px;
        height: 15px;
        background-color: #00ffe7;
        border-radius: 50%;
        margin: 0 10px;
    }
    .branch-line {
        width: 100px;
        height: 2px;
        background-color: #00ffe7;
        margin: 6px 0;
    }
</style>
<div class="branch-connector">
    <div class="branch-node"></div>
    <div class="branch-line"></div>
    <div class="branch-node"></div>
</div>
""", unsafe_allow_html=True)


# --- Update the visualization section to use the selected branches ---
if branch1 and branch2:
    st.subheader(f"Relationship between {branch1.replace('_', ' ')} and {branch2.replace('_', ' ')}")
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0f1117')
    
    # Scatter plot
    sns.scatterplot(
        data=filtered_df,
        x=branch1,
        y=branch2,
        ax=ax1,
        color='#00ffe7',
        alpha=0.7
    )
    ax1.set_title(f"Scatter Plot", color='#00ffe7')
    ax1.set_facecolor('#0f1117')
    ax1.grid(color='#1a1f2e', linestyle='--', linewidth=0.5)
    
    # Regression plot
    sns.regplot(
        data=filtered_df,
        x=branch1,
        y=branch2,
        ax=ax2,
        scatter_kws={'color': '#00ffe7', 'alpha': 0.5},
        line_kws={'color': '#ff6b6b', 'linewidth': 2}
    )
    ax2.set_title(f"Regression Line", color='#00ffe7')
    ax2.set_facecolor('#0f1117')
    ax2.grid(color='#1a1f2e', linestyle='--', linewidth=0.5)
    
    # Style both axes
    for ax in [ax1, ax2]:
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
    
    st.pyplot(fig)
    
    # Calculate and display correlation
    correlation = filtered_df[[branch1, branch2]].corr().iloc[0,1]
    st.metric(
        "Correlation Strength",
        f"{correlation:.2f}",
        help="Values closer to 1 or -1 indicate stronger relationships"
    )


# --- Correlation & Performance Visuals ---
st.header("Correlation between Student Habits and Exam Performance")

# If 2 filters selected, show graph
if len(st.session_state.selected_filters) == 2:
    x_col, y_col = st.session_state.selected_filters
    if x_col in filtered_df.columns and y_col in filtered_df.columns:
        st.subheader(f"Relationship between {label_map[x_col]} and {label_map[y_col]}")
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter plot
        sns.scatterplot(data=filtered_df, x=x_col, y=y_col, ax=ax1)
        ax1.set_title(f"{label_map[x_col]} vs. {label_map[y_col]}")
        
        # Regression plot
        sns.regplot(data=filtered_df, x=x_col, y=y_col, ax=ax2)
        ax2.set_title(f"Regression: {label_map[x_col]} vs. {label_map[y_col]}")
        
        st.pyplot(fig)
        
        # Calculate correlation
        correlation = filtered_df[[x_col, y_col]].corr().iloc[0,1]
        st.metric("Correlation Coefficient", f"{correlation:.2f}")

# Graph selection filter
st.subheader("Select Visualization")

# Define all available visualizations
visualization_options = {
    "Study Hours vs Exam Score": ("Study_Hours_Per_Day", "Exam_Score", "scatter"),
    "Social Media vs Exam Score": ("Social_Media_Hours", "Exam_Score", "scatter"),
    "Netflix Hours vs Exam Score": ("Netflix_Hours", "Exam_Score", "scatter"),
    "Sleep Hours vs Exam Score": ("Sleep_Hours", "Exam_Score", "scatter"),
    "Attendance vs Exam Score": ("Attendance_Percentage", "Exam_Score", "scatter"),
    "Mental Health vs Exam Score": ("Mental_Health_Rating", "Exam_Score", "box"),
    "Diet Quality vs Exam Score": ("Diet_Quality", "Exam_Score", "box"),
    "Internet Quality vs Exam Score": ("Internet_Quality", "Exam_Score", "box"),
    "Part-Time Job vs Exam Score": ("Part_Time_Job", "Exam_Score", "box"),
    "Exercise Frequency vs Exam Score": ("Exercise_Frequency_Per_Week", "Exam_Score", "box"),
    "Extracurriculars vs Exam Score": ("Extracurricular_Participation", "Exam_Score", "box")
}

# Create selection dropdown
selected_viz = st.selectbox("Choose a visualization:", list(visualization_options.keys()))

# Get the selected visualization parameters
x_col, y_col, viz_type = visualization_options[selected_viz]

# Display the selected visualization
fig, ax = plt.subplots(figsize=(8, 5))

if viz_type == "scatter":
    sns.scatterplot(data=filtered_df, x=x_col, y=y_col, ax=ax)
    ax.set_title(f"{x_col.replace('_', ' ')} vs. {y_col.replace('_', ' ')}")
else:  # box plot
    sns.boxplot(data=filtered_df, x=x_col, y=y_col, ax=ax)
    ax.set_title(f"Exam Score by {x_col.replace('_', ' ')}")
    plt.xticks(rotation=45)

st.pyplot(fig)
# Correlation Heatmap (always show)
st.subheader("Correlation Matrix")
numeric_cols = [
    "Study_Hours_Per_Day", "Social_Media_Hours", "Netflix_Hours",
    "Sleep_Hours", "Exercise_Frequency_Per_Week", "Attendance_Percentage", "Exam_Score"
]

heatmap_data = filtered_df[numeric_cols].dropna()
if not heatmap_data.empty:
   # Create the figure with dark background
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0f1117')

    # Calculate correlation matrix
    corr_matrix = heatmap_data.corr()

    # Create a mask for correlations below 0.83
    text_colors = np.where(np.abs(corr_matrix.values) < 0.82, "black", "white")

    # Create heatmap with custom styling
    heatmap = sns.heatmap(
        corr_matrix, 
         annot=True, 
        cmap="coolwarm", 
        ax=ax,
        annot_kws={
             "color": "white",  # Default color (will be overridden)
            "fontsize": 10
        },
        linewidths=0.5,
        linecolor='#1a1f2e',
        vmin=-1, vmax=1
    )

    # Manually set text colors based on correlation strength
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            text = heatmap.texts[i * len(corr_matrix) + j]
            text.set_color(text_colors[i, j])

    # Customize the color bar
    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    # Style the title and axes
    ax.set_title(
        "Correlation Matrix of Habits and Exam Score", 
        color='#00ffe7',
        pad=20,
        fontsize=14
    )

    ax.tick_params(
        axis='both',
        which='both',
        colors='white',
        length=0
    )

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('#1a1f2e')

    st.pyplot(fig)

# Raw data view
st.subheader("Filtered Data Preview")
st.dataframe(filtered_df.head(100), height=300)

# Add some explanatory text
st.markdown("""
### Insights:
- Click on the branches in the tree image above to select two variables to compare
- The dashboard will automatically show the relationship between your selected variables
- Use the sidebar filters to focus on specific student groups
- Hover over charts for detailed information
""")

# Add download button for filtered data
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download filtered data as CSV",
    data=csv,
    file_name="student_habits_performance.csv",
    mime="text/csv"
)