import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import urllib.parse
from matplotlib import cycler
from scipy.stats import f_oneway, chi2_contingency, spearmanr


st.set_page_config(
    page_title="Students Habits VS. Performance", 
    page_icon="🎓", 
    layout="wide"
)

try:
    with open(r"C:\Users\user\Documents\GitHub\Dashboard\back.svg", "r", encoding="utf-8") as f:
        svg_content = f.read()
except Exception as e:
    st.error(f"Error reading SVG file: {e}")
    svg_content = None

if svg_content:
    encoded_svg = urllib.parse.quote(svg_content)
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/svg+xml;utf8,{encoded_svg}");
        background-repeat: no-repeat;
        background-position: center;
        background-size: cover;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

st.title("🎓 Students Habits VS. Performance")
st.markdown("Analyzing the correlation of students' habits and their academic performance.")

st.markdown("""
    <style>

    .element-container:has(.stMetric) {
        background-color: #0E1117;
        color: #0E1117;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
    }

    .stMetric label, .stMetric div {
        color: #3D3D3D !important;
    }
    
    .stMetric > div > div {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        text-shadow: 0 0 5px #DEDCD8;
    }

    .stButton>button, .stSlider .st-cq {
        background-color: #0E1117 !important;
        color: #4285F4 !important;
        border: 1px solid #F0F2F6 !important;
        border-radius: 8px;
    }

    body, div, span, p, a, li, ul {
    color: #DEDCD8 !important;
    }        
    
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

plt.style.use('default')
sns.set_style("whitegrid", {
    "axes.facecolor": "#262730",
    "figure.facecolor": "#0E1117",
    "grid.color": "#1A1C23"
})

plt.rcParams['axes.prop_cycle'] = cycler(color=["#FF4B4B"])
plt.rcParams['axes.edgecolor'] = "#FFFFFF"
plt.rcParams['axes.labelcolor'] = "#FFFFFF"
plt.rcParams['text.color'] = "#FFFFFF"
plt.rcParams['xtick.color'] = "#FFFFFF"
plt.rcParams['ytick.color'] = "#FFFFFF"

# Load data function with caching
@st.cache_data
def load_data():
    # Load data from CSV file
    df = pd.read_csv(r"C:\Users\user\Documents\GitHub\Dashboard\student_habits_performance.csv")
    
    # Clean Data
    df_cleaned = df[df['gender'] != 'Other']
    df_cleaned.to_csv('cleaned_file.csv', index=False)
    
    # Rename columns for consistency
    df_cleaned.columns = [
        "Student_ID", "Age", "Gender", "Study_Hours_Per_Day", "Social_Media_Hours",
        "Netflix_Hours", "Part_Time_Job", "Attendance_Percentage", "Sleep_Hours",
        "Diet_Quality", "Exercise_Frequency_Per_Week", "Parental_Education_Level",
        "Internet_Quality", "Mental_Health_Rating", "Extracurricular_Participation",
        "Exam_Score"
    ]

    # Ensure hour-based columns are float
    df_cleaned["Study_Hours_Per_Day"] = df_cleaned["Study_Hours_Per_Day"].astype(float)
    df_cleaned["Social_Media_Hours"] = df_cleaned["Social_Media_Hours"].astype(float)
    df_cleaned["Netflix_Hours"] = df_cleaned["Netflix_Hours"].astype(float)
    df_cleaned["Sleep_Hours"] = df_cleaned["Sleep_Hours"].astype(float)
    df_cleaned["Exercise_Frequency_Per_Week"] = df_cleaned["Exercise_Frequency_Per_Week"].astype(float)

    return df_cleaned

df = load_data()

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
    st.metric("👨‍👩‍🎓 Common Parental Education", parent_mode[0] if not parent_mode.empty else "N/A")

with col4:
    st.metric("💤 Avg. Sleep Hours", f"{df['Sleep_Hours'].mean():.2f} hrs")
    
    exercise_count = df[df['Exercise_Frequency_Per_Week'] > 0].shape[0]
    percent_exercising = (exercise_count / df.shape[0]) * 100 if df.shape[0] else 0
    st.metric("💪 % Regular Exercise", f"{percent_exercising:.1f}%")

with col5:
    part_time_count = df[df['Part_Time_Job'] == 'Yes'].shape[0]
    part_time_pct = (part_time_count / df.shape[0]) * 100 
    st.metric("📱 Avg. Social Media Hours", f"{df['Social_Media_Hours'].mean():.2f} hrs")
    
    internet_mode = df['Internet_Quality'].mode()
    st.metric("🌐 Most Common Internet Quality", internet_mode[0] if not internet_mode.empty else "N/A")

# Sidebar filters
st.sidebar.header("Filter Data")

# Student ID filter
Student_ID_List = df['Student_ID'].unique().tolist()
Student_ID_List.sort()

Search_ID = st.sidebar.multiselect(
    "Select Student ID(s)",
    options=Student_ID_List,
    help="Disclaimer: If the data for the ID is not available, it may have been removed because they are identified as 'Other'."
)

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

# Apply filters
filtered_df = df.copy()

if Search_ID:
    filtered_df = filtered_df[filtered_df["Student_ID"].isin(Search_ID)]

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

# Branch Selector
st.subheader("🌿 Branch Selector - Choose Two Variables to Explore")

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

# Visualization Logic with Proper Categorical Handling
if branch1 and branch2:
    st.subheader(f"Relationship between {branch1.replace('_', ' ').title()} and {branch2.replace('_', ' ').title()}")
    
    # Determine variable types
    branch1_is_categorical = filtered_df[branch1].dtype == 'object'
    branch2_is_categorical = filtered_df[branch2].dtype == 'object'
    
    if not branch1_is_categorical and not branch2_is_categorical:
        # Both numeric - scatter plot + regression
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter plot
        sns.scatterplot(data=filtered_df, x=branch1, y=branch2, ax=ax1, color='#4CAF50')
        ax1.set_title(f"Scatter Plot")
        
        # Regression plot
        sns.regplot(data=filtered_df, x=branch1, y=branch2, ax=ax2, 
                   scatter_kws={'color': '#4CAF50'}, 
                   line_kws={'color': '#FF5722'})
        ax2.set_title(f"Regression Line")
        
        # Calculate Pearson correlation
        correlation = filtered_df[[branch1, branch2]].corr().iloc[0,1]
        corr_text = f"Pearson r = {correlation:.2f}"
        
    elif branch1_is_categorical and not branch2_is_categorical:
        # Categorical vs numeric - box plot + swarm plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Box plot
        sns.boxplot(data=filtered_df, x=branch1, y=branch2, ax=ax1, palette="viridis")
        ax1.set_title(f"Distribution by {branch1.replace('_', ' ').title()}")
        ax1.tick_params(axis='x', rotation=45)
        
        # Swarm plot
        sns.swarmplot(data=filtered_df, x=branch1, y=branch2, ax=ax2, color='#4CAF50')
        ax2.set_title(f"Value Distribution")
        ax2.tick_params(axis='x', rotation=45)
        
        # Calculate ANOVA p-value
        groups = filtered_df.groupby(branch1)[branch2].apply(list)
        if len(groups) >= 2:
            _, p_value = f_oneway(*groups)
            corr_text = f"ANOVA p-value = {p_value:.4f}"
        else:
            corr_text = "Insufficient groups for ANOVA"
            
    elif not branch1_is_categorical and branch2_is_categorical:
        # Numeric vs categorical - flip the axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Box plot
        sns.boxplot(data=filtered_df, x=branch2, y=branch1, ax=ax1, palette="viridis")
        ax1.set_title(f"Distribution by {branch2.replace('_', ' ').title()}")
        ax1.tick_params(axis='x', rotation=45)
        
        # Swarm plot
        sns.swarmplot(data=filtered_df, x=branch2, y=branch1, ax=ax2, color='#4CAF50')
        ax2.set_title(f"Value Distribution")
        ax2.tick_params(axis='x', rotation=45)
        
        # Calculate ANOVA p-value
        groups = filtered_df.groupby(branch2)[branch1].apply(list)
        if len(groups) >= 2:
            _, p_value = f_oneway(*groups)
            corr_text = f"ANOVA p-value = {p_value:.4f}"
        else:
            corr_text = "Insufficient groups for ANOVA"
            
    else:
        # Both categorical - contingency table and heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Count plot
        sns.countplot(data=filtered_df, x=branch1, hue=branch2, ax=ax1, palette="viridis")
        ax1.set_title(f"Count by {branch1.replace('_', ' ').title()}")
        ax1.legend(title=branch2.replace('_', ' ').title())
        ax1.tick_params(axis='x', rotation=45)
        
        # Heatmap
        crosstab = pd.crosstab(filtered_df[branch1], filtered_df[branch2])
        sns.heatmap(crosstab, annot=True, fmt='d', cmap='viridis', ax=ax2)
        ax2.set_title(f"Contingency Table")
        
        # Chi-square test
        chi2, p, _, _ = chi2_contingency(crosstab)
        corr_text = f"Chi-square p-value = {p:.4f}"
    
    st.pyplot(fig)
    st.metric("Statistical Relationship", corr_text)

# Graph selection filter with correlation
st.subheader("How Each Factor Affect Academic Performance")

# Define all available visualizations with correlation types
visualization_options = {
    "Study Hours": ("Study_Hours_Per_Day", "Exam_Score", "scatter", "pearson"),
    "Social Media Hours": ("Social_Media_Hours", "Exam_Score", "scatter", "pearson"),
    "Netflix Hours": ("Netflix_Hours", "Exam_Score", "scatter", "pearson"),
    "Sleep Hours": ("Sleep_Hours", "Exam_Score", "scatter", "pearson"),
    "Attendance Percentage": ("Attendance_Percentage", "Exam_Score", "scatter", "pearson"),
    "Mental Health Rating": ("Mental_Health_Rating", "Exam_Score", "box", "spearman"),
    "Diet Quality": ("Diet_Quality", "Exam_Score", "box", "spearman"),
    "Internet Quality": ("Internet_Quality", "Exam_Score", "box", "spearman"),
    "Part-Time Job": ("Part_Time_Job", "Exam_Score", "box", "anova"),
    "Exercise Frequency": ("Exercise_Frequency_Per_Week", "Exam_Score", "box", "spearman"),
    "Extracurriculars": ("Extracurricular_Participation", "Exam_Score", "box", "anova")
}

# Create selection dropdown
selected_viz = st.selectbox("Choose a Factor:", list(visualization_options.keys()))

# Get the selected visualization parameters
x_col, y_col, viz_type, corr_type = visualization_options[selected_viz]

# Calculate correlation based on type
def calculate_correlation(data, x, y, method):
    # Handle string/categorical data
    if data[x].dtype == 'object':
        # For categorical variables, use ANOVA if >2 groups, t-test if 2 groups
        groups = data.groupby(x)[y].apply(list).values
        if len(groups) < 2:
            return None
        if method == "anova":
            if len(groups) == 2:
                # Use t-test for binary categories
                from scipy.stats import ttest_ind
                _, p_val = ttest_ind(groups[0], groups[1])
                return p_val
            else:
                _, p_val = f_oneway(*groups)
                return p_val
        else:
            # For non-ANOVA methods with strings, convert to codes
            data = data.copy()
            data[x] = pd.factorize(data[x])[0]
    
    # Handle numeric data
    try:
        if method == "pearson":
            return data[[x, y]].corr(method='pearson').iloc[0,1]
        elif method == "spearman":
            return spearmanr(data[x], data[y])[0]
        elif method == "anova":
            groups = [data[data[x] == val][y] for val in data[x].unique()]
            if len(groups) < 2:
                return None
            _, p_val = f_oneway(*groups)
            return p_val
    except:
        return None
    return None

# Interpretation note
if viz_type == "scatter":
    st.caption("Pearson correlation ranges from -1 (perfect negative) to +1 (perfect positive)")
elif viz_type == "box":
    if corr_type == "spearman":
        st.caption("Spearman correlation ranges from -1 (perfect negative) to +1 (perfect positive)")
    elif corr_type == "anova":
        st.caption("ANOVA p-value < 0.05 suggests significant difference between groups")

# Calculate correlation
correlation = calculate_correlation(filtered_df, x_col, y_col, corr_type)

# Display the selected visualization
fig, ax = plt.subplots(figsize=(8, 5))

if viz_type == "scatter":
    plot_data = filtered_df.copy()
    
    # Convert object (string) to numeric codes if necessary
    if plot_data[x_col].dtype == 'object':
        plot_data[x_col] = pd.factorize(plot_data[x_col])[0]

    sns.scatterplot(data=plot_data, x=x_col, y=y_col, ax=ax)
    title = f"{x_col.replace('_', ' ')} vs. {y_col.replace('_', ' ')}"
    if correlation is not None and not pd.isna(correlation):
        title += f"\nPearson r = {correlation:.2f}" if corr_type == "pearson" else f"\nCorrelation = {correlation:.2f}"
    ax.set_title(title)

elif viz_type == "box":
    sns.boxplot(data=filtered_df, x=x_col, y=y_col, ax=ax)
    title = f"Exam Score by {x_col.replace('_', ' ')}"
    if correlation is not None:
        if corr_type == "anova":
            title += f"\nANOVA p-value = {correlation:.4f}"
            if correlation < 0.05:
                title += " (significant)"
        else:
            title += f"\nSpearman ρ = {correlation:.2f}"
    ax.set_title(title)
    plt.xticks(rotation=45)

st.pyplot(fig)

# Correlation Heatmap
st.subheader("Summarization of Each Factor's Correlation Each Other")
numeric_df = filtered_df.select_dtypes(include='number')
corr_matrix = numeric_df.corr(method=corr_type if corr_type in ["pearson", "spearman"] else "pearson")

heatmap_fig, heatmap_ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True, ax=heatmap_ax)
heatmap_ax.set_title(f"{corr_type.capitalize()} Correlation Heatmap")

# Display the heatmap
st.pyplot(heatmap_fig)
                     
# Raw data view
st.subheader("Filtered Data Preview")
st.dataframe(filtered_df.head(100), height=300)

# Explanatory text
st.markdown("""
### Insights:
- Click on the branches to select two variables to compare
- The dashboard will automatically show the relationship between your selected variables
- Use the sidebar filters to focus on specific student groups
- Hover over charts for detailed information
""")

# Filtered Data Download
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download filtered data as CSV",
    data=csv,
    file_name="student_habits_performance.csv",
    mime="text/csv"
)