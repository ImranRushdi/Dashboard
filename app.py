import streamlit as st
import pandas as pd
import plotly.express as px

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
    # Example: Convert columns to proper types
    df_cleaned = df[df['gender'] != 'Other']
    df_cleaned.to_csv('cleaned_file.csv', index=False)
    
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filter Data")

# Dynamic field selection based on available data
available_fields = df['Field_of_Study'].unique() if 'Field_of_Study' in df.columns else []
selected_fields = st.sidebar.multiselect(
    "Select Fields of Study",
    options=available_fields,
    default=available_fields[:min(3, len(available_fields))] if len(available_fields) > 0 else []
)

# GPA filter (if column exists)
if 'University_GPA' in df.columns:
    gpa_range = st.sidebar.slider(
        "University GPA Range",
        min_value=float(df['University_GPA'].min()),
        max_value=float(df['University_GPA'].max()),
        value=(float(df['University_GPA'].min()), float(df['University_GPA'].max()))
    )
else:
    gpa_range = (0, 4)  # Default range if column doesn't exist

# Salary filter (if column exists)
if 'Starting_Salary' in df.columns:
    salary_range = st.sidebar.slider(
        "Starting Salary Range ($)",
        min_value=int(df['Starting_Salary'].min()),
        max_value=int(df['Starting_Salary'].max()),
        value=(int(df['Starting_Salary'].min()), int(df['Starting_Salary'].max()))
    )
else:
    salary_range = (0, 100000)  # Default range if column doesn't exist

# Gender filter (if column exists)
if 'Gender' in df.columns:
    gender_options = ['All'] + list(df['Gender'].unique())
    gender_filter = st.sidebar.radio(
        "Gender",
        options=gender_options,
        index=0
    )
else:
    gender_filter = 'All'

# Apply filters
filtered_df = df.copy()

if len(selected_fields) > 0 and 'Field_of_Study' in df.columns:
    filtered_df = filtered_df[filtered_df['Field_of_Study'].isin(selected_fields)]

if 'University_GPA' in df.columns:
    filtered_df = filtered_df[
        (filtered_df['University_GPA'] >= gpa_range[0]) &
        (filtered_df['University_GPA'] <= gpa_range[1])
    ]

if 'Starting_Salary' in df.columns:
    filtered_df = filtered_df[
        (filtered_df['Starting_Salary'] >= salary_range[0]) &
        (filtered_df['Starting_Salary'] <= salary_range[1])
    ]

if gender_filter != 'All' and 'Gender' in df.columns:
    filtered_df = filtered_df[filtered_df['Gender'] == gender_filter]

# Main content
st.title("🎓 Postgraduate Employability in the US")
st.markdown("Analyzing the relationship between academic performance, field of study, and employment outcomes")

# KPI cards
st.subheader("Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if 'University_GPA' in filtered_df.columns:
        st.metric("Average CGPA", f"{filtered_df['University_GPA'].mean():.2f}")
    else:
        st.metric("Average CGPA", "N/A")

with col2:
    if 'Starting_Salary' in filtered_df.columns:
        st.metric("Average Salary", f"${filtered_df['Starting_Salary'].mean():,.0f}")
    else:
        st.metric("Average Salary", "N/A")

with col3:
    if 'Job_Offers' in filtered_df.columns:
        st.metric("Total Job Offers", filtered_df['Job_Offers'].sum())
    else:
        st.metric("Total Job Offers", "N/A")

with col4:
    if 'Internships_Completed' in filtered_df.columns:
        st.metric("Total Internships", filtered_df['Internships_Completed'].sum())
    else:
        st.metric("Total Internships", "N/A")

with col5:
    if 'Projects_Completed' in filtered_df.columns:
        st.metric("Total Projects", filtered_df['Projects_Completed'].sum())
    else:
        st.metric("Total Projects", "N/A")

# Charts Section
st.subheader("Data Visualizations")

# First Row of Charts
col1, col2 = st.columns(2)

with col1:
    if 'Field_of_Study' in filtered_df.columns:
        st.markdown("#### Percentage of Courses Majored in US")
        field_counts = filtered_df['Field_of_Study'].value_counts()
        fig = px.pie(field_counts, 
                     values=field_counts.values, 
                     names=field_counts.index,
                     hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Field_of_Study column not found in data")

with col2:
    if 'Field_of_Study' in filtered_df.columns and 'Starting_Salary' in filtered_df.columns:
        st.markdown("#### Annual Starting Salary by Field")
        fig = px.box(filtered_df, 
                     x='Field_of_Study', 
                     y='Starting_Salary',
                     color='Field_of_Study')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Required columns for salary analysis not found")

# Second Row of Charts
col1, col2 = st.columns(2)

with col1:
    if 'University_GPA' in filtered_df.columns and 'Job_Offers' in filtered_df.columns:
        st.markdown("#### University GPA vs. Job Offers")
        fig = px.scatter(filtered_df, 
                         x='University_GPA', 
                         y='Job_Offers',
                         color='Field_of_Study' if 'Field_of_Study' in filtered_df.columns else None,
                         trendline="lowess")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Required columns for GPA vs Job Offers analysis not found")

with col2:
    if 'SAT_Score' in filtered_df.columns:
        st.markdown("#### SAT Score Distribution")
        fig = px.histogram(filtered_df, 
                           x='SAT_Score', 
                           nbins=20,
                           color='Field_of_Study' if 'Field_of_Study' in filtered_df.columns else None)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("SAT_Score column not found in data")

# Third Row of Charts
col1, col2 = st.columns(2)

with col1:
    if 'Internships_Completed' in filtered_df.columns and 'Job_Offers' in filtered_df.columns:
        st.markdown("#### Internships vs. Job Offers")
        fig = px.scatter(filtered_df, 
                         x='Internships_Completed', 
                         y='Job_Offers',
                         size='Starting_Salary' if 'Starting_Salary' in filtered_df.columns else None,
                         color='Field_of_Study' if 'Field_of_Study' in filtered_df.columns else None)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Required columns for internships analysis not found")

with col2:
    if 'Starting_Salary' in filtered_df.columns:
        st.markdown("#### Salary Distribution by Field")
        fig = px.histogram(filtered_df, 
                           x='Starting_Salary', 
                           color='Field_of_Study' if 'Field_of_Study' in filtered_df.columns else None,
                           nbins=20,
                           barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Starting_Salary column not found in data")

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