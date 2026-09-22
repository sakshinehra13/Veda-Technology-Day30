import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Advanced Data Analysis Tool",
    page_icon="📊",
    layout="wide"
)

# App Title & Description
st.title("📊 Advanced Data Analysis Dashboard & CLI Tool")
st.markdown("Upload your CSV dataset to perform summaries, filtering, grouping, statistical reporting, and data visualizations.")

# Sidebar File Uploader
st.sidebar.header("1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)

    df = load_data(uploaded_file)

    # Sidebar Navigation
    st.sidebar.header("2. Navigation")
    options = st.sidebar.radio(
        "Choose an action:", 
        ["Dataset Preview & Summary", "Statistical Analysis", "Data Filtering", "Group By & Aggregation", "Data Visualization"]
    )

    # --- TAB 1: PREVIEW & SUMMARY ---
    if options == "Dataset Preview & Summary":
        st.subheader("📋 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values Count", int(df.isnull().sum().sum()))

        st.markdown("---")
        st.subheader("🔍 Data Preview (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("---")
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Data Types")
            dtypes_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
            st.dataframe(dtypes_df, use_container_width=True)
            
        with col_right:
            st.subheader("Missing Values per Column")
            missing_df = pd.DataFrame(df.isnull().sum(), columns=["Missing Count"])
            st.dataframe(missing_df, use_container_width=True)

    # --- TAB 2: STATISTICAL ANALYSIS ---
    elif options == "Statistical Analysis":
        st.subheader("📈 Statistical Report")
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_cols:
            stat_option = st.selectbox("Select scope:", ["All Numeric Columns", "Specific Column"])
            
            if stat_option == "All Numeric Columns":
                st.dataframe(df.describe(), use_container_width=True)
            else:
                selected_col = st.selectbox("Choose column:", numeric_cols)
                st.write(df[selected_col].describe())
                
                # Extra stats
                col1, col2, col3 = st.columns(3)
                col1.metric("Median", round(df[selected_col].median(), 2))
                col2.metric("Variance", round(df[selected_col].var(), 2))
                col3.metric("Skewness", round(df[selected_col].skew(), 2))
        else:
            st.warning("No numeric columns found in the dataset for statistical reporting.")

    # --- TAB 3: DATA FILTERING ---
    elif options == "Data Filtering":
        st.subheader("🔎 Advanced Data Filtering")
        
        col_name = st.selectbox("Select column to filter by:", df.columns)
        unique_vals = df[col_name].dropna().unique()
        
        if df[col_name].dtype == 'object' or len(unique_vals) < 20:
            selected_vals = st.multiselect("Select value(s) to match:", options=list(unique_vals), default=list(unique_vals)[:min(5, len(unique_vals))])
            filtered_df = df[df[col_name].isin(selected_vals)]
        else:
            op = st.selectbox("Condition:", ["Equal to (==)", "Greater than (>)", "Less than (<)"])
            val = st.number_input("Value:", value=float(df[col_name].mean()))
            if op == "Equal to (==)":
                filtered_df = df[df[col_name] == val]
            elif op == "Greater than (>)":
                filtered_df = df[df[col_name] > val]
            else:
                filtered_df = df[df[col_name] < val]

        st.write(f"Showing **{len(filtered_df)}** matching rows out of {len(df)} total rows.")
        st.dataframe(filtered_df, use_container_width=True)

        # Download Filtered Data Option
        csv_export = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv_export,
            file_name='filtered_data.csv',
            mime='text/csv',
        )

    # --- TAB 4: GROUP BY & AGGREGATION ---
    elif options == "Group By & Aggregation":
        st.subheader("📑 Group By and Aggregate")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if categorical_cols and numeric_cols:
            group_col = st.selectbox("Group By Column:", categorical_cols)
            target_col = st.selectbox("Target Numeric Column:", numeric_cols)
            
            agg_funcs = st.multiselect("Aggregation Functions:", ["mean", "sum", "count", "min", "max"], default=["mean", "sum", "count"])
            
            if agg_funcs:
                grouped_df = df.groupby(group_col)[target_col].agg(agg_funcs).reset_index()
                st.dataframe(grouped_df, use_container_width=True)
                
                # Quick chart for grouped data
                if len(agg_funcs) > 0:
                    st.bar_chart(grouped_df.set_index(group_col)[agg_funcs[0]])
        else:
            st.warning("Dataset must contain both categorical and numeric columns for grouping operations.")

    # --- TAB 5: DATA VISUALIZATION ---
    elif options == "Data Visualization":
        st.subheader("📊 Data Visualizations")
        
        chart_type = st.selectbox("Select Chart Type", ["Histogram", "Scatter Plot", "Correlation Heatmap"])
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if chart_type == "Histogram":
            if numeric_cols:
                col = st.selectbox("Select Numeric Column", numeric_cols)
                bins = st.slider("Number of Bins", 5, 100, 20)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.histplot(df[col], bins=bins, kde=True, ax=ax, color='skyblue')
                st.pyplot(fig)
            else:
                st.warning("No numeric columns available.")
                
        elif chart_type == "Scatter Plot":
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("X-axis Column", numeric_cols, index=0)
                y_col = st.selectbox("Y-axis Column", numeric_cols, index=1)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, color='purple')
                st.pyplot(fig)
            else:
                st.warning("At least two numeric columns are required for a scatter plot.")
                
        elif chart_type == "Correlation Heatmap":
            if len(numeric_cols) >= 2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("At least two numeric columns are required for correlation analysis.")

else:
    st.info("👈 Please upload a CSV file from the sidebar to get started.")