import base64

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import plotly.express as px
import numpy as np
import plotly.graph_objects as go



# Set page configuration
st.set_page_config(layout="wide")

logo_path = "amazon_logo.jpg"

# Encode the image to base64
with open(logo_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Use markdown to add the logo with custom CSS
st.markdown(
    f"""
    <style>
    .logo {{
        position: absolute;
        top: -30px;
        right: 140px;
        width: 0.5px;  # Adjust the width as needed
        height: 10px;
    }}
    </style>
    <div class="logo">
        <img src="data:image/png;base64,{encoded_string}" alt="Logo">
    </div>
    """,
    unsafe_allow_html=True
)


# Load and combine all CSV files into a single DataFrame
@st.cache_data

# Function to extract and load data from a zip file
def load_data_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Extract all the files
        z.extractall()
        # Get the list of extracted files
        extracted_files = z.namelist()

    # Assuming there is only one CSV file in the zip
    csv_file = [file for file in extracted_files if file.endswith('.csv')][0]

    # Load the CSV file into a DataFrame
    all_data = pd.read_csv(csv_file)

    return all_data


def plot_histogram(df, metric, title):
    fig = px.histogram(df, x=metric, nbins=30, histnorm='density',
                       labels={'x': metric},
                       title=title)
    fig.update_layout(
        xaxis_title=metric,
        yaxis_title='Density',
        bargap=0.5,  # Gap between bars of adjacent location coordinates
        showlegend=False  # Hide legend for simplicity
    )
    return fig

zip_path = 'amazon_consolidated_data.zip'

# Load data from the ZIP file
all_data = load_data_from_zip(zip_path)
all_data = all_data.loc[:, ~all_data.columns.str.startswith('Unnamed')]


# Convert ratings and no_of_ratings to numeric and handle non-numeric values
all_data['ratings'] = pd.to_numeric(all_data['ratings'], errors='coerce')
all_data['no_of_ratings'] = pd.to_numeric(all_data['no_of_ratings'], errors='coerce')
all_data['discount_price'] = all_data['discount_price'].replace('[₹,]', '', regex=True).astype(float)
all_data['discount_price'] = pd.to_numeric(all_data['discount_price'], errors='coerce')
all_data['actual_price'] = all_data['actual_price'].replace('[₹,]', '', regex=True).astype(float)
all_data['actual_price'] = pd.to_numeric(all_data['actual_price'], errors='coerce')


avg_rating = all_data['ratings'].mean()
avg_no_of_ratings = all_data['no_of_ratings'].mean()
all_data = all_data.rename(columns={'name': 'product_name'})

# Streamlit app
# Sidebar for navigation
st.sidebar.title('Navigation')
option = st.sidebar.radio('Select a page:', ['Project Overview', 'Data Analysis', 'Product Comparison', 'About'])

if option == 'Project Overview':
    # Centered and beautified title
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='font-size: 36px; color: #0066cc;'>Amazon Sales Analytics Hub</h1>
    </div>
    """, unsafe_allow_html=True)

    st.header("Project Overview")

    st.write("""
        The Amazon Sales Analytics Hub is a comprehensive platform designed to analyze and visualize sales data from Amazon. This project aims to provide real-time insights into customer preferences, pricing strategies, and product performance across diverse categories.

        ### Problem Statement
        Retailers on Amazon face challenges in understanding customer preferences, optimizing pricing strategies, and evaluating product performance across diverse categories. Current methods often lack the capability to provide comprehensive real-time insights into sales data, making it difficult to make informed decisions. An interactive and user-friendly platform is needed to analyze and visualize sales data, helping retailers address these issues effectively.

        ### Objectives
        - **Compare Product Performance:** Examine the performance of various product categories.
        - **Analyze Pricing Strategies:** Evaluate pricing strategies and discount patterns.
        - **Review Distribution Analysis:** Analyze the distribution of product ratings and correlate the number of ratings with discounted prices.
        - **Dynamic Data Exploration:** Enable dynamic data exploration through an interactive platform.

        ### Key Components
        1. **Data Collection:** Utilize the Amazon Products Sales Dataset from Kaggle, which includes comprehensive sales data for various product categories on Amazon.
        2. **Data Processing:** Clean, preprocess, and analyze the data using Python with Pandas.
        3. **Visualization:** Create interactive visualizations using Matplotlib, Seaborn, and Tableau.
        4. **User Interface:** Develop a user-friendly interface with Streamlit for interactive application development and deploy it on AWS.

        ### Technologies Used
        - **Programming Languages:** Python
        - **Data Analysis:** Pandas
        - **Visualization:** Matplotlib, Seaborn, Tableau
        - **Interactive Application Development:** Streamlit
        - **Cloud Hosting:** AWS

        This platform empowers users with valuable insights derived from sales data, combining data analysis, pricing strategy evaluation, and visualization techniques for comprehensive understanding and decision-making.
        """)

elif option == 'Data Analysis':
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='font-size: 36px; color: #0066cc;'>Data Analysis</h1>
    </div>
    """, unsafe_allow_html=True)

    # Basic statistics and data overview
    st.subheader('Basic Statistics and Data Overview')

    # Separate categorical and numerical columns
    categorical_cols = ['product_name', 'main_category', 'sub_category', 'image']
    numerical_cols = [col for col in all_data.columns if col not in categorical_cols]

    # Display basic statistics for categorical data
    st.write('#### Categorical Data Overview')
    categorical_stats = all_data[categorical_cols].describe(include='all').transpose()
    st.dataframe(categorical_stats.style.set_properties(**{'text-align': 'center'}).set_table_styles(
        [dict(selector='th', props=[('text-align', 'center')])]))

    # Display basic statistics for numerical data
    st.write('#### Numerical Data Overview')
    numerical_stats = all_data[numerical_cols].describe().transpose()
    st.dataframe(numerical_stats.style.format("{:.2f}").set_properties(**{'text-align': 'center'}).set_table_styles(
        [dict(selector='th', props=[('text-align', 'center')])]))

    ### Outlier Analysis for Actual Price
    st.write("")
    st.write("#### Outlier Analysis for Actual Price")
    st.write("Standard Deviation of Actual Price is 13550819.54 which indicates high possibilities of Outliers.")
    Q1 = all_data['actual_price'].quantile(0.25)
    Q3 = all_data['actual_price'].quantile(0.75)
    IQR = Q3 - Q1

    st.write("IQR of Actual Price - ", IQR)

    # Define the acceptable range
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers
    all_data = all_data[(all_data['actual_price'] >= lower_bound) & (all_data['actual_price'] <= upper_bound)]

    # Display summary statistics after filtering
    st.write("")
    st.markdown("<u>Statistics after the removal of outliers</u>", unsafe_allow_html=True)

    numerical_stats = all_data[numerical_cols].describe().transpose()
    st.dataframe(numerical_stats.style.format("{:.2f}").set_properties(**{'text-align': 'center'}).set_table_styles(
        [dict(selector='th', props=[('text-align', 'center')])]))

    # Missing values analysis
    st.write("")
    st.subheader('Missing Values Analysis')

    # Check for missing values
    missing_values = all_data.isnull().sum()

    # Convert missing values to a DataFrame
    missing_values_df = pd.DataFrame(missing_values, columns=['Missing Values']).transpose()

    # Display missing values analysis
    st.dataframe(missing_values_df)

    # Price Distribution Analysis
    st.write("")
    st.subheader('Price Distribution Analysis')

    metrics = ['discount_price', 'actual_price']  # Replace with actual metric column names in your dataset

    # Create a streamlit app to display plots side by side
    st.markdown("<h4 style='text-align: center;'>Histograms of Metrics</h4>", unsafe_allow_html=True)
    cols = st.columns(len(metrics))

    for i, metric in enumerate(metrics):
        with cols[i]:
            fig = plot_histogram(all_data, metric, f'Distribution of {metric.capitalize()}')
            st.plotly_chart(fig)

            # Calculate and display key metrics
            mean_value = all_data[metric].mean()
            median_value = all_data[metric].median()
            std_dev_value = all_data[metric].std()
            st.write(f"**Mean {metric.capitalize()}:** {mean_value:.2f}")
            st.write(f"**Median {metric.capitalize()}:** {median_value:.2f}")
            st.write(f"**Standard Deviation {metric.capitalize()}:** {std_dev_value:.2f}")


    ### Ratings Data Analysis ###
    metrics = ['ratings', 'no_of_ratings']  # Replace with actual metric column names in your dataset

    # Create a streamlit app to display plots side by side
    cols = st.columns(len(metrics))

    for i, metric in enumerate(metrics):
        with cols[i]:
            fig = plot_histogram(all_data, metric, f'Distribution of {metric.capitalize()}')
            st.plotly_chart(fig)

            # Calculate and display key metrics
            mean_value = all_data[metric].mean()
            median_value = all_data[metric].median()
            std_dev_value = all_data[metric].std()
            st.write(f"**Mean {metric.capitalize()}:** {mean_value:.2f}")
            st.write(f"**Median {metric.capitalize()}:** {median_value:.2f}")
            st.write(f"**Standard Deviation {metric.capitalize()}:** {std_dev_value:.2f}")

    st.subheader('')


    # Category-wise Analysis
    st.write("")
    st.subheader('Category-wise Analysis')

    # Counting the number of products per category
    category_counts = all_data['main_category'].value_counts()

    # Generate colors dynamically based on number of categories
    num_categories = len(category_counts)
    colors = px.colors.qualitative.Plotly[:num_categories]  # Using Plotly's qualitative color palette

    # Create Plotly bar chart
    fig = go.Figure(go.Bar(
        x=category_counts.index,
        y=category_counts.values,
        marker_color=colors  # Assign colors based on the dynamically generated list
    ))

    fig.update_layout(
        title='Number of Products per Category',
        xaxis=dict(title='Category'),
        yaxis=dict(title='Number of Products'),
        xaxis_tickangle=-45  # Rotate x-axis labels
    )
    st.plotly_chart(fig)

    # Correlation Analysis
    # Calculate correlation matrix
    correlation_matrix = all_data[['discount_price', 'actual_price', 'ratings', 'no_of_ratings']].corr()

    # Create Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',  # Choose colorscale
        zmin=-1, zmax=1,  # Set min and max values for color scale
        colorbar=dict(title='Correlation')
    ))

    fig.update_layout(
        xaxis=dict(title='Metrics'),
        yaxis=dict(title='Metrics'),
        title=dict(
        text='Correlation Heatmap',
        font=dict(size=28)  # Increase the size of the title
    )
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)


elif option == 'Product Comparison':
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='font-size: 36px; color: #0066cc;'>Product Comparison</h1>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar product selection
    st.sidebar.subheader('Select Products for Comparison')

    # Allow selection from any main and sub category
    main_categories = all_data['main_category'].unique()
    selected_main_categories = st.sidebar.multiselect('Select Main Categories:', main_categories)

    if selected_main_categories:
        sub_categories = all_data[all_data['main_category'].isin(selected_main_categories)]['sub_category'].unique()
        selected_sub_categories = st.sidebar.multiselect('Select Sub Categories:', sub_categories)

        if selected_sub_categories:
            selected_data = all_data[(all_data['main_category'].isin(selected_main_categories)) &
                                     (all_data['sub_category'].isin(selected_sub_categories))]

            # Calculate discount percentage
            selected_data['discount_percentage'] = ((selected_data['actual_price'] - selected_data['discount_price']) /
                                                    selected_data['actual_price']) * 100

            # Compute necessary statistics for selected products
            product_cat_stats = selected_data.groupby('sub_category').agg({
                'product_name' : ['count'],
                'ratings': ['mean', 'median', 'min', 'max', 'std'],
                'no_of_ratings': 'sum',
                'discount_price': ['mean', 'median', 'min', 'max', 'std'],
                'actual_price': ['mean', 'median', 'min', 'max', 'std'],
                'discount_percentage': 'mean'
            }).reset_index()

            # Flatten MultiIndex columns
            product_cat_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in
                                         product_cat_stats.columns.values]
            product_cat_stats.rename(columns={'product_name_count': 'product_count'}, inplace=True)


            # Display selected product information in a table
            st.subheader('Selected Product Category Information')
            st.dataframe(product_cat_stats)


            # Create figure for comparisons
            fig_comparisons = go.Figure()

            # Add bar plots for each metric comparison
            for i, metric in enumerate(
                    ['discount_price_mean', 'actual_price_mean']):
                fig_comparisons.add_trace(go.Bar(x=product_cat_stats['sub_category'], y=product_cat_stats[metric],
                                                 name=metric.split('_')[0].capitalize()))

            # Update layout for comparisons
            fig_comparisons.update_layout(barmode='group', title='Metrics Comparisons',
                                          xaxis_title='Sub Category', yaxis_title='Value')

            # Display comparisons
            st.plotly_chart(fig_comparisons)

            # Example using Plotly
            fig_box = px.box(selected_data, x='sub_category', y='ratings', title='Ratings Distribution by Sub Category')
            st.plotly_chart(fig_box)

            # Example using Plotly
            fig_scatter = px.scatter(selected_data, x='no_of_ratings', y='ratings', color='sub_category',
                                     title='Ratings vs Number of Ratings')
            st.plotly_chart(fig_scatter)

            # Create a row for four pie charts
            col1, col2 = st.columns(2)

            # Create pie charts for each metric
            fig_pie1 = px.pie(product_cat_stats, values='no_of_ratings_sum', names='sub_category',
                              title='Number of Ratings Distribution')
            with col1:
                st.plotly_chart(fig_pie1)

            fig_pie2 = px.pie(product_cat_stats, values='ratings_mean', names='sub_category',
                              title='Ratings Distribution')
            with col2:
                st.plotly_chart(fig_pie2)



            fig_pie3 = px.pie(product_cat_stats, values='actual_price_mean', names='sub_category',
                              title='Actual Price Distribution')

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_pie3)

            fig_pie4 = px.pie(product_cat_stats, values='discount_price_mean', names='sub_category',
                              title='Discount Price Distribution')
            with col2:
                st.plotly_chart(fig_pie4)

        else:
            st.write("Please select subcategories from the sidebar.")
    else:
        st.write("Please select main categories from the sidebar.")

elif option == 'About':
    st.header('About')

    st.write("""
    The Amazon Sales Analytics Hub is a cutting-edge platform designed to equip Amazon retailers with real-time insights into customer preferences, pricing strategies, and product performance. This initiative tackles the challenges of informed decision-making by utilizing advanced data science and visualization techniques.

    Through dynamic data exploration and thorough analysis, the platform enables users to comprehend customer behavior, fine-tune pricing strategies, and assess product performance across various categories.

    Developed as part of the BANA 8083 - 006 MS Capstone at the University of Cincinnati, this project aims to offer Amazon retailers actionable insights via a comprehensive, interactive platform for real-time sales data analysis and visualization.
    """)

    st.subheader("Contact")

    st.write("**Name:** Bharath Marturu")
    st.write("**Email Address:** marturbh@mail.uc.edu")
    st.write("**LinkedIn:** [Bharath Marturu](https://www.linkedin.com/in/bharath-marturu-6b08b3a8/)")
    st.write("**Phone Number:** +1 (513) 399 0530")
