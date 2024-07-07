import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page configuration
st.set_page_config(layout="wide")

# Path variables
extracted_files_path = 'C:/Users/bhara/OneDrive - University of Cincinnati/Capstone/Amazon Products Final/'
output_file_path = (
    'C:/Users/bhara/OneDrive - University of Cincinnati/Capstone/Amazon Products Final/amazon_consolidated_data.csv')


# Load and combine all CSV files into a single DataFrame
@st.cache_data
def load_data():
    extracted_files = os.listdir(extracted_files_path)
    dataframes = []
    for file in extracted_files:
        file_path = os.path.join(extracted_files_path, file)
        if file.endswith('.csv'):
            df = pd.read_csv(file_path)
            df['category'] = os.path.splitext(file)[0]
            dataframes.append(df)
    all_data = pd.concat(dataframes, ignore_index=True)
    all_data.to_csv(output_file_path, index=False)
    return all_data

all_data = load_data()

# Convert ratings and no_of_ratings to numeric and handle non-numeric values
all_data['ratings'] = pd.to_numeric(all_data['ratings'], errors='coerce')
all_data['no_of_ratings'] = pd.to_numeric(all_data['no_of_ratings'], errors='coerce')
all_data['discount_price'] = pd.to_numeric(all_data['discount_price'], errors='coerce')
all_data['actual_price'] = pd.to_numeric(all_data['actual_price'], errors='coerce')


avg_rating = all_data['ratings'].mean()
avg_no_of_ratings = all_data['no_of_ratings'].mean()
all_data = all_data.rename(columns={'name': 'product_name'})


# Streamlit app
st.title('Amazon Products Analysis')

# Sidebar for navigation
st.sidebar.title('Navigation')
option = st.sidebar.radio('Select a page:', ['Project Overview', 'Data Analysis', 'Product Comparison', 'About'])

if option == 'Project Overview':
    st.header('Project Overview')
    project_summary = """
    **Amazon Sales Analytics Hub - Exploring Product Data**

    The "Amazon Sales Analytics Hub - Exploring Product Data" project addresses the challenges faced by Amazon retailers in understanding customer preferences, optimizing pricing strategies, and evaluating product performance across diverse categories. It aims to provide a comprehensive, real-time interactive platform to analyze and visualize sales data, facilitating informed decision-making. The project's objectives include comparing product category performance, examining pricing strategies and discount patterns, analyzing product ratings distribution, and correlating ratings with discounted prices. Utilizing Python for data analysis, Scikit-learn for EDA, Matplotlib/Seaborn and Tableau for visualization, Streamlit for application development, and AWS for cloud hosting, the platform will enable users to upload sales data, gain insights, perform custom analyses, and export visualizations. The primary dataset is the Amazon Products Sales Dataset from Kaggle, containing extensive sales data across various product categories. While the project promises valuable insights, limitations include potential dataset coverage gaps, dependency on data refresh rates for real-time updates, and biases in user ratings and reviews.
    """
    st.write(project_summary)

elif option == 'Data Analysis':
    st.header('Data Analysis')

    # Basic statistics and data overview
    st.subheader('Step 1: Basic Statistics and Data Overview')

    # Separate categorical and numerical columns
    categorical_cols = ['product_name', 'main_category', 'sub_category', 'image']
    numerical_cols = [col for col in all_data.columns if col not in categorical_cols]

    # Display basic statistics for categorical data
    st.write('### Categorical Data Overview')
    categorical_stats = all_data[categorical_cols].describe(include='all').transpose()
    st.dataframe(categorical_stats.style.set_properties(**{'text-align': 'center'}).set_table_styles(
        [dict(selector='th', props=[('text-align', 'center')])]))

    # Display basic statistics for numerical data
    st.write('### Numerical Data Overview')
    numerical_stats = all_data[numerical_cols].describe().transpose()
    st.dataframe(numerical_stats.style.format("{:.2f}").set_properties(**{'text-align': 'center'}).set_table_styles(
        [dict(selector='th', props=[('text-align', 'center')])]))



    # Missing values analysis
    st.subheader('Step 2: Missing Values Analysis')

    # Check for missing values
    missing_values = all_data.isnull().sum()

    # Convert missing values to a DataFrame
    missing_values_df = pd.DataFrame(missing_values, columns=['Missing Values']).transpose()

    # Display missing values analysis
    st.dataframe(missing_values_df)



    # Price Distribution Analysis
    st.subheader('Step 3: Price Distribution Analysis')
    all_data['discount_price'] = all_data['discount_price'].replace('[₹,]', '', regex=True).astype(float)
    all_data['actual_price'] = all_data['actual_price'].replace('[₹,]', '', regex=True).astype(float)

    st.subheader('Discount Price Distribution')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(all_data['discount_price'].dropna(), bins=50, kde=False, stat="density", color='skyblue',
                 edgecolor='black', label='Histogram', ax=ax)
    sns.kdeplot(all_data['discount_price'].dropna(), color='blue', linewidth=2, label='KDE', ax=ax)
    ax.set_title('Discount Price Distribution with KDE')
    ax.set_xlabel('Discount Price (₹)')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 10000)
    ax.legend()
    st.pyplot(fig)

    st.subheader('Actual Price Distribution')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(all_data['actual_price'].dropna(), bins=50, kde=False, stat="density", color='lightgreen',
                 edgecolor='black', label='Histogram', ax=ax)
    sns.kdeplot(all_data['actual_price'].dropna(), color='green', linewidth=2, label='KDE', ax=ax)
    ax.set_title('Actual Price Distribution with KDE')
    ax.set_xlabel('Actual Price (₹)')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 50000)
    ax.legend()
    st.pyplot(fig)

    # Ratings Distribution Analysis
    st.subheader('Step 4: Ratings Distribution Analysis')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(all_data['ratings'].dropna(), bins=50, kde=True, ax=ax)
    ax.set_title('Ratings Distribution')
    ax.set_xlabel('Ratings')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Number of Ratings Distribution Analysis
    st.subheader('Step 5: Number of Ratings Distribution Analysis')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(all_data['no_of_ratings'].dropna(), bins=50, kde=True, ax=ax)
    ax.set_title('Number of Ratings Distribution')
    ax.set_xlabel('Number of Ratings')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Category-wise Analysis
    st.subheader('Step 6: Category-wise Analysis')
    category_counts = all_data['category'].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title('Number of Products per Category')
    ax.set_xlabel('Category')
    ax.set_ylabel('Number of Products')
    st.pyplot(fig)

    # Correlation Analysis
    st.subheader('Step 7: Correlation Analysis')
    correlation_matrix = all_data[['discount_price', 'actual_price', 'ratings', 'no_of_ratings']].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)

elif option == 'Product Comparison':
    st.header('Product Comparison')

    # Sidebar product selection
    st.sidebar.subheader('Select Products for Comparison')

    # Allow selection from any main and sub category
    main_categories = all_data['main_category'].unique()
    selected_main_categories = st.sidebar.multiselect('Select Main Categories:', main_categories)

    if selected_main_categories:
        sub_categories = all_data[all_data['main_category'].isin(selected_main_categories)]['sub_category'].unique()
        selected_sub_categories = st.sidebar.multiselect('Select Sub Categories:', sub_categories)

        if selected_sub_categories:
            product_names = all_data[(all_data['main_category'].isin(selected_main_categories)) &
                                     (all_data['sub_category'].isin(selected_sub_categories))]['product_name'].unique()
            selected_products = st.sidebar.multiselect('Select Products:', product_names)

            if selected_products:
                selected_data = all_data[all_data['product_name'].isin(selected_products)].copy()

                # Calculate discount percentage
                selected_data['discount_percentage'] = ((selected_data['actual_price'] - selected_data['discount_price']) / selected_data['actual_price']) * 100

                # Truncate product names to 50 characters for plotting
                selected_data['truncated_product_name'] = selected_data['product_name'].str[:50]

                # Compute necessary statistics for selected products
                product_stats = selected_data.groupby('product_name').agg({
                    'ratings': ['mean', 'median', 'min', 'max', 'std'],
                    'no_of_ratings': 'sum',
                    'discount_price': ['mean', 'median', 'min', 'max', 'std'],
                    'actual_price': ['mean', 'median', 'min', 'max', 'std'],
                    'discount_percentage': 'mean'
                }).reset_index()

                # Flatten MultiIndex columns
                product_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in product_stats.columns.values]

                # Display selected product information in a table
                st.subheader('Selected Product Information')

                # Display product statistics in tabular form
                st.dataframe(product_stats)

                product_stats['truncated_product_name'] = product_stats['product_name'].str[:50]
                # Ratings Distribution
                st.subheader('Ratings Distribution')
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.boxplot(x='truncated_product_name', y='ratings', data=selected_data, ax=ax)
                ax.set_title('Ratings Distribution')
                ax.set_xlabel('Product Name')
                ax.set_ylabel('Ratings')
                st.pyplot(fig)

                # Comparison plots (example)
                if len(selected_products) > 1:
                    st.subheader('Price Comparison')
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.barplot(x='truncated_product_name', y='discount_price_mean', data=product_stats, ax=ax)
                    ax.set_title('Discount Price Comparison')
                    ax.set_xlabel('Product Name')
                    ax.set_ylabel('Discount Price (₹)')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                    st.pyplot(fig)

                    st.subheader('Ratings Comparison')
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.barplot(x='truncated_product_name', y='ratings_mean', data=product_stats, ax=ax)
                    ax.set_title('Ratings Comparison')
                    ax.set_xlabel('Product Name')
                    ax.set_ylabel('Ratings')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                    st.pyplot(fig)
                else:
                    st.write("Please select more than one product for comparison.")
            else:
                st.write("Please select products from the sidebar to compare.")
        else:
            st.write("Please select subcategories from the sidebar.")
    else:
        st.write("Please select main categories from the sidebar.")


elif option == 'About':
    st.header('About')
    about_text = """
        This project was developed as part of the BANA 8083 - 006 MS Capstone at the University of Cincinnati. 
        It aims to provide Amazon retailers with actionable insights through a comprehensive, real-time interactive 
        platform for analyzing and visual"""
