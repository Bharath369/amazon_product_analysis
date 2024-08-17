<p align="right">
  <img src="images/amazon_logo.jpg" alt="Logo" width="100" height="100">
</p>


# Amazon Product Analysis

## Overview
This project focuses on analyzing Amazon product data to uncover insights related to customer reviews, ratings, pricing trends, and product popularity. The goal is to leverage data-driven analysis to help sellers optimize their product listings and improve customer satisfaction.

## Problem Statement
With the vast amount of data available on Amazon products, it becomes challenging for sellers to make informed decisions. This project aims to address key questions such as:
- What factors influence product ratings and reviews?
- How do pricing trends affect product popularity?
- What are the common sentiments expressed in customer reviews?

## Data
The dataset includes various features such as:
- **Product Title**: The name of the product.
- **Category**: The category under which the product is listed.
- **Price**: The selling price of the product.
- **Rating**: Average customer rating for the product.
- **Review Count**: Number of customer reviews.
- **Review Text**: Text of customer reviews.

### Key Data Files
- `products.csv`: Contains information about product titles, categories, and pricing.
- `reviews.csv`: Includes customer reviews, ratings, and sentiments.
- `amazon_consolidated_data.csv`: Includes all the retail data combining products, reviews etc.

## Analysis
The analysis in this project covers:
1. **Exploratory Data Analysis (EDA)**: Understanding the distribution of ratings, prices, and review counts across different categories.
2. **Sentiment Analysis**: Analyzing customer reviews to determine overall sentiment (positive, neutral, or negative).
3. **Pricing Strategy**: Examining the relationship between pricing and product popularity.
4. **Rating Prediction**: Developing models to predict product ratings based on available features.

For detailed analysis, refer to the [scripts/Home.py](scripts/Home.py).

## Tools Used
- **Python**: Core programming language used for data manipulation and analysis.
- **Pandas**: For data handling and manipulation.
- **Numpy**: For numerical computations.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For building and evaluating machine learning models.
- **NLTK/Spacy**: For natural language processing and sentiment analysis.
- **Jupyter Notebook**: For interactive data analysis and visualization.

## Installation
To run this project, install the required dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
