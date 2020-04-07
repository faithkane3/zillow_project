# Zillow Linear Regression Project

This repo will hold all of the files for my zillow project. The purpose of this project is to practice the data science pipeline by filtering and acquiring data using SQL, cleaning data using Pandas, splitting, scaling, and modeling data using functions I created to make my work reproducible. 

# Presentation [Link](https://docs.google.com/presentation/d/10Lq_R0PpSmsHcDTRgRzkBUDK5FHhj5IgdvqI4DcOY8A/edit?usp=sharing)

# Requirements and Planning

- **Write a sql query to acquire data**

    - last transaction during May and June 2017
    
    - single-unit residential properties only
    

- **Figure out state and county for each property**


- **Figure out distribution of tax rates by county**

    - Calculate tax rate using givens of home value and taxes
    
    
- **MVP - Use square feet of home, # of bedrooms, and # of bathrooms to predict home value**

    - Deliverable 1: Presentation slides (Audience == Zillow Team)
    
    - Deliverable 2: Repo containing Jupyter Notebook displaying your work through the pipeline with detailed documentation.
    
    - Deliverable 3: .py modules containing functions that ensure your work is neat, readable, and recreatable.
    
    - Deliverable 4: README.md file with data dictionary

# Hypothesis 
    
- $H_0$: Square Feet, Bedroom Count, and Bathroom Count are not drivers of Home Value.

- $H_a$: Square Feet, Bedroom Count, and Bathroom Count are drivers of Home Value.

# Data Dictionary

Int64Index: 15947 entries

Data columns (total 10 columns):
____
| variable  |     Count and Dtype    |
|:----------|:-----------------------|
|bedrooms    |15947 non-null float64 |
|bathrooms   |15947 non-null float64 |
|square_feet | 15947 non-null int64 |
|taxes       |15947 non-null float64 |
|home_value   | 15947 non-null float64|
|propertylandusedesc  | 15947 non-null object|
|fips_number   |  15947 non-null int64 |
|zip_code      |  15947 non-null float64 |
|tax_rate (calculated) | 15947 non-null float64 |
|county_name  (engineered) | 15947 non-null object|

# Skill Focus:

- Exploratory Data Analysis to discover the drivers of home value and visualize the data.

- Pearson's R testing to find correlations between different features and the target, home value.

- Use Recursive Feature Elimination to select the best features to use in my linear regression model.

- Scale data for my linear regression model.

- Evaluate my models using their RMSE, R-squared, and p-values to see how they perform compared to a baseline created from the median home value of all the homes in the data set.

- Predict home values and recommend the best features for use in predicting the target.

# Findings and Conclusion:

- My Linear Regression Model was better at predicting home value than just using the mean of home values. I believe my model could be improved with Feature Engineering, but the features I are better predictors than using a mean baseline value.

- Based on RMSE and R^2 scores, I reject the Null Hypothesis that square feet, bedroom count, and bathroom count are not drivers of home value in the Zillow dataset.

- The features in my Linear Regression model are highly correlated with each other, and in a future iteration of this project, I believe that I could improve the performance of my model using Feature Engineering to calculate new features that combine bedroom and bathroom count.

# To Recreate This Project:

- All files needed to recreate the work in this notebook are included in this repo.

Note: You will need your own env.py file to access database.
