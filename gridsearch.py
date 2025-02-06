import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import random
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import sklearn
# General Overview:
# Air pollution has always been prominent until to this day. Throughout the countries, every year, it has been
# rising and health of the people are compromised. In this exam, we will be taking a look on the factors
# that causes air pollution deaths around the world. We will be using the capabilities of machine learning,
# specifically the use of Extreme Gradient Boosting to discover and predict the future data and statistics
# regarding air pollutions deaths. In this exam, you will be tested by, first, to get the
# correlation of air pollution deaths vs the different environment and socio-economic factors. Second, is to
# predict, using XGBoost (Extreme Gradient Boosting), the number of air pollution deaths in the year 2018 and
# get the best possible R^2 Score. Third, you are tasked to create an HTML/CSS Application that shows the results
# of the prediction and correlation. And finally, explain the results through graphs (SHAP, and Scatter Plot).
# There will be specific instructions inside the functions to guide you throughout the exam.

# NOTE: There will be an HTML file in this exam where you can see the expected results of the exam.
#       You can use that as your reference.

# NOTE: You are free to create multiple functions for readability and organization of your code.
# NOTE: You are also free to look up different documentations for the libraries you will be using.


# This function defines all the files as a dictionary that will be used throughout the exam.
# This csv files contains your data in the different environment and socioeconomic factors
# These files are located at `data/...`
def define_files():
    files = {
        "air_pollution_death": "data/air_pollution_death.csv",
        "transportation": 'data/road-transportation_country_emissions.csv',
        "coal": 'data/coal-mining_country_emissions.csv',
        "cropland": 'data/cropland-fires_country_emissions.csv',
        "residential_commercial": 'data/residential-and-commercial-onsite-fuel-usage_country_emissions.csv',
        "forest_clearing": 'data/forest-land-clearing_country_emissions.csv',
        "petrochemicals": 'data/petrochemicals_country_emissions.csv',
        "electricity_generation": 'data/electricity-generation_country_emissions.csv',
        "incineration_open_burning": 'data/incineration-and-open-burning-of-waste_country_emissions.csv',
        "health_expenditure": 'data/health-expenditure.csv',
        "urban_population": 'data/urban-population.csv'
    }
    return files

def find_common_country_codes(env_list, socio_list, files):
    env_codes = []
    for key in env_list:
        df = pd.read_csv(files[key])
        codes = set(df['iso3_country'].unique())
        env_codes.append(codes)
    common_env_codes = set.intersection(*env_codes) if env_codes else set()
    
    # Process socioeconomic files; note the header starts on line 5 (index 4)
    socio_codes = []
    for key in env_list:
        df = pd.read_csv(files[key])
        codes = set(df['iso3_country'].unique())
        env_codes.append(codes)
    common_env_codes = set.intersection(*env_codes) if env_codes else set()
    
    # Process socioeconomic files.
    # Skip the first 4 rows so that the header (on line 5) is used.
    socio_codes = []
    for key in socio_list:
        df = pd.read_csv(files[key], skiprows=4)
        # Print out the columns to debug (if needed)
        #print(f"Columns for {key}:", df.columns.tolist())
        
        # Search for the country code column in a case- and space-insensitive way.
        country_col = None
        for col in df.columns:
            if col.strip().replace(" ", "").lower() == "countrycode":
                country_col = col
                break
                
        if country_col is None:
            raise ValueError(f"Expected country code column not found in file {files[key]}. Found columns: {df.columns.tolist()}")
            
        codes = set(df[country_col].unique())
        socio_codes.append(codes)
    common_socio_codes = set.intersection(*socio_codes) if socio_codes else set()
    
    # Now, find the intersection between the environment and socioeconomic country codes.
    common_country_codes = sorted(list(common_env_codes.intersection(common_socio_codes)))
    
    #print("Common Country Codes:", common_country_codes)
    return common_country_codes

# This function should contain your code in getting the R^2 Score using xgboost.
def start_predict_xgboost():

    # Set seed to 42 for consistent results
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Define Files
    files = define_files()

    # These are the the list of environment factors.
    environment_factor_files_list = [
        'transportation',
        'coal',
        'cropland',
        'residential_commercial',
        'forest_clearing',
        'petrochemicals',
        'electricity_generation',
        'incineration_open_burning'
    ]
    # These are the list of socio-economic factors
    socioeconomic_files_list = [
        'health_expenditure',
        'urban_population'
    ]

    # 1. Construct a code that will retrieve the common country codes inside the both csv factor categories.
    # Since there are two factor files categories, combine them and the results should only return a list
    # of common country codes and should be sorted.

    #Your code below:
    common_country_codes = find_common_country_codes(environment_factor_files_list, socioeconomic_files_list, files)
    #Your code above

    # 2. For our air pollution data, group the data by country code and
    #    filter the data by only retrieving the data that are:
    #   A. Present in the common country codes.
    #   B. AND in the period 2018
    #   C. AND Both sexes
    air_pollution_df = pd.read_csv(files["air_pollution_death"])
    #Your code below:
    filtered_air_pollution_df = air_pollution_df[
        (air_pollution_df['SpatialDimValueCode'].isin(common_country_codes))  &  # A. Present in common country codes
        (air_pollution_df['Period'] == 2018) &  # B. Year is 2018
        (air_pollution_df['Dim1'] == 'Both sexes') # C. Data for both sexes
    ]
    filtered_air_pollution_df.to_csv('filtered_air_deaths.csv', index=False)
    filtered_air_pollution_df.info()
    #Your code above:

    # 3. With the use of common_country_codes
    #   A. We would like to create a separate list that consists environment_data and socioeconomc_data
    #   B. (For environment_data only) Iterate to each data and filter them by common country codes and
    #       select data starting from 2018-01-01 00:00:00. Aggregate this data by grouping it by country code and
    #       getting their respective total emissions_quantity. Rename the common country code column
    #       to Country Code and emissions_quantity to 'name'.
    #   C. (For socioeconomic_data) Iterate to each data and filter them by common country codes and
    #       select data in 2018 column. Rename the common country code column
    #       to Country Code and the column '2018' to 'name'.

    # Your code below:
    environment_data = []
    socioeconomic_data = []

    for key in environment_factor_files_list:
        df = pd.read_csv(files[key])

        df_filtered = df[
            (df['iso3_country'].isin(common_country_codes)) &
            (df['start_time'] >= '2018-01-01 00:00:00')
        ]

        df_grouped = df_filtered.groupby('iso3_country', as_index=False)['emissions_quantity'].sum()

        df_grouped.rename(columns={'iso3_country': 'Country Code', 'emissions_quantity': key}, inplace=True)
        environment_data.append(df_grouped)
    
    #print(environment_data)

    for key in socioeconomic_files_list:
        df = pd.read_csv(files[key], skiprows=4)  # Skip metadata rows
        
        # Filter by common country codes
        df_filtered = df[df['Country Code'].isin(common_country_codes)]
        
        # Select only the 2018 column
        df_filtered = df_filtered[['Country Code', '2018']].rename(columns={'2018': key})
        
        # Append to socioeconomic data list
        socioeconomic_data.append(df_filtered)
    #print(socioeconomic_data)

    environment_merged = environment_data[0]  # Start with the first dataframe in the list
    #print(environment_data)
    # You code above:


    # 4. With the use of the environment_data and socioeconomic_data
    #   A. Merge the environment_data and socioeconomic_data on common country codes
    #   B. Do another merge with the air pollution deaths data on common country codes.
    #   C. Create a dataframe using this merged data of A and B.
     
    # Your code below:
    deaths_by_country = filtered_air_pollution_df.groupby(['SpatialDimValueCode'])['FactValueNumeric'].sum().reset_index()
    deaths_by_country.rename(columns={'SpatialDimValueCode': 'Country Code', 'FactValueNumeric': 'deaths'}, inplace=True)

    for df in environment_data[1:]:
        environment_merged = pd.merge(environment_merged, df, on="Country Code", how="outer")

    # Merge socioeconomic data
    final_merged = environment_merged  # Start with the merged environment data
    for df in socioeconomic_data:
        final_merged = pd.merge(final_merged, df, on="Country Code", how="outer")

    final_merged = pd.merge(final_merged, deaths_by_country, on="Country Code", how="outer")
    
    # Now final_merged contains the combined data
    # You code above:


    # 5. For our final dataframe
    #   A. Select the columns that have > 0.5 missing values
    #   B. Drop the drops missing values
    #   C. Based from the example output, there should be an option where CHN and IND are excluded

    # Your code below:
    '''
    missing_threshold = 0.5 * len(final_merged)
    to_drop = final_merged[final_merged.isnull().sum() > missing_threshold]

    final_merged_cleaned = final_merged.drop(columns=to_drop)
    '''
    final_merged_cleaned = final_merged.dropna()
    final_merged_cleaned.to_csv('real_merged_data.csv', index=False)
    # You code above:

    # 6. Determine our X and y values
    #   A. X should be columns other than common country codes and air pollution death
    #   B. y should be the 'air_pollution_death' values
    # Your code below:

    X = final_merged_cleaned.drop(columns=['Country Code','deaths'])
    y = final_merged_cleaned['deaths']

    # In this exam, we shall use 80% as our train data and 20% as our test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=random_seed)

    # 7. For standardization of dataset, a common requirement is to scale the data.
    #   A. Scale the X_train and X_test
    scaler = StandardScaler()
    # Your code below:
    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)
    # You code above:

    

    # 8. Defining our XGB Model with hyperparameters
    #   A. Try to get the best R^2 score as much as possible by tweaking the hyperparameters
    # NOTE: There are pre defined hyperparameters, try not to remove them in tuning for
    #       the best possible R^2 Score.

    print(f"scikit-learn version: {sklearn.__version__}")
    print(f"xgboost version: {xgb.__version__}")
    param_grid = {
    'n_estimators': [100, 300, 500, 600],  # Number of boosting rounds
    'learning_rate': [0.01, 0.05, 0.1],    # Learning rate
    'max_depth': [1, 3, 5, 7],             # Maximum depth of trees
    'gamma': [0, 0.1, 0.2],                 # Regularization term
    'subsample': [0.3, 0.4, 0.5, 0.7],           # Subsample ratio
    'colsample_bytree': [0.5, 0.7, 1.0],    # Subsample ratio for each tree
    'reg_alpha': [0.01, 0.1],               # L1 regularization
    'reg_lambda': [1.0, 5.0, 10.0]          # L2 regularization
    }
    
    xgb_model = xgb.XGBRegressor(
        random_state=random_seed,
    )

    # 9. With the xgb_model.
    #   A. Generate the code to train the model
    #   B. Generate the code to evaluate and predict the Test and Train data
    #   C. Print the results to the HTML File\

    eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
    # Set up the GridSearchCV with cross-validation
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='r2',  # Use R² as the scoring metric
        cv=3,          # Use 3-fold cross-validation
        verbose=2,     # Display progress
        n_jobs=-1      # Use all available CPUs
    )

    grid_search.fit(X_train_scaled, y_train, eval_set=eval_set)

    # Print the best hyperparameters found by grid search
    print("Best Hyperparameters: ", grid_search.best_params_)

    # Train the model with the best parameters
    best_xgb_model = grid_search.best_estimator_

    # Evaluate and predict with the best model
    y_train_pred = best_xgb_model.predict(X_train_scaled)
    y_test_pred = best_xgb_model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    result = best_xgb_model.evals_result()

    print(result)
    print(f"Train R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")

    iterations = len(result['validation_0']['rmse'])
    x_axis = range(0, iterations)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, result['validation_0']['rmse'], label='Train RMSE')  # Train error (RMSE)
    plt.plot(x_axis, result['validation_1']['rmse'], label='Test RMSE')  # Test error (RMSE)
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.title('Learning Curves (Training vs Testing RMSE)')
    plt.legend()
    plt.grid(True)
    plt.show()
    generate_beeswarm_plot(X_train, xgb_model)

def generate_beeswarm_plot(X_train, model):
    # 1. Create a SHAP Explainer object
    explainer = shap.Explainer(model, X_train)

    # 2. Calculate SHAP values for the training set
    shap_values = explainer(X_train)

    # 3. Create the beeswarm plot
    shap.summary_plot(shap_values, X_train, plot_type="dot")  # You can also use "dot" or "bar"

    # Save the plot as a PNG file
    shap_fig = plt.figure()
    shap_fig.savefig("beeswarm_plot.png", bbox_inches='tight')
    plt.close(shap_fig)  # Close the plot to avoid showing it in some environments

    print("Beeswarm plot saved as 'beeswarm_plot.png'")


# This function should contain the generation of your HTML File. Use the coding_test.html as your reference
# This function should also contain the generation of scatter plot and choropleth map
def generate_html_file():

    # 11. For the scatter plot:
    #   A. Show the scatter plot of the different countries' Air Pollution Deaths vs different factors (Refer to the HTML File)
    #   B. Show the correlation between the Air Pollution Deaths vs different factors (Refer to the HTML file)
    #   C. There should be an option to remove outliers (CHN and IND) or not in showing the graph.
    # NOTE: You can visit plotly for the scatter plot documentation
    scatter_html_blocks = []

    # Your code below:

    # You code above:

    # 12. For the choropleth map
    #   A. Show the map with the Air pollution deaths in every country (Refer to the HTML File)
    #   B. When a country is hovered in the map, show the Country Code and Air Pollution Deaths only.
    # NOTE: You can visit plotly for the choropleth map documentation
    choropleth_map = ""

    # Your code below:

    # Your code above:

    # HTML Creation starts here:
    with open("coding_test_output.html", "w", encoding="utf-8") as f:
        f.write("""
        <html>
            <head>
                <title>Air Pollution and Emissions Analysis</title>
                </script>
            </head>
            <body>
                <h1>Air Pollution Deaths by Country</h1>
            </body>
        </html>
        """)

start_predict_xgboost()