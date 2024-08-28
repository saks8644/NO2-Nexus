# NO2 Nexus

**NO2 Nexus** is an advanced machine learning model designed to transform coarse-resolution satellite data into high-resolution maps of Nitrogen Dioxide (NO2) concentrations. By leveraging machine learning techniques, NO2 Nexus enhances air quality monitoring, offering detailed insights into NO2 distribution for researchers, policymakers, and urban planners.
## Data Source

The coarse-resolution spatial data for this model is obtained from Google Earth Engine's Sentinel-5P satellite. Specifically, the dataset used is the Nitrogen Dioxide (NO2) column number density, provided by the Sentinel-5P satellite's TROPOspheric Monitoring Instrument (TROPOMI).




## Features

- **High-Resolution NO2 Mapping:** Converts coarse-resolution satellite NO2 data into fine-resolution maps for more detailed spatial analysis.
- **Machine Learning Model:** Utilizes Random Forest Regression to predict NO2 concentrations based on spatial and temporal features, ensuring high accuracy.
- **Model Validation:** Includes evaluation metrics such as RMSE (Root Mean Squared Error) and R² score to validate model performance.
- **Visualization Tools:** Provides residual plots, feature importance charts, and prediction vs. actual value plots to facilitate comprehensive analysis.
- **Customizable:** Easily adapt the model by updating file paths and parameters to fit different datasets and regions.

## Installation

To use NO2 Nexus, you need Python and the required packages. Follow these steps to set up your environment:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/no2-nexus.git
   cd no2-nexus
2. **Create a Virtual Environment (Optional but Recommended):**
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
3. **Install Dependencies:**
   pip install -r requirements.txt
## Usage
Prepare Your Data:
Ensure you have a TIFF file containing NO2 data. Update the file path in the script to point to your data.
Run the Model:
Execute the Python script to preprocess the data, train the model, and generate high-resolution NO2 maps: python no2_nexus.py
## The script performs the following tasks:

- Loads and preprocesses the TIFF file.
- Trains the Random Forest model with the data.
- Validates the model using metrics and generates visualizations.
- Produces a fine-resolution NO2 map and displays it.
## Model Details
- File Path: Update the tif_file variable in the script with the path to your TIFF file.
- Temporal Features: The current setup is for the year 2019 and month 6. Adjust as needed for other time periods.
- Model Parameters: Default parameters for the Random Forest Regressor are used, but these can be modified in the code.
## Visualizations
The model generates the following visualizations:

- Fine Resolution NO2 Map: Provides a detailed map of predicted NO2 concentrations.
  ![no2 plot](https://github.com/user-attachments/assets/a96bee29-81fb-4e12-ab38-5ed42bc5da4d)

- Residual Plot: Displays residuals versus actual values to evaluate model errors.
  ![residual plt](https://github.com/user-attachments/assets/722fee7d-2b45-450f-acb1-dbd906002fcd)

- Feature Importances: Shows the importance of each feature used in the model.
- ![feature imp](https://github.com/user-attachments/assets/5bf75ee2-bdf9-4d4c-8a30-571204052518)

- Prediction vs Actual Plot: Compares predicted values with actual values to assess model accuracy.
  ![pre vs act](https://github.com/user-attachments/assets/e75c2385-3dbc-45d5-b9cb-3da5d3b051f2)



## Contact
For support, questions, or contributions, please contact sakshambalsane19@gmail.com.        
