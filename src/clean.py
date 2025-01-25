import numpy as np
from sklearn.impute import SimpleImputer

class Cleaner:
    """
    A class for cleaning and preprocessing data, to handle missing 
    values, data formatting, and outlier removal for insurance-related datasets.
    """
    
    def __init__(self):
        self.imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
        
    def clean_data(self, data):
        """
        Cleans the provided dataset by performing the following steps:
        1. Drops unnecessary columns.
        2. Converts the 'AnnualPremium' column to a numeric format.
        3. Handles missing values in categorical and numeric columns.
        4. Removes outliers from the 'AnnualPremium' column.
        
        """
        
        # Drop irrelevant or unnecessary columns
        data.drop(['id', 'SalesChannelID', 'VehicleAge', 'DaysSinceCreated'], axis=1, inplace=True)
        
        # Convert 'AnnualPremium' to a numeric value by removing currency symbols and commas
        data['AnnualPremium'] = (
            data['AnnualPremium']
            .str.replace('£', '', regex=False)
            .str.replace(',', '', regex=False)
            .astype(float)
        )
        
        # Impute missing values for categorical columns 'Gender' and 'RegionID' with the most frequent value
        for col in ['Gender', 'RegionID']:
            data[col] = self.imputer.fit_transform(data[[col]]).flatten()
        
        # Fill missing values for the 'Age' column with the median age
        data['Age'] = data['Age'].fillna(data['Age'].median())
        
        # Fill missing values for 'HasDrivingLicense' with 1 (assuming default is having a license)
        data['HasDrivingLicense'] = data['HasDrivingLicense'].fillna(1)
        
        # Fill missing values for 'Switch' with -1 (representing an unknown state)
        data['Switch'] = data['Switch'].fillna(-1)
        
        # Fill missing values for 'PastAccident' with "Unknown"
        data['PastAccident'] = data['PastAccident'].fillna("Unknown", inplace=False)
        
        # Identify and remove outliers in 'AnnualPremium' using the IQR method
        Q1 = data['AnnualPremium'].quantile(0.25)  # First quartile
        Q3 = data['AnnualPremium'].quantile(0.75)  # Third quartile
        IQR = Q3 - Q1  # Interquartile range
        upper_bound = Q3 + 1.5 * IQR  # Upper bound for outliers
        data = data[data['AnnualPremium'] <= upper_bound]  # Filter outliers
        
        return data
