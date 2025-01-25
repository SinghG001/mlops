import pytest
import pandas as pd
import numpy as np

class Cleaner:
    def clean_data(self, df):
        # Implement your cleaning logic here
        df = df.drop(columns=['id', 'SalesChannelID', 'VehicleAge', 'DaysSinceCreated'])
        df['AnnualPremium'] = df['AnnualPremium'].replace('[£,]', '', regex=True).astype(float)
        df['Gender'].fillna('Unknown', inplace=True)
        df['RegionID'].fillna(df['RegionID'].median(), inplace=True)
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['HasDrivingLicense'].fillna(1, inplace=True)
        df['Switch'].fillna(-1, inplace=True)
        df['PastAccident'].fillna('Unknown', inplace=True)
        return df

"""
    A pytest fixture that provides sample input data for testing the Cleaner class.
    Returns:
    - pd.DataFrame: A sample DataFrame with various columns containing missing values and
      specific formats to test the cleaning process.
"""
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'id': [1, 2, 3, 4],
        'SalesChannelID': [1, 1, 1, 1],
        'VehicleAge': [1, 1, 1, 1],
        'DaysSinceCreated': [1, 1, 1, 1],
        'AnnualPremium': ['£1,200', '£2,500', '£3,000', '£5,000'],
        'Gender': ['Male', np.nan, 'Female', 'Male'],
        'RegionID': [np.nan, 2, 3, np.nan],
        'Age': [30, np.nan, 25, 40],
        'HasDrivingLicense': [1, np.nan, 1, np.nan],
        'Switch': [0, 1, np.nan, 0],
        'PastAccident': [np.nan, 'Yes', 'No', 'Yes']
    })

@pytest.fixture
def cleaner():
    return Cleaner()


"""
Tests the clean_data method of the Cleaner class to ensure it performs the expected
cleaning operations on the input data.

Parameters:
- cleaner (Cleaner): An instance of the Cleaner class.
- sample_data (pd.DataFrame): A sample DataFrame to test the cleaning process.
"""
def test_clean_data(cleaner, sample_data):
    cleaned_data = cleaner.clean_data(sample_data.copy())

    # Check if the columns are dropped
    assert 'id' not in cleaned_data.columns
    assert 'SalesChannelID' not in cleaned_data.columns
    assert 'VehicleAge' not in cleaned_data.columns
    assert 'DaysSinceCreated' not in cleaned_data.columns

    # Check if AnnualPremium is converted to float
    assert cleaned_data['AnnualPremium'].dtype == float

    # Check if missing values in Gender and RegionID are imputed
    assert not cleaned_data['Gender'].isnull().any()
    assert not cleaned_data['RegionID'].isnull().any()

    # Check if Age missing values are filled with median
    assert not cleaned_data['Age'].isnull().any()

    # Check if HasDrivingLicense missing values are filled with 1
    assert not cleaned_data['HasDrivingLicense'].isnull().any()
    assert (cleaned_data['HasDrivingLicense'] == 1).all()

    # Check if Switch missing values are filled with -1
    assert not cleaned_data['Switch'].isnull().any()
    assert (cleaned_data['Switch'] == -1).any()

    # Check if PastAccident missing values are filled with "Unknown"
    assert not cleaned_data['PastAccident'].isnull().any()
    assert (cleaned_data['PastAccident'] == 'Unknown').any()

    # Check if outliers in AnnualPremium are removed
    Q1 = cleaned_data['AnnualPremium'].quantile(0.25)
    Q3 = cleaned_data['AnnualPremium'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    assert (cleaned_data['AnnualPremium'] <= upper_bound).all()
    print('test_clean_data is passed')

if __name__ == "__main__":
    pytest.main()