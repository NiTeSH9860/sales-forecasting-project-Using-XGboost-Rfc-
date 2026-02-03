import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SalesDataPreprocessor:
    def __init__(self):
        self.date_format = "%m/%d/%Y"
        
    def load_data(self, filepath):
        """Loading and basic cleaning sales data"""
        df = pd.read_csv(filepath)
        print(f"Loaded data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    
    def preprocess(self, df):
        """Main preprocessing pipeline"""
        # Converting date column
        df['Sale_Date'] = pd.to_datetime(df['Sale_Date'], format=self.date_format)
        
        # Sorting by date
        df = df.sort_values('Sale_Date')
        
        # Creating time-based features
        df['Year'] = df['Sale_Date'].dt.year
        df['Month'] = df['Sale_Date'].dt.month
        df['Quarter'] = df['Sale_Date'].dt.quarter
        df['Week'] = df['Sale_Date'].dt.isocalendar().week
        df['Day'] = df['Sale_Date'].dt.day
        df['DayOfWeek'] = df['Sale_Date'].dt.dayofweek
        df['DayOfYear'] = df['Sale_Date'].dt.dayofyear
        
        # Handling categorical variables
        categorical_cols = ['Region', 'Product_Category', 'Customer_Type', 
                          'Payment_Method', 'Sales_Channel', 'Sales_Rep']
        
        for col in categorical_cols:
            df[f'{col}_encoded'] = df[col].astype('category').cat.codes
        
        # Calculating profit
        df['Profit'] = df['Sales_Amount'] - (df['Unit_Cost'] * df['Quantity_Sold'])
        
        # Removing potential outliers (optional)
        df = self._remove_outliers(df, 'Sales_Amount')
        
        return df
    
    def _remove_outliers(self, df, column):
        """Removing outliers using IQR method"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        print(f"Removed {len(df) - len(filtered_df)} outliers from {column}")
        return filtered_df
    
    def create_time_series_features(self, df, target_col='Sales_Amount', group_by=None):
        """Create aggregated time series features"""
        if group_by:
            # Aggregating by region, category, etc.
            df_agg = df.groupby([group_by, 'Sale_Date'])[target_col].sum().reset_index()
            return df_agg
        else:
            # Overall time series
            df_agg = df.groupby('Sale_Date')[target_col].sum().reset_index()
            return df_agg
filepath = 'D:\\end to end sales forecasting project\\sales_data.csv'  