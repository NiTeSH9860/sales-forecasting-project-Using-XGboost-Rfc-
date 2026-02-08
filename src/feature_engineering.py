import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.lag_features = [1, 7, 14, 30]  # Days to lag
        self.rolling_windows = [7, 14, 30]  # Rolling window sizes
        
    def create_time_series_features(self, time_series_df):
        """Creating lag and rolling features for time series"""
        df = time_series_df.copy()
        
        # Creating date features
        df['Year'] = df['Sale_Date'].dt.year.astype(int)
        df['Month'] = df['Sale_Date'].dt.month.astype(int)
        df['Day'] = df['Sale_Date'].dt.day.astype(int)
        df['DayOfWeek'] = df['Sale_Date'].dt.dayofweek.astype(int)
        df['DayOfYear'] = df['Sale_Date'].dt.dayofyear.astype(int)
        df['WeekOfYear'] = df['Sale_Date'].dt.isocalendar().week.astype(int)
        
        # Creating lag features
        for lag in self.lag_features:
            df[f'Lag_{lag}'] = df['Sales_Amount'].shift(lag)
        
        # Creating rolling statistics
        for window in self.rolling_windows:
            df[f'Rolling_Mean_{window}'] = df['Sales_Amount'].rolling(window=window).mean()
            df[f'Rolling_Std_{window}'] = df['Sales_Amount'].rolling(window=window).std()
            df[f'Rolling_Min_{window}'] = df['Sales_Amount'].rolling(window=window).min()
            df[f'Rolling_Max_{window}'] = df['Sales_Amount'].rolling(window=window).max()
        
        # Creating difference features
        df['Diff_1'] = df['Sales_Amount'].diff(1)
        df['Diff_7'] = df['Sales_Amount'].diff(7)
        
        # Creating seasonal features
        df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        df['Is_Month_Start'] = (df['Day'] == 1).astype(int)
        df['Is_Month_End'] = df['Day'].isin([28, 29, 30, 31]).astype(int)
        
        # Dropping rows with NaN values created by lag/rolling features
        df = df.dropna()
        
        return df
    
    def create_advanced_features(self, original_df, time_series_df):
        """Creating additional business-relevant features"""
        df = time_series_df.copy()
        
        # Adding day since first sale
        df['Days_Since_First'] = (df['Sale_Date'] - df['Sale_Date'].min()).dt.days
        
        # Adding cyclical features for time
        df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear']/365)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear']/365)
        
        # Adding interaction features from original data
        daily_metrics = original_df.groupby('Sale_Date').agg({
            'Quantity_Sold': 'sum',
            'Discount': 'mean',
            'Profit': 'sum',
            'Customer_Type': lambda x: (x == 'New').sum()
        }).reset_index()
        
        daily_metrics.columns = ['Sale_Date', 'Daily_Quantity', 'Avg_Discount', 
                                'Daily_Profit', 'New_Customers']
        
        df = pd.merge(df, daily_metrics, on='Sale_Date', how='left')
        
        # Filling any NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def prepare_train_test(self, df, test_size=0.2, date_col='Sale_Date'):
        """Splitting data into train and test sets chronologically"""
        df = df.sort_values(date_col)
        
        split_idx = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        # Separating features and target
        X_train = train_df.drop(['Sales_Amount', 'Sale_Date'], axis=1, errors='ignore')
        y_train = train_df['Sales_Amount']
        
        X_test = test_df.drop(['Sales_Amount', 'Sale_Date'], axis=1, errors='ignore')
        y_test = test_df['Sales_Amount']
        
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, train_df, test_df