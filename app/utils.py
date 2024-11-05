# utils.py

import numpy as np
import pandas as pd
from prophet import Prophet

# Mapping for expected column names in the DataFrame
column_mapping = {
    "expected_order_status": [
        "order_status", "status", "order_state", "orderCondition"
    ],
    "expected_order_date": [
        "order_date", "date", "order_time", "purchase_date", "orderDate"
    ],
    "expected_order_count": [
        "order_count", "orders", "num_orders", "order_quantity", "count_of_orders"
    ],
    "expected_customer_id": [
        "customer_id", "cust_id", "client_id", "customerIdentifier"
    ],
    "expected_product_id": [
        "product_id", "prod_id", "item_id", "productIdentifier"
    ],
    "expected_order_amount": [
        "order_amount", "total_amount", "amount", "orderValue"
    ],
    "expected_discount": [
        "discount", "promo_code", "discount_amount", "sale_discount"
    ],
    "expected_shipping_cost": [
        "shipping_cost", "shipping_fee", "delivery_cost", "postage"
    ],
    "expected_order_payment_method": [
        "payment_method", "payment_type", "order_payment", "method_of_payment"
    ],
    "expected_order_region": [
        "region", "location", "order_region", "delivery_region"
    ],
}

def find_column_name(possible_names, df_columns):
    """
    Find the first matching column name from a list of possibilities.
    """
    return next((name for name in possible_names if name in df_columns), None)

def filter_outliers_with_z_score(series, window=4, threshold=2):
    """
    Remove outliers from a Pandas Series using Z-score based on a rolling window.
    """
    rolling_median = series.rolling(window=window, min_periods=1).median()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    z_scores = np.abs((series - rolling_median) / rolling_std)
    return z_scores < threshold

def clean_data(df, actual_order_status, actual_order_date):
    """
    Clean and preprocess the input DataFrame.
    """
    df = df.dropna(subset=[actual_order_status])
    df = df[df[actual_order_status] != "Отказана"]
    df[actual_order_date] = pd.to_datetime(df[actual_order_date], errors="coerce")
    return df.dropna(subset=[actual_order_date])

def aggregate_weekly_orders(df, actual_order_date):
    """
    Aggregate order counts on a weekly basis.
    """
    return (df
            .assign(order_week=df[actual_order_date].dt.to_period("W").apply(lambda r: r.start_time))
            .groupby("order_week")
            .size()
            .reset_index(name="order_count"))

def fit_prophet_model(prophet_df):
    """
    Fit the Prophet model to the prepared DataFrame.
    """
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10.0,
    )
    model.fit(prophet_df)
    return model
