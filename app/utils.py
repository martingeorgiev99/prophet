import numpy as np

column_mapping = {
    "expected_order_status": ["order_status", "status", "order_state", "orderCondition"],
    "expected_order_date": ["order_date", "date", "order_time", "purchase_date", "orderDate"],
    "expected_order_count": ["order_count", "orders", "num_orders", "order_quantity", "count_of_orders"],
}

def find_column_name(possible_names, df_columns):
    for name in possible_names:
        if name in df_columns:
            return name
    return None

def filter_outliers_with_z_score(series, window=4, threshold=2):
    rolling_median = series.rolling(window=window, min_periods=1).median()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    z_scores = np.abs((series - rolling_median) / rolling_std)
    return z_scores < threshold
