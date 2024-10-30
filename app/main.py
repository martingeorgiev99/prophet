from flask import Flask, request, render_template, jsonify
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, r2_score
from prophet.plot import plot_plotly
import numpy as np
import os

app = Flask(__name__)

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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/forecast", methods=["POST"])
def forecast():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Please upload a valid CSV file."})
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Error reading file: {str(e)}"})
    
    actual_order_status = find_column_name(column_mapping["expected_order_status"], df.columns)
    actual_order_date = find_column_name(column_mapping["expected_order_date"], df.columns)
    
    if not actual_order_status or not actual_order_date:
        return jsonify({"error": "Missing required columns in the uploaded CSV file"})
    
    df_cleaned = df.dropna(subset=[actual_order_status])
    df_cleaned = df_cleaned[df_cleaned[actual_order_status] != "Отказана"]
    df_cleaned[actual_order_date] = pd.to_datetime(df_cleaned[actual_order_date], errors="coerce")
    df_cleaned = df_cleaned.dropna(subset=[actual_order_date])
    
    df_cleaned["order_week"] = df_cleaned[actual_order_date].dt.to_period("W").apply(lambda r: r.start_time)
    weekly_orders = df_cleaned.groupby("order_week").size().reset_index(name="order_count")
    
    valid_data = filter_outliers_with_z_score(weekly_orders["order_count"])
    weekly_orders_filtered = weekly_orders[valid_data]
    
    if weekly_orders_filtered.shape[0] < 2:
        return jsonify({"error": "Not enough valid data to make a forecast. Please provide more data."})
    
    weekly_orders_cleaned = weekly_orders_filtered.set_index("order_week").asfreq("W", method="ffill")
    median_order_count = weekly_orders_cleaned["order_count"].median()
    weekly_orders_cleaned["order_count"].fillna(median_order_count, inplace=True)
    prophet_df = weekly_orders_cleaned.reset_index().rename(columns={"order_week": "ds", "order_count": "y"})
    
    try:
        prophet_model = Prophet(
            yearly_seasonality=True, 
            weekly_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.15,
            seasonality_prior_scale=10.0,
        )
        prophet_model.fit(prophet_df)
    except Exception as e:
        return jsonify({"error": f"Error fitting Prophet model: {str(e)}"})
    
    future = prophet_model.make_future_dataframe(periods=12, freq="W")
    forecast = prophet_model.predict(future)
    future_predictions = forecast.tail(2)[["ds", "yhat"]].to_dict(orient="records")
    
    prophet_df["yhat"] = forecast["yhat"].iloc[: len(prophet_df)]
    mae = mean_absolute_error(prophet_df["y"], prophet_df["yhat"])
    r2 = r2_score(prophet_df["y"], prophet_df["yhat"])
    
    try:
        fig = plot_plotly(prophet_model, forecast)
        fig.update_traces(text=forecast["yhat"].astype(str), hoverinfo="text+x+y")
        plot_json = fig.to_json()
    except Exception as e:
        return jsonify({"error": f"Error generating plot: {str(e)}"})
    
    exact_predictions = forecast[["ds", "yhat"]].tail(2).to_dict(orient="records")
    
    return jsonify({
        "predictions": future_predictions,
        "mae": mae,
        "r2": r2,
        "plot": plot_json,
        "exact_predictions": exact_predictions
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
