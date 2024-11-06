# routes.py

from flask import Blueprint, request, render_template, jsonify
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, r2_score
from prophet.plot import plot_plotly
from .utils import (find_column_name, filter_outliers_with_z_score, column_mapping,
                    clean_data, aggregate_weekly_orders, fit_prophet_model)
import logging
import time

# Configure logging to log both to a file and to the console
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])

# Create the main Blueprint for routing
main = Blueprint("main", __name__)

@main.route("/")  # Route for the index page
def index():
    return render_template("index.html")  # Render the main HTML template

@main.route("/forecast", methods=["POST"])  # Route for handling forecast requests
def forecast():
    # Check if a file was uploaded
    if "file" not in request.files:
        logging.error("No file part in the request.")
        return jsonify({"error": "No file part in the request. Please upload a CSV file."}), 400

    file = request.files["file"]
    # Validate that the uploaded file is a CSV
    if not file.filename.endswith(".csv"):
        logging.error("Uploaded file is not a CSV.")
        return jsonify({"error": "Uploaded file is not a valid CSV. Please upload a CSV file."}), 400

    start_time = time.time()  # Start timing the request
    try:
        df = pd.read_csv(file)  # Read the uploaded CSV file
        logging.info("CSV file read successfully.")
    except Exception as e:
        logging.error(f"Error reading file: {str(e)}")
        return jsonify({"error": f"Error reading file: {str(e)}"}), 400

    # Find required columns using the column_mapping in utils
    actual_order_status = find_column_name(column_mapping["expected_order_status"], df.columns)
    actual_order_date = find_column_name(column_mapping["expected_order_date"], df.columns)
    
    # Check if essential columns are missing
    if not actual_order_status or not actual_order_date:
        missing_columns = []
        if not actual_order_status:
            missing_columns.append("order_status")
        if not actual_order_date:
            missing_columns.append("order_date")
        logging.error(f"Missing required column(s): {', '.join(missing_columns)}")
        return jsonify({"error": f"Missing required column(s): {', '.join(missing_columns)}"}), 400

    # Ensure the 'order_status' column is of string type
    if not pd.api.types.is_string_dtype(df[actual_order_status]):
        logging.error("Invalid data type for order status.")
        return jsonify({"error": "Invalid data type for 'order_status'. Expected string values."}), 400

    # Convert 'order_date' to datetime, handling multiple formats
    try:
        df[actual_order_date] = pd.to_datetime(df[actual_order_date], errors='coerce')
        logging.info("Order dates converted to datetime.")
    except Exception as e:
        logging.error(f"Error parsing 'order_date': {str(e)}")
        return jsonify({"error": "Error parsing 'order_date'. Expected datetime values."}), 400

    # Clean and preprocess data
    df = clean_data(df, actual_order_status, actual_order_date)
    if df.empty:
        logging.error("No valid data after cleaning.")
        return jsonify({"error": "No valid data after cleaning. Please check your file."}), 400

    # Aggregate orders on a weekly basis
    weekly_orders = aggregate_weekly_orders(df, actual_order_date)
    if weekly_orders.empty:
        logging.error("No data after aggregation.")
        return jsonify({"error": "No data available after aggregation. Please check your file."}), 400

    # Filter outliers from the weekly data
    weekly_orders = weekly_orders[filter_outliers_with_z_score(weekly_orders["order_count"])]

    # Check if there’s enough data left for model training after filtering
    if weekly_orders.shape[0] < 2:
        logging.warning("Not enough valid data to make a forecast.")
        return jsonify({"error": "Not enough valid data to make a forecast. Please provide more data."}), 400

    # Fill missing weekly values to maintain continuity
    weekly_orders = weekly_orders.set_index("order_week").asfreq("W", method="ffill")
    weekly_orders["order_count"].fillna(weekly_orders["order_count"].median(), inplace=True)

    # Prepare the data for Prophet model
    prophet_df = weekly_orders.reset_index().rename(columns={"order_week": "ds", "order_count": "y"})

    # Fit the Prophet model
    try:
        prophet_model = fit_prophet_model(prophet_df)
    except Exception as e:
        logging.error(f"Error fitting Prophet model: {str(e)}")
        return jsonify({"error": f"Error fitting Prophet model: {str(e)}"}), 400

    # Generate the forecast
    try:
        future = prophet_model.make_future_dataframe(periods=12, freq="W")  # Create future DataFrame for 12 weeks
        forecast = prophet_model.predict(future)  # Generate the forecast
    except Exception as e:
        logging.error(f"Error generating forecast: {str(e)}")
        return jsonify({"error": f"Error generating forecast: {str(e)}"}), 400

    # Calculate MAE and R² for forecast evaluation
    prophet_df["yhat"] = forecast["yhat"].iloc[: len(prophet_df)]
    mae = mean_absolute_error(prophet_df["y"], prophet_df["yhat"])
    r2 = r2_score(prophet_df["y"], prophet_df["yhat"])

    # Generate forecast plot
    try:
        fig = plot_plotly(prophet_model, forecast)  # Create plot
        fig.update_layout(hovermode="x unified")  # Set hover info
        fig.update_traces(
            hovertemplate=(
                "<b>Date</b>: %{x}<br>"
                "<b>Count</b>: %{y:.2f}<br>"
            ),
        )
        plot_json = fig.to_json()  # Convert plot to JSON
        logging.info("Plot generated successfully.")
    except Exception as e:
        logging.error(f"Error generating plot: {str(e)}")
        return jsonify({"error": f"Error generating plot: {str(e)}"}), 400

    # Extract exact predictions for future weeks
    last_historical_date = prophet_df["ds"].max()  # Find the last date in the historical data
    future_predictions_df = forecast[forecast["ds"] > last_historical_date]  # Extract only future predictions
    exact_predictions = future_predictions_df[["ds", "yhat"]].head(5).to_dict(orient="records")  # First 4 predictions

    # Send the final response with predictions, metrics, and plot
    elapsed_time = time.time() - start_time  # Calculate elapsed time for processing
    logging.info(f"Total computation time: {elapsed_time:.2f} seconds.")
    return jsonify({
        "predictions": forecast[["ds", "yhat"]].head(2).to_dict(orient="records"),
        "mae": mae,
        "r2": r2,
        "exact_predictions": exact_predictions,
        "plot": plot_json
    })
