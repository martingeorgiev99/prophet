from flask import Blueprint, request, render_template, jsonify
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, r2_score
from prophet.plot import plot_plotly
from .utils import (find_column_name, filter_outliers_with_z_score, column_mapping,
                    clean_data, aggregate_weekly_orders, fit_prophet_model)
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])

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

    # Validate required columns
    actual_order_status = find_column_name(column_mapping["expected_order_status"], df.columns)
    actual_order_date = find_column_name(column_mapping["expected_order_date"], df.columns)
    
    if not actual_order_status:
        logging.error("Missing required column for order status.")
        return jsonify({"error": "Missing required column: 'order_status'."}), 400
    if not actual_order_date:
        logging.error("Missing required column for order date.")
        return jsonify({"error": "Missing required column: 'order_date'."}), 400

    # Input Validation for Data Types
    if not pd.api.types.is_string_dtype(df[actual_order_status]):
        logging.error("Invalid data type for order status.")
        return jsonify({"error": "Invalid data type for 'order_status'. Expected string values."}), 400

    # Log and Convert 'order_date' to Datetime
    try:
        df[actual_order_date] = pd.to_datetime(df[actual_order_date], errors='raise')  # Convert to datetime
        logging.info("Order dates converted to datetime successfully.")
    except Exception as e:
        logging.error(f"Invalid data type for 'order_date': {str(e)}")
        return jsonify({"error": "Invalid data type for 'order_date'. Expected datetime values."}), 400

    # Data Cleaning and Preprocessing
    try:
        df = clean_data(df, actual_order_status, actual_order_date)  # Clean the data
        logging.info("Data cleaned and preprocessed.")
        logging.info(f"Length after cleaning: {len(df)}")  # Log length after cleaning
    except Exception as e:
        logging.error(f"Error during data preprocessing: {str(e)}")
        return jsonify({"error": f"Error during data preprocessing: {str(e)}"}), 400

    # Weekly aggregation using the defined function
    try:
        weekly_orders = aggregate_weekly_orders(df, actual_order_date)  # Call the aggregation function
        logging.info("Weekly orders aggregated.")
        logging.info(f"Length after aggregation: {len(weekly_orders)}")  # Log length after aggregation
    except Exception as e:
        logging.error(f"Error during weekly aggregation: {str(e)}")
        return jsonify({"error": f"Error during weekly aggregation: {str(e)}"}), 400

    # Outlier Filtering
    try:
        weekly_orders = weekly_orders[filter_outliers_with_z_score(weekly_orders["order_count"])]  # Filter outliers
        logging.info("Outliers filtered from weekly orders.")
        logging.info(f"Length after filtering outliers: {len(weekly_orders)}")  # Log length after filtering
    except Exception as e:
        logging.error(f"Error filtering outliers: {str(e)}")
        return jsonify({"error": f"Error filtering outliers: {str(e)}"}), 400

    # Ensuring data continuity
    if weekly_orders.shape[0] < 2:
        logging.warning("Not enough valid data to make a forecast.")
        return jsonify({"error": "Not enough valid data to make a forecast. Please provide more data."}), 400

    # Fill missing values with the median
    try:
        weekly_orders = weekly_orders.set_index("order_week").asfreq("W", method="ffill")  # Forward fill missing weeks
        weekly_orders["order_count"].fillna(weekly_orders["order_count"].median(), inplace=True)  # Fill missing counts
        prophet_df = weekly_orders.reset_index().rename(columns={"order_week": "ds", "order_count": "y"})  # Prepare DataFrame for Prophet
        logging.info("Missing values filled.")
    except Exception as e:
        logging.error(f"Error filling missing values: {str(e)}")
        return jsonify({"error": f"Error filling missing values: {str(e)}"}), 400

    # Prophet Model
    try:
        prophet_model = fit_prophet_model(prophet_df)  # Fit the Prophet model
        logging.info("Prophet model fitted successfully.")
    except Exception as e:
        logging.error(f"Error fitting Prophet model: {str(e)}")
        return jsonify({"error": f"Error fitting Prophet model: {str(e)}"}), 400

    # Forecast
    try:
        future = prophet_model.make_future_dataframe(periods=12, freq="W")  # Create future DataFrame
        forecast = prophet_model.predict(future)  # Make predictions
        logging.info("Forecast generated successfully.")
    except Exception as e:
        logging.error(f"Error generating forecast: {str(e)}")
        return jsonify({"error": f"Error generating forecast: {str(e)}"}), 400

    # Metrics Calculation
    prophet_df["yhat"] = forecast["yhat"].iloc[: len(prophet_df)]  # Add predicted values to DataFrame
    mae = mean_absolute_error(prophet_df["y"], prophet_df["yhat"])  # Calculate MAE
    r2 = r2_score(prophet_df["y"], prophet_df["yhat"])  # Calculate R²

    # Plotting
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

    # Extracting Predictions
    future_predictions = forecast.head(2)[["ds", "yhat"]].to_dict(orient="records")  # Get future predictions
    last_historical_date = prophet_df["ds"].max()  # Get last historical date
    future_predictions_df = forecast[forecast["ds"] > last_historical_date]  # Get future predictions only
    exact_predictions = future_predictions_df[["ds", "yhat"]].head(4).to_dict(orient="records")  # Prepare exact predictions

    # Final JSON Response
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    logging.info(f"Total computation time: {elapsed_time:.2f} seconds.")
    logging.info("Returning predictions and metrics.")
    return jsonify({
        "predictions": future_predictions,
        "mae": mae,
        "r2": r2,
        "exact_predictions": exact_predictions,
        "plot": plot_json
    })
