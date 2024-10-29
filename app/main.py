from flask import Flask, request, render_template, jsonify

import pandas as pd

from prophet import Prophet

from sklearn.metrics import mean_absolute_error, r2_score

from prophet.plot import plot_plotly


app = Flask(__name__)


# Common column names mapping for different CSV structures

column_mapping = {
    "expected_order_status": [
        "order_status",
        "status",
        "order_state",
        "orderCondition",
    ],
    "expected_order_date": [
        "order_date",
        "date",
        "order_time",
        "purchase_date",
        "orderDate",
    ],
    "expected_order_count": [
        "order_count",
        "orders",
        "num_orders",
        "order_quantity",
        "count_of_orders",
    ],
}


def find_column_name(possible_names, df_columns):
    for name in possible_names:
        if name in df_columns:
            return name

    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/forecast", methods=["POST"])
def forecast():
    # Check if a file is uploaded

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    # Validate if the file is a CSV

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Please upload a valid CSV file."})

    # Read the uploaded CSV file into a DataFrame

    try:
        df = pd.read_csv(file)

    except Exception as e:
        return jsonify({"error": f"Error reading file: {str(e)}"})

    # Get the actual column names from the CSV

    actual_order_status = find_column_name(
        column_mapping["expected_order_status"], df.columns
    )

    actual_order_date = find_column_name(
        column_mapping["expected_order_date"], df.columns
    )

    # Check if required columns are found

    if not actual_order_status or not actual_order_date:
        return jsonify({"error": "Missing required columns in the uploaded CSV file"})

    # Preprocess the data using the dynamically found columns

    df_cleaned = df.dropna(subset=[actual_order_status])

    df_cleaned = df_cleaned[
        df_cleaned[actual_order_status] != "Отказана"
    ]  # Exclude canceled orders

    # Convert order_date to datetime and handle invalid dates

    df_cleaned[actual_order_date] = pd.to_datetime(
        df_cleaned[actual_order_date], errors="coerce"
    )

    df_cleaned = df_cleaned.dropna(subset=[actual_order_date])

    # Handle weekly aggregation

    df_cleaned["order_week"] = (
        df_cleaned[actual_order_date].dt.to_period("W").apply(lambda r: r.start_time)
    )

    weekly_orders = (
        df_cleaned.groupby("order_week").size().reset_index(name="order_count")
    )

    # Clean outliers (weeks with very low order counts)

    weekly_orders_cleaned = weekly_orders[weekly_orders["order_count"] > 75]

    # Fill missing weeks with the median order count

    weekly_orders_cleaned = weekly_orders_cleaned.set_index("order_week").asfreq(
        "W", method="ffill"
    )

    median_order_count = weekly_orders_cleaned["order_count"].median()

    weekly_orders_cleaned["order_count"].fillna(median_order_count, inplace=True)

    # Ensure there are at least 2 non-NaN rows

    if weekly_orders_cleaned.shape[0] < 2:
        return jsonify(
            {
                "error": "Not enough valid data to make a forecast. Please provide more data."
            }
        )

    # Prepare the DataFrame for Prophet (renaming columns)

    prophet_df = weekly_orders_cleaned.reset_index().rename(
        columns={"order_week": "ds", "order_count": "y"}
    )

    # Initialize and fit the Prophet model

    try:
        prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            seasonality_mode="multiplicative",
        )

        prophet_model.fit(prophet_df)

    except Exception as e:
        return jsonify({"error": f"Error fitting Prophet model: {str(e)}"})

    # Make future predictions

    future = prophet_model.make_future_dataframe(periods=2, freq="W")

    forecast = prophet_model.predict(future)

    # Extract future predictions

    future_predictions = forecast.tail(2)[["ds", "yhat"]].to_dict(orient="records")

    # Calculate model evaluation metrics

    prophet_df["yhat"] = forecast["yhat"].iloc[: len(prophet_df)]

    mae = mean_absolute_error(prophet_df["y"], prophet_df["yhat"])

    r2 = r2_score(prophet_df["y"], prophet_df["yhat"])

    # Prepare hovertext for Plotly plot showing real or predicted values

    prophet_df["hover_text"] = "Actual Orders: " + prophet_df["y"].astype(str)

    forecast["hover_text"] = "Predicted Orders: " + forecast["yhat"].astype(str)

    # Generate Plotly interactive forecast plot with hovertext for real and predicted values

    try:
        fig = plot_plotly(prophet_model, forecast)

        fig.update_traces(text=forecast["hover_text"], hoverinfo="text+x+y")

        plot_json = fig.to_json()

    except Exception as e:
        return jsonify({"error": f"Error generating plot: {str(e)}"})

    # Extract exact predictions for display under MAE and R2

    exact_predictions = forecast[["ds", "yhat"]].tail(2).to_dict(orient="records")

    return jsonify(
        {
            "predictions": future_predictions,
            "mae": mae,
            "r2": r2,
            "plot": plot_json,  # Return Plotly plot as JSON to render on the client side
            "exact_predictions": exact_predictions,  # Return exact predictions for display
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
