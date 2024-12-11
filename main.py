import mysql.connector
import json
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_error, r2_score
from werkzeug.security import check_password_hash
from flask import Flask, request, jsonify

app = Flask(__name__)

# MySQL connection setup with dictionary cursor
db = mysql.connector.connect(
    host="localhost",
)

# Create a cursor to interact with the database, now returning results as dictionaries
cursor = db.cursor(dictionary=True)

# Function to validate user credentials
def validate_user(username, password):
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    
    # Accessing the user data if it's in dictionary format
    if user and check_password_hash(user['password_hash'], password):
        return True
    return False

def calculate_forecast(user_id):
    # Retrieve the data for the given user
    cursor.execute("SELECT * FROM orders WHERE user_id = %s", (user_id,))
    orders_data = cursor.fetchall()

    if not orders_data:
        return {"error": "No orders found for this user"}

    # Convert data to a DataFrame
    df = pd.DataFrame(orders_data, columns=["order_id", "user_id", "order_status", "order_date"])
    
    # Ensure 'order_date' is in datetime format
    df['order_date'] = pd.to_datetime(df['order_date'])

    # Aggregate the data weekly
    weekly_orders = df.groupby(pd.Grouper(key='order_date', freq='W')).size().reset_index(name="order_count")

    # Fit the Prophet model
    prophet_df = weekly_orders.rename(columns={"order_date": "ds", "order_count": "y"})
    model = Prophet(yearly_seasonality=True)
    model.fit(prophet_df)

    # Make future predictions
    future = model.make_future_dataframe(prophet_df, periods=12, freq='W')
    forecast = model.predict(future)

    # Calculate metrics
    mae = mean_absolute_error(prophet_df['y'], forecast['yhat'][:len(prophet_df)])
    r2 = r2_score(prophet_df['y'], forecast['yhat'][:len(prophet_df)])

    # Generate plot
    fig = plot_plotly(model, forecast)
    fig.update_layout(hovermode="x unified")
    plot_json = fig.to_json()

    # Store forecast in the database
    forecast_data = {
        "forecast_date": forecast['ds'].iloc[-1].strftime('%Y-%m-%d'),
        "predicted_sales": forecast['yhat'].iloc[-1],
        "forecast_data": json.dumps(forecast.to_dict(orient="records"))
    }
    
    cursor.execute("""
        INSERT INTO forecast_cache (user_id, forecast_date, predicted_sales, forecast_data)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE predicted_sales = VALUES(predicted_sales), forecast_data = VALUES(forecast_data)
    """, (
        user_id,
        forecast_data["forecast_date"],
        forecast_data["predicted_sales"],
        forecast_data["forecast_data"]
    ))
    db.commit()

    # Return forecast and metrics
    return {
        "predictions": forecast[['ds', 'yhat']].head(4).to_dict(orient="records"), # 4 weeks into the future
        "mae": mae, # Lower = Better
        "r2": r2, # Higher = Better
        "plot": plot_json
    }

@app.route('/insertOrders', methods=['POST'])
def insert_orders():
    # Authentication
    auth = request.authorization
    if not auth or not validate_user(auth.username, auth.password):
        return jsonify({"error": "Invalid username or password"}), 401
    
    # Extract user_id based on the authenticated username
    cursor.execute("SELECT user_id FROM users WHERE username = %s", (auth.username,))
    user = cursor.fetchone()
    
    if not user:
        return jsonify({"error": "User does not exist"}), 400
    
    user_id = user["user_id"]

    # Get the JSON data from the request
    try:
        data = request.get_json()

        # Check if the data is a list (bulk insert) or a single object (single insert)
        if isinstance(data, list):
            # Process bulk insert
            if not all('id' in order and 'order_date' in order and 'order_status' in order for order in data):
                return jsonify({"error": "Missing required fields in one or more orders"}), 400

            # Prepare bulk insert query
            query = """
                INSERT INTO orders (order_id, user_id, order_status, order_date)
                VALUES (%s, %s, %s, %s)
            """

            # Prepare list of order data, associating with user_id
            values = [(order['id'], user_id, order['order_status'], order['order_date']) for order in data]

            cursor.executemany(query, values)  # Bulk insert using executemany
            db.commit()
            return jsonify({"message": f"{len(values)} orders inserted successfully, duplicates ignored."}), 200
        
        else:
            return jsonify({"error": "Request body should be a list of orders"}), 400

    except Exception as e:
        return jsonify({"error": f"Invalid JSON format: {str(e)}"}), 400

@app.route('/getForecast', methods=['GET'])
def get_forecast():
    # Authentication
    auth = request.authorization
    if not auth or not validate_user(auth.username, auth.password):
        return jsonify({"error": "Invalid username or password"}), 401

    # Get user_id from the request
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    # Check if the forecast is cached
    cursor.execute("SELECT * FROM forecast_cache WHERE user_id = %s", (user_id,))
    cached_forecast = cursor.fetchone()

    if cached_forecast:
        return jsonify({
            "forecast_data": cached_forecast[3],  # forecast_data (JSON)
            "forecast_date": cached_forecast[2],
            "predicted_sales": cached_forecast[4]
        }), 200

    # If no cached forecast, calculate the forecast
    forecast_data = calculate_forecast(user_id)

    # Cache the forecast
    cursor.execute("""
        INSERT INTO forecast_cache (user_id, forecast_date, predicted_sales, forecast_data)
        VALUES (%s, %s, %s, %s)
    """, (
        user_id,
        forecast_data['forecast_date'],
        forecast_data['predicted_sales'],
        json.dumps(forecast_data)  # Store forecast data as JSON
    ))

    db.commit()
    return jsonify(forecast_data), 200
    
@app.route('/updateForecast', methods=['POST'])
def update_forecast():
    # Extract username and password from request headers
    auth = request.authorization
    if not auth or not validate_user(auth.username, auth.password):
        return jsonify({"error": "Invalid username or password"}), 401

    # Get the user_id and new forecast data from the request
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    # Simulate recalculating the forecast
    forecast_data = {
        "forecast": "Updated",
        "user_id": user_id,
        "predicted_sales": 6000.00,  # This will be recalculated based on new data
        "forecast_date": "2024-12-20"
    }

    try:
        # Check if the forecast for the user already exists in the cache
        cursor.execute("SELECT * FROM forecast_cache WHERE user_id = %s", (user_id,))
        existing_forecast = cursor.fetchone()

        if existing_forecast:
            # If forecast exists, update it
            cursor.execute("""
                UPDATE forecast_cache
                SET forecast_date = %s, predicted_sales = %s, forecast_data = %s
                WHERE user_id = %s
            """, (
                forecast_data['forecast_date'],
                forecast_data['predicted_sales'],
                json.dumps(forecast_data),  # Convert to JSON string
                user_id
            ))

            db.commit()
            return jsonify(forecast_data), 200
        else:
            # If no forecast exists, insert a new one
            cursor.execute("""
                INSERT INTO forecast_cache (user_id, forecast_date, predicted_sales, forecast_data)
                VALUES (%s, %s, %s, %s)
            """, (
                user_id,
                forecast_data['forecast_date'],
                forecast_data['predicted_sales'],
                json.dumps(forecast_data)  # Convert to JSON string
            ))

            db.commit()
            return jsonify(forecast_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/updateOrderStatus', methods=['POST'])
def update_order_status():
    # Extract username and password from request headers
    auth = request.authorization
    if not auth or not validate_user(auth.username, auth.password):
        return jsonify({"error": "Invalid username or password"}), 401

    # Get the order_id and new order status from the request
    order_id = request.json.get('order_id')
    new_status = request.json.get('order_status')

    if not order_id or not new_status:
        return jsonify({"error": "order_id and order_status are required"}), 400

    try:
        # Check if the order exists
        cursor.execute("SELECT * FROM orders WHERE order_id = %s", (order_id,))
        order = cursor.fetchone()

        if not order:
            return jsonify({"error": f"Order with ID {order_id} not found"}), 404

        # Update the order status
        cursor.execute("""
            UPDATE orders
            SET order_status = %s
            WHERE order_id = %s
        """, (new_status, order_id))

        db.commit()
        return jsonify({"message": f"Order {order_id} status updated to {new_status}"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
