from datetime import datetime, timedelta, timezone
import os
import mysql.connector
import json
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from werkzeug.security import check_password_hash
from flask import Flask, request, jsonify, abort
import logging
import time
from functools import wraps
import base64
from apscheduler.schedulers.background import BackgroundScheduler
import pytz
from flask import send_from_directory
from mysql.connector import errorcode
from dateutil.parser import parse as dateutil_parse 

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)

# Database Connection - variables set in Heroku
try:
    db = mysql.connector.connect(
        host=os.environ.get("MYSQL_ADDON_HOST"),
        user=os.environ.get("MYSQL_ADDON_USER"),
        password=os.environ.get("MYSQL_ADDON_PASSWORD"),
        database=os.environ.get("MYSQL_ADDON_DB"),
        port=int(os.environ.get("MYSQL_ADDON_PORT", 3306)),  # 3306 if not set
        charset="utf8mb4"
    )
    
    cursor = db.cursor(dictionary=True)
    logging.info("Database connection established successfully.")
except Exception as e:
    logging.error(f"Database connection error: {str(e)}")
    raise e

# Auth
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get('Authorization')
        if not auth:
            logging.warning("No authorization header provided.")
            abort(403, description="Forbidden: Authentication credentials were not provided.")

        try:
            # Decode Base64 encoded credentials
            auth_type, creds = auth.split(None, 1)
            if auth_type.lower() != 'basic':
                logging.warning("Unsupported authorization type.")
                abort(403, description="Forbidden: Unsupported authorization type.")

            decoded_creds = base64.b64decode(creds).decode('utf-8')
            username, password = decoded_creds.split(':', 1)
        except Exception as e:
            logging.error(f"Error parsing authorization header: {str(e)}")
            abort(403, description="Forbidden: Invalid authorization header.")

        if not validate_user(username, password):
            logging.warning(f"Authentication failed for user: {username}")
            abort(403, description="Forbidden: Invalid username or password.")

        # Attach user_id to request
        cursor.execute("SELECT user_id FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        if user:
            request.user_id = user['user_id']
        else:
            logging.error(f"User '{username}' not found after authentication.")
            abort(403, description="Forbidden: User not found.")

        return f(*args, **kwargs)
    return decorated

def validate_user(username, password):
    """Validates the user credentials."""
    try:
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        if user and check_password_hash(user['password_hash'], password):
            return True
        return False
    except Exception as e:
        logging.error(f"Error validating user '{username}': {str(e)}")
        return False

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "ERROR",
        "error_message": "The requested resource was not found on the server.",
        "suggestion": "Please check the URL for errors or refer to the API documentation."
    }), 404

@app.errorhandler(403)
def not_found(error):
    return jsonify({
        "status": "ERROR",
        "error_message": "You do not have permission to access this resource.",
        "suggestion": "Enter correct login credentials"
    }), 403
    
def calculate_forecast(user_id, reference_datetime=None):
    """
    Perform data retrieval, Prophet fitting, forecast generation,
    performance metrics calculation, and insert into forecast_performance table.
    Returns a dict with 'forecast_data' on success, or an error dict otherwise.
    """
    start_time = time.time()  # Start timing the forecast calculation

    # Use provided reference_datetime or current UTC time
    if reference_datetime is None:
        reference_datetime = datetime.now(timezone.utc)
    else:
        try:
            reference_datetime = pd.to_datetime(reference_datetime).tz_localize('UTC')
        except Exception as e:
            logging.error(f"Invalid reference_datetime format: {str(e)}")
            return {"error": "Invalid reference_datetime format."}

    logging.info(f"Reference DateTime: {reference_datetime}")

    # Retrieve data
    try:
        cursor.execute("SELECT * FROM orders WHERE user_id = %s", (user_id,))
        orders_data = cursor.fetchall()
    except Exception as e:
        logging.error(f"Error retrieving orders for user {user_id}: {str(e)}")
        return {"error": "Database error retrieving orders."}

    if not orders_data:
        logging.error(f"No orders found for user {user_id}.")
        return {"error": "No orders found for this user"}

    # Create DataFrame
    df = pd.DataFrame(
        orders_data,
        columns=["order_id", "user_id", "order_status", "order_date",
                 "old_status", "status_last_changed_at", "created_at"]
    )
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df = df.dropna(subset=['order_date', 'order_status'])

    logging.info("Raw data sample:")
    logging.info(df.head())

    #  Only flag "Cancelled" statuses
    #  If 'Отказана', 'Терминирана', or 'Cancelled', label as Cancelled.
    cancelled_keywords = {"Отказана", "Терминирана", "Cancelled"}
    df['order_status'] = df['order_status'].apply(
        lambda s: "Cancelled" if s in cancelled_keywords else s
    )

    # Filter out cancelled orders
    df = df[df['order_status'] != 'Cancelled']
    logging.info("After filtering canceled orders, sample:")
    logging.info(df.head())

    # Aggregate weekly, Sunday-based
    weekly_orders = (
        df.groupby(pd.Grouper(key='order_date', freq='W-SUN'))
          .size()
          .reset_index(name='order_count')
    )
    logging.info("Weekly aggregation sample:")
    logging.info(weekly_orders.head())

    if weekly_orders.empty:
        logging.error("No data after aggregation.")
        return {"error": "No data after aggregation"}

    if weekly_orders.shape[0] < 2:
        logging.error("Not enough data to make a forecast.")
        return {"error": "Not enough data to make a forecast"}

    # Rename columns for Prophet (required)
    prophet_df = weekly_orders.rename(columns={"order_date": "ds", "order_count": "y"})[['ds', 'y']]
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], errors='coerce').dt.tz_localize('UTC')
    prophet_df = prophet_df.dropna(subset=['ds'])
    
    # Remove timezone information to make 'ds' timezone-naive
    prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
    logging.info(f"Prophet DF 'ds' timezone info: {prophet_df['ds'].dt.tz}")

    # Define cutoff date
    now = reference_datetime
    today = now.astimezone(timezone.utc).date()
    
    # Calculate last Sunday
    last_sunday = today - timedelta(days=(today.weekday() + 1) % 7)

    # If today is Sunday, check time
    if now.astimezone(timezone.utc).date().weekday() == 6:
        if now.astimezone(timezone.utc).time() >= datetime.strptime('00:00:01', '%H:%M:%S').time():
            last_completed_sunday = today
        else:
            last_completed_sunday = today - timedelta(weeks=1)
    else:
        last_completed_sunday = last_sunday

    cutoff_date = last_completed_sunday
    logging.info(f"Cutoff Date: {cutoff_date}")

    # Filter only complete weeks
    prophet_df = prophet_df[prophet_df['ds'].dt.date <= cutoff_date]
    logging.info("Prophet DF after cutoff filtering:")
    logging.info(prophet_df)

    if prophet_df.empty:
        logging.error("No complete weeks of data available for forecasting.")
        return {"error": "No complete weeks of data available for forecasting"}

    logging.info("Prophet DF sample before forecasting:")
    logging.info(prophet_df.head())

    # Fit Prophet
    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.15,
            seasonality_prior_scale=10.0
        )
        model.fit(prophet_df)
        logging.info("Prophet model fitted successfully.")
    except Exception as e:
        logging.error(f"Error fitting Prophet model: {str(e)}")
        return {"error": "Error fitting Prophet model."}

    # Future predictions (4 weeks, W-SUN)
    try:
        future = model.make_future_dataframe(periods=4, freq='W-SUN')
        forecast = model.predict(future)
        logging.info("Forecast generated.")
    except Exception as e:
        logging.error(f"Error generating forecast: {str(e)}")
        return {"error": "Error generating forecast."}

    # Ensure predictions are non-negative
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

    # Filter forecast to only include weeks after cutoff_date
    forecast = forecast[forecast['ds'].dt.date > cutoff_date]
    logging.info("Forecast after filtering to include only future weeks:")
    logging.info(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(6))

    # Select only required columns (next 4 weeks)
    forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(4)

    # Performance metrics on historical portion
    try:
        historical_forecast = model.predict(prophet_df[['ds']])
        historical_forecast = historical_forecast.set_index('ds')
        prophet_df = prophet_df.set_index('ds')

        # Align actual and predicted
        historical_forecast_aligned = historical_forecast.loc[prophet_df.index]
        y_true = prophet_df['y']
        y_pred = historical_forecast_aligned['yhat']

        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        logging.info(f"Performance Metrics - MAE: {mae}, R²: {r2}, MAPE: {mape}")
    except Exception as e:
        logging.error(f"Error calculating performance metrics: {str(e)}")
        mae, r2, mape = None, None, None

    # Prepare forecast_data for response
    output = []
    for _, row in forecast_data.iterrows():
        ds = pd.to_datetime(row['ds'])
        monday_date = ds - timedelta(days=6)  # ds is Sunday
        sunday_date = ds

        forecast_entry = {
            "week_start": monday_date.strftime('%B %d, %Y'),
            "week_end": sunday_date.strftime('%B %d, %Y'),
            "predicted_sales": float(row['yhat']),
            "lower_bound": float(row['yhat_lower']),
            "upper_bound": float(row['yhat_upper']),
        }
        output.append(forecast_entry)

    # Determine last_forecast_date_db
    try:
        # Assuming 'week_end' is in the forecast_entries
        last_forecast_date = output[-1]['week_end']
        last_forecast_date_db = pd.to_datetime(last_forecast_date).date()
    except Exception as e:
        logging.error(f"Error determining last_forecast_date_db: {str(e)}")
        last_forecast_date_db = datetime.now(timezone.utc).date()  # Fallback to current date

    # Insert into forecast_performance once per forecast
    try:
        # Convert forecast_data to JSON string
        forecast_json = json.dumps(output)
        logging.debug(f"Forecast JSON: {forecast_json}")

        insert_query = """
            INSERT INTO forecast_performance 
            (user_id, forecast_date, forecast_data, r2, mae, mape, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                forecast_data = VALUES(forecast_data),
                r2 = VALUES(r2),
                mae = VALUES(mae),
                mape = VALUES(mape),
                created_at = VALUES(created_at)
        """
        cursor.execute(insert_query, (
            user_id,
            datetime.now(timezone.utc),
            forecast_json,
            r2 if r2 is not None else 0,
            mae if mae is not None else 0,
            mape if mape is not None else 0,
            datetime.now(timezone.utc)
        ))
        db.commit()
        logging.info(f"Forecast performance metrics inserted/updated for user {user_id}.")
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_DUP_ENTRY:
            logging.error(f"Duplicate entry for user {user_id} and forecast_date. Skipping insertion.")
        else:
            logging.error(f"Error inserting forecast performance: {str(err)}")
    except Exception as e:
        logging.error(f"Error inserting forecast performance: {str(e)}")

    # Insert forecast data into forecast_cache
    try:
        # Update or Insert into forecast_cache
        cursor.execute("SELECT * FROM forecast_cache WHERE user_id = %s", (user_id,))
        existing = cursor.fetchone()

        if existing:
            cursor.execute("""
                UPDATE forecast_cache
                SET forecast_date = %s,
                    forecast_data = %s,
                    created_at = %s
                WHERE user_id = %s
            """, (
                last_forecast_date_db,
                forecast_json,
                datetime.now(timezone.utc),
                user_id
            ))
            logging.info(f"Updated forecast_cache for user {user_id}.")
        else:
            cursor.execute("""
                INSERT INTO forecast_cache (user_id, forecast_date, forecast_data, created_at)
                VALUES (%s, %s, %s, %s)
            """, (
                user_id,
                last_forecast_date_db,
                forecast_json,
                datetime.now(timezone.utc)
            ))
            logging.info(f"Inserted new forecast_cache entry for user {user_id}.")

        db.commit()
    except Exception as e:
        logging.error(f"Error updating forecast_cache: {str(e)}")

    elapsed_time = time.time() - start_time
    logging.info(f"Total computation time: {elapsed_time:.2f} seconds.")

    return {"forecast_data": output}

@app.route('/insertOrders', methods=['POST'])
@require_auth
def insert_orders():
    """Insert single or multiple orders for a given user."""
    try:
        data = request.get_json()
        if not data:
            logging.error("Request body cannot be empty.")
            return jsonify({
                "status": "ERROR",
                "error_message": "Request body cannot be empty"
            }), 400

        user_id = request.user_id  # from @require_auth

        def parse_and_validate_date(date_str, index=None):
            """
            Parses date_str into datetime, ensures it's not in the future.
            Returns the normalized string "YYYY-MM-DD HH:MM:SS" or raises ValueError.
            """
            raw_date = date_str.strip()
            try:
                dt = dateutil_parse(raw_date, fuzzy=False)
            except Exception as ex:
                logging.error(
                    "Failed to parse date in order%s: raw=%r, ASCII=%s, error=%s",
                    f" at index {index}" if index is not None else "",
                    raw_date,
                    [ord(ch) for ch in raw_date],
                    ex
                )
                raise ValueError(
                    f"Invalid date format for order at index {index}. "
                    "Expected format: YYYY-MM-DD HH:MM:SS (or parseable equivalent)."
                )

            if dt > datetime.now():
                raise ValueError(
                    f"Order date cannot be in the future for order at index {index}."
                )
            
            return dt.strftime("%Y-%m-%d %H:%M:%S")

        def validate_single_order(order, index=None):
            """
            Checks for required fields, validates date, no future date, etc.
            Returns (True, formatted_date_string) or (False, error_message).
            """
            required = ['id', 'order_date', 'order_status']
            if not all(k in order for k in required):
                msg = (f"Missing fields in order at index {index}: {order}"
                       if index is not None else
                       "Missing required fields (id, order_date, order_status).")
                logging.error(msg)
                return False, msg

            try:
                good_date_str = parse_and_validate_date(order['order_date'], index=index)
            except ValueError as ve:
                return False, str(ve)

            return True, good_date_str

        # Bulk insert scenario
        if 'orders' in data and isinstance(data['orders'], list):
            orders_list = data['orders']
            if not orders_list:
                msg = "Orders list cannot be empty"
                logging.error(msg)
                return jsonify({"status": "ERROR", "error_message": msg}), 400

            values = []
            for index, order in enumerate(orders_list):
                ok, result = validate_single_order(order, index=index)
                if not ok:
                    return jsonify({"status": "ERROR", "error_message": result}), 400
                
                # Store the order_status as-is (Bulgarian text)
                formatted_date = result
                values.append((order['id'], user_id, order['order_status'], formatted_date))

            query = """
                INSERT IGNORE INTO orders (order_id, user_id, order_status, order_date)
                VALUES (%s, %s, %s, %s)
            """
            cursor.executemany(query, values)
            db.commit()
            inserted_count = cursor.rowcount

            logging.info(f"{inserted_count} orders inserted successfully, duplicates ignored.")
            return jsonify({
                "status": "OK",
                "message": f"{inserted_count} orders inserted successfully, duplicates ignored."
            }), 200

        # Single order scenario
        else:
            ok, result = validate_single_order(data)
            if not ok:
                return jsonify({"status": "ERROR", "error_message": result}), 400

            formatted_date = result
            query = """
                INSERT IGNORE INTO orders (order_id, user_id, order_status, order_date)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (
                data['id'],
                user_id,
                data['order_status'],
                formatted_date
            ))
            db.commit()

            if cursor.rowcount > 0:
                logging.info("Order inserted successfully.")
                return jsonify({"status": "OK", "message": "Order inserted successfully."}), 200
            else:
                logging.info("Order already exists and was ignored.")
                return jsonify({"status": "OK", "message": "Order already exists and was ignored."}), 200

    except Exception as e:
        logging.error(f"Error inserting orders: {str(e)}", exc_info=True)
        return jsonify({
            "status": "ERROR",
            "error_message": f"Internal server error: {str(e)}"
        }), 500

def recalculate_and_cache_forecast(user_id, reference_datetime=None):
    """
    Runs the full forecast calculation, then upserts (updates/inserts) 
    the forecast_cache table.
    Returns the forecast_data list (each element is a dict with week_start, etc.)
    """
    new_forecast_data = calculate_forecast(user_id, reference_datetime)
    if 'forecast_data' not in new_forecast_data:
        # If there's an error instead, log and return empty list
        logging.error(f"Forecast not generated for user {user_id}: {new_forecast_data}")
        return []

    forecast_entries = new_forecast_data['forecast_data']

    # Since forecast_cache is handled within calculate_forecast, no need to handle it here
    return forecast_entries

@app.route('/getForecastByDate', methods=['POST'])
@require_auth
def get_forecast_by_date():
    """
    Returns the forecast for a specific reference_datetime.
    Useful for testing or showcasing purposes.
    """
    user_id = request.user_id
    data = request.get_json()

    if not data or 'reference_datetime' not in data:
        return jsonify({"status": "ERROR", "error_message": "reference_datetime is required"}), 400

    reference_datetime = data['reference_datetime']

    # Calculate forecast as of the provided reference_datetime
    forecast_data = recalculate_and_cache_forecast(user_id, reference_datetime)
    
    if not forecast_data:
        return jsonify({"status": "ERROR", "error_message": "Could not generate forecast for the provided reference_datetime."}), 500

    return jsonify({
        "status": "OK",
        "message": f"Forecast generated for reference_datetime {reference_datetime}",
        "forecasts": format_forecast_for_response(forecast_data)["forecasts"]
    }), 200

@app.route('/updateForecast', methods=['POST'])
@require_auth
def update_forecast():
    """
    Manually triggers a forecast update regardless of conditions.
    Returns updated forecast after recalculation.
    """
    user_id = request.user_id
    logging.info(f"User {user_id} requested a manual forecast update.")

    # Always recalculate and cache forecast
    forecast_data = recalculate_and_cache_forecast(user_id)

    if not forecast_data:
        logging.error(f"Forecast not generated for user {user_id} (possibly insufficient data).")
        return jsonify({"status": "ERROR", "error_message": "Forecast not generated."}), 500

    logging.info(f"Forecast updated successfully for user {user_id}.")
    return jsonify({
        "status": "OK",
        "message": f"Forecast updated successfully for user {user_id}",
        "forecasts": format_forecast_for_response(forecast_data)["forecasts"]
    }), 200

def count_status_changes_in_last_full_week(user_id):
    """
    Counts how many distinct status changes occurred in the most recently
    completed Monday–Sunday. The 'last full week' is the prior Monday..Sunday
    relative to today's Monday.
    """
    try:
        now = datetime.now(timezone.utc)
        today = now.date()
        this_weeks_monday = today - timedelta(days=today.weekday())
        last_weeks_monday = this_weeks_monday - timedelta(weeks=1)
        last_weeks_sunday = last_weeks_monday + timedelta(days=6)

        query = """
            SELECT COUNT(*) AS cnt
            FROM orders
            WHERE user_id = %s
              AND status_last_changed_at >= %s
              AND status_last_changed_at <= %s
        """
        cursor.execute(query, (user_id, last_weeks_monday, last_weeks_sunday))
        result = cursor.fetchone()

        if not result or 'cnt' not in result:
            return 0
        return result['cnt']
    except Exception as e:
        logging.error(f"Error counting status changes: {str(e)}")
        return 0

def format_forecast_for_response(forecast_data):
    """Helper to transform forecast entries into a suitable JSON structure."""
    transformed = []
    for entry in forecast_data:
        transformed.append({
            "week_start": entry['week_start'],
            "week_end": entry['week_end'],
            "predicted_sales": entry['predicted_sales'],
            "lower_bound": entry['lower_bound'],
            "upper_bound": entry['upper_bound']
        })
    return {"forecasts": transformed}

@app.route('/updateOrderStatus', methods=['POST'])
@require_auth
def update_order_status():
    user_id = request.user_id
    data = request.get_json()
    order_id = data.get('order_id')
    new_status = data.get('order_status')

    if not order_id or not new_status:
        return jsonify({"status": "ERROR", "error_message": "Both order_id and order_status are required."}), 400

    try:
        cursor.execute("""
            SELECT order_status, order_date, user_id 
            FROM orders 
            WHERE order_id = %s
        """, (order_id,))
        order = cursor.fetchone()

        if not order:
            return jsonify({"status": "ERROR", "error_message": f"Order with ID {order_id} not found."}), 404

        old_status_db = order['order_status']
        # Check if 'order_date' is present and valid
        if 'order_date' not in order or order['order_date'] is None:
            return jsonify({"status": "ERROR", "error_message": f"Order date for ID {order_id} is missing."}), 500
        
        order_date = pd.to_datetime(order['order_date']).date()
        user_id_from_order = order['user_id']

        # Ensure the authenticated user matches the order's user
        if user_id != user_id_from_order:
            return jsonify({"status": "ERROR", "error_message": "You do not have permission to update this order."}), 403

        # If there's no actual change in status, do nothing
        if new_status == old_status_db:
            return jsonify({"status": "OK", "message": f"Order {order_id} status is unchanged."}), 200

        # Check if the order's week is fully complete
        now = datetime.now(timezone.utc)
        today = now.date()
        week_start_date = order_date - timedelta(days=order_date.weekday())  # Monday
        week_end_date = week_start_date + timedelta(days=6)                  # Sunday

        if week_end_date <= today:
            # Allow the update for a completed week
            cursor.execute("""
                UPDATE orders
                SET order_status = %s,
                    old_status = %s,
                    status_last_changed_at = %s
                WHERE order_id = %s
            """, (new_status, old_status_db, now, order_id))
            db.commit()

            return jsonify({
                "status": "OK",
                "message": f"Order {order_id} status updated from {old_status_db} to {new_status}."
            }), 200
        else:
            # The week is not yet complete
            return jsonify({
                "status": "INFO",
                "message": f"Order {order_id} update ignored (incomplete week)."
            }), 200

    except Exception as e:
        logging.error(f"Error updating order status: {str(e)}")
        return jsonify({"status": "ERROR", "error_message": f"Internal server error: {str(e)}"}), 500

@app.route('/getForecast', methods=['POST'])
@require_auth
def get_forecast():
    """
    Returns the latest cached forecast if it exists for the given user_id.
    If no cache is found, calculates once, caches it, then returns it.
    DOES NOT automatically recalc if there's only 1 new status change, etc.
    """
    user_id = request.user_id

    # Fetch latest forecast for this user
    try:
        cursor.execute("""
            SELECT forecast_date, forecast_data
            FROM forecast_cache
            WHERE user_id = %s
            ORDER BY forecast_date DESC
            LIMIT 1
        """, (user_id,))
        cached_forecast = cursor.fetchone()

        if cached_forecast is None:
            logging.info(f"No cached forecast found for user {user_id}. Calling updateForecast to generate a new forecast.")
            # Call the existing update_forecast function to generate a new forecast
            return update_forecast()  # Call the function directly

    except Exception as e:
        logging.error(f"Error retrieving forecast_cache for user {user_id}: {str(e)}")
        return jsonify({"status": "ERROR", "error_message": "Error retrieving forecast cache."}), 500

    # If forecast is cached, return it immediately
    try:
        forecast_data = json.loads(cached_forecast['forecast_data'])
        logging.info(f"Returning cached forecast for user {user_id}.")
        return jsonify({"status": "OK", "forecast_data": format_forecast_for_response(forecast_data)}), 200
    except (json.JSONDecodeError, TypeError) as e:
        logging.error(f"Error decoding cached forecast for user {user_id}: {str(e)}")
        return jsonify({"status": "ERROR", "error_message": "Cached forecast data corrupted; please update forecast."}), 500

def schedule_jobs():
    """
    Schedules automatic forecast updates based on the following conditions:
    1. More than 5 order status changes.
    2. It's Monday at 00:00:01 UTC.
    """
    scheduler = BackgroundScheduler(timezone=pytz.UTC)

    # Schedule the job to run every 360 minutes / 6 hours
    scheduler.add_job(
        func=auto_update_forecast_based_on_status_changes,
        trigger="interval",
        minutes=360,
        id='status_change_checker',
        max_instances=1,
        coalesce=True,
        replace_existing=True
    )

    # Schedule the job to run every Monday at 00:00:01 UTC
    scheduler.add_job(
        func=auto_update_forecast_on_monday,
        trigger='cron',
        day_of_week='mon',
        hour=0,
        minute=0,
        second=1,
        id='monday_forecast_updater',
        max_instances=1,
        coalesce=True,
        replace_existing=True
    )

    scheduler.start()
    logging.info(f"Scheduler started with jobs for automatic forecast updates. {scheduler.minutes} until next run.")

def auto_update_forecast_based_on_status_changes():
    """
    Automatically update forecast if more than 5 order status changes have occurred per user.
    """
    try:
        cursor.execute("SELECT user_id FROM users")
        users = cursor.fetchall()
        logging.info(f"Fetched {len(users)} users for status change checking.")

        for user in users:
            user_id = user['user_id']
            recent_changes = count_status_changes_in_last_full_week(user_id)
            threshold = 5
            logging.info(f"User {user_id}: Recent status changes in last full week: {recent_changes}")

            if recent_changes >= threshold:
                logging.info(f"Automatic Trigger: User {user_id} has {recent_changes} status changes. Updating forecast.")
                forecast_data = recalculate_and_cache_forecast(user_id)
                if forecast_data:
                    logging.info(f"Forecast updated automatically for user {user_id} due to status changes.")
                else:
                    logging.error(f"Failed to update forecast automatically for user {user_id}.")
            else:
                logging.info(f"No forecast update needed for user {user_id}.")
    except Exception as e:
        logging.error(f"Error in auto_update_forecast_based_on_status_changes: {str(e)}")

def auto_update_forecast_on_monday():
    """
    Automatically update forecast every Monday at 00:00:01 UTC.
    """
    try:
        cursor.execute("SELECT user_id FROM users")
        users = cursor.fetchall()
        logging.info(f"Fetched {len(users)} users for Monday forecast updating.")

        for user in users:
            user_id = user['user_id']
            logging.info(f"Automatic Trigger: Monday midnight UTC. Updating forecast for user {user_id}.")
            forecast_data = recalculate_and_cache_forecast(user_id)
            if forecast_data:
                logging.info(f"Forecast updated automatically for user {user_id} on Monday.")
            else:
                logging.error(f"Failed to update forecast automatically for user {user_id} on Monday.")
    except Exception as e:
        logging.error(f"Error in auto_update_forecast_on_monday: {str(e)}")

@app.route('/getForecastChart', methods=['POST'])
@require_auth
def get_forecast_chart():
    """
    Returns a Chart.js-compatible JSON (graph.json) for the latest forecast
    for the authenticated user.
    
    Example of returned JSON structure:

    {
      "type": "line",
      "data":{
          "label": ["2025-01-20","2025-01-27","2025-02-03","2025-02-10"],
          "datasets": [
              {
                  "yAxisID": "upper_bound",
                  "backgroundColor": "rgba(0, 0, 0, 0)",
                  "label": "Горна Граница",
                  "borderColor": "green",
                  "data": [247,260,268,256]
              },
              {
                  "yAxisID": "orders_count",
                  "backgroundColor": "rgba(0, 0, 0, 0)",
                  "label": "Поръчки (седмица)",
                  "borderColor": "red",
                  "data": [216,229,236,224]
              },
              {
                  "yAxisID": "lower_bound",
                  "backgroundColor": "rgba(0, 0, 0, 0)",
                  "label": "Ниска Граница",
                  "borderColor": "blue",
                  "data": [186,198,205,192]
              }
          ]
      },
      "options": {
          "scales": {
              "yAxes": [
                  {"id": "orders_count", "type": "linear", "position": "left"},
                  {"id": "orders_count", "type": "linear", "position": "right"}
              ]
          }
      }
    }
    """
    user_id = request.user_id 
    try:
        # Retrieve latest forecast_data from forecast_performance (or forecast_cache)
        cursor.execute("""
            SELECT forecast_data 
            FROM forecast_performance
            WHERE user_id = %s
            ORDER BY forecast_date DESC
            LIMIT 1
        """, (user_id,))
        row = cursor.fetchone()

        if not row:
            return jsonify({
                "status": "ERROR",
                "error_message": "No forecast found for this user."
            }), 404

        # Parse forecast_data JSON from the database
        forecast_data = json.loads(row['forecast_data'])  # Should be a list of forecast entries

        # Prepare arrays for Chart.js
        labels = []
        data_predicted = []  # "Поръчки (седмица)"
        data_upper = []      # "Горна Граница"
        data_lower = []      # "Долна Граница"

        for f_entry in forecast_data:
            # Each f_entry has structure:
            # {
            #   "week_start": "January 20, 2025",
            #   "week_end":   "January 26, 2025",
            #   "predicted_sales": 216.0,
            #   "lower_bound": 186.0,
            #   "upper_bound": 247.0
            # }
            
            # Convert "week_start" string (e.g. "January 20, 2025") to YYYY-MM-DD
            start_dt = datetime.strptime(f_entry["week_start"], "%B %d, %Y")
            label_str = start_dt.strftime("%Y-%m-%d")
            
            labels.append(label_str)
            data_predicted.append(f_entry["predicted_sales"])
            data_upper.append(f_entry["upper_bound"])
            data_lower.append(f_entry["lower_bound"])

        # Construct Chart.js JSON
        chart_json = {
            "type": "line",
            "data": {
                # The example uses "label" (singular) instead of "labels" (plural). 
                "label": labels,
                "datasets": [
                    {
                        "yAxisID": "upper_bound",
                        "backgroundColor": "rgba(0, 0, 0, 0)",
                        "label": "Горна Граница",
                        "borderColor": "green",
                        "data": data_upper
                    },
                    {
                        "yAxisID": "orders_count",
                        "backgroundColor": "rgba(0, 0, 0, 0)",
                        "label": "Поръчки (седмица)",
                        "borderColor": "red",
                        "data": data_predicted
                    },
                    {
                        "yAxisID": "lower_bound",
                        "backgroundColor": "rgba(0, 0, 0, 0)",
                        "label": "Долна Граница",
                        "borderColor": "blue",
                        "data": data_lower
                    }
                ]
            },
            "options": {
                "scales": {
                    "yAxes": [
                        {
                            "id": "orders_count",
                            "type": "linear",
                            "position": "left"
                        },
                        {
                            "id": "orders_count",
                            "type": "linear",
                            "position": "right"
                        }
                    ]
                }
            }
        }

        return jsonify(chart_json), 200

    except Exception as e:
        logging.error(f"Error retrieving forecast chart data: {str(e)}")
        return jsonify({"status": "ERROR", "error_message": str(e)}), 500

@app.route('/index')
def serve_index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    schedule_jobs()
    port = int(os.environ.get('PORT'))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)