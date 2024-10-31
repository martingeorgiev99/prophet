# Flask Sales Forecast App

This application provides a web interface for uploading a CSV file containing sales data. It processes the data, performs time series forecasting using Facebook Prophet, and visualizes the forecast with Plotly. This app is useful for predicting weekly sales figures based on historical data.

## Features

- **File Upload**: Allows users to upload CSV files containing sales data.
- **Data Preprocessing**: Cleans and preprocesses the data by filtering and handling outliers.
- **Time Series Forecasting**: Utilizes Facebook Prophet to forecast weekly sales.
- **Forecast Visualization**: Interactive visualization of forecasted values with Plotly.
- **Error and Metric Display**: Shows Mean Absolute Error (MAE) and R² for forecast accuracy.

## File Structure

```plaintext
FLASKAPP123/
├── app/
│   ├── __init__.py          # Initializes Flask app and registers blueprint
│   ├── routes.py            # Defines API routes, including forecasting logic
│   └── utils.py             # Helper functions for column mapping and outlier detection
├── static/
│   ├── scripts.js           # JavaScript for form submission and Plotly visualization
│   └── styles.css           # CSS for styling the frontend
├── templates/
│   └── index.html           # Main HTML page for file upload and forecast display
├── main.py                  # Entry point for running the Flask app
├── Procfile                 # Procfile for deploying with Gunicorn
├── Dockerfile               # Dockerfile for containerizing the application
├── docker-compose.yml       # Docker Compose file for running the app with services
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation (this file)
```

## Installation

1. **Clone the repository**:
  ```bash
  git clone https://github.com/martingeorgiev99/flaskapp123.git
  cd flaskapp123
  ```

2. **Install dependencies**:
  ```bash
  pip install -r requirements.txt
  ```

3. **Run the app locally**:
  ```bash
  python main.py
  ```

  The app will be available at `http://127.0.0.1:5000`.

## Using Docker

Build and run the application using Docker: Make sure Docker is installed on your machine. Then run:
  ```bash
  docker-compose up --build
  ```
The app will be available at `http://127.0.0.1:5000`.

## Usage

1. Open the app in your browser.
2. Upload a CSV file with the required columns:
  - `order_status` (or other valid names such as `status`, `order_state`)
  - `order_date` (or other valid names like `date`, `purchase_date`)
3. Click "Upload" to submit the file.
4. View the forecast results, including the Mean Absolute Error (MAE), R² score, and a plot of the forecasted sales.

## API Endpoints

### `POST /forecast`

**Description**: Accepts a CSV file, processes it, and returns sales forecasts.

- **Request Body**: Form-data with a single file field (`file`), which should contain the CSV file.
- **Response**:
  - **Success**: JSON with predictions, MAE, R² score, and plot data.
  - **Error**: JSON with an error message explaining any issues with the file or forecasting.

## Configuration

- **Templates Directory**: `app/__init__.py` specifies the templates directory (`../templates`).
- **Static Files Directory**: `app/__init__.py` specifies the static files directory (`../static`).

## Deployment

To deploy the app, ensure you have **Gunicorn** installed and use the `Procfile`:

```bash
web: gunicorn app.main:app
```
Deploy to a platform that supports Procfiles, such as **Heroku**. Here’s a basic guide to deploying on Heroku:

1. **Log in to Heroku**:
  ```bash
  heroku login
  ```

2. **Create a Heroku app**:
  ```bash
  heroku create your-app-name
  ```

3. **Push the code to Heroku**:
  ```bash
  git push heroku main
  ```

4. **Set environment variables** (if needed):
  You can set environment variables on Heroku using:
  ```bash
  heroku config:set VAR_NAME=value
  ```

5. **Scale the app**:
  ```bash
  heroku ps:scale web=1
  ```

6. **Open the app**:
```bash
  heroku open
```

  Your application should now be running on Heroku at `https://your-app-name.herokuapp.com`.

## Dependencies

The project relies on the following libraries:

- **Flask**: For creating the web application.
- **Pandas**: For data manipulation and CSV handling.
- **Prophet**: For time series forecasting.
- **Scikit-learn**: For evaluating the forecast with MAE and R².
- **Plotly**: For interactive forecast visualization.
- **Gunicorn**: For deployment in production.

All dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```
