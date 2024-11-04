from app import create_app
import os

app = create_app()  # Initialize the Flask application

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get the port from environment or default to 5000
    app.run(host="0.0.0.0", port=port)  # Run the app on all IP addresses at the specified port
