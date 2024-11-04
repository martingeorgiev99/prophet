from flask import Flask

def create_app():
    """
    Create and configure the Flask application.
    """
    app = Flask(__name__, template_folder="../templates", static_folder="../static")  # Initialize Flask app with specified template and static folder paths

    # Register blueprints
    from .routes import main  # Import the main blueprint from routes
    app.register_blueprint(main)  # Register the main blueprint with the app

    return app  # Return the configured Flask app
