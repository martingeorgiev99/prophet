from flask import Flask

def create_app():
    # Specify both template and static folders
    app = Flask(__name__, template_folder="../templates", static_folder="../static")

    from .routes import main
    app.register_blueprint(main)

    return app
