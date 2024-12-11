import mysql.connector
from werkzeug.security import generate_password_hash

# Database connection
db = mysql.connector.connect(
    host="localhost",
)
cursor = db.cursor()

# Function to create user
def create_user(username, password):
    hashed_password = generate_password_hash(password)
    try:
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, hashed_password))
        db.commit()
        return {"message": f"User {username} created successfully."}
    except mysql.connector.IntegrityError:
        return {"error": "Username already exists."}

# Function to change the password of an existing user
def change_user_password(username, new_password):
    hashed_password = generate_password_hash(new_password)
    cursor.execute("UPDATE users SET password_hash = %s WHERE username = %s", (hashed_password, username))
    db.commit()
    return {"message": f"Password for {username} updated successfully."}

# Function to retrieve a user by their username (for validation)
def get_user_by_username(username):
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    return cursor.fetchone()

# Function to delete a user by username (if needed)
def delete_user(username):
    cursor.execute("DELETE FROM users WHERE username = %s", (username,))
    db.commit()
    return {"message": f"User {username} deleted successfully."}

def manage_users():
    action = input("Enter action (create/change/delete): ").strip().lower()
    username = input("Enter username: ").strip()
    
    if action == "create":
        password = input("Enter password: ").strip()
        print(create_user(username, password))
    elif action == "change":
        new_password = input("Enter new password: ").strip()
        print(change_user_password(username, new_password))
    elif action == "delete":
        print(delete_user(username))
    else:
        print("Invalid action.")
        
manage_users()
