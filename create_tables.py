import os
import mysql.connector
from mysql.connector import errorcode

# Database configuration read from Heroku environment variables
DB_CONFIG = {
    'host': os.environ.get("MYSQL_ADDON_HOST"),
    'user': os.environ.get("MYSQL_ADDON_USER"),
    'password': os.environ.get("MYSQL_ADDON_PASSWORD"),
    'database': os.environ.get("MYSQL_ADDON_DB"),
    'port': int(os.environ.get("MYSQL_ADDON_PORT", 3306)),
    'charset': 'utf8mb4'
}

TABLES = {}
TABLES['users'] = (
    "CREATE TABLE IF NOT EXISTS users ("
    "  user_id INT AUTO_INCREMENT PRIMARY KEY,"
    "  username VARCHAR(255) NOT NULL UNIQUE,"
    "  password_hash VARCHAR(255) NOT NULL,"
    "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    ") ENGINE=InnoDB"
)

TABLES['orders'] = (
    "CREATE TABLE IF NOT EXISTS orders ("
    "  order_id INT PRIMARY KEY,"
    "  user_id INT NOT NULL,"
    "  order_status VARCHAR(50) NOT NULL,"
    "  order_date DATE NOT NULL,"
    "  old_status VARCHAR(50),"
    "  status_last_changed_at DATETIME,"
    "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
    "  FOREIGN KEY (user_id) REFERENCES users(user_id)"
    ") ENGINE=InnoDB"
)

TABLES['forecast_cache'] = (
    "CREATE TABLE IF NOT EXISTS forecast_cache ("
    "  forecast_id INT AUTO_INCREMENT PRIMARY KEY,"
    "  user_id INT NOT NULL,"
    "  forecast_date DATE NOT NULL,"
    "  forecast_data JSON NOT NULL,"
    "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
    "  FOREIGN KEY (user_id) REFERENCES users(user_id)"
    ") ENGINE=InnoDB"
)

TABLES['forecast_performance'] = (
    "CREATE TABLE IF NOT EXISTS forecast_performance ("
    "  id INT AUTO_INCREMENT PRIMARY KEY,"
    "  user_id INT NOT NULL,"
    "  forecast_date DATETIME NOT NULL,"
    "  forecast_data JSON NOT NULL,"
    "  r2 FLOAT,"
    "  mae FLOAT,"
    "  mape FLOAT,"
    "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
    "  FOREIGN KEY (user_id) REFERENCES users(user_id)"
    ") ENGINE=InnoDB"
)

def create_tables():
    try:
        cnx = mysql.connector.connect(**DB_CONFIG)
        cursor = cnx.cursor()
        for table_name, table_sql in TABLES.items():
            try:
                print(f"Creating table {table_name}: ", end='')
                cursor.execute(table_sql)
                print("OK")
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                    print("already exists.")
                else:
                    print(err.msg)

        cursor.close()
        cnx.close()
    except mysql.connector.Error as err:
        print(err)
        exit(1)

if __name__ == '__main__':
    create_tables()
