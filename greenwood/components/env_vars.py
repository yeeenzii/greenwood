import os
from dotenv import load_dotenv

load_dotenv()

env_dict = {
    'DB_USER': os.getenv('DB_USER'),
    'DB_PASSWORD': os.getenv('DB_PASSWORD'),
    'DB_CONNECTION_STRING': os.getenv('DB_CONNECTION_STRING'),
}