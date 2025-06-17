import os
from pathlib import Path
from dotenv import load_dotenv

# Load the .env file from backend/.env
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

class Settings:
    DATABASE_URL = os.getenv("DB_URL")

settings = Settings()
