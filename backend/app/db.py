import os
import time
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from app.config import settings

if os.getenv("SQLALCHEMY_DEBUG", "false").lower() == "true":
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "sqlalchemy.log")),
            logging.StreamHandler()
        ]
    )
    logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)

MAX_RETRIES = 10
RETRY_DELAY = 2  # seconds

engine = None
for i in range(MAX_RETRIES):
    try:
        engine = create_engine(settings.DATABASE_URL)
        with engine.connect() as conn:
            print(f"[✓] Database connected on attempt {i+1}")
        break
    except OperationalError:
        print(f"[✗] Waiting for database... attempt {i+1}/{MAX_RETRIES}")
        time.sleep(RETRY_DELAY)
else:
    raise RuntimeError("Could not connect to the database.")

SessionLocal = sessionmaker(bind=engine)
