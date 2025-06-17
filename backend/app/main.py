from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from app.websocket.buildings import building_ws_handler
from app.models import Base
from app.db import engine

import logging
import os

# --- Logging Setup ---
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# --- FastAPI App Setup ---
app = FastAPI()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider restricting this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Table Initialization ---
@app.on_event("startup")
def on_startup():
    logger.info("Creating database tables (if not exists)...")
    Base.metadata.create_all(bind=engine)
    logger.info("Startup complete.")

# --- Routes ---
@app.get("/", tags=["Health Check"])
def health_check() -> dict:
    return {"status": "ok", "message": "FastAPI backend running"}

@app.websocket("/ws/buildings")
async def buildings_ws(websocket: WebSocket):
    await building_ws_handler(websocket)
