from fastapi import WebSocket, WebSocketDisconnect
from app.db import SessionLocal
from app.crud.building_query import get_safe_buildings_nearby, log_all_safe_buildings
import logging

# async def building_ws_handler(websocket: WebSocket):
#     await websocket.accept()
#     db = SessionLocal()
#     try:
#         while True:
#             data = await websocket.receive_json()
#             logging.info(f"Received WebSocket data: {data}")
#             lat = data["latitude"]
#             lng = data["longitude"]
#             radius = data.get("radius_km", 5)  # default radius 5 km

#             buildings = get_safe_buildings_nearby(db, lat, lng, radius)
#             logging.info(f"Sending {len(buildings)} buildings in response")
#             await websocket.send_json({"buildings": buildings})

#     except WebSocketDisconnect:
#         logging.info("WebSocket disconnected")
#     except Exception as e:
#         logging.error(f"WebSocket handler error: {e}")
#         await websocket.send_json({"error": str(e)})
#     finally:
#         db.close()

async def building_ws_handler(websocket: WebSocket):
    await websocket.accept()
    db = SessionLocal()
    try:
        while True:
            try:
                data = await websocket.receive_json()
                lat = data["latitude"]
                lng = data["longitude"]
                radius = data.get("radius_km", 5)
                buildings = get_safe_buildings_nearby(db, lat, lng, radius)
                await websocket.send_json({"buildings": buildings})
            except Exception as e:
                import traceback
                traceback.print_exc()
                await websocket.send_json({"error": str(e)})
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    finally:
        db.close()

