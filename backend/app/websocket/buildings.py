from fastapi import WebSocket, WebSocketDisconnect
from app.db import SessionLocal
from app.crud.building_query import get_safe_buildings_nearby_clustered
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
    session = SessionLocal()
    try:
        while True:
            data = await websocket.receive_json()
            try:
                # 例: 建物検索
                buildings = get_safe_buildings_nearby_clustered(session, data["latitude"], data["longitude"], data.get("radius_km", 5))
                await websocket.send_json({"buildings": buildings})
            except Exception as e:
                session.rollback()  # 例外が起きたらロールバック
                await websocket.send_json({"error": str(e)})
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    finally:
        session.close()


