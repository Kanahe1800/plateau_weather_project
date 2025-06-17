import logging
from sqlalchemy import cast, func
from sqlalchemy.orm import Session
from geoalchemy2 import Geography
from geoalchemy2.functions import ST_DWithin, ST_Transform, ST_SetSRID, ST_MakePoint, ST_MakeEnvelope
from geoalchemy2.shape import to_shape
from shapely.ops import transform
import pyproj

from app.models import SafeBuilding

def get_safe_buildings_nearby(session: Session, lat: float, lng: float, radius_km: float):
    radius_m = radius_km * 1000
    logging.info(f"Received query for buildings near lat={lat}, lng={lng}, radius={radius_km} km")

    query = session.query(SafeBuilding).filter(
        ST_DWithin(
            cast(ST_Transform(SafeBuilding.geom, 4326), Geography),
            cast(ST_SetSRID(ST_MakePoint(lng, lat), 4326), Geography),
            radius_m
        )
    )
    buildings = query.all()
    logging.info(f"Query returned {len(buildings)} buildings")

    # Setup transformer for centroid coordinate conversion
    transformer = pyproj.Transformer.from_crs(6668, 4326, always_xy=True).transform

    results = []
    for b in buildings:
        try:
            logging.debug(f"Processing building id={b.id} name={b.name}")
            shape = to_shape(b.geom)
            centroid = shape.centroid
            lonlat = transform(transformer, centroid)
            logging.debug(f"Centroid WGS84: ({lonlat.y}, {lonlat.x})")

            results.append({
                "id": b.id,
                "name": b.name or "",
                "latitude": lonlat.y,
                "longitude": lonlat.x
            })
        except Exception as e:
            logging.warning(f"Failed to transform building {b.id}: {e}")

    logging.info(f"Returning {len(results)} buildings to client")
    return results


import logging
from app.db import SessionLocal
from app.models import SafeBuilding

def log_all_safe_buildings():
    session = SessionLocal()
    try:
        buildings = session.query(SafeBuilding).all()
        logging.info(f"Logging all {len(buildings)} safe buildings:")
        for b in buildings:
            logging.info(f"ID: {b.id}, Name: {b.name}, Usage: {b.usage_code}")
    finally:
        session.close()