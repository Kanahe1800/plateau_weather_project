import logging
from collections import defaultdict
from sqlalchemy import text
from geoalchemy2.elements import WKBElement
from geoalchemy2.shape import to_shape
from shapely.ops import transform
import pyproj

def get_safe_buildings_nearby_clustered(session, lat, lng, radius_km, cluster_dist_m=20):
    radius_m = radius_km * 1000
    cluster_dist_deg = cluster_dist_m / 111320  # approximate meters to degrees conversion

    sql = text("""
        WITH nearby_buildings AS (
            SELECT id, name, geom
            FROM safe_buildings
            WHERE ST_DWithin(
                CAST(ST_Transform(geom, 4326) AS geography),
                CAST(ST_SetSRID(ST_MakePoint(:lng, :lat), 4326) AS geography),
                :radius_m
            )
        ),
        clusters AS (
            SELECT
                unnest(ST_ClusterWithin(ST_Transform(geom, 4326), :cluster_dist_deg)) AS cluster_geom
            FROM nearby_buildings
        ),
        cluster_reps AS (
            SELECT cluster_geom, MIN(nb.id) AS rep_id
            FROM clusters c
            JOIN nearby_buildings nb
                ON ST_Intersects(ST_Transform(nb.geom, 4326), c.cluster_geom)
            GROUP BY cluster_geom
        ),
        risks_agg AS (
            SELECT cr.rep_id,
                   json_agg(DISTINCT jsonb_build_object(
                       'hazard_type', bra.hazard_type,
                       'description_code', bra.description_code,
                       'rank', bra.rank
                   )) AS risks
            FROM nearby_buildings nb
            JOIN building_risk_attributes bra ON nb.id = bra.building_id
            JOIN cluster_reps cr ON nb.id = cr.rep_id OR nb.id IN (
                SELECT nb2.id FROM nearby_buildings nb2 WHERE ST_Intersects(ST_Transform(nb2.geom, 4326), cr.cluster_geom)
            )
            GROUP BY cr.rep_id
        )
        SELECT sb.id, sb.name, ST_AsEWKB(sb.geom) AS geom, COALESCE(ra.risks, '[]'::json) AS risks
        FROM safe_buildings sb
        JOIN cluster_reps cr ON sb.id = cr.rep_id
        LEFT JOIN risks_agg ra ON sb.id = ra.rep_id
        ORDER BY sb.id;
    """)

    params = {
        "lng": lng,
        "lat": lat,
        "radius_m": radius_m,
        "cluster_dist_deg": cluster_dist_deg,
    }

    rows = session.execute(sql, params).mappings().all()
    logging.info(f"Clustering query returned {len(rows)} rows")

    transformer = pyproj.Transformer.from_crs(6668, 4326, always_xy=True).transform

    buildings_dict = defaultdict(lambda: {
        "id": None,
        "name": "",
        "latitude": None,
        "longitude": None,
        "risks": []
    })

    for row in rows:
        b = buildings_dict[row['id']]
        if b["id"] is None:
            geom_raw = row['geom']
            geom = WKBElement(geom_raw, srid=6668)
            shape = to_shape(geom)
            centroid = shape.centroid
            lonlat = transform(transformer, centroid)

            b["id"] = row['id']
            b["name"] = row['name']
            b["latitude"] = lonlat.y
            b["longitude"] = lonlat.x

        # Deduplicate risks by converting list to set of tuples (hashable)
        if row['risks']:
            unique_risks = { 
                (risk['hazard_type'], risk.get('description_code'), risk.get('rank')) 
                for risk in row['risks']
            }
            # Convert back to dicts and assign
            b["risks"] = [ 
                {"hazard_type": r[0], "description_code": r[1], "rank": r[2]} 
                for r in unique_risks
            ]

    buildings = list(buildings_dict.values())
    logging.info(f"Returning {len(buildings)} clustered buildings with deduplicated risks")
    return buildings



# def get_safe_buildings_nearby(session: Session, lat: float, lng: float, radius_km: float):
#     radius_m = radius_km * 1000
#     logging.info(f"Received query for buildings near lat={lat}, lng={lng}, radius={radius_km} km")

#     query = session.query(SafeBuilding).filter(
#         ST_DWithin(
#             cast(ST_Transform(SafeBuilding.geom, 4326), Geography),
#             cast(ST_SetSRID(ST_MakePoint(lng, lat), 4326), Geography),
#             radius_m
#         )
#     )
#     buildings = query.all()
#     logging.info(f"Query returned {len(buildings)} buildings")

#     # Setup transformer for centroid coordinate conversion
#     transformer = pyproj.Transformer.from_crs(6668, 4326, always_xy=True).transform

#     results = []
#     for b in buildings:
#         try:
#             logging.debug(f"Processing building id={b.id} name={b.name}")
#             shape = to_shape(b.geom)
#             centroid = shape.centroid
#             lonlat = transform(transformer, centroid)
#             logging.debug(f"Centroid WGS84: ({lonlat.y}, {lonlat.x})")

#             results.append({
#                 "id": b.id,
#                 "name": b.name or "",
#                 "latitude": lonlat.y,
#                 "longitude": lonlat.x
#             })
#         except Exception as e:
#             logging.warning(f"Failed to transform building {b.id}: {e}")

#     logging.info(f"Returning {len(results)} buildings to client")
#     return results


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