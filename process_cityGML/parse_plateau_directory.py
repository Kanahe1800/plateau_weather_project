import os
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from shapely.wkt import dumps as wkt_dumps
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from process_cityGML.models import Base, SafeBuilding, BuildingRiskAttribute
from dotenv import load_dotenv
import logging
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "plateau_data"))

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("skipped_buildings.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# --- Load DB config ---
load_dotenv()
url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(url)
Session = sessionmaker(bind=engine)
session = Session()
Base.metadata.create_all(engine)

# --- Namespace ---
ns = {
    "bldg": "http://www.opengis.net/citygml/building/2.0",
    "gml": "http://www.opengis.net/gml",
    "core": "http://www.opengis.net/citygml/2.0",
    "xlink": "http://www.w3.org/1999/xlink",
    "uro": "https://www.geospatial.jp/iur/uro/3.1"
}

SAFE_USAGE_CODES = {"403", "421", "422"}

def extract_risk_attributes(bldg_elem, ns):
    risk_attributes = []
    for risk_elem in bldg_elem.findall(".//uro:bldgDisasterRiskAttribute", ns):
        for child in risk_elem:
            hazard_type = child.tag.split("}")[-1].replace("RiskAttribute", "")
            entry = {
                "hazard_type": hazard_type,
                "description_code": child.findtext("uro:description", default=None, namespaces=ns),
                "rank": to_int(child.findtext("uro:rank", default=None, namespaces=ns)),
                "depth": to_float(child.findtext("uro:depth", default=None, namespaces=ns)),
                "depth_unit": child.find("uro:depth", ns).attrib.get("uom") if child.find("uro:depth", ns) is not None else None,
                "admin_type": child.findtext("uro:adminType", default=None, namespaces=ns),
                "scale": child.findtext("uro:scale", default=None, namespaces=ns),
                "duration": to_float(child.findtext("uro:duration", default=None, namespaces=ns)),
                "duration_unit": child.find("uro:duration", ns).attrib.get("uom") if child.find("uro:duration", ns) is not None else None,
            }
            risk_attributes.append(entry)
    # logging.info(risk_attributes)
    return risk_attributes

def to_int(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        return None

def to_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None

def extract_footprint_polygon(solid_elem):
    if solid_elem is None:
        return None
    min_z = float('inf')
    best_polygon = None
    for pos_list_elem in solid_elem.findall(".//gml:posList", ns):
        coords = list(map(float, pos_list_elem.text.strip().split()))
        if len(coords) < 9:
            continue
        try:
            points = [(coords[i+1], coords[i]) for i in range(0, len(coords), 3)]
            z_values = [coords[i+2] for i in range(0, len(coords), 3)]
            avg_z = sum(z_values) / len(z_values)
            if avg_z < min_z and len(points) >= 3:
                poly = Polygon(points)
                if poly.is_valid:
                    best_polygon = poly
                    min_z = avg_z
        except Exception:
            continue
    return best_polygon

def process_gml_file(filepath):
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except Exception as e:
        logging.error(f"Failed to parse {filepath}: {e}")
        return

    for bldg in root.findall(".//bldg:Building", ns):
        gml_id = bldg.attrib.get("{http://www.opengis.net/gml}id", "UNKNOWN")
        name = bldg.findtext("gml:name", default="", namespaces=ns)
        usage_elem = bldg.find("bldg:usage", ns)
        usage_code = usage_elem.text.strip() if usage_elem is not None else ""
        if usage_code not in SAFE_USAGE_CODES:
            continue

        height = bldg.findtext("bldg:measuredHeight", default=None, namespaces=ns)
        height = float(height) if height else None

        solid = bldg.find("bldg:lod2Solid", ns)
        if solid is None:
            solid = bldg.find("bldg:lod1Solid", ns)

        if solid is None:
            logging.error(f"Skipped building {gml_id}: missing lod1/lod2 solid")
            continue

        polygon = extract_footprint_polygon(solid)

        if polygon is None or not polygon.is_valid or polygon.is_empty:
            # logging.error(f"Skipped building {gml_id}: invalid or missing geometry")
            continue

        risks = extract_risk_attributes(bldg, ns)
        risk_objects = [BuildingRiskAttribute(
            hazard_type=r["hazard_type"],
            description_code=r["description_code"],
            rank=r["rank"],
            depth=r["depth"],
            depth_unit=r["depth_unit"],
            admin_type=r["admin_type"],
            scale=r["scale"],
            duration=r["duration"],
            duration_unit=r["duration_unit"]
        ) for r in risks]

        building = SafeBuilding(
            gml_id=gml_id,
            name=name,
            usage_code=int(usage_code),
            height=height,
            geom=f"SRID=6668;{wkt_dumps(polygon)}",
            risk_attributes=risk_objects  # ここでリスクを一括追加
        )
        session.add(building)


def walk_and_process_all():
    session.query(BuildingRiskAttribute).delete()
    session.query(SafeBuilding).delete()
    session.commit()
    for root_dir, _, files in os.walk(DATA_DIR):
        for fname in files:
            if fname.endswith(".gml"):
                fpath = os.path.join(root_dir, fname)
                logging.info(f"Processing: {fpath}")
                process_gml_file(fpath)
    session.commit()
    logging.info("✅ 全ファイルの処理と登録が完了しました。")

def reset_tables():
    from sqlalchemy import text
    session.execute(text("TRUNCATE TABLE building_risk_attributes, safe_buildings RESTART IDENTITY CASCADE"))
    session.commit()
    logging.info("✅ All tables truncated and IDs reset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CityGML and insert safe buildings.")
    parser.add_argument("--reset", action="store_true", help="Truncate tables and reset IDs before processing")
    args = parser.parse_args()

    if args.reset:
        reset_tables()

    walk_and_process_all()
