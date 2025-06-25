import os
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
import numpy as np

def parse_citygml_to_geotiff(gml_path, output_tif_path, resolution=10):
    import os
    import xml.etree.ElementTree as ET
    from shapely.geometry import Polygon
    import geopandas as gpd
    import rasterio
    from rasterio import features
    from rasterio.transform import from_bounds

    ns = {
        "gml": "http://www.opengis.net/gml/3.2",
        "ksj": "http://nlftp.mlit.go.jp/ksj/schemas/ksj-app",
        "xlink": "http://www.w3.org/1999/xlink"
    }

    try:
        tree = ET.parse(gml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERROR] Failed to parse {gml_path}: {e}")
        return

    # Step 1: Extract curves
    curves = {}
    for curve in root.findall(".//gml:Curve", ns):
        cid = curve.attrib.get("{http://www.opengis.net/gml/3.2}id")
        pos_list = curve.find(".//gml:posList", ns)
        if not cid or pos_list is None:
            continue
        coords = list(map(float, pos_list.text.strip().split()))
        latlon = [(coords[i+1], coords[i]) for i in range(0, len(coords), 2)]
        curves[cid] = latlon

    # Step 2: Map surfaces to curves
    surfaces = {}
    for surface in root.findall(".//gml:Surface", ns):
        sid = surface.attrib.get("{http://www.opengis.net/gml/3.2}id")
        exterior = surface.find(".//gml:curveMember", ns)
        if sid and exterior is not None:
            cid = exterior.attrib.get("{http://www.w3.org/1999/xlink}href", "").lstrip("#")
            surfaces[sid] = cid

    # Step 3: Build polygons
    label_mapping = {"1": 3, "2": 2, "3": 1}  # 土石流, 急傾斜地崩壊, 地すべり

    records = []
    for hazard in root.findall(".//ksj:SedimentRelatedDisasterWarningAreasPolygon", ns):
        coz_elem = hazard.find("ksj:coz", ns)
        bounds_elem = hazard.find("ksj:bounds", ns)
        if coz_elem is None or bounds_elem is None:
            continue

        coz = coz_elem.text.strip()
        sid = bounds_elem.attrib.get("{http://www.w3.org/1999/xlink}href", "").lstrip("#")
        cid = surfaces.get(sid)
        coords = curves.get(cid)
        if not coords or len(coords) < 4:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        try:
            polygon = Polygon(coords)
            if not polygon.is_valid or polygon.is_empty:
                continue
        except Exception:
            continue

        risk_label = label_mapping.get(coz, 0)
        records.append({"geometry": polygon, "hazard_type": coz, "risk": risk_label})

    if not records:
        print(f"[SKIP] No valid polygons parsed from {gml_path}")
        return

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326").to_crs("EPSG:3857")

    bounds = gdf.total_bounds
    if (bounds[2] - bounds[0]) < 1e-6 or (bounds[3] - bounds[1]) < 1e-6:
        print(f"[WARNING] Very narrow bounds in {gml_path}: {bounds}")
        return

    width = max(1, int((bounds[2] - bounds[0]) / resolution))
    height = max(1, int((bounds[3] - bounds[1]) / resolution))
    transform = from_bounds(*bounds, width, height)

    raster = features.rasterize(
        ((geom, value) for geom, value in zip(gdf.geometry, gdf["risk"])),
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    os.makedirs(os.path.dirname(output_tif_path), exist_ok=True)
    with rasterio.open(
        output_tif_path, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs=gdf.crs,
        transform=transform
    ) as dst:
        dst.write(raster, 1)

    print(f"[SUCCESS] GeoTIFF written: {output_tif_path}")



def process_all_gml_files(root_dir="./A33-23_00_GML", output_root="output_geotiff", resolution=10):
    """
    Recursively process all GML (.xml) files under each prefecture folder and convert to GeoTIFFs.
    """
    for prefecture in os.listdir(root_dir):
        pref_path = os.path.join(root_dir, prefecture, "GML")
        if not os.path.isdir(pref_path):
            continue

        for filename in os.listdir(pref_path):
            if filename.endswith(".xml"):
                gml_path = os.path.join(pref_path, filename)
                output_path = os.path.join(output_root, filename.replace(".xml", ".tif"))
                try:
                    parse_citygml_to_geotiff(gml_path, output_path, resolution)
                except Exception as e:
                    print(f"[ERROR] Failed to process {gml_path}: {e}")


# 実行（コメントを外して使う）
process_all_gml_files()
