# process_cityGML/models.py

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from geoalchemy2 import Geometry
from sqlalchemy.orm import relationship

Base = declarative_base()

class SafeBuilding(Base):
    __tablename__ = "safe_buildings"

    id = Column(Integer, primary_key=True)
    gml_id = Column(String, unique=True)
    name = Column(String)
    usage_code = Column(Integer)
    height = Column(Float)
    geom = Column(Geometry(geometry_type="POLYGON", srid=6668))
    risk_attributes = relationship("BuildingRiskAttribute", back_populates="building", cascade="all, delete-orphan")


class BuildingRiskAttribute(Base):
    __tablename__ = "building_risk_attributes"
    id = Column(Integer, primary_key=True)
    building_id = Column(Integer, ForeignKey("safe_buildings.id", ondelete="CASCADE"))
    hazard_type = Column(String)
    description_code = Column(String)
    rank = Column(Integer)
    depth = Column(Float)
    depth_unit = Column(String)
    admin_type = Column(String)
    scale = Column(String)
    duration = Column(Float)
    duration_unit = Column(String)

    building = relationship("SafeBuilding", back_populates="risk_attributes")
