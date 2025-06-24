# plateau_weather_project

This project explores the generation of flood risk maps using deep learning models trained on rainfall data, without incorporating land use or soil conditions. The primary objective is to visualize areas vulnerable to flooding based solely on short-term precipitation events, such as 60-minute rainfall intensity.

> ⚠️ **Note:** This project is a **work in progress** and is not yet functional in React Native. Current efforts focus on backend development and AI model training using rainfall data only.

---

## 🌧️ Objective

To generate a flood risk map based on short-term rainfall intensity (e.g., 60-minute rainfall), trained via a CNN (Convolutional Neural Network). The resulting map aims to highlight flood-prone areas **without considering land use**, making it a simplified risk indicator for sudden, intense rain events.

---

## 🔍 What Works

- ✅ Parsing and preprocessing rainfall data
- ✅ Generating flood risk predictions using a basic CNN model
- ✅ Serving risk GeoTIFFs through a FastAPI backend
- ✅ CityGML parser to extract building information (e.g., for potential evacuation sites)

---

## 🚧 What Doesn't Work Yet

- ❌ React Native frontend is not functional
- ❌ Risk model does **not** consider land use, elevation, soil, or hydrological flow
- ❌ No real-time flood forecasting
- ❌ No integration with official hazard data

---

## 🧠 AI Model Notes

- Trained on rainfall data only (e.g., JMA 60-minute cumulative rainfall)
- Output: simplified flood risk raster (GeoTIFF)
- Intended for proof-of-concept visualization — **not suitable for real-world decision-making**

---

## 🗺️ Visualization Example

Below is an example of the output flood risk map displayed in QGIS:

![Flood risk map example in QGIS](docs/images/risk_map.png)

- Transparent or black = No data
- Brighter colors = Higher predicted flood risk (CNN output)

---
