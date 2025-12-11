"""
In this file we handle the logic of the discretized map of Sweden

"""
import geopandas as gpd
from shapely.geometry import Point


TARGET_CRS = "EPSG:3006"
WGS84_CRS = "EPSG:4326"

class SwedenMap:
    def __init__(self):
        self.gdf_wgs84 = gpd.read_file("Data/geo/svenska-landskap-klippt.geo.json")
        self.gdf_projected = self.gdf_wgs84.to_crs(TARGET_CRS)

    def point_to_region(self, lon, lat):
        
        point_geom = Point(lon, lat)
        inside = self.gdf_wgs84[self.gdf_wgs84.contains(point_geom)]
        if len(inside) > 0:
            return inside.iloc[0]["landskap"]
        else:
            # --- Closest Region Logic (Requires Projected CRS) ---
            point_gs = gpd.GeoSeries([point_geom], crs=WGS84_CRS)
            point_projected = point_gs.to_crs(TARGET_CRS).iloc[0]
            distances = self.gdf_projected.distance(point_projected)
            closest_region_idx = distances.idxmin()
            return self.gdf_wgs84.loc[closest_region_idx]["landskap"]
        

    def middle_point_of_region(self, region_name):
        region = self.gdf_wgs84[self.gdf_wgs84["landskap"] == region_name]
        if len(region) == 0:
            raise ValueError(f"Region '{region_name}' not found.")
        centroid = region.geometry.iloc[0].centroid
        return centroid.y, centroid.x  # Return as (lat, lon)
    