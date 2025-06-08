"""Create test data for the satellite analysis"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
import numpy as np

def create_test_regions():
    """Create test regions around a known area (e.g., parts of India)"""
    
    # Create a grid of regions around a test area (Bihar/Jharkhand border - known sand mining area)
    base_lat, base_lon = 24.5, 85.0  # Approximate center
    
    regions = []
    region_names = []
    
    # Create 3x3 grid of regions
    for i in range(3):
        for j in range(3):
            # Each region is ~0.1 degree (about 10km)
            min_lon = base_lon + j * 0.1 - 0.15
            max_lon = base_lon + j * 0.1 - 0.05
            min_lat = base_lat + i * 0.1 - 0.15
            max_lat = base_lat + i * 0.1 - 0.05
            
            # Create polygon
            polygon = Polygon([
                (min_lon, min_lat),
                (max_lon, min_lat),
                (max_lon, max_lat),
                (min_lon, max_lat)
            ])
            
            regions.append(polygon)
            region_names.append(f"Region_{i}_{j}")
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'NAME': region_names,
        'AREA_KM2': [100] * len(regions),  # Approximate area
        'geometry': regions
    }, crs='EPSG:4326')
    
    return gdf

def create_test_mining_sites():
    """Create test mining sites"""
    
    # Create some random mining sites in the area
    base_lat, base_lon = 24.5, 85.0
    
    np.random.seed(42)  # For reproducible results
    
    mining_sites = []
    site_names = []
    
    for i in range(5):
        # Random location within the broader area
        lat = base_lat + np.random.uniform(-0.2, 0.2)
        lon = base_lon + np.random.uniform(-0.2, 0.2)
        
        mining_sites.append(Point(lon, lat))
        site_names.append(f"Mining_Site_{i+1}")
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'NAME': site_names,
        'TYPE': 'Sand Mining',
        'geometry': mining_sites
    }, crs='EPSG:4326')
    
    return gdf

def main():
    """Create and save test data"""
    
    print("Creating test regions...")
    regions = create_test_regions()
    regions.to_file('../data/test_regions.geojson', driver='GeoJSON')
    print(f"✓ Created {len(regions)} test regions")
    
    print("Creating test mining sites...")
    mining_sites = create_test_mining_sites()
    mining_sites.to_file('../data/test_mining_sites.geojson', driver='GeoJSON')
    print(f"✓ Created {len(mining_sites)} test mining sites")
    
    print("\nTest data created in ../data/ folder")
    print("You can now run the analysis with:")
    print("python main.py --regions data/test_regions.geojson --start_date 2020-01-01 --end_date 2023-12-31 --mining_sites data/test_mining_sites.geojson")

if __name__ == "__main__":
    main()