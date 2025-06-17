# create_test_data.py - Create test regions
import geopandas as gpd
from shapely.geometry import Point, Polygon
import os

def create_test_regions():
    """Create 6 test regions with known poverty levels"""
    
    locations = [
        # POOR AREAS
        {'name': 'Rural_Bihar_Poor', 'lat': 25.0961, 'lon': 85.3131, 'poverty': 'Very High'},
        {'name': 'Dhaka_Slum', 'lat': 23.8103, 'lon': 90.4125, 'poverty': 'Very High'},
        
        # MEDIUM AREAS  
        {'name': 'Semi_Urban_UP', 'lat': 26.8467, 'lon': 80.9462, 'poverty': 'Medium'},
        {'name': 'Small_Town_Jharkhand', 'lat': 23.6102, 'lon': 85.2799, 'poverty': 'Medium'},
        
        # RICH AREAS
        {'name': 'Mumbai_Bandra_Rich', 'lat': 19.0596, 'lon': 72.8295, 'poverty': 'Very Low'},
        {'name': 'Bangalore_IT_Hub', 'lat': 12.9352, 'lon': 77.6245, 'poverty': 'Very Low'}
    ]
    
    regions = []
    names = []
    poverty_levels = []
    
    for loc in locations:
        # Create 5km x 5km region around each location
        size = 0.045
        min_lon = loc['lon'] - size/2
        max_lon = loc['lon'] + size/2
        min_lat = loc['lat'] - size/2
        max_lat = loc['lat'] + size/2
        
        polygon = Polygon([
            (min_lon, min_lat), (max_lon, min_lat),
            (max_lon, max_lat), (min_lon, max_lat)
        ])
        
        regions.append(polygon)
        names.append(loc['name'])
        poverty_levels.append(loc['poverty'])
    
    gdf = gpd.GeoDataFrame({
        'NAME': names,
        'POVERTY_LEVEL': poverty_levels,
        'geometry': regions
    }, crs='EPSG:4326')
    
    return gdf

def create_test_mining_sites():
    """Create mining sites"""
    
    mining_locations = [
        {'name': 'Jharia_Coal_Mines', 'lat': 23.7644, 'lon': 86.4152, 'type': 'Coal'},
        {'name': 'Sand_Mining_WB', 'lat': 24.3792, 'lon': 88.3115, 'type': 'Sand'}
    ]
    
    sites = []
    names = []
    types = []
    
    for mining in mining_locations:
        sites.append(Point(mining['lon'], mining['lat']))
        names.append(mining['name'])
        types.append(mining['type'])
    
    gdf = gpd.GeoDataFrame({
        'NAME': names,
        'TYPE': types,
        'geometry': sites
    }, crs='EPSG:4326')
    
    return gdf

def main():
    """Create and save test data"""
    os.makedirs('data', exist_ok=True)
    
    print("🗺️  Creating test regions...")
    regions = create_test_regions()
    regions.to_file('data/test_regions.geojson', driver='GeoJSON')
    print(f"✅ Created {len(regions)} test regions")
    
    print("⛏️  Creating mining sites...")
    mining_sites = create_test_mining_sites()
    mining_sites.to_file('data/test_mining_sites.geojson', driver='GeoJSON')
    print(f"✅ Created {len(mining_sites)} mining sites")
    
    print("\n📋 REGIONS CREATED:")
    for _, region in regions.iterrows():
        print(f"   • {region['NAME']}: {region['POVERTY_LEVEL']} poverty")

if __name__ == "__main__":
    main()