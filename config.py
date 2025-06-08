"""Configuration settings for satellite analysis"""

# Satellite data settings
SATELLITE_CONFIG = {
    'nighttime_lights': {
        'collection': 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG',
        'band': 'avg_rad',
        'scale': 500
    },
    'landsat': {
        'collection': 'LANDSAT/LC08/C02/T1_L2',
        'scale': 30,
        'cloud_filter': 20
    },
    'sentinel2': {
        'collection': 'COPERNICUS/S2_SR',
        'scale': 10,
        'cloud_filter': 20
    },
    'precipitation': {
        'collection': 'NASA/GPM_L3/IMERG_MONTHLY_V06',
        'band': 'precipitation',
        'scale': 10000
    },
    'temperature': {
        'collection': 'MODIS/006/MOD11A1',
        'band': 'LST_Day_1km',
        'scale': 1000
    },
    'elevation': {
        'collection': 'USGS/SRTMGL1_003',
        'band': 'elevation',
        'scale': 30
    }
}

# Analysis settings
ANALYSIS_CONFIG = {
    'distance_buffers': [1000, 5000, 10000, 20000],  # meters from mining sites
    'temporal_frequency': 'quarterly',  # monthly, quarterly, yearly
    'variables': [
        # Economic Activity Indicators
        'nighttime_lights_mean', 'nighttime_lights_sum', 'nighttime_lights_max',
        'electricity_access_proxy', 'lit_pixels_count',
        
        # Vegetation & Agriculture
        'ndvi_mean', 'ndvi_std', 'ndvi_max', 'evi_mean', 'savi_mean',
        'agricultural_productivity', 'crop_productivity_max', 'agricultural_intensity',
        
        # Built Environment & Housing Quality
        'built_up_area', 'built_up_ratio', 'building_count', 'building_density',
        'roof_quality_score', 'estimated_good_roof_ratio', 'urban_compactness',
        
        # Infrastructure & Access
        'road_density_km_per_km2', 'infrastructure_access_score', 'hospital_count',
        'school_count', 'travel_time_to_cities', 'water_access_proxy',
        
        # Population & Demographics
        'population_count', 'population_density', 'settlement_type',
        
        # Environmental Conditions
        'precip_mean', 'temp_mean', 'elevation_mean', 'climate_stress_index',
        'water_stress_index', 'no2_concentration', 'co_concentration',
        
        # Composite Indices
        'economic_activity_index', 'infrastructure_quality_index',
        'environmental_conditions_index', 'overall_development_index',
        'poverty_likelihood_score', 'development_category',
        
        # Spatial Relationships
        'distance_to_mining'
    ]
}

# Variable descriptions for documentation
VARIABLE_DESCRIPTIONS = {
    # Economic Activity
    'nighttime_lights_mean': 'Average nighttime light radiance (economic activity proxy)',
    'electricity_access_proxy': 'Estimated electricity access (0-1 scale)',
    
    # Housing Quality
    'roof_quality_score': 'Roof material quality indicator (0-1, higher = better)',
    'estimated_good_roof_ratio': 'Estimated proportion of good quality roofs',
    
    # Infrastructure
    'road_density_km_per_km2': 'Road network density (km of roads per km²)',
    'water_access_proxy': 'Water access indicator based on water body proximity',
    
    # Composite Indices
    'poverty_likelihood_score': 'Overall poverty likelihood (0-1, higher = more likely poor)',
    'overall_development_index': 'Composite development index (0-1, higher = more developed)'
}