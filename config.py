# config.py - Configuration settings
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
    }
}

ANALYSIS_CONFIG = {
    'temporal_frequency': 'quarterly'
}