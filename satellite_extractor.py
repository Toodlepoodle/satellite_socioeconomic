"""Satellite data extraction functions"""

import ee
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import numpy as np
from config import SATELLITE_CONFIG

class SatelliteExtractor:
    def __init__(self):
        """Initialize Google Earth Engine"""
        try:
            # Initialize with your project ID
            ee.Initialize(project='ed-sayandasgupta97')
            print("✓ Google Earth Engine initialized with project: ed-sayandasgupta97")
            self.gee_available = True
        except Exception as e:
            print(f"✗ Error initializing GEE: {e}")
            print("Solutions:")
            print("1. Run 'earthengine authenticate --project ed-sayandasgupta97' in terminal")
            print("2. Make sure the project 'ed-sayandasgupta97' has Earth Engine API enabled")
            print("3. Check your authentication: earthengine authenticate")
            self.gee_available = False
            return
    
    def extract_nighttime_lights(self, geometry, start_date, end_date):
        """Extract nighttime lights data - proxy for electricity access and economic activity"""
        try:
            collection = ee.ImageCollection(SATELLITE_CONFIG['nighttime_lights']['collection']) \
                          .filterDate(start_date, end_date) \
                          .filterBounds(geometry) \
                          .select(SATELLITE_CONFIG['nighttime_lights']['band'])
            
            # Calculate multiple statistics
            mean_lights = collection.mean()
            sum_lights = collection.sum()
            max_lights = collection.max()
            std_lights = collection.reduce(ee.Reducer.stdDev())
            
            # Count of lit pixels (electricity access proxy)
            lit_pixels = collection.mean().gt(0).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=geometry,
                scale=SATELLITE_CONFIG['nighttime_lights']['scale'],
                maxPixels=1e9
            )
            
            # Extract values
            stats = mean_lights.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=SATELLITE_CONFIG['nighttime_lights']['scale'],
                maxPixels=1e9
            )
            
            sum_stats = sum_lights.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=geometry,
                scale=SATELLITE_CONFIG['nighttime_lights']['scale'],
                maxPixels=1e9
            )
            
            max_stats = max_lights.reduceRegion(
                reducer=ee.Reducer.max(),
                geometry=geometry,
                scale=SATELLITE_CONFIG['nighttime_lights']['scale'],
                maxPixels=1e9
            )
            
            std_stats = std_lights.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=SATELLITE_CONFIG['nighttime_lights']['scale'],
                maxPixels=1e9
            )
            
            return {
                'nighttime_lights_mean': stats.getInfo().get('avg_rad', 0),
                'nighttime_lights_sum': sum_stats.getInfo().get('avg_rad', 0),
                'nighttime_lights_max': max_stats.getInfo().get('avg_rad', 0),
                'nighttime_lights_std': std_stats.getInfo().get('avg_rad_stdDev', 0),
                'lit_pixels_count': lit_pixels.getInfo().get('avg_rad', 0),
                'electricity_access_proxy': min(1.0, stats.getInfo().get('avg_rad', 0) / 10.0)  # Normalized 0-1
            }
        except Exception as e:
            print(f"Error extracting nighttime lights: {e}")
            return {'nighttime_lights_mean': 0, 'nighttime_lights_sum': 0, 'nighttime_lights_max': 0, 
                   'nighttime_lights_std': 0, 'lit_pixels_count': 0, 'electricity_access_proxy': 0}
    
    def extract_vegetation_indices(self, geometry, start_date, end_date):
        """Extract vegetation indices and land cover - proxies for agriculture, food security"""
        try:
            collection = ee.ImageCollection(SATELLITE_CONFIG['landsat']['collection']) \
                          .filterDate(start_date, end_date) \
                          .filterBounds(geometry) \
                          .filter(ee.Filter.lt('CLOUD_COVER', SATELLITE_CONFIG['landsat']['cloud_filter']))
            
            def calculate_indices(image):
                # Scale Landsat Collection 2 data
                optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
                
                # NDVI (vegetation density)
                ndvi = optical.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
                
                # EVI (Enhanced Vegetation Index - better for dense vegetation)
                evi = optical.expression(
                    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                    {
                        'NIR': optical.select('SR_B5'),
                        'RED': optical.select('SR_B4'),
                        'BLUE': optical.select('SR_B2')
                    }
                ).rename('EVI')
                
                # NDBI (Normalized Difference Built-up Index)
                ndbi = optical.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
                
                # NDWI (Normalized Difference Water Index)
                ndwi = optical.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
                
                # SAVI (Soil Adjusted Vegetation Index - better for sparse vegetation)
                savi = optical.expression(
                    '((NIR - RED) / (NIR + RED + 0.5)) * (1 + 0.5)',
                    {
                        'NIR': optical.select('SR_B5'),
                        'RED': optical.select('SR_B4')
                    }
                ).rename('SAVI')
                
                # MNDWI (Modified NDWI - better for water detection)
                mndwi = optical.normalizedDifference(['SR_B3', 'SR_B6']).rename('MNDWI')
                
                return image.addBands([ndvi, evi, ndbi, ndwi, savi, mndwi])
            
            # Apply calculations and get statistics
            indices = collection.map(calculate_indices)
            
            # Calculate multiple statistics for each index
            stats = {}
            for band in ['NDVI', 'EVI', 'NDBI', 'NDWI', 'SAVI', 'MNDWI']:
                mean_img = indices.select(band).mean()
                std_img = indices.select(band).reduce(ee.Reducer.stdDev())
                max_img = indices.select(band).max()
                
                mean_stats = mean_img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=SATELLITE_CONFIG['landsat']['scale'],
                    maxPixels=1e9
                )
                
                std_stats = std_img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=SATELLITE_CONFIG['landsat']['scale'],
                    maxPixels=1e9
                )
                
                max_stats = max_img.reduceRegion(
                    reducer=ee.Reducer.max(),
                    geometry=geometry,
                    scale=SATELLITE_CONFIG['landsat']['scale'],
                    maxPixels=1e9
                )
                
                stats[f'{band.lower()}_mean'] = mean_stats.getInfo().get(band, 0)
                stats[f'{band.lower()}_std'] = std_stats.getInfo().get(f'{band}_stdDev', 0)
                stats[f'{band.lower()}_max'] = max_stats.getInfo().get(band, 0)
            
            # Agricultural productivity proxy
            stats['agricultural_productivity'] = max(0, stats.get('ndvi_mean', 0)) * max(0, stats.get('evi_mean', 0))
            
            # Water access proxy (higher MNDWI = more water bodies nearby)
            stats['water_access_proxy'] = min(1.0, max(0, stats.get('mndwi_mean', 0) + 0.3))
            
            return stats
            
        except Exception as e:
            print(f"Error extracting vegetation indices: {e}")
            return {f'{band.lower()}_{stat}': 0 for band in ['ndvi', 'evi', 'ndbi', 'ndwi', 'savi', 'mndwi'] 
                   for stat in ['mean', 'std', 'max']} | {'agricultural_productivity': 0, 'water_access_proxy': 0}
    
    def extract_building_characteristics(self, geometry):
        """Extract building density, roof quality indicators, and urban morphology"""
        try:
            # Global Human Settlement Layer data
            population = ee.Image('JRC/GHSL/P2016/POP_GPW_GLOBE_V1')
            built_up = ee.Image('JRC/GHSL/P2016/BUILT_LDSMT_GLOBE_V1')
            settlement = ee.Image('JRC/GHSL/P2016/SMOD_POP_GLOBE_V1')
            
            # High-resolution building footprints (Google/Open Buildings)
            try:
                buildings = ee.FeatureCollection('GOOGLE/Research/open-buildings/v3/polygons') \
                             .filterBounds(geometry)
                
                # Building density and characteristics
                building_count = buildings.size()
                building_area = buildings.aggregate_sum('area_in_meters')
                building_confidence = buildings.aggregate_mean('confidence')
                
                # Convert to image for area calculations
                building_raster = buildings.reduceToImage(['confidence'], ee.Reducer.mean())
                
                building_density_stats = building_raster.reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=geometry,
                    scale=10,
                    maxPixels=1e9
                )
                
                building_stats = {
                    'building_count': building_count.getInfo(),
                    'total_building_area': building_area.getInfo(),
                    'avg_building_confidence': building_confidence.getInfo(),
                    'building_density': building_density_stats.getInfo().get('confidence', 0)
                }
                
            except:
                # Fallback if Open Buildings not available
                building_stats = {
                    'building_count': 0,
                    'total_building_area': 0,
                    'avg_building_confidence': 0,
                    'building_density': 0
                }
            
            # Population and settlement statistics
            pop_stats = population.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=geometry,
                scale=250,
                maxPixels=1e9
            )
            
            built_stats = built_up.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=geometry,
                scale=30,
                maxPixels=1e9
            )
            
            settlement_stats = settlement.reduceRegion(
                reducer=ee.Reducer.mode(),
                geometry=geometry,
                scale=1000,
                maxPixels=1e9
            )
            
            # Calculate derived indicators
            total_area = geometry.area().getInfo()
            population_count = pop_stats.getInfo().get('population_count', 0)
            built_area = built_stats.getInfo().get('built', 0)
            
            result = {
                'population_count': population_count,
                'built_up_area': built_area,
                'settlement_type': settlement_stats.getInfo().get('smod_code', 0),
                'population_density': population_count / (total_area / 1e6) if total_area > 0 else 0,  # per km2
                'built_up_ratio': built_area / total_area if total_area > 0 else 0,
                'urban_compactness': built_area / max(1, population_count) if population_count > 0 else 0
            }
            
            result.update(building_stats)
            return result
            
        except Exception as e:
            print(f"Error extracting building characteristics: {e}")
            return {
                'population_count': 0, 'built_up_area': 0, 'settlement_type': 0,
                'population_density': 0, 'built_up_ratio': 0, 'urban_compactness': 0,
                'building_count': 0, 'total_building_area': 0, 'avg_building_confidence': 0,
                'building_density': 0
            }
    
    def extract_infrastructure_indicators(self, geometry):
        """Extract road density, market access, and infrastructure quality indicators"""
        try:
            # OpenStreetMap road data
            roads = ee.FeatureCollection('projects/sat-io/open-datasets/OSM/OSM_roads') \
                     .filterBounds(geometry)
            
            # Calculate road statistics
            road_length = roads.aggregate_sum('length_m')
            road_count = roads.size()
            
            # Road density calculation
            area_km2 = geometry.area().divide(1e6)
            road_density = road_length.divide(area_km2)
            
            # Market access using Global Friction Surface
            try:
                friction = ee.Image('Oxford/MAP/friction_surface_2019')
                travel_time = ee.Image('Oxford/MAP/accessibility_to_cities_2015_v1_0')
                
                friction_stats = friction.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=1000,
                    maxPixels=1e9
                )
                
                travel_stats = travel_time.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=1000,
                    maxPixels=1e9
                )
                
                market_access_stats = {
                    'travel_friction': friction_stats.getInfo().get('friction', 0),
                    'travel_time_to_cities': travel_stats.getInfo().get('accessibility', 0)
                }
            except:
                market_access_stats = {
                    'travel_friction': 0,
                    'travel_time_to_cities': 0
                }
            
            # Healthcare and education facility access (if available)
            try:
                hospitals = ee.FeatureCollection('projects/sat-io/open-datasets/OSM/OSM_hospitals') \
                             .filterBounds(geometry)
                schools = ee.FeatureCollection('projects/sat-io/open-datasets/OSM/OSM_schools') \
                           .filterBounds(geometry)
                
                facility_stats = {
                    'hospital_count': hospitals.size().getInfo(),
                    'school_count': schools.size().getInfo()
                }
            except:
                facility_stats = {
                    'hospital_count': 0,
                    'school_count': 0
                }
            
            result = {
                'road_length_km': road_length.getInfo() / 1000 if road_length.getInfo() else 0,
                'road_count': road_count.getInfo(),
                'road_density_km_per_km2': road_density.getInfo() if road_density.getInfo() else 0,
                'infrastructure_access_score': min(1.0, (road_density.getInfo() or 0) / 10.0)  # Normalized
            }
            
            result.update(market_access_stats)
            result.update(facility_stats)
            return result
            
        except Exception as e:
            print(f"Error extracting infrastructure indicators: {e}")
            return {
                'road_length_km': 0, 'road_count': 0, 'road_density_km_per_km2': 0,
                'infrastructure_access_score': 0, 'travel_friction': 0,
                'travel_time_to_cities': 0, 'hospital_count': 0, 'school_count': 0
            }
    
    def extract_roof_material_indicators(self, geometry, start_date, end_date):
        """Extract roof material quality indicators from high-resolution imagery"""
        try:
            # Use Sentinel-2 for higher resolution analysis
            s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
                  .filterDate(start_date, end_date) \
                  .filterBounds(geometry) \
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            
            def calculate_roof_indices(image):
                # Calculate indices that correlate with roof materials
                # Bright surfaces (metal, concrete) vs dark surfaces (thatch, low-quality materials)
                
                # Brightness index
                brightness = image.expression(
                    '(B2 + B3 + B4) / 3',
                    {'B2': image.select('B2'), 'B3': image.select('B3'), 'B4': image.select('B4')}
                ).rename('brightness')
                
                # Metal roof indicator (high reflectance in NIR and SWIR)
                metal_index = image.expression(
                    '(B8 + B11) / 2',
                    {'B8': image.select('B8'), 'B11': image.select('B11')}
                ).rename('metal_index')
                
                # Concrete/cement indicator
                concrete_index = image.expression(
                    '(B3 + B4) / 2',
                    {'B3': image.select('B3'), 'B4': image.select('B4')}
                ).rename('concrete_index')
                
                return image.addBands([brightness, metal_index, concrete_index])
            
            # Apply calculations
            roof_indices = s2.map(calculate_roof_indices).median()
            
            # Extract statistics
            brightness_stats = roof_indices.select('brightness').reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=10,
                maxPixels=1e9
            )
            
            metal_stats = roof_indices.select('metal_index').reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=10,
                maxPixels=1e9
            )
            
            concrete_stats = roof_indices.select('concrete_index').reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=10,
                maxPixels=1e9
            )
            
            # Calculate roof quality score (higher = better quality materials)
            brightness_val = brightness_stats.getInfo().get('brightness', 0)
            metal_val = metal_stats.getInfo().get('metal_index', 0)
            concrete_val = concrete_stats.getInfo().get('concrete_index', 0)
            
            roof_quality_score = (brightness_val * 0.4 + metal_val * 0.3 + concrete_val * 0.3) / 3000  # Normalized
            
            return {
                'roof_brightness': brightness_val,
                'metal_roof_indicator': metal_val,
                'concrete_roof_indicator': concrete_val,
                'roof_quality_score': min(1.0, max(0.0, roof_quality_score)),
                'estimated_good_roof_ratio': min(1.0, roof_quality_score * 2)  # Proxy for % of good roofs
            }
            
        except Exception as e:
            print(f"Error extracting roof material indicators: {e}")
            return {
                'roof_brightness': 0, 'metal_roof_indicator': 0, 'concrete_roof_indicator': 0,
                'roof_quality_score': 0, 'estimated_good_roof_ratio': 0
            }
    
    def extract_environmental_data(self, geometry, start_date, end_date):
        """Extract comprehensive environmental variables"""
        try:
            # Precipitation (water availability)
            precip = ee.ImageCollection(SATELLITE_CONFIG['precipitation']['collection']) \
                      .filterDate(start_date, end_date) \
                      .select(SATELLITE_CONFIG['precipitation']['band'])
            
            precip_mean = precip.mean()
            precip_sum = precip.sum()
            precip_std = precip.reduce(ee.Reducer.stdDev())
            
            # Temperature (heat stress, energy needs)
            temp = ee.ImageCollection(SATELLITE_CONFIG['temperature']['collection']) \
                    .filterDate(start_date, end_date) \
                    .select(SATELLITE_CONFIG['temperature']['band'])
            
            temp_mean = temp.mean().multiply(0.02).subtract(273.15)  # Convert to Celsius
            temp_max = temp.max().multiply(0.02).subtract(273.15)
            temp_std = temp.reduce(ee.Reducer.stdDev()).multiply(0.02)
            
            # Elevation and terrain
            elevation = ee.Image(SATELLITE_CONFIG['elevation']['collection'])
            slope = ee.Terrain.slope(elevation)
            
            # Air quality indicators (if available)
            try:
                # Sentinel-5P air quality data
                no2 = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2') \
                       .filterDate(start_date, end_date) \
                       .select('NO2_column_number_density') \
                       .mean()
                
                co = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_CO') \
                      .filterDate(start_date, end_date) \
                      .select('CO_column_number_density') \
                      .mean()
                
                air_quality_stats = {
                    'no2_concentration': no2.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=geometry,
                        scale=1000,
                        maxPixels=1e9
                    ).getInfo().get('NO2_column_number_density', 0),
                    
                    'co_concentration': co.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=geometry,
                        scale=1000,
                        maxPixels=1e9
                    ).getInfo().get('CO_column_number_density', 0)
                }
            except:
                air_quality_stats = {'no2_concentration': 0, 'co_concentration': 0}
            
            # Extract all environmental statistics
            env_stats = {}
            
            # Precipitation stats
            for name, img in [('precip_mean', precip_mean), ('precip_sum', precip_sum), ('precip_std', precip_std)]:
                stats = img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=SATELLITE_CONFIG['precipitation']['scale'],
                    maxPixels=1e9
                )
                key = 'precipitation_stdDev' if 'std' in name else 'precipitation'
                env_stats[name] = stats.getInfo().get(key, 0)
            
            # Temperature stats
            for name, img in [('temp_mean', temp_mean), ('temp_max', temp_max), ('temp_std', temp_std)]:
                stats = img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=SATELLITE_CONFIG['temperature']['scale'],
                    maxPixels=1e9
                )
                key = 'LST_Day_1km_stdDev' if 'std' in name else 'LST_Day_1km'
                env_stats[name] = stats.getInfo().get(key, 0)
            
            # Topographic stats
            elev_stats = elevation.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=SATELLITE_CONFIG['elevation']['scale'],
                maxPixels=1e9
            )
            
            slope_stats = slope.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=SATELLITE_CONFIG['elevation']['scale'],
                maxPixels=1e9
            )
            
            env_stats.update({
                'elevation_mean': elev_stats.getInfo().get('elevation', 0),
                'slope_mean': slope_stats.getInfo().get('slope', 0),
                'climate_stress_index': abs(env_stats.get('temp_mean', 20) - 25) / 10,  # Deviation from optimal temp
                'water_stress_index': max(0, 1 - env_stats.get('precip_mean', 0) / 100),  # Water scarcity proxy
            })
            
            env_stats.update(air_quality_stats)
            return env_stats
            
        except Exception as e:
            print(f"Error extracting environmental data: {e}")
            return {
                'precip_mean': 0, 'precip_sum': 0, 'precip_std': 0,
                'temp_mean': 0, 'temp_max': 0, 'temp_std': 0,
                'elevation_mean': 0, 'slope_mean': 0,
                'climate_stress_index': 0, 'water_stress_index': 0,
                'no2_concentration': 0, 'co_concentration': 0
            }
    
    def extract_economic_activity_indicators(self, geometry, start_date, end_date):
        """Extract additional economic activity indicators"""
        try:
            # Crop productivity indicators
            modis_ndvi = ee.ImageCollection('MODIS/006/MOD13Q1') \
                          .filterDate(start_date, end_date) \
                          .select('NDVI')
            
            # Calculate growing season metrics
            ndvi_max = modis_ndvi.max()
            ndvi_range = modis_ndvi.max().subtract(modis_ndvi.min())
            
            # Livestock indicators (using grassland/pasture detection)
            grassland_mask = modis_ndvi.mean().gt(2000).And(modis_ndvi.mean().lt(6000))
            
            # Industrial activity indicators (using thermal data)
            thermal = ee.ImageCollection('MODIS/006/MOD11A1') \
                       .filterDate(start_date, end_date) \
                       .select('LST_Day_1km')
            
            thermal_anomalies = thermal.mean().subtract(thermal.mean().focal_mean(5000))
            
            # Extract statistics
            crop_stats = ndvi_max.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=250,
                maxPixels=1e9
            )
            
            range_stats = ndvi_range.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=250,
                maxPixels=1e9
            )
            
            grassland_stats = grassland_mask.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=geometry,
                scale=250,
                maxPixels=1e9
            )
            
            thermal_stats = thermal_anomalies.reduceRegion(thermal_stats = thermal_anomalies.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=1000,
                maxPixels=1e9
            )
            
            return {
                'crop_productivity_max': crop_stats.getInfo().get('NDVI', 0),
                'crop_seasonality': range_stats.getInfo().get('NDVI', 0),
                'grassland_area': grassland_stats.getInfo().get('NDVI', 0),
                'thermal_industrial_activity': thermal_stats.getInfo().get('LST_Day_1km', 0),
                'agricultural_intensity': crop_stats.getInfo().get('NDVI', 0) / 10000,  # Normalized
                'livestock_potential': min(1.0, grassland_stats.getInfo().get('NDVI', 0) / 1000)
            }
            
        except Exception as e:
            print(f"Error extracting economic activity indicators: {e}")
            return {
                'crop_productivity_max': 0, 'crop_seasonality': 0, 'grassland_area': 0,
                'thermal_industrial_activity': 0, 'agricultural_intensity': 0, 'livestock_potential': 0
            }
    
    def extract_poverty_composite_indicators(self, all_variables):
        """Create composite poverty indicators from all extracted variables"""
        try:
            # Normalize variables to 0-1 scale for composite indices
            def normalize(value, min_val, max_val):
                if max_val == min_val:
                    return 0
                return max(0, min(1, (value - min_val) / (max_val - min_val)))
            
            # Economic activity composite
            economic_components = [
                normalize(all_variables.get('nighttime_lights_mean', 0), 0, 10),
                normalize(all_variables.get('building_density', 0), 0, 1000),
                normalize(all_variables.get('road_density_km_per_km2', 0), 0, 5),
                all_variables.get('electricity_access_proxy', 0)
            ]
            economic_activity_index = sum(economic_components) / len(economic_components)
            
            # Infrastructure quality composite
            infrastructure_components = [
                all_variables.get('roof_quality_score', 0),
                normalize(all_variables.get('road_density_km_per_km2', 0), 0, 5),
                all_variables.get('infrastructure_access_score', 0),
                1 - normalize(all_variables.get('travel_time_to_cities', 0), 0, 500)  # Inverted
            ]
            infrastructure_quality_index = sum(infrastructure_components) / len(infrastructure_components)
            
            # Environmental conditions composite
            environmental_components = [
                all_variables.get('water_access_proxy', 0),
                1 - all_variables.get('climate_stress_index', 0),  # Inverted
                1 - all_variables.get('water_stress_index', 0),   # Inverted
                normalize(all_variables.get('agricultural_productivity', 0), 0, 1)
            ]
            environmental_conditions_index = sum(environmental_components) / len(environmental_components)
            
            # Overall development index (weighted average)
            overall_development_index = (
                economic_activity_index * 0.4 +
                infrastructure_quality_index * 0.35 +
                environmental_conditions_index * 0.25
            )
            
            # Poverty likelihood (inverted development)
            poverty_likelihood = 1 - overall_development_index
            
            return {
                'economic_activity_index': economic_activity_index,
                'infrastructure_quality_index': infrastructure_quality_index,
                'environmental_conditions_index': environmental_conditions_index,
                'overall_development_index': overall_development_index,
                'poverty_likelihood_score': poverty_likelihood,
                'development_category': self._categorize_development(overall_development_index)
            }
            
        except Exception as e:
            print(f"Error creating composite indicators: {e}")
            return {
                'economic_activity_index': 0, 'infrastructure_quality_index': 0,
                'environmental_conditions_index': 0, 'overall_development_index': 0,
                'poverty_likelihood_score': 1, 'development_category': 'Low'
            }
    
    def _categorize_development(self, index):
        """Categorize development level based on composite index"""
        if index >= 0.7:
            return 'High'
        elif index >= 0.4:
            return 'Medium'
        else:
            return 'Low'