import ee
import geemap
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import warnings
warnings.filterwarnings('ignore')

class EnhancedSatellitePovertyAnalyzer:
    """
    Enhanced poverty analysis with additional socioeconomic indicators
    Built on working SatellitePovertyAnalyzer base class
    """
    
    def __init__(self, project_id='ed-sayandasgupta97'):
        """Initialize GEE and authenticate"""
        try:
            # Initialize Google Earth Engine with project ID
            ee.Initialize(project=project_id)
            print(f"✅ Google Earth Engine initialized with project: {project_id}")
            # Test connection to ensure everything works
            self.test_connection()
        except Exception as e:
            print(f"❌ GEE Authentication failed: {e}")
            print("Please run: earthengine authenticate")
            raise
            
    def test_connection(self):
        """Test GEE connection with actual data"""
        try:
            # Test with a simple collection query
            test_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').limit(1)
            size = test_collection.size().getInfo()
            print(f"✅ GEE connection verified. Test collection size: {size}")
        except Exception as e:
            print(f"❌ GEE connection test failed: {e}")
            raise
    
    def create_buffer_zone(self, lat, lon, radius_km):
        """Create circular buffer around point"""
        # Create point geometry from coordinates
        point = ee.Geometry.Point([lon, lat])
        # Create buffer zone with specified radius in meters
        buffer = point.buffer(radius_km * 1000)
        return buffer
    
    def get_poverty_from_nighttime_lights(self, geometry, start_date, end_date):
        """
        Extract poverty indicators from VIIRS nighttime lights
        Literature: Elvidge et al. (2009), Jean et al. (2016) - NTL as GDP/wealth proxy
        Low nighttime lights = Lower economic activity = Higher poverty risk
        """
        # Print progress message
        print("   🌙 Analyzing nighttime lights for economic poverty indicators...")
        try:
            # Get VIIRS Nighttime Day/Night Band Composites
            viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG') \
                     .filterDate(start_date, end_date) \
                     .filterBounds(geometry)
            
            # Check collection size
            collection_size = viirs.size().getInfo()
            print(f"   📊 Found {collection_size} VIIRS nighttime images")
            
            # Return empty results if no data
            if collection_size == 0:
                return self._empty_ntl_results()
            
            # Get median composite to reduce noise
            ntl_composite = viirs.select('avg_rad').median()
            
            # Calculate poverty-relevant statistics
            stats = ntl_composite.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    reducer2=ee.Reducer.median(),
                    sharedInputs=True
                ).combine(
                    reducer2=ee.Reducer.stdDev(),
                    sharedInputs=True
                ).combine(
                    reducer2=ee.Reducer.percentile([10, 25, 75, 90]),
                    sharedInputs=True
                ).combine(
                    reducer2=ee.Reducer.count(),
                    sharedInputs=True
                ),
                geometry=geometry,
                scale=500,
                maxPixels=1e9,
                bestEffort=True
            ).getInfo()
            
            # Calculate poverty indicators based on literature
            ntl_mean = stats.get('avg_rad_mean', 0) or 0
            ntl_median = stats.get('avg_rad_median', 0) or 0
            ntl_p10 = stats.get('avg_rad_p10', 0) or 0  # Bottom 10% - extreme poverty areas
            ntl_p25 = stats.get('avg_rad_p25', 0) or 0  # Bottom 25% - poverty areas
            
            # Poverty indicators (higher values = more poverty)
            electrification_deficit = max(0, 1 - ntl_mean)  # Lack of electrification
            extreme_poverty_ratio = 1.0 if ntl_p10 < 0.1 else 0.0  # Very dark areas
            economic_isolation_index = max(0, 1 - ntl_median)  # Economic isolation
            
            return {
                'ntl_mean_radiance': ntl_mean,
                'ntl_median_radiance': ntl_median,
                'electrification_deficit': electrification_deficit,
                'extreme_poverty_ratio': extreme_poverty_ratio,
                'economic_isolation_index': economic_isolation_index,
                'dark_area_percentage': (stats.get('avg_rad_p25', 0) or 0) < 0.1,
                'ntl_inequality': (stats.get('avg_rad_stdDev', 0) or 0) / max(ntl_mean, 0.001),
                'ntl_pixel_count': stats.get('avg_rad_count', 0)
            }
            
        except Exception as e:
            print(f"   ❌ Error analyzing nighttime lights: {e}")
            return self._empty_ntl_results()
    
    def get_housing_quality_indicators(self, geometry, start_date, end_date):
        """
        Extract housing quality from Sentinel-2 & SAR data
        Literature: Duque et al. (2015), Kuffer et al. (2016) - roof materials, building density
        Poor housing = Higher poverty
        """
        # Print progress message
        print("   🏠 Analyzing housing quality indicators...")
        try:
            # Sentinel-2 for roof material analysis
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                  .filterDate(start_date, end_date) \
                  .filterBounds(geometry) \
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            
            # Sentinel-1 SAR for building structure analysis
            s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
                  .filterDate(start_date, end_date) \
                  .filterBounds(geometry) \
                  .filter(ee.Filter.eq('instrumentMode', 'IW'))
            
            # Initialize results dictionary
            results = {}
            
            # Process Sentinel-2 data if available
            if s2.size().getInfo() > 0:
                # Calculate indices for roof material assessment
                s2_composite = s2.median()
                
                # Metal roof index (higher reflectance in NIR)
                metal_roof_index = s2_composite.select('B8').divide(s2_composite.select('B4'))
                
                # Concrete/cement roof index 
                concrete_index = s2_composite.select('B11').divide(s2_composite.select('B8'))
                
                # Thatch/organic roof index (higher in red, lower in NIR)
                organic_roof_index = s2_composite.select('B4').divide(s2_composite.select('B8'))
                
                # Calculate statistics
                roof_stats = ee.Image([metal_roof_index, concrete_index, organic_roof_index]).reduceRegion(
                    reducer=ee.Reducer.mean().combine(
                        reducer2=ee.Reducer.percentile([25, 75]),
                        sharedInputs=True
                    ),
                    geometry=geometry,
                    scale=10,
                    maxPixels=1e9,
                    bestEffort=True
                ).getInfo()
                
                # Poor housing indicators (higher = more poverty)
                organic_roof_prevalence = roof_stats.get('B4_mean', 0) or 0
                poor_roof_materials = max(0, organic_roof_prevalence - 0.5)  # High organic/thatch ratio
                
                # Update results with roof material indicators
                results.update({
                    'metal_roof_ratio': roof_stats.get('B8_mean', 0) or 0,
                    'concrete_roof_ratio': roof_stats.get('B11_mean', 0) or 0,
                    'organic_roof_ratio': organic_roof_prevalence,
                    'poor_roof_materials_index': poor_roof_materials,
                    'roof_material_diversity': roof_stats.get('B4_p75', 0) - roof_stats.get('B4_p25', 0)
                })
            
            # Process Sentinel-1 SAR data if available
            if s1.size().getInfo() > 0:
                # SAR analysis for building density and structure
                s1_composite = s1.select(['VV', 'VH']).median()
                
                # Urban density from SAR backscatter
                urban_density = s1_composite.select('VV').subtract(s1_composite.select('VH'))
                
                # Building structure quality (more regular structures have higher VV/VH ratio)
                structure_quality = s1_composite.select('VV').divide(s1_composite.select('VH'))
                
                # Calculate SAR statistics
                sar_stats = ee.Image([urban_density, structure_quality]).reduceRegion(
                    reducer=ee.Reducer.mean().combine(
                        reducer2=ee.Reducer.stdDev(),
                        sharedInputs=True
                    ),
                    geometry=geometry,
                    scale=10,
                    maxPixels=1e9,
                    bestEffort=True
                ).getInfo()
                
                # Informal settlement indicators
                building_density = sar_stats.get('VV_mean', 0) or 0
                structure_irregularity = sar_stats.get('VV_stdDev', 0) or 0
                poor_building_quality = max(0, 1 - (sar_stats.get('VV_mean', 0) or 0))
                
                # Update results with building indicators
                results.update({
                    'building_density_index': building_density,
                    'structure_irregularity': structure_irregularity,
                    'informal_settlement_index': poor_building_quality,
                    'building_quality_variance': structure_irregularity
                })
            
            return results
            
        except Exception as e:
            print(f"   ❌ Error analyzing housing quality: {e}")
            return {}
    
    def get_infrastructure_poverty_indicators(self, geometry):
        """
        Extract infrastructure-based poverty indicators
        Literature: Watmough et al. (2019) - road access, market access as poverty indicators
        Poor infrastructure access = Higher poverty
        """
        # Print progress message
        print("   🛣️ Analyzing infrastructure poverty indicators...")
        try:
            # Initialize results dictionary
            results = {}
            
            # Water access analysis using Global Surface Water
            gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
            water_occurrence = gsw.select('occurrence')
            
            # Distance to reliable water sources (occurrence > 50%)
            reliable_water = water_occurrence.gt(50)
            water_distance = reliable_water.fastDistanceTransform(5000).sqrt()
            
            # Calculate water statistics
            water_stats = water_distance.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    reducer2=ee.Reducer.min(),
                    sharedInputs=True
                ),
                geometry=geometry,
                scale=30,
                maxPixels=1e9,
                bestEffort=True
            ).getInfo()
            
            # Water poverty indicators
            water_access_deficit = min(1.0, (water_stats.get('distance_mean', 5000) or 5000) / 5000)
            severe_water_shortage = 1.0 if (water_stats.get('distance_min', 5000) or 5000) > 2000 else 0.0
            
            # Terrain accessibility analysis using SRTM DEM
            dem = ee.Image('USGS/SRTMGL1_003')
            terrain = ee.Algorithms.Terrain(dem)
            slope = terrain.select('slope')
            
            # Calculate terrain statistics
            terrain_stats = slope.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    reducer2=ee.Reducer.percentile([75, 90]),
                    sharedInputs=True
                ),
                geometry=geometry,
                scale=30,
                maxPixels=1e9,
                bestEffort=True
            ).getInfo()
            
            # Terrain-based isolation index
            mean_slope = terrain_stats.get('slope_mean', 0) or 0
            terrain_isolation = min(1.0, mean_slope / 15.0)  # Normalize by 15 degrees
            geographic_isolation = 1.0 if mean_slope > 10 else 0.0
            
            # Update results with infrastructure indicators
            results.update({
                'water_access_deficit': water_access_deficit,
                'severe_water_shortage': severe_water_shortage,
                'terrain_isolation_index': terrain_isolation,
                'geographic_isolation': geographic_isolation,
                'mean_slope_degrees': mean_slope,
                'water_distance_km': (water_stats.get('distance_mean', 0) or 0) / 1000
            })
            
            return results
            
        except Exception as e:
            print(f"   ❌ Error analyzing infrastructure: {e}")
            return {}
    
    def get_environmental_poverty_indicators(self, geometry, start_date, end_date):
        """
        Extract environmental poverty indicators
        Literature: Chakraborty et al. (2017) - environmental quality and poverty correlation
        Poor environmental conditions = Higher poverty risk
        """
        # Print progress message
        print("   🌱 Analyzing environmental poverty indicators...")
        try:
            # Initialize results dictionary
            results = {}
            
            # Vegetation health analysis using Sentinel-2
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                  .filterDate(start_date, end_date) \
                  .filterBounds(geometry) \
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            
            # Process Sentinel-2 data if available
            if s2.size().getInfo() > 0:
                # Create median composite
                s2_composite = s2.median()
                
                # NDVI for vegetation health
                ndvi = s2_composite.normalizedDifference(['B8', 'B4'])
                
                # Environmental degradation indices
                bare_soil_index = s2_composite.normalizedDifference(['B11', 'B8'])  # NDBI
                
                # Calculate environmental statistics
                env_stats = ee.Image([ndvi, bare_soil_index]).reduceRegion(
                    reducer=ee.Reducer.mean().combine(
                        reducer2=ee.Reducer.percentile([10, 25]),
                        sharedInputs=True
                    ),
                    geometry=geometry,
                    scale=10,
                    maxPixels=1e9,
                    bestEffort=True
                ).getInfo()
                
                # Environmental poverty indicators
                vegetation_deficit = max(0, 0.3 - (env_stats.get('nd_mean', 0) or 0))  # Low NDVI
                environmental_degradation = max(0, (env_stats.get('nd_1_mean', 0) or 0) - 0.1)  # High bare soil
                green_space_deprivation = 1.0 if (env_stats.get('nd_mean', 0) or 0) < 0.2 else 0.0
                
                # Update results with environmental indicators
                results.update({
                    'vegetation_deficit': vegetation_deficit,
                    'environmental_degradation': environmental_degradation,
                    'green_space_deprivation': green_space_deprivation,
                    'mean_ndvi': env_stats.get('nd_mean', 0) or 0,
                    'bare_soil_ratio': env_stats.get('nd_1_mean', 0) or 0
                })
            
            # Air quality proxy from Sentinel-5P
            try:
                # Get NO2 data from Sentinel-5P TROPOMI
                no2 = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2') \
                       .filterDate(start_date, end_date) \
                       .filterBounds(geometry) \
                       .select('NO2_column_number_density')
                
                # Process NO2 data if available
                if no2.size().getInfo() > 0:
                    # Calculate NO2 statistics
                    no2_stats = no2.median().reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=geometry,
                        scale=1000,
                        maxPixels=1e9,
                        bestEffort=True
                    ).getInfo()
                    
                    # Air pollution poverty indicator
                    air_pollution_burden = min(1.0, max(0, (no2_stats.get('NO2_column_number_density', 0) or 0) * 1e6))
                    
                    # Update results with air quality indicators
                    results.update({
                        'air_pollution_burden': air_pollution_burden,
                        'no2_density': no2_stats.get('NO2_column_number_density', 0) or 0
                    })
            except:
                # Default air quality values if data unavailable
                results.update({
                    'air_pollution_burden': 0,
                    'no2_density': 0
                })
            
            return results
            
        except Exception as e:
            print(f"   ❌ Error analyzing environmental indicators: {e}")
            return {}
    
    def get_population_poverty_indicators(self, geometry, year=2020):
        """
        Extract population-based poverty indicators
        Literature: Steele et al. (2017) - population density patterns and poverty
        """
        # Print progress message
        print("   👥 Analyzing population poverty indicators...")
        try:
            # WorldPop population data for India
            population = ee.ImageCollection('WorldPop/GP/100m/pop') \
                          .filter(ee.Filter.eq('year', year)) \
                          .filter(ee.Filter.eq('country', 'IND'))
            
            # Return empty results if no population data
            if population.size().getInfo() == 0:
                return {}
            
            # Create population mosaic
            pop_image = population.mosaic()
            
            # Population statistics
            pop_stats = pop_image.reduceRegion(
                reducer=ee.Reducer.sum().combine(
                    reducer2=ee.Reducer.mean(),
                    sharedInputs=True
                ).combine(
                    reducer2=ee.Reducer.stdDev(),
                    sharedInputs=True
                ).combine(
                    reducer2=ee.Reducer.percentile([10, 90]),
                    sharedInputs=True
                ),
                geometry=geometry,
                scale=100,
                maxPixels=1e9,
                bestEffort=True
            ).getInfo()
            
            # Calculate area and derived indicators
            area_km2 = geometry.area().divide(1e6).getInfo()
            total_pop = pop_stats.get('population_sum', 0) or 0
            pop_density = total_pop / area_km2 if area_km2 > 0 else 0
            
            # Population poverty indicators
            pop_std = pop_stats.get('population_stdDev', 0) or 0
            pop_mean = pop_stats.get('population_mean', 0) or 0
            
            # Calculate inequality and overcrowding
            population_inequality = pop_std / max(pop_mean, 1)  # High inequality may indicate slums
            overcrowding_index = min(1.0, pop_density / 10000)  # Normalize by 10k people/km²
            
            # Settlement pattern analysis
            pop_p90 = pop_stats.get('population_p90', 0) or 0
            pop_p10 = pop_stats.get('population_p10', 0) or 0
            settlement_disparity = (pop_p90 - pop_p10) / max(pop_mean, 1)
            
            return {
                'total_population': total_pop,
                'population_density_per_km2': pop_density,
                'population_inequality': population_inequality,
                'overcrowding_index': overcrowding_index,
                'settlement_disparity': settlement_disparity,
                'area_km2': area_km2
            }
            
        except Exception as e:
            print(f"   ❌ Error analyzing population indicators: {e}")
            return {}
    
    def get_enhanced_access_indicators(self, geometry):
        """
        Analyze access to hospitals, airports, and markets using road networks
        Literature: Alegana et al. (2012) - healthcare access; Kwan (2006) - transportation access
        """
        # Print progress message
        print("   🏥 Analyzing enhanced access indicators (hospitals, airports, markets)...")
        try:
            # Initialize results dictionary
            results = {}
            
            # Try to get OSM healthcare facilities
            try:
                # Healthcare facilities analysis (using point locations as proxy)
                # Distance calculation based on region centroid
                region_centroid = geometry.centroid()
                
                # Major cities in India with hospitals (as reference points)
                major_cities = [
                    [77.2090, 28.6139],  # Delhi
                    [72.8777, 19.0760],  # Mumbai  
                    [88.3639, 22.5726],  # Kolkata
                    [80.2707, 13.0827],  # Chennai
                    [77.5946, 12.9716],  # Bangalore
                    [78.4867, 17.3850],  # Hyderabad
                    [72.5714, 23.0225],  # Ahmedabad
                    [75.7873, 26.9124]   # Jaipur
                ]
                
                # Calculate distance to nearest major city (proxy for hospital access)
                min_city_distance = float('inf')
                for city_coords in major_cities:
                    # Create city point
                    city_point = ee.Geometry.Point(city_coords)
                    # Calculate distance to region centroid
                    distance = region_centroid.distance(city_point)
                    # Update minimum distance
                    distance_km = distance.divide(1000).getInfo()
                    min_city_distance = min(min_city_distance, distance_km)
                
                # Healthcare access indicators
                hospital_access_km = min_city_distance
                hospital_access_deficit = min(1.0, hospital_access_km / 50.0)  # Normalize by 50km
                
                # Update results with healthcare access
                results.update({
                    'distance_to_nearest_major_city_km': hospital_access_km,
                    'hospital_access_deficit': hospital_access_deficit,
                    'healthcare_isolation_index': hospital_access_deficit
                })
                
            except Exception as e:
                print(f"   ⚠️ Healthcare access calculation issue: {e}")
                # Default values if calculation fails
                results.update({
                    'distance_to_nearest_major_city_km': 25.0,
                    'hospital_access_deficit': 0.5,
                    'healthcare_isolation_index': 0.5
                })
            
            # Airport access analysis
            try:
                # Major airports in India
                major_airports = [
                    [77.1025, 28.5562],  # Delhi Airport
                    [72.8682, 19.0896],  # Mumbai Airport
                    [88.4467, 22.6542],  # Kolkata Airport
                    [80.1694, 12.9900],  # Chennai Airport
                    [77.7081, 13.1986],  # Bangalore Airport
                    [78.4291, 17.2403],  # Hyderabad Airport
                    [72.6358, 23.0776],  # Ahmedabad Airport
                    [75.8121, 26.8244]   # Jaipur Airport
                ]
                
                # Calculate distance to nearest airport
                min_airport_distance = float('inf')
                region_centroid = geometry.centroid()
                
                for airport_coords in major_airports:
                    # Create airport point
                    airport_point = ee.Geometry.Point(airport_coords)
                    # Calculate distance
                    distance = region_centroid.distance(airport_point)
                    distance_km = distance.divide(1000).getInfo()
                    min_airport_distance = min(min_airport_distance, distance_km)
                
                # Airport access indicators
                airport_access_km = min_airport_distance
                airport_access_deficit = min(1.0, airport_access_km / 100.0)  # Normalize by 100km
                
                # Update results with airport access
                results.update({
                    'distance_to_nearest_airport_km': airport_access_km,
                    'airport_access_deficit': airport_access_deficit,
                    'transportation_isolation': airport_access_deficit
                })
                
            except Exception as e:
                print(f"   ⚠️ Airport access calculation issue: {e}")
                # Default values if calculation fails
                results.update({
                    'distance_to_nearest_airport_km': 50.0,
                    'airport_access_deficit': 0.5,
                    'transportation_isolation': 0.5
                })
            
            # Market access analysis (using populated areas as proxy)
            try:
                # Use population density as proxy for market access
                # Areas with higher population typically have better market access
                population = ee.ImageCollection('WorldPop/GP/100m/pop') \
                              .filter(ee.Filter.eq('year', 2020)) \
                              .filter(ee.Filter.eq('country', 'IND'))
                
                if population.size().getInfo() > 0:
                    # Get population image
                    pop_image = population.mosaic()
                    
                    # Calculate population within larger area (market catchment)
                    larger_area = geometry.buffer(10000)  # 10km buffer for market analysis
                    
                    # Calculate population statistics in larger area
                    market_stats = pop_image.reduceRegion(
                        reducer=ee.Reducer.mean().combine(
                            reducer2=ee.Reducer.sum(),
                            sharedInputs=True
                        ),
                        geometry=larger_area,
                        scale=100,
                        maxPixels=1e9,
                        bestEffort=True
                    ).getInfo()
                    
                    # Market access indicators based on surrounding population
                    surrounding_population = market_stats.get('population_sum', 0) or 0
                    market_access_score = min(1.0, surrounding_population / 50000)  # Normalize by 50k people
                    market_access_deficit = 1 - market_access_score
                    
                    # Economic opportunities index
                    economic_opportunities = market_access_score
                    
                    # Update results with market access
                    results.update({
                        'surrounding_population_10km': surrounding_population,
                        'market_access_score': market_access_score,
                        'market_access_deficit': market_access_deficit,
                        'economic_opportunities_index': economic_opportunities
                    })
                    
                else:
                    # Default market access values
                    results.update({
                        'surrounding_population_10km': 25000,
                        'market_access_score': 0.5,
                        'market_access_deficit': 0.5,
                        'economic_opportunities_index': 0.5
                    })
                    
            except Exception as e:
                print(f"   ⚠️ Market access calculation issue: {e}")
                # Default market access values
                results.update({
                    'surrounding_population_10km': 25000,
                    'market_access_score': 0.5,
                    'market_access_deficit': 0.5,
                    'economic_opportunities_index': 0.5
                })
            
            return results
            
        except Exception as e:
            print(f"   ❌ Error analyzing enhanced access indicators: {e}")
            return {}
    
    def get_road_quality_indicators(self, geometry, start_date, end_date):
        """
        Analyze road quality using Sentinel-2 spectral characteristics
        Literature: Engstrom et al. (2020) - road surface material detection
        """
        # Print progress message
        print("   🛣️ Analyzing road quality indicators...")
        try:
            # Get Sentinel-2 data
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                  .filterDate(start_date, end_date) \
                  .filterBounds(geometry) \
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            
            # Return empty results if no data
            if s2.size().getInfo() == 0:
                return {}
            
            # Create median composite
            s2_composite = s2.median()
            
            # Road surface indices
            # Paved road index (higher SWIR reflectance)
            paved_road_index = s2_composite.select('B11').divide(s2_composite.select('B4'))
            
            # Unpaved road index (higher visible, lower SWIR)
            unpaved_road_index = s2_composite.select('B4').divide(s2_composite.select('B11'))
            
            # Calculate road quality statistics
            road_stats = ee.Image([paved_road_index, unpaved_road_index]).reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    reducer2=ee.Reducer.percentile([25, 75]),
                    sharedInputs=True
                ),
                geometry=geometry,
                scale=10,
                maxPixels=1e9,
                bestEffort=True
            ).getInfo()
            
            # Road quality indicators
            paved_ratio = road_stats.get('B11_mean', 0) or 0
            unpaved_ratio = road_stats.get('B4_mean', 0) or 0
            
            # Road quality deficit (higher unpaved ratio = higher poverty)
            road_quality_deficit = min(1.0, unpaved_ratio / max(paved_ratio, 0.1))
            road_infrastructure_score = 1 - road_quality_deficit
            
            return {
                'paved_road_index': paved_ratio,
                'unpaved_road_index': unpaved_ratio,
                'road_quality_deficit': road_quality_deficit,
                'road_infrastructure_score': road_infrastructure_score,
                'road_surface_quality': "Good" if road_quality_deficit < 0.3 else "Moderate" if road_quality_deficit < 0.7 else "Poor"
            }
            
        except Exception as e:
            print(f"   ❌ Error analyzing road quality: {e}")
            return {}
    
    def calculate_multidimensional_poverty_index(self, results):
        """
        Calculate Multidimensional Poverty Index (MPI) based on satellite indicators
        Literature: Alkire & Foster (2011), adapted for satellite data
        """
        # Print progress message
        print("   📊 Calculating Multidimensional Poverty Index...")
        
        # Define poverty dimensions and weights (based on literature)
        dimensions = {
            'economic_deprivation': {
                'weight': 0.33,
                'indicators': {
                    'electrification_deficit': 0.4,
                    'economic_isolation_index': 0.3,
                    'extreme_poverty_ratio': 0.3
                }
            },
            'living_standards': {
                'weight': 0.33,
                'indicators': {
                    'poor_roof_materials_index': 0.25,
                    'informal_settlement_index': 0.25,
                    'water_access_deficit': 0.25,
                    'severe_water_shortage': 0.25
                }
            },
            'environmental_deprivation': {
                'weight': 0.34,
                'indicators': {
                    'vegetation_deficit': 0.3,
                    'environmental_degradation': 0.3,
                    'green_space_deprivation': 0.2,
                    'air_pollution_burden': 0.2
                }
            }
        }
        
        # Calculate dimension scores
        dimension_scores = {}
        
        for dim_name, dim_config in dimensions.items():
            # Initialize dimension score
            dim_score = 0
            weight_sum = 0
            
            # Calculate weighted average of indicators within dimension
            for indicator, weight in dim_config['indicators'].items():
                if indicator in results and results[indicator] is not None:
                    value = results[indicator]
                    # Normalize to 0-1 scale where 1 = maximum poverty
                    normalized_value = min(1.0, max(0.0, float(value)))
                    dim_score += weight * normalized_value
                    weight_sum += weight
            
            # Store dimension score
            dimension_scores[dim_name] = dim_score / weight_sum if weight_sum > 0 else 0
        
        # Calculate overall MPI
        mpi_score = sum(dimension_scores[dim] * dimensions[dim]['weight'] 
                       for dim in dimension_scores)
        
        # Poverty classification based on MPI score
        poverty_level = "Low" if mpi_score < 0.33 else "Moderate" if mpi_score < 0.66 else "High"
        
        return {
            'multidimensional_poverty_index': mpi_score,
            'poverty_level_classification': poverty_level,
            'economic_deprivation_score': dimension_scores.get('economic_deprivation', 0),
            'living_standards_score': dimension_scores.get('living_standards', 0),
            'environmental_deprivation_score': dimension_scores.get('environmental_deprivation', 0),
            'poverty_intensity': mpi_score,  # Alias for MPI
            'vulnerable_to_poverty': 1.0 if mpi_score > 0.5 else 0.0
        }
    
    def calculate_enhanced_mpi(self, results):
        """
        Enhanced MPI including new socioeconomic indicators
        """
        # Get base MPI first
        base_mpi = self.calculate_multidimensional_poverty_index(results)
        
        # Enhanced dimensions with new indicators
        enhanced_dimensions = {
            'access_deprivation': {
                'weight': 0.15,
                'indicators': {
                    'hospital_access_deficit': 0.3,
                    'airport_access_deficit': 0.2,
                    'market_access_deficit': 0.3,
                    'road_quality_deficit': 0.2
                }
            },
            'connectivity_isolation': {
                'weight': 0.10,
                'indicators': {
                    'healthcare_isolation_index': 0.4,
                    'transportation_isolation': 0.3,
                    'economic_opportunities_index': 0.3  # Inverted
                }
            }
        }
        
        # Calculate enhanced dimension scores
        enhanced_scores = {}
        for dim_name, dim_config in enhanced_dimensions.items():
            dim_score = 0
            weight_sum = 0
            
            for indicator, weight in dim_config['indicators'].items():
                if indicator in results and results[indicator] is not None:
                    value = results[indicator]
                    normalized_value = min(1.0, max(0.0, float(value)))
                    
                    # Invert economic opportunities (higher = better)
                    if 'opportunities' in indicator:
                        normalized_value = 1 - normalized_value
                    
                    dim_score += weight * normalized_value
                    weight_sum += weight
            
            enhanced_scores[dim_name] = dim_score / weight_sum if weight_sum > 0 else 0
        
        # Calculate enhanced MPI (adjust base weights to make room for new dimensions)
        enhanced_mpi = (
            0.25 * base_mpi.get('economic_deprivation_score', 0) +
            0.25 * base_mpi.get('living_standards_score', 0) +
            0.25 * base_mpi.get('environmental_deprivation_score', 0) +
            0.15 * enhanced_scores.get('access_deprivation', 0) +
            0.10 * enhanced_scores.get('connectivity_isolation', 0)
        )
        
        # Enhanced poverty classification
        enhanced_poverty_level = "Low" if enhanced_mpi < 0.33 else "Moderate" if enhanced_mpi < 0.66 else "High"
        
        # Add enhanced results to base MPI
        base_mpi.update({
            'enhanced_mpi': enhanced_mpi,
            'enhanced_poverty_level': enhanced_poverty_level,
            'access_deprivation_score': enhanced_scores.get('access_deprivation', 0),
            'connectivity_isolation_score': enhanced_scores.get('connectivity_isolation', 0)
        })
        
        return base_mpi
    
    def analyze_poverty_conditions(self, lat, lon, radius_km, start_date='2023-01-01', end_date='2023-12-31'):
        """
        Comprehensive poverty analysis using satellite data with enhanced indicators
        """
        # Print analysis header
        print(f"\n🔍 COMPREHENSIVE ENHANCED POVERTY ANALYSIS")
        print(f"📍 Location: ({lat:.4f}, {lon:.4f})")
        print(f"📏 Analysis Radius: {radius_km} km")
        print(f"📅 Analysis Period: {start_date} to {end_date}")
        print(f"🛰️ Using real satellite data for poverty measurement...")
        
        # Create geometry
        geometry = self.create_buffer_zone(lat, lon, radius_km)
        
        # Initialize results
        results = {
            'latitude': lat,
            'longitude': lon,
            'radius_km': radius_km,
            'analysis_date': datetime.now().isoformat(),
            'start_date': start_date,
            'end_date': end_date,
            'data_sources': []
        }
        
        print(f"\n📡 EXTRACTING POVERTY INDICATORS:")
        print("="*50)
        
        # 1. Economic poverty from nighttime lights
        try:
            ntl_poverty = self.get_poverty_from_nighttime_lights(geometry, start_date, end_date)
            results.update(ntl_poverty)
            results['data_sources'].append('VIIRS_Economic_Poverty')
        except Exception as e:
            print(f"   ❌ Economic poverty analysis failed: {e}")
        
        # 2. Housing quality poverty indicators
        try:
            housing_poverty = self.get_housing_quality_indicators(geometry, start_date, end_date)
            results.update(housing_poverty)
            results['data_sources'].append('Sentinel_Housing_Quality')
        except Exception as e:
            print(f"   ❌ Housing quality analysis failed: {e}")
        
        # 3. Infrastructure poverty indicators
        try:
            infra_poverty = self.get_infrastructure_poverty_indicators(geometry)
            results.update(infra_poverty)
            results['data_sources'].append('Infrastructure_Poverty')
        except Exception as e:
            print(f"   ❌ Infrastructure poverty analysis failed: {e}")
        
        # 4. Environmental poverty indicators
        try:
            env_poverty = self.get_environmental_poverty_indicators(geometry, start_date, end_date)
            results.update(env_poverty)
            results['data_sources'].append('Environmental_Poverty')
        except Exception as e:
            print(f"   ❌ Environmental poverty analysis failed: {e}")
        
        # 5. Population poverty indicators
        try:
            pop_poverty = self.get_population_poverty_indicators(geometry)
            results.update(pop_poverty)
            results['data_sources'].append('Population_Poverty')
        except Exception as e:
            print(f"   ❌ Population poverty analysis failed: {e}")
        
        # 6. Enhanced access indicators (NEW)
        try:
            access_poverty = self.get_enhanced_access_indicators(geometry)
            results.update(access_poverty)
            results['data_sources'].append('Enhanced_Access_Indicators')
        except Exception as e:
            print(f"   ❌ Enhanced access analysis failed: {e}")
        
        # 7. Road quality indicators (NEW)
        try:
            road_poverty = self.get_road_quality_indicators(geometry, start_date, end_date)
            results.update(road_poverty)
            results['data_sources'].append('Road_Quality_Indicators')
        except Exception as e:
            print(f"   ❌ Road quality analysis failed: {e}")
        
        # 8. Calculate Enhanced Multidimensional Poverty Index
        try:
            mpi_results = self.calculate_enhanced_mpi(results)
            results.update(mpi_results)
        except Exception as e:
            print(f"   ❌ Enhanced MPI calculation failed: {e}")
        
        print(f"\n✅ ENHANCED POVERTY ANALYSIS COMPLETE!")
        print(f"📊 Data sources used: {len(results['data_sources'])}")
        
        return results
    
    def generate_poverty_report(self, results):
        """Generate comprehensive poverty analysis report with enhanced indicators"""
        
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE ENHANCED POVERTY ANALYSIS REPORT")
        print("="*80)
        print(f"📍 Location: ({results['latitude']}, {results['longitude']})")
        print(f"📏 Analysis Radius: {results['radius_km']} km")
        print(f"📅 Analysis Period: {results['start_date']} to {results['end_date']}")
        
        # Enhanced Multidimensional Poverty Index Summary
        if 'enhanced_mpi' in results:
            mpi = results['enhanced_mpi']
            poverty_level = results.get('enhanced_poverty_level', 'Unknown')
            
            print(f"\n🎯 ENHANCED POVERTY ASSESSMENT SUMMARY:")
            print(f"   📊 Enhanced Multidimensional Poverty Index (MPI): {mpi:.3f}")
            print(f"   📈 Enhanced Poverty Level: {poverty_level}")
            print(f"   📊 Original MPI: {results.get('multidimensional_poverty_index', 0):.3f}")
            print(f"   ⚠️ Vulnerable to Poverty: {'Yes' if results.get('vulnerable_to_poverty', 0) > 0.5 else 'No'}")
        
        # Economic Poverty Indicators
        print(f"\n💰 ECONOMIC POVERTY INDICATORS:")
        if 'electrification_deficit' in results:
            print(f"   ⚡ Electrification Deficit: {results['electrification_deficit']:.3f}")
            print(f"   🌙 Economic Isolation Index: {results.get('economic_isolation_index', 0):.3f}")
            print(f"   🔴 Extreme Poverty Areas: {results.get('extreme_poverty_ratio', 0):.3f}")
            print(f"   💡 Mean Nighttime Radiance: {results.get('ntl_mean_radiance', 0):.4f} nW/cm²/sr")
        
        # Housing & Living Standards
        print(f"\n🏠 HOUSING & LIVING STANDARDS:")
        if 'poor_roof_materials_index' in results:
            print(f"   🏚️ Poor Roof Materials Index: {results['poor_roof_materials_index']:.3f}")
            print(f"   🏘️ Informal Settlement Index: {results.get('informal_settlement_index', 0):.3f}")
            print(f"   🏗️ Building Quality Issues: {'High' if results.get('structure_irregularity', 0) > 0.5 else 'Low'}")
        
        # Water & Infrastructure Access  
        print(f"\n💧 WATER & INFRASTRUCTURE ACCESS:")
        if 'water_access_deficit' in results:
            print(f"   🚰 Water Access Deficit: {results['water_access_deficit']:.3f}")
            print(f"   🌊 Severe Water Shortage: {'Yes' if results.get('severe_water_shortage', 0) > 0.5 else 'No'}")
            print(f"   🛣️ Terrain Isolation: {results.get('terrain_isolation_index', 0):.3f}")
            print(f"   📏 Distance to Water: {results.get('water_distance_km', 0):.1f} km")
        
        # Enhanced Access Indicators (NEW)
        print(f"\n🏥 ENHANCED ACCESS INDICATORS:")
        if 'hospital_access_deficit' in results:
            print(f"   🏥 Hospital Access Deficit: {results['hospital_access_deficit']:.3f}")
            print(f"   🏨 Distance to Major City: {results.get('distance_to_nearest_major_city_km', 0):.1f} km")
            print(f"   ✈️ Airport Access Deficit: {results.get('airport_access_deficit', 0):.3f}")
            print(f"   ✈️ Distance to Airport: {results.get('distance_to_nearest_airport_km', 0):.1f} km")
            print(f"   🏪 Market Access Deficit: {results.get('market_access_deficit', 0):.3f}")
            print(f"   💼 Economic Opportunities: {results.get('economic_opportunities_index', 0):.3f}")
        
        # Road Quality Indicators (NEW)
        print(f"\n🛣️ ROAD QUALITY INDICATORS:")
        if 'road_quality_deficit' in results:
            print(f"   🛣️ Road Quality Deficit: {results['road_quality_deficit']:.3f}")
            print(f"   🛤️ Road Infrastructure Score: {results.get('road_infrastructure_score', 0):.3f}")
            print(f"   🚧 Road Surface Quality: {results.get('road_surface_quality', 'Unknown')}")
            print(f"   📊 Paved Road Index: {results.get('paved_road_index', 0):.3f}")
        
        # Environmental Conditions
        print(f"\n🌍 ENVIRONMENTAL POVERTY INDICATORS:")
        if 'vegetation_deficit' in results:
            print(f"   🌱 Vegetation Deficit: {results['vegetation_deficit']:.3f}")
            print(f"   🏜️ Environmental Degradation: {results.get('environmental_degradation', 0):.3f}")
            print(f"   🌳 Green Space Deprivation: {'Yes' if results.get('green_space_deprivation', 0) > 0.5 else 'No'}")
            print(f"   🌫️ Air Pollution Burden: {results.get('air_pollution_burden', 0):.3f}")
        
        # Population & Settlement Patterns
        print(f"\n👥 POPULATION & SETTLEMENT ANALYSIS:")
        if 'total_population' in results:
            print(f"   👨‍👩‍👧‍👦 Total Population: {results['total_population']:.0f} people")
            print(f"   🏘️ Population Density: {results.get('population_density_per_km2', 0):.1f} people/km²")
            print(f"   📊 Population Inequality: {results.get('population_inequality', 0):.3f}")
            print(f"   🏠 Overcrowding Index: {results.get('overcrowding_index', 0):.3f}")
            print(f"   👥 Surrounding Population (10km): {results.get('surrounding_population_10km', 0):.0f}")
        
        # Enhanced Dimension Scores (NEW)
        if 'access_deprivation_score' in results:
            print(f"\n📊 ENHANCED POVERTY DIMENSION SCORES:")
            print(f"   💰 Economic Deprivation: {results.get('economic_deprivation_score', 0):.3f}")
            print(f"   🏠 Living Standards: {results.get('living_standards_score', 0):.3f}")
            print(f"   🌍 Environmental Deprivation: {results.get('environmental_deprivation_score', 0):.3f}")
            print(f"   🔗 Access Deprivation: {results['access_deprivation_score']:.3f}")
            print(f"   📡 Connectivity Isolation: {results.get('connectivity_isolation_score', 0):.3f}")
        
        # Risk Assessment
        mpi = results.get('enhanced_mpi', results.get('multidimensional_poverty_index', 0))
        print(f"\n⚠️ POVERTY RISK ASSESSMENT:")
        if mpi < 0.2:
            print(f"   ✅ Low Poverty Risk - Well-developed area with good access")
        elif mpi < 0.4:
            print(f"   🟡 Moderate Poverty Risk - Some development and access gaps")
        elif mpi < 0.6:
            print(f"   🟠 High Poverty Risk - Significant deprivations and poor access")
        else:
            print(f"   🔴 Extreme Poverty Risk - Multiple severe deprivations and isolation")
        
        # Access Quality Summary (NEW)
        print(f"\n🔗 ACCESS QUALITY SUMMARY:")
        hospital_access = 1 - results.get('hospital_access_deficit', 0.5)
        airport_access = 1 - results.get('airport_access_deficit', 0.5)
        market_access = results.get('market_access_score', 0.5)
        road_quality = results.get('road_infrastructure_score', 0.5)
        
        print(f"   🏥 Healthcare Access Quality: {'Good' if hospital_access > 0.7 else 'Moderate' if hospital_access > 0.4 else 'Poor'}")
        print(f"   ✈️ Transportation Access: {'Good' if airport_access > 0.7 else 'Moderate' if airport_access > 0.4 else 'Poor'}")
        print(f"   🏪 Market Access: {'Good' if market_access > 0.7 else 'Moderate' if market_access > 0.4 else 'Poor'}")
        print(f"   🛣️ Road Infrastructure: {'Good' if road_quality > 0.7 else 'Moderate' if road_quality > 0.4 else 'Poor'}")
        
        # Data Quality Assessment
        print(f"\n📈 DATA QUALITY & SOURCES:")
        print(f"   🛰️ Satellite Data Sources: {len(results.get('data_sources', []))}")
        total_indicators = len([k for k in results.keys() if k not in ['latitude', 'longitude', 'radius_km', 'analysis_date', 'start_date', 'end_date', 'data_sources']])
        print(f"   📊 Total Indicators Extracted: {total_indicators}")
        print(f"   📅 Analysis Completed: {results.get('analysis_date', 'Unknown')}")
        
        print("="*80)
        
        return results
    
    def _empty_ntl_results(self):
        """Return empty nighttime lights results"""
        return {
            'ntl_mean_radiance': 0,
            'ntl_median_radiance': 0,
            'electrification_deficit': 1.0,  # Assume high deficit if no data
            'extreme_poverty_ratio': 1.0,
            'economic_isolation_index': 1.0,
            'dark_area_percentage': True,
            'ntl_inequality': 0,
            'ntl_pixel_count': 0
        }

# Additional utility functions for enhanced socioeconomic analysis
def compare_poverty_levels(analyzer, locations):
    """Compare poverty levels across multiple locations with enhanced indicators"""
    print("🔍 ENHANCED COMPARATIVE POVERTY ANALYSIS")
    print("="*60)
    
    all_results = []
    
    for location in locations:
        print(f"\n📍 Analyzing: {location['name']}")
        results = analyzer.analyze_poverty_conditions(**{k: v for k, v in location.items() if k != 'name'})
        results['location_name'] = location['name']
        all_results.append(results)
    
    # Compare Enhanced MPI scores
    print(f"\n📊 ENHANCED POVERTY COMPARISON SUMMARY:")
    print(f"{'Location':<25} {'Enhanced MPI':<15} {'Poverty Level':<15} {'Access Quality':<15} {'Risk Level'}")
    print("-" * 90)
    
    for result in sorted(all_results, key=lambda x: x.get('enhanced_mpi', x.get('multidimensional_poverty_index', 0)), reverse=True):
        enhanced_mpi = result.get('enhanced_mpi', result.get('multidimensional_poverty_index', 0))
        poverty_level = result.get('enhanced_poverty_level', result.get('poverty_level_classification', 'Unknown'))
        
        # Calculate access quality score
        hospital_access = 1 - result.get('hospital_access_deficit', 0.5)
        market_access = result.get('market_access_score', 0.5)
        road_quality = result.get('road_infrastructure_score', 0.5)
        access_quality = (hospital_access + market_access + road_quality) / 3
        
        access_qual_text = "Good" if access_quality > 0.7 else "Moderate" if access_quality > 0.4 else "Poor"
        risk = "🔴 High" if enhanced_mpi > 0.6 else "🟠 Moderate" if enhanced_mpi > 0.3 else "🟢 Low"
        
        print(f"{result['location_name']:<25} {enhanced_mpi:<15.3f} {poverty_level:<15} {access_qual_text:<15} {risk}")
    
    return all_results

def create_enhanced_poverty_dataset(results_list, output_file='enhanced_poverty_dataset.csv'):
    """Create a comprehensive dataset from multiple enhanced poverty analyses"""
    
    # Define comprehensive poverty indicators to extract
    poverty_indicators = [
        # Location and basic info
        'latitude', 'longitude', 'radius_km',
        
        # Enhanced MPI indicators
        'enhanced_mpi', 'enhanced_poverty_level',
        'multidimensional_poverty_index', 'poverty_level_classification',
        
        # Original poverty dimensions
        'electrification_deficit', 'economic_isolation_index', 'extreme_poverty_ratio',
        'poor_roof_materials_index', 'informal_settlement_index', 
        'water_access_deficit', 'severe_water_shortage',
        'vegetation_deficit', 'environmental_degradation', 'air_pollution_burden',
        'population_density_per_km2', 'overcrowding_index',
        'economic_deprivation_score', 'living_standards_score', 'environmental_deprivation_score',
        
        # Enhanced access indicators (NEW)
        'hospital_access_deficit', 'distance_to_nearest_major_city_km',
        'airport_access_deficit', 'distance_to_nearest_airport_km',
        'market_access_deficit', 'market_access_score', 'economic_opportunities_index',
        'surrounding_population_10km',
        
        # Road quality indicators (NEW)
        'road_quality_deficit', 'road_infrastructure_score', 'road_surface_quality',
        'paved_road_index', 'unpaved_road_index',
        
        # Enhanced dimension scores (NEW)
        'access_deprivation_score', 'connectivity_isolation_score',
        
        # Isolation indices
        'healthcare_isolation_index', 'transportation_isolation'
    ]
    
    # Extract data for each location
    dataset = []
    for results in results_list:
        row = {}
        for indicator in poverty_indicators:
            row[indicator] = results.get(indicator, None)
        
        # Add location name if available
        if 'location_name' in results:
            row['location_name'] = results['location_name']
            
        dataset.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(dataset)
    df.to_csv(output_file, index=False)
    
    print(f"💾 Enhanced poverty indicators dataset saved to: {output_file}")
    print(f"📊 Dataset contains {len(df)} locations and {len(poverty_indicators)} indicators")
    print(f"🆕 New indicators include: hospital access, airport access, market access, road quality")
    
    return df

# Testing functions for enhanced poverty analysis
def test_enhanced_poverty_analysis():
    """Test enhanced poverty analysis with known locations"""
    print("🧪 TESTING ENHANCED SATELLITE-BASED POVERTY ANALYSIS")
    print("="*70)
    
    analyzer = EnhancedSatellitePovertyAnalyzer()
    
    # Test locations representing different poverty and access levels
    test_locations = [
        # Urban slum area (expected high poverty, poor access)
        {"name": "Mumbai Slum Area", "lat": 19.0370, "lon": 72.8570, "radius_km": 2},
        
        # Urban developed area (expected low poverty, good access)
        {"name": "Mumbai Business District", "lat": 19.0608, "lon": 72.8347, "radius_km": 2},
        
        # Rural area (expected moderate poverty, poor access)
        {"name": "Rural Bihar", "lat": 25.0961, "lon": 85.3131, "radius_km": 5},
        
        # Tier-2 city (expected moderate poverty and access)
        {"name": "Jaipur Suburban", "lat": 26.9124, "lon": 75.7873, "radius_km": 3},
        
        # Remote area (expected high isolation)
        {"name": "Remote Mountain Area", "lat": 30.7333, "lon": 79.0667, "radius_km": 5}
    ]
    
    # Analyze each location
    results = compare_poverty_levels(analyzer, test_locations)
    
    # Create enhanced research dataset
    dataset = create_enhanced_poverty_dataset(results)
    
    return results, dataset

# Main execution
if __name__ == "__main__":
    print("🚀 ENHANCED SATELLITE-BASED POVERTY ANALYSIS SYSTEM")
    print("="*70)
    print("🌟 Now including hospital, airport, market access and road quality indicators")
    
    # Initialize enhanced analyzer
    analyzer = EnhancedSatellitePovertyAnalyzer()
    
    # Define test locations for comparison
    test_locations = [
        {"name": "Wealthy Urban Area", "lat": 19.0608, "lon": 72.8347, "radius_km": 2},
        {"name": "Rural Poor Area", "lat": 22.97013056, "lon": 85.26155000, "radius_km": 2}
    ]
    
     # Run enhanced comparative analysis
    print("\n🔍 Running enhanced poverty analysis with access indicators...")
    results = compare_poverty_levels(analyzer, test_locations)
    
    # Save results with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"enhanced_poverty_analysis_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Create enhanced dataset
    dataset = create_enhanced_poverty_dataset(results, f"enhanced_poverty_dataset_{timestamp}.csv")
    
    print(f"📊 Enhanced analysis complete! Dataset has {len(dataset)} locations with {len(dataset.columns)} indicators.")