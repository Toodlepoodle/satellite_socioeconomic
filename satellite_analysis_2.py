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
    Now includes individual measure inspection capabilities
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
    
    def get_poverty_from_nighttime_lights(self, geometry, start_date, end_date, detailed_output=False):
        """
        Extract poverty indicators from VIIRS nighttime lights
        Literature: Elvidge et al. (2009), Jean et al. (2016) - NTL as GDP/wealth proxy
        Low nighttime lights = Lower economic activity = Higher poverty risk
        
        Args:
            detailed_output: If True, prints detailed analysis of each step
        """
        if detailed_output:
            print("\n🌙 DETAILED NIGHTTIME LIGHTS POVERTY ANALYSIS")
            print("="*60)
            print("📚 Method: Using VIIRS Day/Night Band for economic activity assessment")
            print("📊 Logic: Low nighttime luminosity indicates limited electrification and economic activity")
        
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
            
            if detailed_output:
                print(f"   📅 Date range: {start_date} to {end_date}")
                print(f"   🛰️ VIIRS collection size: {collection_size} images")
            
            # Return empty results if no data
            if collection_size == 0:
                if detailed_output:
                    print("   ⚠️ No VIIRS data available - using default high poverty indicators")
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
            
            if detailed_output:
                print(f"   📈 Raw Statistics:")
                print(f"      Mean radiance: {ntl_mean:.4f} nW/cm²/sr")
                print(f"      Median radiance: {ntl_median:.4f} nW/cm²/sr")
                print(f"      10th percentile: {ntl_p10:.4f} nW/cm²/sr")
                print(f"      25th percentile: {ntl_p25:.4f} nW/cm²/sr")
            
            # Poverty indicators (higher values = more poverty)
            electrification_deficit = max(0, 1 - ntl_mean)  # Lack of electrification
            extreme_poverty_ratio = 1.0 if ntl_p10 < 0.1 else 0.0  # Very dark areas
            economic_isolation_index = max(0, 1 - ntl_median)  # Economic isolation
            
            if detailed_output:
                print(f"   🔍 Derived Poverty Indicators:")
                print(f"      Electrification Deficit: {electrification_deficit:.3f} (0=good, 1=poor)")
                print(f"      Extreme Poverty Ratio: {extreme_poverty_ratio:.3f} (0=none, 1=present)")
                print(f"      Economic Isolation: {economic_isolation_index:.3f} (0=connected, 1=isolated)")
                
                # Interpretation
                if electrification_deficit > 0.7:
                    print("      ⚠️ High electrification deficit indicates poor economic development")
                elif electrification_deficit > 0.4:
                    print("      🟡 Moderate electrification deficit")
                else:
                    print("      ✅ Good electrification levels")
            
            results = {
                'ntl_mean_radiance': ntl_mean,
                'ntl_median_radiance': ntl_median,
                'electrification_deficit': electrification_deficit,
                'extreme_poverty_ratio': extreme_poverty_ratio,
                'economic_isolation_index': economic_isolation_index,
                'dark_area_percentage': (stats.get('avg_rad_p25', 0) or 0) < 0.1,
                'ntl_inequality': (stats.get('avg_rad_stdDev', 0) or 0) / max(ntl_mean, 0.001),
                'ntl_pixel_count': stats.get('avg_rad_count', 0)
            }
            
            if detailed_output:
                print(f"   ✅ Nighttime lights analysis complete")
                
            return results
            
        except Exception as e:
            print(f"   ❌ Error analyzing nighttime lights: {e}")
            return self._empty_ntl_results()
    
    def get_housing_quality_indicators(self, geometry, start_date, end_date, detailed_output=False):
        """
        Extract housing quality from Sentinel-2 & SAR data
        Literature: Duque et al. (2015), Kuffer et al. (2016) - roof materials, building density
        Poor housing = Higher poverty
        
        Args:
            detailed_output: If True, prints detailed analysis of each step
        """
        if detailed_output:
            print("\n🏠 DETAILED HOUSING QUALITY ANALYSIS")
            print("="*60)
            print("📚 Method: Sentinel-2 optical + Sentinel-1 SAR for roof materials and building structure")
            print("📊 Logic: Poor roof materials and irregular structures indicate poverty")
        
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
            
            if detailed_output:
                s2_size = s2.size().getInfo()
                s1_size = s1.size().getInfo()
                print(f"   🛰️ Sentinel-2 images: {s2_size}")
                print(f"   📡 Sentinel-1 images: {s1_size}")
            
            # Initialize results dictionary
            results = {}
            
            # Process Sentinel-2 data if available
            if s2.size().getInfo() > 0:
                if detailed_output:
                    print("   🔍 Analyzing roof materials with Sentinel-2...")
                
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
                
                if detailed_output:
                    print(f"      Metal roof ratio: {roof_stats.get('B8_mean', 0):.3f}")
                    print(f"      Concrete roof ratio: {roof_stats.get('B11_mean', 0):.3f}")
                    print(f"      Organic roof prevalence: {organic_roof_prevalence:.3f}")
                    print(f"      Poor roof materials index: {poor_roof_materials:.3f}")
                
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
                if detailed_output:
                    print("   📡 Analyzing building structure with Sentinel-1 SAR...")
                
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
                
                if detailed_output:
                    print(f"      Building density index: {building_density:.3f}")
                    print(f"      Structure irregularity: {structure_irregularity:.3f}")
                    print(f"      Informal settlement index: {poor_building_quality:.3f}")
                
                # Update results with building indicators
                results.update({
                    'building_density_index': building_density,
                    'structure_irregularity': structure_irregularity,
                    'informal_settlement_index': poor_building_quality,
                    'building_quality_variance': structure_irregularity
                })
            
            if detailed_output:
                print(f"   ✅ Housing quality analysis complete")
                
            return results
            
        except Exception as e:
            print(f"   ❌ Error analyzing housing quality: {e}")
            return {}
    
    def get_infrastructure_poverty_indicators(self, geometry, detailed_output=False):
        """
        Extract infrastructure-based poverty indicators
        Literature: Watmough et al. (2019) - road access, market access as poverty indicators
        Poor infrastructure access = Higher poverty
        
        Args:
            detailed_output: If True, prints detailed analysis of each step
        """
        if detailed_output:
            print("\n🛣️ DETAILED INFRASTRUCTURE ANALYSIS")
            print("="*60)
            print("📚 Method: Global Surface Water + SRTM DEM for water access and terrain analysis")
            print("📊 Logic: Distance to water and difficult terrain indicate infrastructure poverty")
        
        # Print progress message
        print("   🛣️ Analyzing infrastructure poverty indicators...")
        try:
            # Initialize results dictionary
            results = {}
            
            # Water access analysis using Global Surface Water
            gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
            water_occurrence = gsw.select('occurrence')
            
            if detailed_output:
                print("   💧 Analyzing water access...")
            
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
            water_distance_m = water_stats.get('distance_mean', 5000) or 5000
            water_access_deficit = min(1.0, water_distance_m / 5000)
            severe_water_shortage = 1.0 if (water_stats.get('distance_min', 5000) or 5000) > 2000 else 0.0
            
            if detailed_output:
                print(f"      Average distance to water: {water_distance_m/1000:.1f} km")
                print(f"      Water access deficit: {water_access_deficit:.3f}")
                print(f"      Severe water shortage: {'Yes' if severe_water_shortage > 0.5 else 'No'}")
            
            # Terrain accessibility analysis using SRTM DEM
            dem = ee.Image('USGS/SRTMGL1_003')
            terrain = ee.Algorithms.Terrain(dem)
            slope = terrain.select('slope')
            
            if detailed_output:
                print("   ⛰️ Analyzing terrain accessibility...")
            
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
            
            if detailed_output:
                print(f"      Mean slope: {mean_slope:.1f} degrees")
                print(f"      Terrain isolation index: {terrain_isolation:.3f}")
                print(f"      Geographic isolation: {'Yes' if geographic_isolation > 0.5 else 'No'}")
            
            # Update results with infrastructure indicators
            results.update({
                'water_access_deficit': water_access_deficit,
                'severe_water_shortage': severe_water_shortage,
                'terrain_isolation_index': terrain_isolation,
                'geographic_isolation': geographic_isolation,
                'mean_slope_degrees': mean_slope,
                'water_distance_km': water_distance_m / 1000
            })
            
            if detailed_output:
                print(f"   ✅ Infrastructure analysis complete")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Error analyzing infrastructure: {e}")
            return {}
    
    def get_environmental_poverty_indicators(self, geometry, start_date, end_date, detailed_output=False):
        """
        Extract environmental poverty indicators
        Literature: Chakraborty et al. (2017) - environmental quality and poverty correlation
        Poor environmental conditions = Higher poverty risk
        
        Args:
            detailed_output: If True, prints detailed analysis of each step
        """
        if detailed_output:
            print("\n🌱 DETAILED ENVIRONMENTAL ANALYSIS")
            print("="*60)
            print("📚 Method: Sentinel-2 NDVI + Sentinel-5P air quality for environmental conditions")
            print("📊 Logic: Poor vegetation and high pollution correlate with poverty")
        
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
                if detailed_output:
                    print(f"   🌿 Analyzing vegetation with {s2.size().getInfo()} Sentinel-2 images...")
                
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
                ndvi_mean = env_stats.get('nd_mean', 0) or 0
                vegetation_deficit = max(0, 0.3 - ndvi_mean)  # Low NDVI
                environmental_degradation = max(0, (env_stats.get('nd_1_mean', 0) or 0) - 0.1)  # High bare soil
                green_space_deprivation = 1.0 if ndvi_mean < 0.2 else 0.0
                
                if detailed_output:
                    print(f"      Mean NDVI: {ndvi_mean:.3f}")
                    print(f"      Vegetation deficit: {vegetation_deficit:.3f}")
                    print(f"      Environmental degradation: {environmental_degradation:.3f}")
                    print(f"      Green space deprivation: {'Yes' if green_space_deprivation > 0.5 else 'No'}")
                    
                    # Interpretation
                    if ndvi_mean > 0.4:
                        print("      ✅ Good vegetation health")
                    elif ndvi_mean > 0.2:
                        print("      🟡 Moderate vegetation")
                    else:
                        print("      ⚠️ Poor vegetation - indicates degraded environment")
                
                # Update results with environmental indicators
                results.update({
                    'vegetation_deficit': vegetation_deficit,
                    'environmental_degradation': environmental_degradation,
                    'green_space_deprivation': green_space_deprivation,
                    'mean_ndvi': ndvi_mean,
                    'bare_soil_ratio': env_stats.get('nd_1_mean', 0) or 0
                })
            
            # Air quality proxy from Sentinel-5P
            try:
                if detailed_output:
                    print("   🌫️ Analyzing air quality with Sentinel-5P...")
                
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
                    no2_density = no2_stats.get('NO2_column_number_density', 0) or 0
                    air_pollution_burden = min(1.0, max(0, no2_density * 1e6))
                    
                    if detailed_output:
                        print(f"      NO2 density: {no2_density:.2e} mol/m²")
                        print(f"      Air pollution burden: {air_pollution_burden:.3f}")
                    
                    # Update results with air quality indicators
                    results.update({
                        'air_pollution_burden': air_pollution_burden,
                        'no2_density': no2_density
                    })
                else:
                    if detailed_output:
                        print("      ⚠️ No Sentinel-5P data available")
                    results.update({
                        'air_pollution_burden': 0,
                        'no2_density': 0
                    })
            except:
                # Default air quality values if data unavailable
                results.update({
                    'air_pollution_burden': 0,
                    'no2_density': 0
                })
            
            if detailed_output:
                print(f"   ✅ Environmental analysis complete")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Error analyzing environmental indicators: {e}")
            return {}
    
    def get_population_poverty_indicators(self, geometry, year=2020, detailed_output=False):
        """
        Extract population-based poverty indicators
        Literature: Steele et al. (2017) - population density patterns and poverty
        
        Args:
            detailed_output: If True, prints detailed analysis of each step
        """
        if detailed_output:
            print("\n👥 DETAILED POPULATION ANALYSIS")
            print("="*60)
            print("📚 Method: WorldPop high-resolution population data")
            print("📊 Logic: Population density patterns and inequality indicate settlement quality")
        
        # Print progress message
        print("   👥 Analyzing population poverty indicators...")
        try:
            # WorldPop population data for India
            population = ee.ImageCollection('WorldPop/GP/100m/pop') \
                          .filter(ee.Filter.eq('year', year)) \
                          .filter(ee.Filter.eq('country', 'IND'))
            
            # Return empty results if no population data
            if population.size().getInfo() == 0:
                if detailed_output:
                    print("   ⚠️ No WorldPop data available")
                return {}
            
            if detailed_output:
                print(f"   📊 Using WorldPop {year} data for India")
            
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
            
            if detailed_output:
                print(f"      Total population: {total_pop:.0f} people")
                print(f"      Population density: {pop_density:.1f} people/km²")
                print(f"      Population inequality: {population_inequality:.3f}")
                print(f"      Overcrowding index: {overcrowding_index:.3f}")
                print(f"      Settlement disparity: {settlement_disparity:.3f}")
                
                # Interpretation
                if pop_density > 5000:
                    print("      ⚠️ High density - potential overcrowding")
                elif pop_density > 1000:
                    print("      🟡 Moderate density")
                else:
                    print("      ✅ Low to moderate density")
            
            results = {
                'total_population': total_pop,
                'population_density_per_km2': pop_density,
                'population_inequality': population_inequality,
                'overcrowding_index': overcrowding_index,
                'settlement_disparity': settlement_disparity,
                'area_km2': area_km2
            }
            
            if detailed_output:
                print(f"   ✅ Population analysis complete")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Error analyzing population indicators: {e}")
            return {}
    
    def get_enhanced_access_indicators(self, geometry, detailed_output=False):
        """
        Analyze access to hospitals, airports, and markets using road networks
        Literature: Alegana et al. (2012) - healthcare access; Kwan (2006) - transportation access
        
        Args:
            detailed_output: If True, prints detailed analysis of each step
        """
        if detailed_output:
            print("\n🏥 DETAILED ACCESS ANALYSIS")
            print("="*60)
            print("📚 Method: Distance to major cities/airports + population density for market access")
            print("📊 Logic: Distance to services and low population indicate poor access")
        
        # Print progress message
        print("   🏥 Analyzing enhanced access indicators (hospitals, airports, markets)...")
        try:
            # Initialize results dictionary
            results = {}
            
            # Try to get OSM healthcare facilities
            try:
                if detailed_output:
                    print("   🏥 Calculating healthcare access...")
                
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
                
                if detailed_output:
                    print(f"      Distance to nearest major city: {hospital_access_km:.1f} km")
                    print(f"      Hospital access deficit: {hospital_access_deficit:.3f}")
                    if hospital_access_km > 30:
                        print("      ⚠️ Poor healthcare access - far from major medical centers")
                    elif hospital_access_km > 15:
                        print("      🟡 Moderate healthcare access")
                    else:
                        print("      ✅ Good healthcare access")
                
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
                if detailed_output:
                    print("   ✈️ Calculating airport access...")
                
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
                
                if detailed_output:
                    print(f"      Distance to nearest airport: {airport_access_km:.1f} km")
                    print(f"      Airport access deficit: {airport_access_deficit:.3f}")
                    if airport_access_km > 80:
                        print("      ⚠️ Poor transportation access - far from airports")
                    elif airport_access_km > 40:
                        print("      🟡 Moderate transportation access")
                    else:
                        print("      ✅ Good transportation access")
                
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
                if detailed_output:
                    print("   🏪 Calculating market access...")
                
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
                    
                    if detailed_output:
                        print(f"      Surrounding population (10km): {surrounding_population:.0f}")
                        print(f"      Market access score: {market_access_score:.3f}")
                        print(f"      Market access deficit: {market_access_deficit:.3f}")
                        if market_access_score > 0.7:
                            print("      ✅ Good market access - high surrounding population")
                        elif market_access_score > 0.4:
                            print("      🟡 Moderate market access")
                        else:
                            print("      ⚠️ Poor market access - low surrounding population")
                    
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
            
            if detailed_output:
                print(f"   ✅ Access analysis complete")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Error analyzing enhanced access indicators: {e}")
            return {}
    
    def get_road_quality_indicators(self, geometry, start_date, end_date, detailed_output=False):
        """
        Analyze road quality using Sentinel-2 spectral characteristics
        Literature: Engstrom et al. (2020) - road surface material detection
        
        Args:
            detailed_output: If True, prints detailed analysis of each step
        """
        if detailed_output:
            print("\n🛣️ DETAILED ROAD QUALITY ANALYSIS")
            print("="*60)
            print("📚 Method: Sentinel-2 spectral analysis for road surface materials")
            print("📊 Logic: Paved roads have different spectral signatures than unpaved roads")
        
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
                if detailed_output:
                    print("   ⚠️ No Sentinel-2 data available for road analysis")
                return {}
            
            if detailed_output:
                print(f"   🛰️ Using {s2.size().getInfo()} Sentinel-2 images")
            
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
            
            # Road surface quality classification
            if road_quality_deficit < 0.3:
                road_surface_quality = "Good"
            elif road_quality_deficit < 0.7:
                road_surface_quality = "Moderate"
            else:
                road_surface_quality = "Poor"
            
            if detailed_output:
                print(f"      Paved road index: {paved_ratio:.3f}")
                print(f"      Unpaved road index: {unpaved_ratio:.3f}")
                print(f"      Road quality deficit: {road_quality_deficit:.3f}")
                print(f"      Road infrastructure score: {road_infrastructure_score:.3f}")
                print(f"      Road surface quality: {road_surface_quality}")
                
                if road_quality_deficit > 0.7:
                    print("      ⚠️ Poor road quality - predominantly unpaved surfaces")
                elif road_quality_deficit > 0.3:
                    print("      🟡 Moderate road quality - mixed surface types")
                else:
                    print("      ✅ Good road quality - predominantly paved surfaces")
            
            results = {
                'paved_road_index': paved_ratio,
                'unpaved_road_index': unpaved_ratio,
                'road_quality_deficit': road_quality_deficit,
                'road_infrastructure_score': road_infrastructure_score,
                'road_surface_quality': road_surface_quality
            }
            
            if detailed_output:
                print(f"   ✅ Road quality analysis complete")
            
            return results
            
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
    
    # ========================================
    # FLEXIBLE SELECTIVE ANALYSIS METHODS
    # ========================================
    
    def analyze_selective_measures(self, lat, lon, radius_km, measures=None, start_date='2023-01-01', end_date='2023-12-31', detailed_output=False):
        """
        Flexible analysis - select specific measures to analyze at any lat/long
        
        Args:
            lat, lon, radius_km: Location parameters
            measures: List of measures to analyze. Options:
                     'ntl' or 'nighttime_lights' - Nighttime lights poverty indicators
                     'housing' - Housing quality indicators
                     'infrastructure' - Water access and terrain
                     'environmental' - Environmental conditions
                     'population' - Population patterns
                     'access' - Hospital/airport/market access
                     'roads' - Road quality
                     'mpi' - Calculate MPI from available measures
                     If None, analyzes all measures (same as comprehensive analysis)
            start_date, end_date: Date range for analysis
            detailed_output: If True, shows detailed step-by-step analysis
            
        Returns:
            Dictionary with results for selected measures only
        """
        
        print(f"\n🎯 SELECTIVE POVERTY ANALYSIS")
        print(f"📍 Location: ({lat:.4f}, {lon:.4f}) | Radius: {radius_km} km")
        
        # Default to all measures if none specified
        if measures is None:
            measures = ['ntl', 'housing', 'infrastructure', 'environmental', 'population', 'access', 'roads', 'mpi']
            print(f"🔍 Analyzing: ALL MEASURES (comprehensive analysis)")
        else:
            # Ensure measures is a list
            if isinstance(measures, str):
                measures = [measures]
            print(f"🔍 Analyzing: {', '.join(measures).upper()}")
        
        print("="*70)
        
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
            'selected_measures': measures,
            'data_sources': []
        }
        
        # Normalize measure names for flexible input
        measure_mapping = {
            'ntl': 'nighttime_lights',
            'nighttime_lights': 'nighttime_lights',
            'economic': 'nighttime_lights',
            'housing': 'housing',
            'housing_quality': 'housing',
            'infrastructure': 'infrastructure',
            'infra': 'infrastructure',
            'water': 'infrastructure',
            'environmental': 'environmental',
            'environment': 'environmental',
            'env': 'environmental',
            'vegetation': 'environmental',
            'population': 'population',
            'pop': 'population',
            'demographic': 'population',
            'access': 'access',
            'accessibility': 'access',
            'services': 'access',
            'roads': 'roads',
            'road_quality': 'roads',
            'transportation': 'roads',
            'mpi': 'mpi',
            'index': 'mpi',
            'poverty_index': 'mpi'
        }
        
        # Normalize measure names
        normalized_measures = []
        for measure in measures:
            normalized = measure_mapping.get(measure.lower(), measure.lower())
            if normalized not in normalized_measures:
                normalized_measures.append(normalized)
        
        # Analysis methods mapping
        analysis_methods = {
            'nighttime_lights': lambda: self.get_poverty_from_nighttime_lights(geometry, start_date, end_date, detailed_output),
            'housing': lambda: self.get_housing_quality_indicators(geometry, start_date, end_date, detailed_output),
            'infrastructure': lambda: self.get_infrastructure_poverty_indicators(geometry, detailed_output),
            'environmental': lambda: self.get_environmental_poverty_indicators(geometry, start_date, end_date, detailed_output),
            'population': lambda: self.get_population_poverty_indicators(geometry, 2020, detailed_output),
            'access': lambda: self.get_enhanced_access_indicators(geometry, detailed_output),
            'roads': lambda: self.get_road_quality_indicators(geometry, start_date, end_date, detailed_output)
        }
        
        # Run selected analyses
        for measure in normalized_measures:
            if measure == 'mpi':
                continue  # Handle MPI separately after other measures
                
            if measure in analysis_methods:
                try:
                    print(f"\n🔍 Analyzing {measure.replace('_', ' ').title()}...")
                    measure_results = analysis_methods[measure]()
                    results.update(measure_results)
                    results['data_sources'].append(f"{measure.title()}_Analysis")
                    
                    if detailed_output:
                        print(f"   ✅ {measure.replace('_', ' ').title()} analysis complete")
                        
                except Exception as e:
                    print(f"   ❌ {measure.replace('_', ' ').title()} analysis failed: {e}")
            else:
                print(f"   ⚠️ Unknown measure: {measure}")
        
        # Calculate MPI if requested and we have data
        if 'mpi' in normalized_measures and len([k for k in results.keys() if k not in ['latitude', 'longitude', 'radius_km', 'analysis_date', 'start_date', 'end_date', 'selected_measures', 'data_sources']]) > 0:
            try:
                print(f"\n📊 Calculating poverty indices...")
                mpi_results = self.calculate_enhanced_mpi(results)
                results.update(mpi_results)
                results['data_sources'].append('Enhanced_MPI')
                
                if detailed_output:
                    print(f"   ✅ MPI calculation complete")
                    
            except Exception as e:
                print(f"   ❌ MPI calculation failed: {e}")
        
        # Summary
        print(f"\n✅ SELECTIVE ANALYSIS COMPLETE!")
        print(f"📊 Measures analyzed: {len(normalized_measures)}")
        print(f"📈 Data sources used: {len(results['data_sources'])}")
        
        # Quick summary for selected measures
        self._print_selective_summary(results, normalized_measures)
        
        return results
    
    def _print_selective_summary(self, results, measures):
        """Print a quick summary of selected measures"""
        print(f"\n📋 QUICK SUMMARY:")
        
        if 'nighttime_lights' in measures and 'electrification_deficit' in results:
            electrification = 1 - results.get('electrification_deficit', 0)
            print(f"   🌙 Economic Activity (NTL): {'Good' if electrification > 0.7 else 'Moderate' if electrification > 0.4 else 'Poor'}")
            
        if 'housing' in measures and 'poor_roof_materials_index' in results:
            housing_quality = 1 - results.get('poor_roof_materials_index', 0)
            print(f"   🏠 Housing Quality: {'Good' if housing_quality > 0.7 else 'Moderate' if housing_quality > 0.4 else 'Poor'}")
            
        if 'infrastructure' in measures and 'water_access_deficit' in results:
            water_access = 1 - results.get('water_access_deficit', 0)
            print(f"   💧 Water Access: {'Good' if water_access > 0.7 else 'Moderate' if water_access > 0.4 else 'Poor'}")
            
        if 'environmental' in measures and 'mean_ndvi' in results:
            env_health = results.get('mean_ndvi', 0)
            print(f"   🌱 Environmental Health: {'Good' if env_health > 0.4 else 'Moderate' if env_health > 0.2 else 'Poor'}")
            
        if 'population' in measures and 'population_density_per_km2' in results:
            density = results.get('population_density_per_km2', 0)
            print(f"   👥 Population Density: {density:.0f} people/km² ({'High' if density > 5000 else 'Moderate' if density > 1000 else 'Low'})")
            
        if 'access' in measures and 'hospital_access_deficit' in results:
            health_access = 1 - results.get('hospital_access_deficit', 0)
            print(f"   🏥 Healthcare Access: {'Good' if health_access > 0.7 else 'Moderate' if health_access > 0.4 else 'Poor'}")
            
        if 'roads' in measures and 'road_infrastructure_score' in results:
            road_quality = results.get('road_infrastructure_score', 0)
            print(f"   🛣️ Road Quality: {'Good' if road_quality > 0.7 else 'Moderate' if road_quality > 0.4 else 'Poor'}")
            
        if 'mpi' in measures and 'enhanced_mpi' in results:
            mpi = results.get('enhanced_mpi', results.get('multidimensional_poverty_index', 0))
            poverty_level = results.get('enhanced_poverty_level', results.get('poverty_level_classification', 'Unknown'))
            print(f"   📊 Overall Poverty Level: {poverty_level} (MPI: {mpi:.3f})")
    
    def analyze_ntl_only(self, lat, lon, radius_km, start_date='2023-01-01', end_date='2023-12-31', detailed=False):
        """Quick method to analyze only nighttime lights at any location"""
        return self.analyze_selective_measures(lat, lon, radius_km, measures=['ntl'], 
                                             start_date=start_date, end_date=end_date, 
                                             detailed_output=detailed)
    
    def analyze_housing_only(self, lat, lon, radius_km, start_date='2023-01-01', end_date='2023-12-31', detailed=False):
        """Quick method to analyze only housing quality at any location"""
        return self.analyze_selective_measures(lat, lon, radius_km, measures=['housing'], 
                                             start_date=start_date, end_date=end_date, 
                                             detailed_output=detailed)
    
    def analyze_access_only(self, lat, lon, radius_km, detailed=False):
        """Quick method to analyze only access indicators at any location"""
        return self.analyze_selective_measures(lat, lon, radius_km, measures=['access'], 
                                             detailed_output=detailed)
    
    def analyze_custom_combination(self, lat, lon, radius_km, measures, start_date='2023-01-01', end_date='2023-12-31', detailed=False):
        """Analyze any custom combination of measures at any location"""
        return self.analyze_selective_measures(lat, lon, radius_km, measures=measures, 
                                             start_date=start_date, end_date=end_date, 
                                             detailed_output=detailed)
    
    # ========================================
    # INDIVIDUAL MEASURE INSPECTION METHODS (PRESERVED)
    # ========================================
    
    def inspect_nighttime_lights_only(self, lat, lon, radius_km, start_date='2023-01-01', end_date='2023-12-31'):
        """Analyze only nighttime lights poverty indicators with detailed output"""
        print(f"\n🌙 NIGHTTIME LIGHTS POVERTY INSPECTION")
        print(f"📍 Location: ({lat:.4f}, {lon:.4f}) | Radius: {radius_km} km")
        print("="*70)
        
        geometry = self.create_buffer_zone(lat, lon, radius_km)
        results = self.get_poverty_from_nighttime_lights(geometry, start_date, end_date, detailed_output=True)
        
        print(f"\n📋 NIGHTTIME LIGHTS SUMMARY:")
        print(f"   Economic Isolation: {'High' if results.get('economic_isolation_index', 0) > 0.6 else 'Moderate' if results.get('economic_isolation_index', 0) > 0.3 else 'Low'}")
        print(f"   Electrification Level: {'Poor' if results.get('electrification_deficit', 0) > 0.6 else 'Moderate' if results.get('electrification_deficit', 0) > 0.3 else 'Good'}")
        
        return results
    
    def inspect_housing_quality_only(self, lat, lon, radius_km, start_date='2023-01-01', end_date='2023-12-31'):
        """Analyze only housing quality indicators with detailed output"""
        print(f"\n🏠 HOUSING QUALITY INSPECTION")
        print(f"📍 Location: ({lat:.4f}, {lon:.4f}) | Radius: {radius_km} km")
        print("="*70)
        
        geometry = self.create_buffer_zone(lat, lon, radius_km)
        results = self.get_housing_quality_indicators(geometry, start_date, end_date, detailed_output=True)
        
        print(f"\n📋 HOUSING QUALITY SUMMARY:")
        if 'poor_roof_materials_index' in results:
            housing_score = 1 - results.get('poor_roof_materials_index', 0)
            print(f"   Housing Quality: {'Good' if housing_score > 0.7 else 'Moderate' if housing_score > 0.4 else 'Poor'}")
        if 'informal_settlement_index' in results:
            settlement_quality = 1 - results.get('informal_settlement_index', 0)
            print(f"   Settlement Quality: {'Formal' if settlement_quality > 0.7 else 'Mixed' if settlement_quality > 0.4 else 'Informal'}")
        
        return results
    
    def inspect_infrastructure_only(self, lat, lon, radius_km):
        """Analyze only infrastructure indicators with detailed output"""
        print(f"\n🛣️ INFRASTRUCTURE INSPECTION")
        print(f"📍 Location: ({lat:.4f}, {lon:.4f}) | Radius: {radius_km} km")
        print("="*70)
        
        geometry = self.create_buffer_zone(lat, lon, radius_km)
        results = self.get_infrastructure_poverty_indicators(geometry, detailed_output=True)
        
        print(f"\n📋 INFRASTRUCTURE SUMMARY:")
        if 'water_access_deficit' in results:
            water_quality = 1 - results.get('water_access_deficit', 0)
            print(f"   Water Access: {'Good' if water_quality > 0.7 else 'Moderate' if water_quality > 0.4 else 'Poor'}")
        if 'terrain_isolation_index' in results:
            terrain_access = 1 - results.get('terrain_isolation_index', 0)
            print(f"   Terrain Accessibility: {'Easy' if terrain_access > 0.7 else 'Moderate' if terrain_access > 0.4 else 'Difficult'}")
        
        return results
    
    def inspect_environmental_only(self, lat, lon, radius_km, start_date='2023-01-01', end_date='2023-12-31'):
        """Analyze only environmental indicators with detailed output"""
        print(f"\n🌱 ENVIRONMENTAL INSPECTION")
        print(f"📍 Location: ({lat:.4f}, {lon:.4f}) | Radius: {radius_km} km")
        print("="*70)
        
        geometry = self.create_buffer_zone(lat, lon, radius_km)
        results = self.get_environmental_poverty_indicators(geometry, start_date, end_date, detailed_output=True)
        
        print(f"\n📋 ENVIRONMENTAL SUMMARY:")
        if 'mean_ndvi' in results:
            veg_health = results.get('mean_ndvi', 0)
            print(f"   Vegetation Health: {'Good' if veg_health > 0.4 else 'Moderate' if veg_health > 0.2 else 'Poor'}")
        if 'air_pollution_burden' in results:
            air_quality = 1 - results.get('air_pollution_burden', 0)
            print(f"   Air Quality: {'Good' if air_quality > 0.7 else 'Moderate' if air_quality > 0.4 else 'Poor'}")
        
        return results
    
    def inspect_population_only(self, lat, lon, radius_km, year=2020):
        """Analyze only population indicators with detailed output"""
        print(f"\n👥 POPULATION INSPECTION")
        print(f"📍 Location: ({lat:.4f}, {lon:.4f}) | Radius: {radius_km} km")
        print("="*70)
        
        geometry = self.create_buffer_zone(lat, lon, radius_km)
        results = self.get_population_poverty_indicators(geometry, year, detailed_output=True)
        
        print(f"\n📋 POPULATION SUMMARY:")
        if 'population_density_per_km2' in results:
            density = results.get('population_density_per_km2', 0)
            print(f"   Population Density: {'High' if density > 5000 else 'Moderate' if density > 1000 else 'Low'}")
        if 'overcrowding_index' in results:
            crowding = results.get('overcrowding_index', 0)
            print(f"   Overcrowding Level: {'High' if crowding > 0.6 else 'Moderate' if crowding > 0.3 else 'Low'}")
        
        return results
    
    def inspect_access_only(self, lat, lon, radius_km):
        """Analyze only access indicators with detailed output"""
        print(f"\n🏥 ACCESS INSPECTION")
        print(f"📍 Location: ({lat:.4f}, {lon:.4f}) | Radius: {radius_km} km")
        print("="*70)
        
        geometry = self.create_buffer_zone(lat, lon, radius_km)
        results = self.get_enhanced_access_indicators(geometry, detailed_output=True)
        
        print(f"\n📋 ACCESS SUMMARY:")
        if 'hospital_access_deficit' in results:
            health_access = 1 - results.get('hospital_access_deficit', 0)
            print(f"   Healthcare Access: {'Good' if health_access > 0.7 else 'Moderate' if health_access > 0.4 else 'Poor'}")
        if 'airport_access_deficit' in results:
            transport_access = 1 - results.get('airport_access_deficit', 0)
            print(f"   Transportation Access: {'Good' if transport_access > 0.7 else 'Moderate' if transport_access > 0.4 else 'Poor'}")
        if 'market_access_score' in results:
            market_access = results.get('market_access_score', 0)
            print(f"   Market Access: {'Good' if market_access > 0.7 else 'Moderate' if market_access > 0.4 else 'Poor'}")
        
        return results
    
    def inspect_road_quality_only(self, lat, lon, radius_km, start_date='2023-01-01', end_date='2023-12-31'):
        """Analyze only road quality indicators with detailed output"""
        print(f"\n🛣️ ROAD QUALITY INSPECTION")
        print(f"📍 Location: ({lat:.4f}, {lon:.4f}) | Radius: {radius_km} km")
        print("="*70)
        
        geometry = self.create_buffer_zone(lat, lon, radius_km)
        results = self.get_road_quality_indicators(geometry, start_date, end_date, detailed_output=True)
        
        print(f"\n📋 ROAD QUALITY SUMMARY:")
        if 'road_infrastructure_score' in results:
            road_score = results.get('road_infrastructure_score', 0)
            print(f"   Road Infrastructure: {'Good' if road_score > 0.7 else 'Moderate' if road_score > 0.4 else 'Poor'}")
        if 'road_surface_quality' in results:
            print(f"   Surface Quality: {results.get('road_surface_quality', 'Unknown')}")
        
        return results
    
    def inspect_all_measures_detailed(self, lat, lon, radius_km, start_date='2023-01-01', end_date='2023-12-31'):
        """Run detailed inspection of all individual measures"""
        print(f"\n🔍 COMPREHENSIVE DETAILED INSPECTION")
        print(f"📍 Location: ({lat:.4f}, {lon:.4f}) | Radius: {radius_km} km")
        print("="*80)
        
        all_results = {}
        
        # Run each individual inspection
        all_results.update(self.inspect_nighttime_lights_only(lat, lon, radius_km, start_date, end_date))
        all_results.update(self.inspect_housing_quality_only(lat, lon, radius_km, start_date, end_date))
        all_results.update(self.inspect_infrastructure_only(lat, lon, radius_km))
        all_results.update(self.inspect_environmental_only(lat, lon, radius_km, start_date, end_date))
        all_results.update(self.inspect_population_only(lat, lon, radius_km))
        all_results.update(self.inspect_access_only(lat, lon, radius_km))
        all_results.update(self.inspect_road_quality_only(lat, lon, radius_km, start_date, end_date))
        
        # Calculate MPI
        mpi_results = self.calculate_enhanced_mpi(all_results)
        all_results.update(mpi_results)
        
        print(f"\n🎯 FINAL COMPREHENSIVE ASSESSMENT:")
        print(f"   Enhanced MPI: {all_results.get('enhanced_mpi', 0):.3f}")
        print(f"   Poverty Level: {all_results.get('enhanced_poverty_level', 'Unknown')}")
        
        return all_results
    
    # ========================================
    # ORIGINAL COMPREHENSIVE ANALYSIS METHOD (UNCHANGED)
    # ========================================
    
    def analyze_poverty_conditions(self, lat, lon, radius_km, start_date='2023-01-01', end_date='2023-12-31'):
        """
        Comprehensive poverty analysis using satellite data with enhanced indicators
        (Original method - unchanged to maintain flow integrity)
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

# ========================================
# ENHANCED UTILITY FUNCTIONS WITH SELECTIVE ANALYSIS
# ========================================

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

def compare_selective_measures(analyzer, locations, measures):
    """Compare specific measures across multiple locations"""
    print(f"🔍 SELECTIVE MEASURES COMPARISON: {', '.join(measures).upper()}")
    print("="*80)
    
    all_results = []
    
    for location in locations:
        print(f"\n📍 Analyzing {location['name']} - Selected Measures Only")
        
        # Extract location parameters
        params = {k: v for k, v in location.items() if k != 'name'}
        
        # Run selective analysis
        results = analyzer.analyze_selective_measures(measures=measures, **params)
        results['location_name'] = location['name']
        all_results.append(results)
    
    # Create comparison table
    print(f"\n📊 SELECTIVE COMPARISON SUMMARY:")
    print(f"{'Location':<20}", end='')
    
    # Dynamic headers based on measures
    if 'ntl' in measures or 'nighttime_lights' in measures:
        print(f"{'NTL Score':<12}", end='')
    if 'housing' in measures:
        print(f"{'Housing':<12}", end='')
    if 'infrastructure' in measures:
        print(f"{'Water':<12}", end='')
    if 'environmental' in measures:
        print(f"{'Environment':<12}", end='')
    if 'population' in measures:
        print(f"{'Pop Density':<12}", end='')
    if 'access' in measures:
        print(f"{'Access':<12}", end='')
    if 'roads' in measures:
        print(f"{'Roads':<12}", end='')
    if 'mpi' in measures:
        print(f"{'MPI':<12}", end='')
    
    print()  # New line
    print("-" * (20 + 12 * len([m for m in measures if m in ['ntl', 'nighttime_lights', 'housing', 'infrastructure', 'environmental', 'population', 'access', 'roads', 'mpi']])))
    
    # Print data rows
    for result in all_results:
        print(f"{result['location_name']:<20}", end='')
        
        if 'ntl' in measures or 'nighttime_lights' in measures:
            ntl_score = 1 - result.get('electrification_deficit', 0.5)
            print(f"{ntl_score:<12.3f}", end='')
        if 'housing' in measures:
            housing_score = 1 - result.get('poor_roof_materials_index', 0.5)
            print(f"{housing_score:<12.3f}", end='')
        if 'infrastructure' in measures:
            water_score = 1 - result.get('water_access_deficit', 0.5)
            print(f"{water_score:<12.3f}", end='')
        if 'environmental' in measures:
            env_score = result.get('mean_ndvi', 0.3)
            print(f"{env_score:<12.3f}", end='')
        if 'population' in measures:
            pop_density = result.get('population_density_per_km2', 0)
            print(f"{pop_density:<12.0f}", end='')
        if 'access' in measures:
            access_score = result.get('market_access_score', 0.5)
            print(f"{access_score:<12.3f}", end='')
        if 'roads' in measures:
            road_score = result.get('road_infrastructure_score', 0.5)
            print(f"{road_score:<12.3f}", end='')
        if 'mpi' in measures:
            mpi_score = result.get('enhanced_mpi', result.get('multidimensional_poverty_index', 0))
            print(f"{mpi_score:<12.3f}", end='')
        
        print()  # New line
    
    return all_results

def inspect_individual_measures_comparison(analyzer, locations):
    """Compare individual poverty measures across multiple locations"""
    print("🔍 INDIVIDUAL MEASURES COMPARISON ACROSS LOCATIONS")
    print("="*80)
    
    all_individual_results = {}
    
    for location in locations:
        location_name = location['name']
        print(f"\n📍 Individual Inspection: {location_name}")
        print("-" * 50)
        
        # Extract location parameters
        params = {k: v for k, v in location.items() if k != 'name'}
        
        # Run individual inspections
        individual_results = {
            'nighttime_lights': analyzer.inspect_nighttime_lights_only(**params),
            'housing_quality': analyzer.inspect_housing_quality_only(**params),
            'infrastructure': analyzer.inspect_infrastructure_only(params['lat'], params['lon'], params['radius_km']),
            'environmental': analyzer.inspect_environmental_only(**params),
            'population': analyzer.inspect_population_only(params['lat'], params['lon'], params['radius_km']),
            'access': analyzer.inspect_access_only(params['lat'], params['lon'], params['radius_km']),
            'road_quality': analyzer.inspect_road_quality_only(**params)
        }
        
        all_individual_results[location_name] = individual_results
    
    # Summary comparison table
    print(f"\n📊 INDIVIDUAL MEASURES COMPARISON TABLE:")
    print("-" * 120)
    header = f"{'Location':<20} {'NTL Score':<12} {'Housing':<12} {'Water':<12} {'Environment':<12} {'Population':<12} {'Access':<12} {'Roads':<12}"
    print(header)
    print("-" * 120)
    
    for location_name, measures in all_individual_results.items():
        # Extract key scores for comparison
        ntl_score = 1 - measures['nighttime_lights'].get('electrification_deficit', 0.5)
        housing_score = 1 - measures['housing_quality'].get('poor_roof_materials_index', 0.5)
        water_score = 1 - measures['infrastructure'].get('water_access_deficit', 0.5)
        env_score = measures['environmental'].get('mean_ndvi', 0.3)
        pop_score = min(1.0, measures['population'].get('population_density_per_km2', 1000) / 3000)
        access_score = measures['access'].get('market_access_score', 0.5)
        road_score = measures['road_quality'].get('road_infrastructure_score', 0.5)
        
        print(f"{location_name:<20} {ntl_score:<12.3f} {housing_score:<12.3f} {water_score:<12.3f} {env_score:<12.3f} {pop_score:<12.3f} {access_score:<12.3f} {road_score:<12.3f}")
    
    return all_individual_results

def quick_ntl_comparison(analyzer, locations):
    """Quick comparison of only nighttime lights across locations"""
    print("🌙 QUICK NIGHTTIME LIGHTS COMPARISON")
    print("="*50)
    
    results = []
    for location in locations:
        params = {k: v for k, v in location.items() if k != 'name'}
        ntl_result = analyzer.analyze_ntl_only(**params)
        ntl_result['location_name'] = location['name']
        results.append(ntl_result)
    
    # Simple comparison table
    print(f"\n{'Location':<25} {'Electrification':<15} {'Economic Activity':<18} {'Poverty Risk'}")
    print("-" * 75)
    
    for result in results:
        electrification = 1 - result.get('electrification_deficit', 0)
        economic_activity = 1 - result.get('economic_isolation_index', 0)
        poverty_risk = "Low" if electrification > 0.7 else "Moderate" if electrification > 0.4 else "High"
        
        print(f"{result['location_name']:<25} {electrification:<15.3f} {economic_activity:<18.3f} {poverty_risk}")
    
    return results

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
    
    return dfscore', 'living_standards_score', 'environmental_deprivation_score',
        
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

# ========================================
# TESTING FUNCTIONS WITH SELECTIVE ANALYSIS
# ========================================

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
    
    # Test individual measures comparison
    individual_comparison = inspect_individual_measures_comparison(analyzer, test_locations[:3])  # First 3 for demo
    
    # Create enhanced research dataset
    dataset = create_enhanced_poverty_dataset(results)
    
    return results, individual_comparison, dataset

def demo_selective_analysis():
    """Demonstrate selective analysis capabilities"""
    print("🎯 DEMONSTRATION: SELECTIVE POVERTY ANALYSIS")
    print("="*70)
    
    analyzer = EnhancedSatellitePovertyAnalyzer()
    
    # Test location
    lat, lon, radius = 23.75263611, 87.21012500, 4  # Bardhaman
    
    print(f"\n🎯 SELECTIVE ANALYSIS DEMO")
    print(f"📍 Test Location: Bardhaman ({lat:.4f}, {lon:.4f})")
    
    # Demo different selective analyses
    print(f"\n" + "="*60)
    print("🌙 1. NIGHTTIME LIGHTS ONLY")
    ntl_only = analyzer.analyze_ntl_only(lat, lon, radius, detailed=True)
    
    print(f"\n" + "="*60)
    print("🏠 2. HOUSING QUALITY ONLY")
    housing_only = analyzer.analyze_housing_only(lat, lon, radius, detailed=True)
    
    print(f"\n" + "="*60)
    print("🏥 3. ACCESS INDICATORS ONLY")
    access_only = analyzer.analyze_access_only(lat, lon, radius, detailed=True)
    
    print(f"\n" + "="*60)
    print("🎯 4. CUSTOM COMBINATION: NTL + HOUSING + ACCESS")
    custom_combo = analyzer.analyze_custom_combination(lat, lon, radius, 
                                                      measures=['ntl', 'housing', 'access', 'mpi'], 
                                                      detailed=True)
    
    print(f"\n" + "="*60)
    print("🔍 5. SELECTIVE MEASURES WITH FLEXIBLE INPUT")
    # Test flexible input formats
    flexible_analysis = analyzer.analyze_selective_measures(lat, lon, radius, 
                                                           measures=['economic', 'env', 'roads'], 
                                                           detailed_output=True)
    
    return {
        'ntl_only': ntl_only,
        'housing_only': housing_only,
        'access_only': access_only,
        'custom_combo': custom_combo,
        'flexible_analysis': flexible_analysis
    }

def demo_individual_inspections():
    """Demonstrate individual measure inspection capabilities"""
    print("🎯 DEMONSTRATION: INDIVIDUAL MEASURE INSPECTIONS")
    print("="*70)
    
    analyzer = EnhancedSatellitePovertyAnalyzer()
    
    # Test location
    lat, lon, radius = 23.75263611, 87.21012500, 4  # Bardhaman
    
    print(f"\n🎯 INDIVIDUAL MEASURE INSPECTION DEMO")
    print(f"📍 Test Location: Bardhaman ({lat:.4f}, {lon:.4f})")
    
    # Demo each individual inspection
    print(f"\n" + "="*50)
    print("🔍 1. NIGHTTIME LIGHTS ONLY")
    ntl_results = analyzer.inspect_nighttime_lights_only(lat, lon, radius)
    
    print(f"\n" + "="*50)
    print("🔍 2. HOUSING QUALITY ONLY") 
    housing_results = analyzer.inspect_housing_quality_only(lat, lon, radius)
    
    print(f"\n" + "="*50)
    print("🔍 3. INFRASTRUCTURE ONLY")
    infra_results = analyzer.inspect_infrastructure_only(lat, lon, radius)
    
    print(f"\n" + "="*50)
    print("🔍 4. ENVIRONMENTAL ONLY")
    env_results = analyzer.inspect_environmental_only(lat, lon, radius)
    
    print(f"\n" + "="*50)
    print("🔍 5. POPULATION ONLY")
    pop_results = analyzer.inspect_population_only(lat, lon, radius)
    
    print(f"\n" + "="*50)
    print("🔍 6. ACCESS ONLY")
    access_results = analyzer.inspect_access_only(lat, lon, radius)
    
    print(f"\n" + "="*50)
    print("🔍 7. ROAD QUALITY ONLY")
    road_results = analyzer.inspect_road_quality_only(lat, lon, radius)
    
    print(f"\n" + "="*50)
    print("🔍 8. ALL MEASURES WITH DETAILED OUTPUT")
    detailed_results = analyzer.inspect_all_measures_detailed(lat, lon, radius)
    
    return {
        'nighttime_lights': ntl_results,
        'housing': housing_results, 
        'infrastructure': infra_results,
        'environmental': env_results,
        'population': pop_results,
        'access': access_results,
        'road_quality': road_results,
        'comprehensive_detailed': detailed_results
    }

def demo_comparison_methods():
    """Demonstrate different comparison methods"""
    print("📊 DEMONSTRATION: COMPARISON METHODS")
    print("="*70)
    
    analyzer = EnhancedSatellitePovertyAnalyzer()
    
    # Test locations
    test_locations = [
        {"name": "Bardhaman", "lat": 23.75263611, "lon": 87.21012500, "radius_km": 4},
        {"name": "Darjeeling", "lat": 27.04170278, "lon": 88.26640556, "radius_km": 4},
        {"name": "Kolkata Central", "lat": 22.5726, "lon": 88.3639, "radius_km": 3}
    ]
    
    print(f"\n🌙 1. QUICK NTL COMPARISON")
    ntl_comparison = quick_ntl_comparison(analyzer, test_locations)
    
    print(f"\n🎯 2. SELECTIVE MEASURES COMPARISON")
    selective_comparison = compare_selective_measures(analyzer, test_locations, 
                                                    measures=['ntl', 'housing', 'access'])
    
    print(f"\n🔍 3. INDIVIDUAL MEASURES COMPARISON")
    individual_comparison = inspect_individual_measures_comparison(analyzer, test_locations)
    
    print(f"\n📊 4. COMPREHENSIVE COMPARISON")
    comprehensive_comparison = compare_poverty_levels(analyzer, test_locations)
    
    return {
        'ntl_comparison': ntl_comparison,
        'selective_comparison': selective_comparison,
        'individual_comparison': individual_comparison,
        'comprehensive_comparison': comprehensive_comparison
    }

# ========================================
# MAIN EXECUTION WITH SELECTIVE ANALYSIS OPTIONS
# ========================================

if __name__ == "__main__":
    print("🚀 ENHANCED SATELLITE-BASED POVERTY ANALYSIS SYSTEM")
    print("="*70)
    print("🌟 NOW WITH SELECTIVE ANALYSIS CAPABILITIES!")
    print("🎯 Analyze ANY specific measure at ANY lat/long")
    print("🔍 Flexible input: 'ntl', 'housing', 'access', 'environmental', etc.")
    print("📊 Quick methods: analyze_ntl_only(), analyze_housing_only(), etc.")
    
    # Initialize enhanced analyzer
    analyzer = EnhancedSatellitePovertyAnalyzer()
    
    # Demo selective analysis capabilities
    print(f"\n🎯 RUNNING SELECTIVE ANALYSIS DEMO...")
    selective_demo = demo_selective_analysis()
    
    # Demo individual inspections
    print(f"\n🔍 RUNNING INDIVIDUAL INSPECTION DEMO...")
    individual_demo = demo_individual_inspections()
    
    # Demo comparison methods
    print(f"\n📊 RUNNING COMPARISON METHODS DEMO...")
    comparison_demo = demo_comparison_methods()
    
    # Define test locations for final comprehensive analysis
    test_locations = [
        {"name": "Bardhaman", "lat": 23.75263611, "lon": 87.21012500, "radius_km": 4},
        {"name": "Darjeeling", "lat": 27.04170278, "lon": 88.26640556, "radius_km": 4}
    ]
    
    # Run enhanced comparative analysis
    print(f"\n🔍 Running enhanced poverty analysis with access indicators...")
    results = compare_poverty_levels(analyzer, test_locations)
    
    # Save results with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"enhanced_poverty_analysis_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Create enhanced dataset
    dataset = create_enhanced_poverty_dataset(results, f"enhanced_poverty_dataset_{timestamp}.csv")
    
    print(f"\n🎯 ANALYSIS COMPLETE!")
    print(f"📊 Enhanced analysis with {len(dataset)} locations and {len(dataset.columns)} indicators.")
    print(f"🔍 Individual inspection capabilities demonstrated.")
    print(f"🎯 Selective analysis capabilities demonstrated.")
    print(f"✨ You can now analyze ANY specific measure at ANY location!")
    
    # Usage examples
    print(f"\n💡 USAGE EXAMPLES:")
    print(f"   # Quick NTL analysis at any location:")
    print(f"   analyzer.analyze_ntl_only(lat, lon, radius)")
    print(f"   ")
    print(f"   # Housing quality only with details:")
    print(f"   analyzer.analyze_housing_only(lat, lon, radius, detailed=True)")
    print(f"   ")
    print(f"   # Custom combination of measures:")
    print(f"   analyzer.analyze_custom_combination(lat, lon, radius, ['ntl', 'access', 'mpi'])")
    print(f"   ")
    print(f"   # Flexible selective analysis:")
    print(f"   analyzer.analyze_selective_measures(lat, lon, radius, measures=['economic', 'env'])")
    print(f"   ")
    print(f"   # Individual inspection with details:")
    print(f"   analyzer.inspect_nighttime_lights_only(lat, lon, radius)")
    print(f"   ")
    print(f"   # Original comprehensive analysis (unchanged):")
    print(f"   analyzer.analyze_poverty_conditions(lat, lon, radius)")
    print(f"   ")
    print(f"   # Quick comparisons:")
    print(f"   quick_ntl_comparison(analyzer, locations)")
    print(f"   compare_selective_measures(analyzer, locations, ['ntl', 'housing'])")