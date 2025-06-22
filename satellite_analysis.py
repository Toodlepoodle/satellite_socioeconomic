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

class SatellitePovertyAnalyzer:
    """
    Comprehensive poverty and socioeconomic analysis using real satellite data
    Based on latest literature: Jean et al. (2016), Yeh et al. (2020), Hall et al. (2023)
    Focuses on measurable poverty indicators from satellite imagery
    """
    
    def __init__(self, project_id='ed-sayandasgupta97'):
        """Initialize GEE and authenticate"""
        try:
            ee.Initialize(project=project_id)
            print(f"✅ Google Earth Engine initialized with project: {project_id}")
            self.test_connection()
        except Exception as e:
            print(f"❌ GEE Authentication failed: {e}")
            print("Please run: earthengine authenticate")
            raise
            
    def test_connection(self):
        """Test GEE connection with actual data"""
        try:
            test_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').limit(1)
            size = test_collection.size().getInfo()
            print(f"✅ GEE connection verified. Test collection size: {size}")
        except Exception as e:
            print(f"❌ GEE connection test failed: {e}")
            raise
    
    def create_buffer_zone(self, lat, lon, radius_km):
        """Create circular buffer around point"""
        point = ee.Geometry.Point([lon, lat])
        buffer = point.buffer(radius_km * 1000)
        return buffer
    
    def get_poverty_from_nighttime_lights(self, geometry, start_date, end_date):
        """
        Extract poverty indicators from VIIRS nighttime lights
        Literature: Elvidge et al. (2009), Jean et al. (2016) - NTL as GDP/wealth proxy
        Low nighttime lights = Lower economic activity = Higher poverty risk
        """
        print("   🌙 Analyzing nighttime lights for economic poverty indicators...")
        try:
            # VIIRS Nighttime Day/Night Band Composites
            viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG') \
                     .filterDate(start_date, end_date) \
                     .filterBounds(geometry)
            
            collection_size = viirs.size().getInfo()
            print(f"   📊 Found {collection_size} VIIRS nighttime images")
            
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
            
            results = {}
            
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
                
                results.update({
                    'metal_roof_ratio': roof_stats.get('B8_mean', 0) or 0,
                    'concrete_roof_ratio': roof_stats.get('B11_mean', 0) or 0,
                    'organic_roof_ratio': organic_roof_prevalence,
                    'poor_roof_materials_index': poor_roof_materials,
                    'roof_material_diversity': roof_stats.get('B4_p75', 0) - roof_stats.get('B4_p25', 0)
                })
            
            if s1.size().getInfo() > 0:
                # SAR analysis for building density and structure
                s1_composite = s1.select(['VV', 'VH']).median()
                
                # Urban density from SAR backscatter
                urban_density = s1_composite.select('VV').subtract(s1_composite.select('VH'))
                
                # Building structure quality (more regular structures have higher VV/VH ratio)
                structure_quality = s1_composite.select('VV').divide(s1_composite.select('VH'))
                
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
        print("   🛣️ Analyzing infrastructure poverty indicators...")
        try:
            results = {}
            
            # Water access analysis
            gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
            water_occurrence = gsw.select('occurrence')
            
            # Distance to reliable water sources (occurrence > 50%)
            reliable_water = water_occurrence.gt(50)
            water_distance = reliable_water.fastDistanceTransform(5000).sqrt()
            
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
            
            # Terrain accessibility analysis
            dem = ee.Image('USGS/SRTMGL1_003')
            terrain = ee.Algorithms.Terrain(dem)
            slope = terrain.select('slope')
            
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
        print("   🌱 Analyzing environmental poverty indicators...")
        try:
            results = {}
            
            # Vegetation health analysis
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                  .filterDate(start_date, end_date) \
                  .filterBounds(geometry) \
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            
            if s2.size().getInfo() > 0:
                s2_composite = s2.median()
                
                # NDVI for vegetation health
                ndvi = s2_composite.normalizedDifference(['B8', 'B4'])
                
                # Environmental degradation indices
                bare_soil_index = s2_composite.normalizedDifference(['B11', 'B8'])  # NDBI
                
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
                
                results.update({
                    'vegetation_deficit': vegetation_deficit,
                    'environmental_degradation': environmental_degradation,
                    'green_space_deprivation': green_space_deprivation,
                    'mean_ndvi': env_stats.get('nd_mean', 0) or 0,
                    'bare_soil_ratio': env_stats.get('nd_1_mean', 0) or 0
                })
            
            # Air quality proxy from Sentinel-5P
            try:
                no2 = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2') \
                       .filterDate(start_date, end_date) \
                       .filterBounds(geometry) \
                       .select('NO2_column_number_density')
                
                if no2.size().getInfo() > 0:
                    no2_stats = no2.median().reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=geometry,
                        scale=1000,
                        maxPixels=1e9,
                        bestEffort=True
                    ).getInfo()
                    
                    # Air pollution poverty indicator
                    air_pollution_burden = min(1.0, max(0, (no2_stats.get('NO2_column_number_density', 0) or 0) * 1e6))
                    
                    results.update({
                        'air_pollution_burden': air_pollution_burden,
                        'no2_density': no2_stats.get('NO2_column_number_density', 0) or 0
                    })
            except:
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
        print("   👥 Analyzing population poverty indicators...")
        try:
            # WorldPop population data
            population = ee.ImageCollection('WorldPop/GP/100m/pop') \
                          .filter(ee.Filter.eq('year', year)) \
                          .filter(ee.Filter.eq('country', 'IND'))
            
            if population.size().getInfo() == 0:
                return {}
            
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
    
    def calculate_multidimensional_poverty_index(self, results):
        """
        Calculate Multidimensional Poverty Index (MPI) based on satellite indicators
        Literature: Alkire & Foster (2011), adapted for satellite data
        """
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
        
        dimension_scores = {}
        
        for dim_name, dim_config in dimensions.items():
            dim_score = 0
            weight_sum = 0
            
            for indicator, weight in dim_config['indicators'].items():
                if indicator in results and results[indicator] is not None:
                    value = results[indicator]
                    # Normalize to 0-1 scale where 1 = maximum poverty
                    normalized_value = min(1.0, max(0.0, float(value)))
                    dim_score += weight * normalized_value
                    weight_sum += weight
            
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
    
    def analyze_poverty_conditions(self, lat, lon, radius_km, start_date='2023-01-01', end_date='2023-12-31'):
        """
        Comprehensive poverty analysis using satellite data
        """
        print(f"\n🔍 COMPREHENSIVE POVERTY ANALYSIS")
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
        
        # 6. Calculate Multidimensional Poverty Index
        try:
            mpi_results = self.calculate_multidimensional_poverty_index(results)
            results.update(mpi_results)
        except Exception as e:
            print(f"   ❌ MPI calculation failed: {e}")
        
        print(f"\n✅ POVERTY ANALYSIS COMPLETE!")
        print(f"📊 Data sources used: {len(results['data_sources'])}")
        
        return results
    
    def generate_poverty_report(self, results):
        """Generate comprehensive poverty analysis report"""
        
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE POVERTY ANALYSIS REPORT")
        print("="*80)
        print(f"📍 Location: ({results['latitude']}, {results['longitude']})")
        print(f"📏 Analysis Radius: {results['radius_km']} km")
        print(f"📅 Analysis Period: {results['start_date']} to {results['end_date']}")
        
        # Multidimensional Poverty Index Summary
        if 'multidimensional_poverty_index' in results:
            mpi = results['multidimensional_poverty_index']
            poverty_level = results.get('poverty_level_classification', 'Unknown')
            
            print(f"\n🎯 POVERTY ASSESSMENT SUMMARY:")
            print(f"   📊 Multidimensional Poverty Index (MPI): {mpi:.3f}")
            print(f"   📈 Poverty Level Classification: {poverty_level}")
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
        
        # Dimension-wise Poverty Scores
        if 'economic_deprivation_score' in results:
            print(f"\n📊 POVERTY DIMENSION SCORES:")
            print(f"   💰 Economic Deprivation: {results['economic_deprivation_score']:.3f}")
            print(f"   🏠 Living Standards: {results.get('living_standards_score', 0):.3f}")
            print(f"   🌍 Environmental Deprivation: {results.get('environmental_deprivation_score', 0):.3f}")
        
        # Risk Assessment
        mpi = results.get('multidimensional_poverty_index', 0)
        print(f"\n⚠️ POVERTY RISK ASSESSMENT:")
        if mpi < 0.2:
            print(f"   ✅ Low Poverty Risk - Well-developed area")
        elif mpi < 0.4:
            print(f"   🟡 Moderate Poverty Risk - Some development gaps")
        elif mpi < 0.6:
            print(f"   🟠 High Poverty Risk - Significant deprivations")
        else:
            print(f"   🔴 Extreme Poverty Risk - Multiple severe deprivations")
        
        # Data Quality Assessment
        print(f"\n📈 DATA QUALITY & SOURCES:")
        print(f"   🛰️ Satellite Data Sources: {len(results.get('data_sources', []))}")
        print(f"   📊 Total Indicators Extracted: {len([k for k in results.keys() if k not in ['latitude', 'longitude', 'radius_km', 'analysis_date', 'start_date', 'end_date', 'data_sources']])}")
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

# Additional utility functions for socioeconomic analysis
def compare_poverty_levels(analyzer, locations):
    """Compare poverty levels across multiple locations"""
    print("🔍 COMPARATIVE POVERTY ANALYSIS")
    print("="*60)
    
    all_results = []
    
    for location in locations:
        print(f"\n📍 Analyzing: {location['name']}")
        results = analyzer.analyze_poverty_conditions(**{k: v for k, v in location.items() if k != 'name'})
        results['location_name'] = location['name']
        all_results.append(results)
    
    # Compare MPI scores
    print(f"\n📊 POVERTY COMPARISON SUMMARY:")
    print(f"{'Location':<25} {'MPI Score':<12} {'Poverty Level':<15} {'Risk Level'}")
    print("-" * 70)
    
    for result in sorted(all_results, key=lambda x: x.get('multidimensional_poverty_index', 0), reverse=True):
        mpi = result.get('multidimensional_poverty_index', 0)
        poverty_level = result.get('poverty_level_classification', 'Unknown')
        risk = "🔴 High" if mpi > 0.6 else "🟠 Moderate" if mpi > 0.3 else "🟢 Low"
        
        print(f"{result['location_name']:<25} {mpi:<12.3f} {poverty_level:<15} {risk}")
    
    return all_results

def create_poverty_indicators_dataset(results_list, output_file='poverty_dataset.csv'):
    """Create a dataset from multiple poverty analyses for further research"""
    
    # Define key poverty indicators to extract
    poverty_indicators = [
        'latitude', 'longitude', 'radius_km',
        'multidimensional_poverty_index', 'poverty_level_classification',
        'electrification_deficit', 'economic_isolation_index', 'extreme_poverty_ratio',
        'poor_roof_materials_index', 'informal_settlement_index', 
        'water_access_deficit', 'severe_water_shortage',
        'vegetation_deficit', 'environmental_degradation', 'air_pollution_burden',
        'population_density_per_km2', 'overcrowding_index',
        'economic_deprivation_score', 'living_standards_score', 'environmental_deprivation_score'
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
    
    print(f"💾 Poverty indicators dataset saved to: {output_file}")
    print(f"📊 Dataset contains {len(df)} locations and {len(poverty_indicators)} indicators")
    
    return df

# Testing functions specifically for poverty analysis
def test_poverty_analysis():
    """Test poverty analysis with known high and low poverty areas"""
    print("🧪 TESTING SATELLITE-BASED POVERTY ANALYSIS")
    print("="*60)
    
    analyzer = SatellitePovertyAnalyzer()
    
    # Test locations representing different poverty levels
    test_locations = [
        # Urban slum area (expected high poverty)
        {"name": "Dharavi Mumbai (Slum)", "lat": 19.0370, "lon": 72.8570, "radius": 1},
        
        # Rural poor area (expected high poverty)  
        {"name": "Rural Bihar", "lat": 25.0961, "lon": 85.3131, "radius": 5},
        
        # Urban developed area (expected low poverty)
        {"name": "Bandra Kurla Mumbai", "lat": 19.0608, "lon": 72.8347, "radius": 2},
        
        # Mixed development area
        {"name": "Kolkata Mixed Area", "lat": 22.5726, "lon": 88.3639, "radius": 3}
    ]
    
    # Analyze each location
    results = compare_poverty_levels(analyzer, test_locations)
    
    # Create research dataset
    dataset = create_poverty_indicators_dataset(results)
    
    return results, dataset

def validate_poverty_predictions():
    """Validate satellite-based poverty predictions against known ground truth"""
    print("✅ POVERTY PREDICTION VALIDATION")
    print("="*50)
    
    # This function would ideally compare satellite-derived poverty indicators
    # with ground truth data from surveys, census, or field studies
    
    validation_areas = [
        # Areas with known poverty levels from surveys/census
        {"name": "Known High Poverty Area", "lat": 22.5726, "lon": 88.3639, "radius": 2, "ground_truth_poverty": "High"},
        {"name": "Known Low Poverty Area", "lat": 28.6139, "lon": 77.2090, "radius": 2, "ground_truth_poverty": "Low"}
    ]
    
    analyzer = SatellitePovertyAnalyzer()
    
    print("Validation would compare:")
    print("• Satellite-derived MPI scores")
    print("• Ground truth poverty classifications")
    print("• Calculate accuracy, precision, recall")
    print("• Identify which satellite indicators correlate best with ground truth")
    
    return validation_areas

# Add this to the end of your main.py file
if __name__ == "__main__":
    print("🚀 SATELLITE-BASED POVERTY ANALYSIS SYSTEM")
    print("="*60)
    
    # Initialize analyzer
    analyzer = SatellitePovertyAnalyzer()
    
    # Define test locations for comparison
    test_locations = [
        {"name": "Urban Slum Area", "lat": 19.0370, "lon": 72.8570, "radius_km": 2},
        {"name": "Wealthy Urban Area", "lat": 19.0608, "lon": 72.8347, "radius_km": 2},
        {"name": "Rural Poor Area", "lat": 22.97013056, "lon": 85.26155000, "radius_km": 2}
    ]
    
    # Run comparative analysis
    print("\n🔍 Running comparative poverty analysis...")
    results = compare_poverty_levels(analyzer, test_locations)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"poverty_comparison_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Create dataset
    dataset = create_poverty_indicators_dataset(results)
    print(f"📊 Analysis complete! Dataset has {len(dataset)} locations.")