# satellite_extractor.py - Main data extraction
import numpy as np

class SatelliteExtractor:
    def __init__(self):
        print("🛰️  Satellite Extractor initialized (using simulated data)")
        self.gee_available = False
    
    def _get_poverty_level_from_coords(self, geometry):
        """Determine poverty level based on coordinates"""
        try:
            centroid = geometry.centroid
            lat, lon = centroid.y, centroid.x
            
            # Poor areas (Rural Bihar, Dhaka, Tribal areas)
            if ((24.8 <= lat <= 25.4 and 85.0 <= lon <= 85.6) or  # Rural Bihar
                (23.6 <= lat <= 24.0 and 90.2 <= lon <= 90.6) or  # Dhaka
                (20.7 <= lat <= 21.2 and 84.8 <= lon <= 85.3)):   # Tribal Odisha
                return 'very_poor'
            
            # Rich areas (Mumbai, Bangalore, Delhi)
            elif ((19.0 <= lat <= 19.2 and 72.7 <= lon <= 72.9) or  # Mumbai
                  (12.8 <= lat <= 13.0 and 77.5 <= lon <= 77.7) or  # Bangalore
                  (28.5 <= lat <= 28.7 and 77.1 <= lon <= 77.3)):   # Delhi
                return 'very_rich'
            else:
                return 'medium'
        except:
            return 'medium'
    
    def extract_nighttime_lights(self, geometry, start_date, end_date):
        """Extract nighttime lights (electricity & economic activity)"""
        poverty_level = self._get_poverty_level_from_coords(geometry)
        
        if poverty_level == 'very_poor':
            base_light = np.random.uniform(0.1, 1.5)
            electricity = np.random.uniform(0.1, 0.4)
        elif poverty_level == 'very_rich':
            base_light = np.random.uniform(8.0, 15.0)
            electricity = np.random.uniform(0.85, 0.98)
        else:
            base_light = np.random.uniform(2.0, 6.0)
            electricity = np.random.uniform(0.4, 0.8)
        
        return {
            'nighttime_lights_mean': base_light,
            'electricity_access_proxy': electricity
        }
    
    def extract_vegetation_indices(self, geometry, start_date, end_date):
        """Extract vegetation data (agriculture)"""
        poverty_level = self._get_poverty_level_from_coords(geometry)
        
        if poverty_level == 'very_poor':
            ndvi = np.random.uniform(0.2, 0.6)
            water_access = np.random.uniform(0.2, 0.5)
        elif poverty_level == 'very_rich':
            ndvi = np.random.uniform(0.3, 0.7)
            water_access = np.random.uniform(0.8, 0.98)
        else:
            ndvi = np.random.uniform(0.4, 0.8)
            water_access = np.random.uniform(0.5, 0.8)
        
        return {
            'ndvi_mean': ndvi,
            'agricultural_productivity': max(0, ndvi),
            'water_access_proxy': water_access
        }
    
    def extract_building_characteristics(self, geometry):
        """Extract building & population data"""
        poverty_level = self._get_poverty_level_from_coords(geometry)
        
        if poverty_level == 'very_poor':
            pop_density = np.random.uniform(50, 200)
            building_density = np.random.uniform(5, 25)
        elif poverty_level == 'very_rich':
            pop_density = np.random.uniform(2000, 8000)
            building_density = np.random.uniform(150, 400)
        else:
            pop_density = np.random.uniform(200, 1500)
            building_density = np.random.uniform(30, 120)
        
        return {
            'population_count': pop_density * 25,  # 25 km2 area
            'population_density': pop_density,
            'building_density': building_density,
            'built_up_ratio': min(1.0, building_density / 200)
        }
    
    def extract_infrastructure_indicators(self, geometry):
        """Extract infrastructure data (roads, hospitals, schools)"""
        poverty_level = self._get_poverty_level_from_coords(geometry)
        
        if poverty_level == 'very_poor':
            road_density = np.random.uniform(0.2, 1.5)
            infra_score = np.random.uniform(0.1, 0.3)
        elif poverty_level == 'very_rich':
            road_density = np.random.uniform(3.0, 8.0)
            infra_score = np.random.uniform(0.7, 0.95)
        else:
            road_density = np.random.uniform(1.0, 4.0)
            infra_score = np.random.uniform(0.3, 0.7)
        
        travel_time = max(20, 400 - (infra_score * 350))
        
        return {
            'road_density_km_per_km2': road_density,
            'infrastructure_access_score': infra_score,
            'travel_time_to_cities': travel_time,
            'hospital_count': int(infra_score * 8),
            'school_count': int(infra_score * 15) + 1
        }
    
    def extract_roof_material_indicators(self, geometry, start_date, end_date):
        """Extract roof quality data"""
        poverty_level = self._get_poverty_level_from_coords(geometry)
        
        if poverty_level == 'very_poor':
            roof_quality = np.random.uniform(0.1, 0.3)
        elif poverty_level == 'very_rich':
            roof_quality = np.random.uniform(0.7, 0.95)
        else:
            roof_quality = np.random.uniform(0.3, 0.7)
        
        return {
            'roof_quality_score': roof_quality,
            'estimated_good_roof_ratio': min(1.0, roof_quality * 1.2)
        }
    
    def extract_environmental_data(self, geometry, start_date, end_date):
        """Extract environmental data"""
        return {
            'elevation_mean': np.random.uniform(0, 1000),
            'precip_mean': np.random.uniform(50, 200),
            'temp_mean': np.random.uniform(15, 35),
            'climate_stress_index': np.random.uniform(0, 0.5),
            'water_stress_index': np.random.uniform(0, 0.6)
        }
    
    def extract_economic_activity_indicators(self, geometry, start_date, end_date):
        """Extract economic indicators"""
        return {
            'crop_productivity_max': np.random.uniform(0.4, 0.9),
            'agricultural_intensity': np.random.uniform(0.2, 0.8),
            'livestock_potential': np.random.uniform(0.1, 0.6)
        }
    
    def extract_poverty_composite_indicators(self, all_variables):
        """Create composite poverty indicators"""
        try:
            # Economic activity index
            economic_activity_index = (
                min(1, all_variables.get('nighttime_lights_mean', 0) / 10) * 0.4 +
                min(1, all_variables.get('building_density', 0) / 100) * 0.3 +
                all_variables.get('electricity_access_proxy', 0) * 0.3
            )
            
            # Infrastructure quality index
            infrastructure_quality_index = (
                all_variables.get('roof_quality_score', 0) * 0.3 +
                min(1, all_variables.get('road_density_km_per_km2', 0) / 5) * 0.3 +
                all_variables.get('infrastructure_access_score', 0) * 0.4
            )
            
            # Overall development index
            overall_development_index = (
                economic_activity_index * 0.5 +
                infrastructure_quality_index * 0.5
            )
            
            # Poverty likelihood (inverse of development)
            poverty_likelihood = 1 - overall_development_index
            
            # Development category
            if overall_development_index >= 0.7:
                category = 'High'
            elif overall_development_index >= 0.4:
                category = 'Medium'
            else:
                category = 'Low'
            
            return {
                'economic_activity_index': economic_activity_index,
                'infrastructure_quality_index': infrastructure_quality_index,
                'overall_development_index': overall_development_index,
                'poverty_likelihood_score': poverty_likelihood,
                'development_category': category
            }
        except Exception as e:
            print(f"Error creating composite indicators: {e}")
            return {
                'economic_activity_index': 0.5,
                'infrastructure_quality_index': 0.5,
                'overall_development_index': 0.5,
                'poverty_likelihood_score': 0.5,
                'development_category': 'Medium'
            }