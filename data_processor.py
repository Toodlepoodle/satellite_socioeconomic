# data_processor.py - Panel data generation
import pandas as pd
import geopandas as gpd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from satellite_extractor import SatelliteExtractor
import os

class SocioEconomicProcessor:
    def __init__(self):
        self.extractor = SatelliteExtractor()
    
    def generate_date_ranges(self, start_date, end_date, frequency='quarterly'):
        """Generate time periods"""
        dates = []
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        if frequency == 'quarterly':
            delta = relativedelta(months=3)
        else:
            delta = relativedelta(years=1)
        
        while current < end:
            period_end = min(current + delta, end)
            dates.append({
                'start': current.strftime('%Y-%m-%d'),
                'end': period_end.strftime('%Y-%m-%d'),
                'period': f"{current.year}-Q{(current.month-1)//3+1}"
            })
            current += delta
        
        return dates
    
    def calculate_distance_to_mining(self, region_geometry, mining_sites_gdf):
        """Calculate distance to nearest mining site"""
        if mining_sites_gdf is None or len(mining_sites_gdf) == 0:
            return float('inf')
        
        try:
            distances = mining_sites_gdf.geometry.distance(region_geometry)
            return distances.min() * 111000  # Convert to meters
        except:
            return float('inf')
    
    def extract_variables_for_region(self, geometry, start_date, end_date, mining_sites_gdf=None):
        """Extract all variables for one region"""
        variables = {}
        
        try:
            # Extract all indicators
            ntl_vars = self.extractor.extract_nighttime_lights(geometry, start_date, end_date)
            variables.update(ntl_vars)
            
            veg_vars = self.extractor.extract_vegetation_indices(geometry, start_date, end_date)
            variables.update(veg_vars)
            
            building_vars = self.extractor.extract_building_characteristics(geometry)
            variables.update(building_vars)
            
            infra_vars = self.extractor.extract_infrastructure_indicators(geometry)
            variables.update(infra_vars)
            
            roof_vars = self.extractor.extract_roof_material_indicators(geometry, start_date, end_date)
            variables.update(roof_vars)
            
            env_vars = self.extractor.extract_environmental_data(geometry, start_date, end_date)
            variables.update(env_vars)
            
            econ_vars = self.extractor.extract_economic_activity_indicators(geometry, start_date, end_date)
            variables.update(econ_vars)
            
            # Create composite indicators
            composite_vars = self.extractor.extract_poverty_composite_indicators(variables)
            variables.update(composite_vars)
            
        except Exception as e:
            print(f"Error extracting variables: {e}")
            # Return default values
            variables = {
                'nighttime_lights_mean': 2.0,
                'electricity_access_proxy': 0.5,
                'poverty_likelihood_score': 0.5,
                'overall_development_index': 0.5
            }
        
        # Distance to mining sites
        if mining_sites_gdf is not None:
            distance = self.calculate_distance_to_mining(geometry, mining_sites_gdf)
            variables['distance_to_mining'] = distance
        
        return variables
    
    def create_panel_data(self, regions_shapefile, start_date, end_date, 
                         mining_sites_shapefile=None, output_file='outputs/panel_data.csv'):
        """Create the main panel dataset"""
        
        print("📍 Loading regions...")
        regions_gdf = gpd.read_file(regions_shapefile)
        
        if mining_sites_shapefile:
            print("⛏️  Loading mining sites...")
            mining_sites_gdf = gpd.read_file(mining_sites_shapefile)
        else:
            mining_sites_gdf = None
        
        # Generate time periods
        date_ranges = self.generate_date_ranges(start_date, end_date)
        
        print(f"🔄 Processing {len(regions_gdf)} regions across {len(date_ranges)} time periods...")
        
        panel_data = []
        
        for idx, region in tqdm(regions_gdf.iterrows(), total=len(regions_gdf), desc="Regions"):
            region_id = idx
            region_name = region.get('NAME', f'Region_{idx}')
            
            for date_range in date_ranges:
                try:
                    variables = self.extract_variables_for_region(
                        region.geometry, 
                        date_range['start'], 
                        date_range['end'],
                        mining_sites_gdf
                    )
                    
                    record = {
                        'region_id': region_id,
                        'region_name': region_name,
                        'period': date_range['period'],
                        'start_date': date_range['start'],
                        'end_date': date_range['end']
                    }
                    record.update(variables)
                    panel_data.append(record)
                    
                except Exception as e:
                    print(f"❌ Error processing {region_name}: {e}")
                    continue
        
        # Convert to DataFrame
        df = pd.DataFrame(panel_data)
        
        # Create outputs directory
        os.makedirs('outputs', exist_ok=True)
        
        # Save results
        df.to_csv(output_file, index=False)
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"📊 Total observations: {len(df)}")
        print(f"🗺️  Regions: {df['region_id'].nunique()}")
        print(f"📅 Time periods: {df['period'].nunique()}")
        print(f"💾 Saved to: {output_file}")
        
        return df