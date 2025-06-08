"""Data processing and panel data generation"""

import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
from tqdm import tqdm
from satellite_extractor import SatelliteExtractor
from config import ANALYSIS_CONFIG

class SocioEconomicProcessor:
    def __init__(self):
        self.extractor = SatelliteExtractor()
    
    def generate_date_ranges(self, start_date, end_date, frequency='quarterly'):
        """Generate date ranges based on frequency"""
        dates = []
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        if frequency == 'monthly':
            delta = relativedelta(months=1)
        elif frequency == 'quarterly':
            delta = relativedelta(months=3)
        elif frequency == 'yearly':
            delta = relativedelta(years=1)
        else:
            raise ValueError("Frequency must be 'monthly', 'quarterly', or 'yearly'")
        
        while current < end:
            period_end = min(current + delta, end)
            dates.append({
                'start': current.strftime('%Y-%m-%d'),
                'end': period_end.strftime('%Y-%m-%d'),
                'period': current.strftime('%Y-%m') if frequency == 'monthly' else 
                         f"{current.year}-Q{(current.month-1)//3+1}" if frequency == 'quarterly' else
                         str(current.year)
            })
            current += delta
        
        return dates
    
    def calculate_distance_to_mining(self, region_geometry, mining_sites_gdf):
        """Calculate minimum distance to nearest mining site"""
        if mining_sites_gdf is None or len(mining_sites_gdf) == 0:
            return float('inf')
        
        try:
            # Calculate distance to all mining sites and return minimum
            distances = mining_sites_gdf.geometry.distance(region_geometry)
            return distances.min()
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return float('inf')
    
    def extract_variables_for_region(self, geometry, start_date, end_date, mining_sites_gdf=None):
        """Extract all socioeconomic variables for a single region and time period"""
        
        # Convert geometry to Earth Engine geometry
        if hasattr(geometry, '__geo_interface__'):
            ee_geometry = ee.Geometry(geometry.__geo_interface__)
        else:
            ee_geometry = ee.Geometry(geometry)
        
        # Extract satellite variables
        variables = {}
        
        # Nighttime lights (economic activity & electricity)
        ntl_vars = self.extractor.extract_nighttime_lights(ee_geometry, start_date, end_date)
        variables.update(ntl_vars)
        
        # Vegetation indices (agriculture & environment)
        veg_vars = self.extractor.extract_vegetation_indices(ee_geometry, start_date, end_date)
        variables.update(veg_vars)
        
        # Building characteristics (housing quality & urban development)
        building_vars = self.extractor.extract_building_characteristics(ee_geometry)
        variables.update(building_vars)
        
        # Infrastructure indicators (roads, healthcare, education access)
        infra_vars = self.extractor.extract_infrastructure_indicators(ee_geometry)
        variables.update(infra_vars)
        
        # Roof material quality indicators
        roof_vars = self.extractor.extract_roof_material_indicators(ee_geometry, start_date, end_date)
        variables.update(roof_vars)
        
        # Environmental variables (climate, air quality, terrain)
        env_vars = self.extractor.extract_environmental_data(ee_geometry, start_date, end_date)
        variables.update(env_vars)
        
        # Economic activity indicators (agriculture, industry)
        econ_vars = self.extractor.extract_economic_activity_indicators(ee_geometry, start_date, end_date)
        variables.update(econ_vars)
        
        # Create composite poverty indicators
        composite_vars = self.extractor.extract_poverty_composite_indicators(variables)
        variables.update(composite_vars)
        
        # Distance to mining sites
        if mining_sites_gdf is not None:
            distance = self.calculate_distance_to_mining(geometry, mining_sites_gdf)
            variables['distance_to_mining'] = distance
        else:
            variables['distance_to_mining'] = float('inf')
        
        return variables
    
    def create_panel_data(self, regions_shapefile, start_date, end_date, 
                         mining_sites_shapefile=None, output_file='panel_data.csv'):
        """
        Create panel data for all regions across time periods
        
        Args:
            regions_shapefile (str): Path to shapefile with regions to analyze
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            mining_sites_shapefile (str, optional): Path to mining sites shapefile
            output_file (str): Output CSV filename
        
        Returns:
            pandas.DataFrame: Panel dataset
        """
        
        print("Loading spatial data...")
        # Load shapefiles
        regions_gdf = gpd.read_file(regions_shapefile)
        
        if mining_sites_shapefile:
            mining_sites_gdf = gpd.read_file(mining_sites_shapefile)
        else:
            mining_sites_gdf = None
        
        # Generate time periods
        date_ranges = self.generate_date_ranges(
            start_date, end_date, 
            ANALYSIS_CONFIG['temporal_frequency']
        )
        
        print(f"Processing {len(regions_gdf)} regions across {len(date_ranges)} time periods...")
        
        # Create panel data
        panel_data = []
        
        for idx, region in tqdm(regions_gdf.iterrows(), total=len(regions_gdf), desc="Processing regions"):
            region_id = idx
            region_name = region.get('NAME', f'Region_{idx}')
            
            for date_range in tqdm(date_ranges, desc=f"Time periods for {region_name}", leave=False):
                try:
                    # Extract variables for this region and time period
                    variables = self.extract_variables_for_region(
                        region.geometry, 
                        date_range['start'], 
                        date_range['end'],
                        mining_sites_gdf
                    )
                    
                    # Create record
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
                    print(f"Error processing region {region_name}, period {date_range['period']}: {e}")
                    continue
        
        # Convert to DataFrame
        df = pd.DataFrame(panel_data)
        
        # Save to file
        print(f"Saving results to {output_file}...")
        df.to_csv(output_file, index=False)
        
        print(f"✓ Panel data created successfully!")
        print(f"  - {len(df)} total observations")
        print(f"  - {df['region_id'].nunique()} regions")
        print(f"  - {df['period'].nunique()} time periods")
        print(f"  - {len([col for col in df.columns if col not in ['region_id', 'region_name', 'period', 'start_date', 'end_date']])} variables")
        
        return df