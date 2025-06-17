# main.py
"""Main script to run socioeconomic analysis"""

import argparse
from data_processor import SocioEconomicProcessor
import pandas as pd
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description='Generate socioeconomic variables from satellite imagery')
    parser.add_argument('--regions', required=True, help='Path to regions shapefile')
    parser.add_argument('--start_date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--mining_sites', help='Path to mining sites shapefile (optional)')
    parser.add_argument('--output', default='outputs/panel_data.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    # Create processor and generate panel data
    processor = SocioEconomicProcessor()
    
    df = processor.create_panel_data(
        regions_shapefile=args.regions,
        start_date=args.start_date,
        end_date=args.end_date,
        mining_sites_shapefile=args.mining_sites,
        output_file=args.output
    )
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(df.describe())
    
    # Show correlation with distance to mining (if available)
    if 'distance_to_mining' in df.columns and not df['distance_to_mining'].isna().all():
        print("\n" + "="*50)
        print("CORRELATION WITH DISTANCE TO MINING")
        print("="*50)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'distance_to_mining']
        
        correlations = df[numeric_cols + ['distance_to_mining']].corr()['distance_to_mining'].drop('distance_to_mining')
        print(correlations.sort_values(key=abs, ascending=False))

if __name__ == "__main__":
    main()
