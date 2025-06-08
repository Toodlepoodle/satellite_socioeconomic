"""Example usage of the satellite socioeconomic analysis tool"""

from data_processor import SocioEconomicProcessor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def run_comprehensive_analysis():
    """Run a comprehensive example analysis with all poverty indicators"""
    
    print("🛰️  COMPREHENSIVE Satellite Socioeconomic Analysis")
    print("=" * 60)
    
    # Initialize processor
    processor = SocioEconomicProcessor()
    
    # Run analysis
    df = processor.create_panel_data(
        regions_shapefile='../data/test_regions.geojson',
        start_date='2022-01-01',
        end_date='2023-12-31',
        mining_sites_shapefile='../data/test_mining_sites.geojson',
        output_file='../outputs/comprehensive_panel_data.csv'
    )
    
    print(f"\n📊 DATASET OVERVIEW")
    print("-" * 40)
    print(f"Total observations: {len(df)}")
    print(f"Regions: {df['region_id'].nunique()}")
    print(f"Time periods: {df['period'].nunique()}")
    print(f"Variables: {len([col for col in df.columns if col not in ['region_id', 'region_name', 'period', 'start_date', 'end_date']])}")
    
    # Poverty indicators analysis
    poverty_indicators = [
        'poverty_likelihood_score', 'electricity_access_proxy', 'roof_quality_score',
        'water_access_proxy', 'infrastructure_access_score', 'overall_development_index'
    ]
    
    available_poverty_indicators = [col for col in poverty_indicators if col in df.columns]
    
    if available_poverty_indicators:
        print(f"\n🏠 POVERTY INDICATORS ANALYSIS")
        print("-" * 40)
        poverty_stats = df[available_poverty_indicators].describe()
        print(poverty_stats)
        
        # Create poverty indicators heatmap
        plt.figure(figsize=(12, 8))
        correlation_matrix = df[available_poverty_indicators].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                   square=True, fmt='.2f')
        plt.title('Poverty Indicators Correlation Matrix')
        plt.tight_layout()
        plt.savefig('../outputs/poverty_indicators_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Housing quality analysis
    housing_indicators = [
        'roof_quality_score', 'estimated_good_roof_ratio', 'building_density',
        'built_up_ratio', 'urban_compactness'
    ]
    
    available_housing = [col for col in housing_indicators if col in df.columns]
    
    if available_housing:
        print(f"\n🏘️  HOUSING QUALITY ANALYSIS")
        print("-" * 40)
        
        # Housing quality distribution
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, indicator in enumerate(available_housing[:6]):
            if i < len(axes):
                df[indicator].hist(bins=20, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'{indicator.replace("_", " ").title()}')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
        
        # Remove empty subplots
        for j in range(len(available_housing), len(axes)):
            axes[j].remove()
        
        plt.tight_layout()
        plt.savefig('../outputs/housing_quality_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Infrastructure and access analysis
    infrastructure_indicators = [
        'road_density_km_per_km2', 'hospital_count', 'school_count',
        'travel_time_to_cities', 'infrastructure_access_score'
    ]
    
    available_infra = [col for col in infrastructure_indicators if col in df.columns]
    
    if available_infra:
        print(f"\n🛣️  INFRASTRUCTURE ACCESS ANALYSIS")
        print("-" * 40)
        infra_stats = df[available_infra].describe()
        print(infra_stats)
    
    # Mining proximity impact analysis
    if 'distance_to_mining' in df.columns:
        print(f"\n⛏️  MINING PROXIMITY IMPACT ANALYSIS")
        print("-" * 50)
        
        # Create distance categories
        df['distance_category'] = pd.cut(
            df['distance_to_mining'], 
            bins=[0, 2000, 5000, 10000, np.inf],
            labels=['Very Close (<2km)', 'Close (2-5km)', 'Medium (5-10km)', 'Far (>10km)']
        )
        
        # Key indicators by distance
        key_indicators = ['poverty_likelihood_score', 'electricity_access_proxy', 
                         'roof_quality_score', 'nighttime_lights_mean']
        available_key = [col for col in key_indicators if col in df.columns]
        
        if available_key:
            comparison = df.groupby('distance_category')[available_key].mean()
            print(comparison)
            
            # Visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, indicator in enumerate(available_key[:4]):
                if i < len(axes):
                    comparison[indicator].plot(kind='bar', ax=axes[i], color='skyblue')
                    axes[i].set_title(f'{indicator.replace("_", " ").title()} by Distance to Mining')
                    axes[i].set_xlabel('Distance Category')
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('../outputs/mining_proximity_impact.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # Time series analysis
    if 'period' in df.columns and len(df['period'].unique()) > 1:
        print(f"\n📈 TIME SERIES ANALYSIS")
        print("-" * 30)
        
        # Time trends for key indicators
        time_series = df.groupby('period')[available_key].mean()
        
        plt.figure(figsize=(15, 10))
        for i, indicator in enumerate(available_key):
            plt.subplot(2, 2, i+1)
            time_series[indicator].plot(marker='o')
            plt.title(f'{indicator.replace("_", " ").title()} Over Time')
            plt.xlabel('Period')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../outputs/time_series_poverty_indicators.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Comprehensive correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['region_id']]
    
    if len(numeric_cols) > 5:
        print(f"\n🔗 COMPREHENSIVE CORRELATION ANALYSIS")
        print("-" * 45)
        
        # Focus on most important correlations
        important_vars = [
            'poverty_likelihood_score', 'overall_development_index', 'nighttime_lights_mean',
            'roof_quality_score', 'electricity_access_proxy', 'water_access_proxy',
            'road_density_km_per_km2', 'distance_to_mining'
        ]
        
        available_important = [col for col in important_vars if col in df.columns]
        
        if len(available_important) > 3:
            correlation_matrix = df[available_important].corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
                       square=True, fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
            plt.title('Key Socioeconomic Variables Correlation Matrix')
            plt.tight_layout()
            plt.savefig('../outputs/comprehensive_correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # Development categorization analysis
    if 'development_category' in df.columns:
        print(f"\n🏆 DEVELOPMENT CATEGORIZATION")
        print("-" * 35)
        
        category_counts = df['development_category'].value_counts()
        print(category_counts)
        
        # Development category by distance to mining
        if 'distance_category' in df.columns:
            crosstab = pd.crosstab(df['distance_category'], df['development_category'], normalize='index')
            print(f"\nDevelopment levels by mining proximity:")
            print(crosstab)
            
            crosstab.plot(kind='bar', stacked=True, figsize=(10, 6))
            plt.title('Development Categories by Distance to Mining')
            plt.xlabel('Distance to Mining Sites')
            plt.ylabel('Proportion')
            plt.legend(title='Development Level')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('../outputs/development_by_mining_distance.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"📁 All visualizations saved to outputs/ folder")
    print(f"📊 Full dataset: {len(df)} observations with {len(df.columns)} variables")
    
    # Summary recommendations
    if 'poverty_likelihood_score' in df.columns:
        high_poverty_regions = df[df['poverty_likelihood_score'] > 0.6]['region_name'].unique()
        low_poverty_regions = df[df['poverty_likelihood_score'] < 0.3]['region_name'].unique()
        
        print(f"\n🎯 POLICY RECOMMENDATIONS")
        print("-" * 30)
        print(f"High poverty risk regions ({len(high_poverty_regions)}): {', '.join(high_poverty_regions[:3])}...")
        print(f"Low poverty risk regions ({len(low_poverty_regions)}): {', '.join(low_poverty_regions[:3])}...")
        
        if 'distance_to_mining' in df.columns:
            mining_impact = df.groupby('distance_category')['poverty_likelihood_score'].mean()
            if mining_impact.iloc[0] > mining_impact.iloc[-1]:
                print("⚠️  Areas closer to mining show higher poverty likelihood")
            else:
                print("💡 Areas closer to mining show economic benefits")

    return df

if __name__ == "__main__":
    run_comprehensive_analysis()