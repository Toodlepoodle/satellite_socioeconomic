# run_simple_example.py - Simple example to run everything
import pandas as pd
import matplotlib.pyplot as plt
from data_processor import SocioEconomicProcessor

def main():
    print("🛰️  SATELLITE SOCIOECONOMIC ANALYSIS")
    print("=" * 50)
    
    # Step 1: Create test data
    print("\n1️⃣  Creating test data...")
    from create_test_data import main as create_data
    create_data()
    
    # Step 2: Run analysis
    print("\n2️⃣  Running analysis...")
    processor = SocioEconomicProcessor()
    
    df = processor.create_panel_data(
        regions_shapefile='data/test_regions.geojson',
        start_date='2022-01-01',
        end_date='2023-12-31',
        mining_sites_shapefile='data/test_mining_sites.geojson',
        output_file='outputs/results.csv'
    )
    
    # Step 3: Show results
    print("\n3️⃣  RESULTS SUMMARY")
    print("-" * 30)
    
    # Get latest period
    latest = df[df['period'] == df['period'].max()].copy()
    
    print("POVERTY ANALYSIS:")
    print("Region Name                | Poverty Score | Development Score | Classification")
    print("-" * 75)
    
    for _, row in latest.iterrows():
        name = row['region_name']
        poverty = row.get('poverty_likelihood_score', 0.5)
        development = row.get('overall_development_index', 0.5)
        
        if poverty > 0.6:
            classification = "HIGH POVERTY"
        elif poverty < 0.4:
            classification = "LOW POVERTY"
        else:
            classification = "MEDIUM POVERTY"
        
        print(f"{name:25} | {poverty:11.3f} | {development:15.3f} | {classification}")
    
    # Step 4: Create simple visualization
    print("\n4️⃣  Creating visualization...")
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Poverty scores
    plt.subplot(1, 2, 1)
    regions = [name[:15] for name in latest['region_name']]
    poverty_scores = latest['poverty_likelihood_score']
    
    colors = ['red' if score > 0.6 else 'orange' if score > 0.4 else 'green' 
              for score in poverty_scores]
    
    plt.bar(regions, poverty_scores, color=colors, alpha=0.7)
    plt.title('Poverty Likelihood by Region')
    plt.ylabel('Poverty Score (0-1)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Development vs Poverty
    plt.subplot(1, 2, 2)
    plt.scatter(latest['poverty_likelihood_score'], 
                latest['overall_development_index'],
                s=100, alpha=0.7, c=colors)
    
    for _, row in latest.iterrows():
        plt.annotate(row['region_name'][:10], 
                    (row['poverty_likelihood_score'], row['overall_development_index']),
                    fontsize=8, ha='center')
    
    plt.xlabel('Poverty Likelihood')
    plt.ylabel('Development Index')
    plt.title('Development vs Poverty')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/analysis_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Step 5: Mining analysis
    if 'distance_to_mining' in df.columns:
        print("\n5️⃣  MINING PROXIMITY ANALYSIS")
        print("-" * 35)
        
        # Create distance categories
        latest['distance_km'] = latest['distance_to_mining'] / 1000
        latest['distance_category'] = pd.cut(
            latest['distance_km'], 
            bins=[0, 100, 500, float('inf')],
            labels=['Near (<100km)', 'Medium (100-500km)', 'Far (>500km)']
        )
        
        mining_analysis = latest.groupby('distance_category')['poverty_likelihood_score'].mean()
        print("Average poverty score by distance to mining:")
        for category, score in mining_analysis.items():
            print(f"  {category}: {score:.3f}")
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📁 Files created:")
    print(f"   • outputs/results.csv (full dataset)")
    print(f"   • outputs/analysis_results.png (visualization)")
    print(f"   • data/test_regions.geojson (test regions)")
    print(f"   • data/test_mining_sites.geojson (mining sites)")
    
    return df

if __name__ == "__main__":
    main()