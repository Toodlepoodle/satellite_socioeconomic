📊 The Indicators Used
Original 3 Core Dimensions (85% weight)
1. Economic Deprivation (25% weight)

electrification_deficit (40%) - Lack of nighttime lights = no electricity
economic_isolation_index (30%) - Low economic activity from satellite data
extreme_poverty_ratio (30%) - Areas with almost no nighttime lights

2. Living Standards (25% weight)

poor_roof_materials_index (25%) - Thatch/organic roofs vs metal/concrete
informal_settlement_index (25%) - Irregular building patterns from SAR data
water_access_deficit (25%) - Distance to reliable water sources
severe_water_shortage (25%) - No water access within 2km

3. Environmental Deprivation (25% weight)

vegetation_deficit (30%) - Low vegetation health (NDVI)
environmental_degradation (30%) - High bare soil, land degradation
green_space_deprivation (20%) - Lack of green spaces
air_pollution_burden (20%) - High NO2 levels from Sentinel-5P

Enhanced Dimensions (15% weight) - NEW
4. Access Deprivation (15% weight)

hospital_access_deficit (30%) - Distance to nearest major city with hospitals
airport_access_deficit (20%) - Distance to nearest airport
market_access_deficit (30%) - Low surrounding population = poor market access
road_quality_deficit (20%) - Unpaved vs paved road ratios

5. Connectivity Isolation (10% weight)

healthcare_isolation_index (40%) - Same as hospital access deficit
transportation_isolation (30%) - Same as airport access deficit
economic_opportunities_index (30%) - Market access score (inverted)

🧮 How the Enhanced MPI is Calculated
python# Step 1: Calculate dimension scores (weighted average within each dimension)
economic_score = (0.4 × electrification_deficit + 
                 0.3 × economic_isolation + 
                 0.3 × extreme_poverty_ratio)

living_standards_score = (0.25 × poor_roof_materials + 
                         0.25 × informal_settlement + 
                         0.25 × water_access_deficit + 
                         0.25 × severe_water_shortage)

environmental_score = (0.3 × vegetation_deficit + 
                      0.3 × environmental_degradation + 
                      0.2 × green_space_deprivation + 
                      0.2 × air_pollution_burden)

access_score = (0.3 × hospital_access_deficit + 
               0.2 × airport_access_deficit + 
               0.3 × market_access_deficit + 
               0.2 × road_quality_deficit)

connectivity_score = (0.4 × healthcare_isolation + 
                     0.3 × transportation_isolation + 
                     0.3 × (1 - economic_opportunities))

# Step 2: Calculate Enhanced MPI (weighted average across dimensions)
Enhanced_MPI = (0.25 × economic_score + 
                0.25 × living_standards_score + 
                0.25 × environmental_score + 
                0.15 × access_score + 
                0.10 × connectivity_score)
🛰️ Data Sources Used

VIIRS Nighttime Lights → Economic indicators
Sentinel-2 → Housing, vegetation, road quality
Sentinel-1 SAR → Building structure, informal settlements
Sentinel-5P → Air pollution (NO2)
WorldPop → Population density, market access
Global Surface Water → Water access
SRTM DEM → Terrain isolation
Distance calculations → Hospital, airport access



Here's a detailed breakdown of each satellite data source and how it's used for poverty analysis:

🌙 VIIRS Nighttime Lights → Economic Indicators
What it is: NASA/NOAA satellite that captures visible and near-infrared light at night
GEE Dataset: 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG'
How it works for poverty:

Bright areas = Economic activity, electricity access, wealth
Dark areas = No electricity, economic isolation, poverty
Scientific basis: Strong correlation between nighttime lights and GDP per capita

Specific indicators extracted:
python# Economic poverty indicators from darkness
electrification_deficit = max(0, 1 - ntl_mean)  # Less light = more poverty
extreme_poverty_ratio = 1.0 if ntl_p10 < 0.1 else 0.0  # Very dark areas
economic_isolation_index = max(0, 1 - ntl_median)  # Economic isolation

🛰️ Sentinel-2 → Housing, Vegetation, Road Quality
What it is: European Space Agency optical satellite with 13 spectral bands, 10m resolution
GEE Dataset: 'COPERNICUS/S2_SR_HARMONIZED'
How it works for poverty:
Housing Quality Analysis:

Metal roofs = Higher NIR reflectance → Better housing
Thatch/organic roofs = Higher red, lower NIR → Poor housing
Concrete roofs = Specific spectral signature

python# Roof material detection
metal_roof_index = NIR_band / Red_band  # Higher = metal roofs
organic_roof_index = Red_band / NIR_band  # Higher = poor materials

Vegetation Health (NDVI):

High NDVI = Healthy vegetation, better environment
Low NDVI = Degraded land, poor living conditions

pythonndvi = (NIR - Red) / (NIR + Red)
vegetation_deficit = max(0, 0.3 - ndvi_mean)  # Low vegetation = poverty

Road Quality:

Paved roads = Higher SWIR reflectance
Unpaved roads = Higher visible, lower SWIR

pythonpaved_road_index = SWIR_band / Red_band
road_quality_deficit = unpaved_ratio / paved_ratio

📡 Sentinel-1 SAR → Building Structure, Informal Settlements
What it is: Synthetic Aperture Radar that penetrates clouds, works day/night
GEE Dataset: 'COPERNICUS/S1_GRD'
How it works for poverty:

SAR backscatter reveals building density and structure patterns
Regular structures = Formal settlements, better planning
Irregular structures = Informal settlements, slums

python# Building analysis from radar
urban_density = VV_band - VH_band  # Building density
structure_quality = VV_band / VH_band  # Building regularity
informal_settlement_index = max(0, 1 - structure_quality)  # Irregular = informal
Why SAR is valuable: Can detect building materials, density, and settlement patterns that optical sensors miss

🌫️ Sentinel-5P → Air Pollution (NO2)
What it is: Atmospheric monitoring satellite measuring trace gases
GEE Dataset: 'COPERNICUS/S5P/NRTI/L3_NO2'
How it works for poverty:

High NO2 = Industrial pollution, traffic, poor air quality
Environmental poverty: Poor communities often live in polluted areas
Health impacts: Air pollution correlates with poverty and health outcomes

python# Air pollution burden
no2_density = NO2_column_number_density
air_pollution_burden = min(1.0, max(0, no2_density * 1e6))
Poverty connection: Environmental justice - poor communities disproportionately exposed to pollution

👥 WorldPop → Population Density, Market Access
What it is: High-resolution population distribution maps using census + satellite data
GEE Dataset: 'WorldPop/GP/100m/pop'
How it works for poverty:
Population Patterns:

Very high density = Overcrowding, slums
Very low density = Isolation, poor access
Population inequality = Mixed formal/informal settlements

python# Population-based poverty indicators
overcrowding_index = min(1.0, pop_density / 10000)  # Normalize by 10k/km²
population_inequality = pop_std / pop_mean  # High inequality = slums
Market Access:

Surrounding population = Economic opportunities, markets
More people nearby = Better market access, jobs, services

python# Market access from population
surrounding_pop = population_sum_within_10km
market_access_score = min(1.0, surrounding_pop / 50000)

💧 Global Surface Water → Water Access
What it is: Maps showing where water has been detected over 30+ years
GEE Dataset: 'JRC/GSW1_4/GlobalSurfaceWater'
How it works for poverty:

Distance to water = Basic need access
Reliable water sources = Areas with >50% water occurrence over time
Water poverty: Far from reliable water = higher poverty risk

python# Water access analysis
reliable_water = water_occurrence > 50%  # Reliable water sources
water_distance = distance_to_reliable_water
water_access_deficit = min(1.0, water_distance / 5000)  # Normalize by 5km
Poverty connection: Water access is fundamental - long distances = time burden, especially for women

🏔️ SRTM DEM → Terrain Isolation
What it is: Shuttle Radar Topography Mission - global elevation data
GEE Dataset: 'USGS/SRTMGL1_003'
How it works for poverty:

Steep terrain = Geographic isolation, poor road access
High slopes = Difficult transportation, service delivery
Terrain barriers = Economic isolation

python# Terrain-based isolation
slope = calculate_slope_from_elevation
terrain_isolation = min(1.0, mean_slope / 15.0)  # Normalize by 15 degrees
geographic_isolation = 1.0 if mean_slope > 10 else 0.0
Poverty connection: Mountainous/hilly areas often have poor infrastructure access, isolated communities

📊 How These Combine for Comprehensive Poverty Analysis
Each data source captures a different dimension of poverty:

VIIRS → Economic activity/electricity
Sentinel-2 → Living conditions, environment, infrastructure
Sentinel-1 → Settlement patterns, housing structure
Sentinel-5P → Environmental health
WorldPop → Social patterns, market access
Global Surface Water → Basic needs access
SRTM → Geographic barriers


📊 Step 2: Dimension Score Calculation
Economic Deprivation Score
pythonEconomic_Score = (0.4 × electrification_deficit + 
                  0.3 × economic_isolation_index + 
                  0.3 × extreme_poverty_ratio)
Living Standards Score
pythonLiving_Standards_Score = (0.25 × poor_roof_materials_index + 
                          0.25 × informal_settlement_index + 
                          0.25 × water_access_deficit + 
                          0.25 × severe_water_shortage)
Environmental Deprivation Score
pythonEnvironmental_Score = (0.30 × vegetation_deficit + 
                       0.30 × environmental_degradation + 
                       0.20 × green_space_deprivation + 
                       0.20 × air_pollution_burden)
Access Deprivation Score (NEW)
python# Distance calculations first
hospital_access_deficit = min(1.0, distance_to_major_city_km / 50.0)
airport_access_deficit = min(1.0, distance_to_airport_km / 100.0)
market_access_score = min(1.0, surrounding_population_10km / 50000)
market_access_deficit = 1 - market_access_score

# Road quality from Sentinel-2
paved_road_index = B11_SWIR / B4_Red
unpaved_road_index = B4_Red / B11_SWIR
road_quality_deficit = min(1.0, unpaved_road_index / max(paved_road_index, 0.1))

# Combine access indicators
Access_Score = (0.30 × hospital_access_deficit + 
                0.20 × airport_access_deficit + 
                0.30 × market_access_deficit + 
                0.20 × road_quality_deficit)
Connectivity Isolation Score (NEW)
python# Same as access but different weighting
healthcare_isolation_index = hospital_access_deficit
transportation_isolation = airport_access_deficit
economic_opportunities_index = market_access_score  # Higher = better, so invert

Connectivity_Score = (0.40 × healthcare_isolation_index + 
                      0.30 × transportation_isolation + 
                      0.30 × (1 - economic_opportunities_index))  # Invert good indicator

🎯 Step 3: Final Enhanced MPI Calculation
Traditional MPI (Literature-based)
pythonTraditional_MPI = (0.33 × Economic_Score + 
                   0.33 × Living_Standards_Score + 
                   0.34 × Environmental_Score)
Enhanced MPI (With New Access Dimensions)
pythonEnhanced_MPI = (0.25 × Economic_Score + 
                0.25 × Living_Standards_Score + 
                0.25 × Environmental_Score + 
                0.15 × Access_Score + 
                0.10 × Connectivity_Score)

🔄 Step 4: Normalization and Classification
Ensure 0-1 Scale
python# All individual indicators normalized to 0-1 where 1 = maximum poverty
normalized_indicator = min(1.0, max(0.0, raw_value))
Poverty Level Classification
pythonif Enhanced_MPI < 0.33:
    poverty_level = "Low"
elif Enhanced_MPI < 0.66:
    poverty_level = "Moderate"
else:
    poverty_level = "High"