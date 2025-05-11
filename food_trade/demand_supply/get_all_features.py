# Usage: python -m food_trade.demand_supply.get_all_features

import geopandas as gpd
import pandas as pd
import numpy as np
import os

admin = gpd.read_file('../../data/admin_polygons/admin_polygons.shp')

# trade data
def get_trade_data(log_y=True, trade_thresh=10):

    def get_crop_trade(trade_type, crop, log_y, trade_thresh):
        # re-exported trade matrix (203 x 203)
        trade_2017 = pd.read_csv(f"../../data/{trade_type}_matrix_{crop}_2017.csv")
        trade_2018 = pd.read_csv(f"../../data/{trade_type}_matrix_{crop}_2018.csv")
        trade_2019 = pd.read_csv(f"../../data/{trade_type}_matrix_{crop}_2019.csv")
        trade_2020 = pd.read_csv(f"../../data/{trade_type}_matrix_{crop}_2020.csv")
        trade_2021 = pd.read_csv(f"../../data/{trade_type}_matrix_{crop}_2021.csv")
        
        trade = pd.concat([trade_2017, trade_2018, trade_2019, trade_2020, trade_2021]).groupby('iso3').mean().reset_index()
        
        # melt trade data 
        trade = trade.rename(columns={'iso3': 'from_iso3'})
        trade = trade.melt(id_vars=['from_iso3'], value_vars=trade['from_iso3'].values.tolist()).rename(
            columns={'variable': 'to_iso3', 'value': f"{trade_type.split('/')[-1]}_{crop}"})
        trade = trade[trade['from_iso3']!=trade['to_iso3']]

        trade.loc[trade[f"{trade_type.split('/')[-1]}_{crop}"]<trade_thresh, f"{trade_type.split('/')[-1]}_{crop}"] = 0
        trade[f"{trade_type.split('/')[-1]}_{crop}_exists"] = 0
        trade.loc[trade[f"{trade_type.split('/')[-1]}_{crop}"]>0, f"{trade_type.split('/')[-1]}_{crop}_exists"] = 1
        # log trade for where it exists (for volume prediction)
        if log_y:
            trade.loc[trade[f"{trade_type.split('/')[-1]}_{crop}_exists"]==1, f"{trade_type.split('/')[-1]}_{crop}"] = trade[
            trade[f"{trade_type.split('/')[-1]}_{crop}_exists"]==1].apply(lambda row: np.log(row[f"{trade_type.split('/')[-1]}_{crop}"]), axis=1)
        return trade

    crops = ['Wheat', 'Maize (corn)', 'Rye', 'Barley', 'Oats', 'Sorghum', 
             'Rice, paddy (rice milled equivalent)', 'Buckwheat', 
             'Millet', 'Quinoa', 'Cereals n.e.c.', 
             'Canary seed', 'Fonio', 'Mixed grain', 'Triticale', 'cereals_all']
    
    df_list = []
    for crop in crops:
        df = get_crop_trade('FAO_re_export/supply', crop, log_y, trade_thresh)
        df_list.append(df)
        df = get_crop_trade('FAO_bal_trade_mat/trade', crop, log_y, trade_thresh)
        df_list.append(df)
        
    dfs = [df.set_index(['from_iso3', 'to_iso3']) for df in df_list]
    trade = pd.concat(dfs, axis=1).reset_index()
    return trade

# admin reductions
def get_admin_reductions():
    # available for all boundaries, i.e. 240 countries, 3787 admins [only country in FAO data not in boundaries data is TKL but it doesn't have much trade]
    
    def _clean_cols(filepath):
        df_admin = pd.read_csv(filepath).drop(['system:index', '.geo', 'admin', 'admin_name', 'country', 'latitude', 'longitude'], axis=1)
        col = [c for c in df_admin.columns.tolist() if c not in ['iso3', 'ID']][0]
        # agg at country level 
        df_country = df_admin[['iso3', col]].groupby('iso3').sum().reset_index()
        
        if col=='rice_production': # conversion factor for milling
            df_admin[col] = df_admin[col] * 0.7
            df_country[col] = df_country[col] * 0.7
            
        return df_admin, df_country
    
    # crop areas
    barley_a_admin, barley_a_country = _clean_cols('../../data/admin_reductions/barley_areas.csv')
    maize_a_admin, maize_a_country = _clean_cols('../../data/admin_reductions/maize_areas.csv')
    millet_a_admin, millet_a_country = _clean_cols('../../data/admin_reductions/millet_areas.csv')
    rice_a_admin, rice_a_country = _clean_cols('../../data/admin_reductions/rice_areas.csv')
    sorghum_a_admin, sorghum_a_country = _clean_cols('../../data/admin_reductions/sorghum_areas.csv')
    wheat_a_admin, wheat_a_country = _clean_cols('../../data/admin_reductions/wheat_areas.csv')
    other_a_admin, other_a_country = _clean_cols('../../data/admin_reductions/other_cereals_areas.csv')
    
    # crop productions
    barley_p_admin, barley_p_country = _clean_cols('../../data/admin_reductions/barley_production.csv')
    maize_p_admin, maize_p_country = _clean_cols('../../data/admin_reductions/maize_production.csv')
    millet_p_admin, millet_p_country = _clean_cols('../../data/admin_reductions/millet_production.csv')
    rice_p_admin, rice_p_country = _clean_cols('../../data/admin_reductions/rice_production.csv')
    sorghum_p_admin, sorghum_p_country = _clean_cols('../../data/admin_reductions/sorghum_production.csv')
    wheat_p_admin, wheat_p_country = _clean_cols('../../data/admin_reductions/wheat_production.csv')
    other_p_admin, other_p_country = _clean_cols('../../data/admin_reductions/other_cereals_production.csv')
    
    # livestock
    buffaloes_admin, buffaloes_country = _clean_cols('../../data/admin_reductions/buffaloes_totals.csv')
    cattle_admin, cattle_country = _clean_cols('../../data/admin_reductions/cattle_totals.csv')
    chickens_admin, chickens_country = _clean_cols('../../data/admin_reductions/chickens_totals.csv')
    ducks_admin, ducks_country = _clean_cols('../../data/admin_reductions/ducks_totals.csv')
    goats_admin, goats_country = _clean_cols('../../data/admin_reductions/goats_totals.csv')
    horses_admin, horses_country = _clean_cols('../../data/admin_reductions/horses_totals.csv')
    pigs_admin, pigs_country = _clean_cols('../../data/admin_reductions/pigs_totals.csv')
    sheep_admin, sheep_country = _clean_cols('../../data/admin_reductions/sheep_totals.csv')
    
    # population
    pop_admin, pop_country = _clean_cols('../../data/admin_reductions/pop_totals.csv')

    # gdp
    gdp_admin, gdp_country = _clean_cols('../../data/admin_reductions/gdp_totals.csv')

    # ghsl
    area_admin, area_country = _clean_cols('../../data/admin_reductions/area_total.csv')
    built_surface_admin, built_surface_country = _clean_cols('../../data/admin_reductions/built_surface.csv')
    built_volume_total_admin, built_volume_total_country = _clean_cols('../../data/admin_reductions/built_volume_total.csv')
    built_volume_nres_admin, built_volume_nres_country = _clean_cols('../../data/admin_reductions/built_volume_nres.csv')
    
    # combine all
    reductions_admin = barley_a_admin.merge(maize_a_admin, how='left').merge(millet_a_admin, how='left').merge(rice_a_admin, how='left').merge(
        sorghum_a_admin, how='left').merge(wheat_a_admin, how='left').merge(other_a_admin, how='left').merge(
        barley_p_admin, how='left').merge(maize_p_admin, how='left').merge(millet_p_admin, how='left').merge(rice_p_admin, how='left').merge(
        sorghum_p_admin, how='left').merge(wheat_p_admin, how='left').merge(other_p_admin, how='left').merge(
        buffaloes_admin, how='left').merge(cattle_admin, how='left').merge(chickens_admin, how='left').merge(ducks_admin, how='left').merge(
        goats_admin, how='left').merge(horses_admin, how='left').merge(pigs_admin, how='left').merge(sheep_admin, how='left').merge(
        pop_admin, how='left').merge(gdp_admin, how='left').merge(area_admin, how='left').merge(built_surface_admin, how='left').merge(
        built_volume_total_admin, how='left').merge(built_volume_nres_admin, how='left')

    reductions_country = barley_a_country.merge(maize_a_country, how='left').merge(millet_a_country, how='left').merge(rice_a_country, how='left').merge(
        sorghum_a_country, how='left').merge(wheat_a_country, how='left').merge(other_a_country, how='left').merge(
        barley_p_country, how='left').merge(maize_p_country, how='left').merge(millet_p_country, how='left').merge(rice_p_country, how='left').merge(
        sorghum_p_country, how='left').merge(wheat_p_country, how='left').merge(other_p_country, how='left').merge(
        buffaloes_country, how='left').merge(cattle_country, how='left').merge(chickens_country, how='left').merge(ducks_country, how='left').merge(
        goats_country, how='left').merge(horses_country, how='left').merge(pigs_country, how='left').merge(sheep_country, how='left').merge(
        pop_country, how='left').merge(gdp_country, how='left').merge(area_country, how='left').merge(built_surface_country, how='left').merge(
        built_volume_total_country, how='left').merge(built_volume_nres_country, how='left')

    return  reductions_admin, reductions_country

def get_fao_prod(crop, years=[2017, 2018, 2019, 2020, 2021]):

    excl = ['COK', 'FRO', 'GLP', 'GUF', 'KIR', 'MTQ', 'REU', 'TKL']

    fao_prod = pd.concat([pd.read_csv(f"../../data/FAO_prod_mat/prod_matrix_{crop}_{year}.csv") for year in years], axis=0, ignore_index=True)
    fao_prod = fao_prod.groupby('iso3').mean().reset_index()
    fao_prod = fao_prod[(~fao_prod['iso3'].isin(excl))]

    df = fao_prod[['iso3', 'prod']].rename(columns={'prod': f'{crop}_prod'})

    return df

# transport data and import tariffs
def get_transport_data(reductions_admin):
    # 219 countries (both from and to), 3642 from admins, 3641 to admins [overall 216 countries, 3642 admins]
    transport_admin = pd.read_parquet('../../data/transport_data/global_lowest_transport_cost.parquet')
    transport_admin = transport_admin[['from_id', 'to_id', 'from_iso3', 'to_iso3', 'freight_USD_t', 'transport_USD_t', 'time_h', 'distance_km',
                           'border_USD_t', 'mode', 'trade_USD_t', 'customs_cost']]

    # adding the within-country transport files
    files = os.listdir('../../data/transport_data/Country_admin_transport/road_rail/')
    countries = list(set([f.split('.')[0].split('_')[-1] for f in files]))
    
    df_list = []

    for c in countries:
        
        transport_files = [f for f in files if c in f]
        transport_country = pd.concat([pd.read_parquet(f"../../data/transport_data/Country_admin_transport/road_rail/{f}") for f in transport_files])
        transport_country['to_id'] = transport_country.apply(lambda row: row['to_id_edge'].split('-')[0], axis=1)
        
        transport_country = transport_country[['from_id', 'to_id', 'from_iso3', 'to_iso3', 
               'transport_USD_t', 'time_h', 'distance_km', 'border_USD_t', 'mode']]
        transport_country['trade_USD_t'] = transport_country['transport_USD_t'] + transport_country['border_USD_t']
        transport_country['customs_cost'] = np.NaN
        transport_country['freight_USD_t'] = np.NaN
        
        transport_country = transport_country.sort_values(by=['from_id', 'to_id', 'transport_USD_t']).reset_index(drop=True)
        transport_country = transport_country.drop_duplicates(subset=['from_id', 'to_id'], keep='first')
        
        df_list.append(transport_country)

    transport_admin_country = pd.concat(df_list, ignore_index=True)
    
    maritime_incl_bulk = pd.read_parquet('../../data/transport_data/Country_admin_transport/maritime/domestic_maritime_transport_including_bulk.parquet')
    maritime_no_bulk = pd.read_parquet('../../data/transport_data/Country_admin_transport/maritime/domestic_maritime_transport_no_bulk.parquet')
    maritime_country = pd.concat([maritime_incl_bulk, maritime_no_bulk], ignore_index=True).drop_duplicates()
    maritime_country = maritime_country.sort_values(by=['from_id', 'to_id', 'transport_USD_t']).reset_index(drop=True)
    maritime_country = maritime_country.drop_duplicates(subset=['from_id', 'to_id'], keep='first')
    maritime_country = maritime_country[['from_id', 'to_id', 'from_iso3', 'to_iso3', 
               'transport_USD_t', 'time_h', 'distance_km', 'mode']]
    maritime_country['border_USD_t'] = 0
    maritime_country['trade_USD_t'] = maritime_country['transport_USD_t'] + maritime_country['border_USD_t']
    maritime_country['customs_cost'] = np.NaN
    maritime_country['freight_USD_t'] = np.NaN
    transport_admin_country = pd.concat([transport_admin_country, maritime_country], ignore_index=True)
    transport_admin_country = transport_admin_country.sort_values(by=['from_id', 'to_id', 'transport_USD_t']).reset_index(drop=True)
    transport_admin_country = transport_admin_country.drop_duplicates(subset=['from_id', 'to_id'], keep='first')
    
    transport_admin = pd.concat([transport_admin_country, transport_admin], ignore_index=True)
    transport_admin = transport_admin.sort_values(by=['from_id', 'to_id', 'transport_USD_t']).reset_index(drop=True)
    transport_admin = transport_admin.drop_duplicates(subset=['from_id', 'to_id'], keep='first').reset_index(drop=True)

    # getting country level transport data through aggregations weighted by production in exporting region and population+livestock in importing region
    transport_admin = transport_admin.merge(reductions_admin[['ID', 'barley_production', 'maize_production', 'millet_production', 'rice_production',
           'sorghum_production', 'wheat_production', 'other_cereals_production']], left_on='from_id', right_on='ID', how='left').drop('ID', axis=1)
    transport_admin = transport_admin.merge(reductions_admin[['ID', 'buffaloes_totals', 'cattle_totals', 'chickens_totals', 'ducks_totals', 'goats_totals',
           'horses_totals', 'pigs_totals', 'sheep_totals', 'pop_totals']], left_on='to_id', right_on='ID', how='left').drop('ID', axis=1)
    
    transport_admin['pop_livestock'] = transport_admin['pop_totals'] + transport_admin['buffaloes_totals'] + transport_admin['cattle_totals'] \
        + transport_admin['chickens_totals'] + transport_admin['ducks_totals'] + transport_admin['goats_totals'] + transport_admin['horses_totals'] \
        + transport_admin['pigs_totals'] + transport_admin['sheep_totals']
    transport_admin['production'] = transport_admin['barley_production'] + transport_admin['maize_production'] + transport_admin['millet_production'] \
        + transport_admin['rice_production'] + transport_admin['sorghum_production'] + transport_admin['wheat_production'] \
        + transport_admin['other_cereals_production'] 
    transport_admin['weight'] = transport_admin['pop_livestock'] * transport_admin['production']
    
    transport_admin = transport_admin.drop(['production', 'barley_production', 'maize_production', 'millet_production', 
                                'rice_production', 'sorghum_production', 'wheat_production', 'other_cereals_production', 'buffaloes_totals', 
                                'cattle_totals', 'chickens_totals', 'ducks_totals', 'goats_totals',
                                'horses_totals', 'pigs_totals', 'sheep_totals', 'pop_totals', 'pop_livestock'], axis=1)
    
    transport_admin.loc[transport_admin['weight']==0, 'weight'] = transport_admin[transport_admin['weight']>0]['weight'].min()
    transport_admin.loc[transport_admin['customs_cost'].isna(), 'customs_cost'] = 0

    wm = lambda x: np.average(x, weights=transport_admin.loc[x.index, "weight"])
    transport_country = transport_admin.groupby(['from_iso3', 'to_iso3']).agg({'freight_USD_t': wm,
                                                                         'transport_USD_t': wm,
                                                                         'time_h': wm,
                                                                         'distance_km': wm,
                                                                         'border_USD_t': wm,
                                                                         'trade_USD_t': wm,
                                                                         'customs_cost': wm,
                                                                         'mode': lambda x: pd.Series.mode(x).iloc[0]})

    
    transport_country = transport_country.reset_index()
    transport_country = transport_country[transport_country['from_iso3']!=transport_country['to_iso3']].reset_index(drop=True)
    
    transport_admin = transport_admin.drop('weight', axis=1)

    # import tariffs data
    mm = pd.read_csv('../../data/Import_tariffs/mm2019.csv', sep=';')
    mm = mm[(mm['hs6_rev2017']>=100000) & (mm['hs6_rev2017']<110000)].reset_index(drop=True)
    mm = mm.groupby(['importer', 'exporter']).mean()[['Pref_Applied_AVE']].reset_index()
    mm_country = pd.read_csv('../../data/Import_tariffs/Countries.csv')
    mm = mm.merge(mm_country, right_on='code', left_on='importer').drop('code', axis=1).rename(columns={'iso3': 'to_iso3'})
    mm = mm.merge(mm_country, right_on='code', left_on='exporter').drop('code', axis=1).rename(columns={'iso3': 'from_iso3'})
    mm = mm.drop(['importer', 'exporter'], axis=1)

    # combining transport and tariffs data (replacing nans with zeros)
    transport_admin = transport_admin.merge(mm, how='left')
    transport_admin.loc[transport_admin['Pref_Applied_AVE'].isna(), 'Pref_Applied_AVE'] = 0
    transport_admin['Pref_Applied_AVE'] = transport_admin['Pref_Applied_AVE'] * 0.8 * (transport_admin['transport_USD_t'] + transport_admin['border_USD_t'])
    
    transport_country = transport_country.merge(mm, how='left')
    transport_country.loc[transport_country['Pref_Applied_AVE'].isna(), 'Pref_Applied_AVE'] = 0
    transport_country['Pref_Applied_AVE'] = transport_country['Pref_Applied_AVE'] * 0.8 * (transport_country['transport_USD_t'] + transport_country['border_USD_t'])
    
    return transport_admin, transport_country

# merge data to get master dataframe
def merge_data(log_x=True, log_y=True, trade_thresh=10, get_trade_only=False):
    # trade data, replacing values less than thresh with zero
    trade = get_trade_data(log_y, trade_thresh)
    print('trade data done')

    if get_trade_only:
        return trade
    
    # x variables (admin reductions and transport)
    reductions_admin, reductions_country = get_admin_reductions()
    admin_cols = reductions_admin.columns
    country_cols = reductions_country.columns

    # correcting production numbers
    crops = ['Wheat', 'Maize (corn)', 'Rye', 'Barley', 'Oats', 'Sorghum', 
             'Rice, paddy (rice milled equivalent)', 'Buckwheat', 
             'Millet', 'Quinoa', 'Canary seed', 'Fonio', 'Mixed grain', 'Triticale', 'Cereals n.e.c.']
    df_list = []
    for crop in crops:
        df = get_fao_prod(crop)
        df_list.append(df)
    dfs = [df.set_index('iso3') for df in df_list]
    prod_df = pd.concat(dfs, axis=1).reset_index()
    prod_df = prod_df.rename(columns={
        'Maize (corn)_prod': 'Maize_prod',
        'Rice, paddy (rice milled equivalent)_prod': 'Rice_prod',
        'Cereals n.e.c._prod': 'other_cereals_prod'
    })

    reductions_country = reductions_country.merge(prod_df)
    # barley
    reductions_country['barley_factor'] = reductions_country['Barley_prod'] / (reductions_country['barley_production'])
    reductions_country['barley_production'] = reductions_country['Barley_prod']
    
    # maize
    reductions_country['maize_factor'] = reductions_country['Maize_prod'] / (reductions_country['maize_production'])
    reductions_country['maize_production'] = reductions_country['Maize_prod']
    
    # millet
    reductions_country['millet_factor'] = reductions_country['Millet_prod'] / (reductions_country['millet_production'])
    reductions_country['millet_production'] = reductions_country['Millet_prod']
    
    # rice
    reductions_country['rice_factor'] = reductions_country['Rice_prod'] / (reductions_country['rice_production'])
    reductions_country['rice_production'] = reductions_country['Rice_prod']
    
    # sorghum
    reductions_country['sorghum_factor'] = reductions_country['Sorghum_prod'] / (reductions_country['sorghum_production'])
    reductions_country['sorghum_production'] = reductions_country['Sorghum_prod']
    
    # wheat
    reductions_country['wheat_factor'] = reductions_country['Wheat_prod'] / (reductions_country['wheat_production'])
    reductions_country['wheat_production'] = reductions_country['Wheat_prod']
    
    # other
    reductions_country['other_cereals_factor'] = (reductions_country['Rye_prod'] + reductions_country['Oats_prod'] + reductions_country['Buckwheat_prod'] + reductions_country['Quinoa_prod']  \
                                     + reductions_country['Canary seed_prod'] + reductions_country['Fonio_prod'] + reductions_country['Mixed grain_prod'] + reductions_country['Triticale_prod']  \
                                     + reductions_country['other_cereals_prod']) / (reductions_country['other_cereals_production'])
    reductions_country['other_cereals_production'] = reductions_country['Rye_prod'] + reductions_country['Oats_prod'] + reductions_country['Buckwheat_prod'] \
    + reductions_country['Canary seed_prod'] + reductions_country['Fonio_prod'] + reductions_country['Mixed grain_prod'] + reductions_country['Triticale_prod']  \
    + reductions_country['Quinoa_prod'] + reductions_country['other_cereals_prod']
    
    reductions_country = reductions_country.fillna(0)
    
    reductions_admin = reductions_admin.merge(reductions_country[['iso3', 'barley_factor', 'maize_factor', 'millet_factor', 
                                                'rice_factor', 'sorghum_factor', 'wheat_factor', 'other_cereals_factor',
                                                'barley_production', 'maize_production', 'millet_production',
                                                'rice_production', 'sorghum_production', 'wheat_production',
                                                'other_cereals_production']].rename(columns={
        'barley_production': 'barley_production_country', 'maize_production': 'maize_production_country', 'millet_production': 'millet_production_country',
        'rice_production': 'rice_production_country', 'sorghum_production': 'sorghum_production_country', 'wheat_production': 'wheat_production_country',
        'other_cereals_production': 'other_cereals_production_country'}))
    reductions_admin = reductions_admin.merge(reductions_admin.groupby('iso3')['ID'].count().reset_index().rename(columns={'ID': 'num_admins'}))
    
    for crop in ['barley', 'maize', 'millet', 'rice', 'sorghum', 'wheat', 'other_cereals']:
        reductions_admin[f'{crop}_production'] = (reductions_admin[f'{crop}_production']) * reductions_admin[f'{crop}_factor']
        reductions_admin.loc[~np.isfinite(reductions_admin[f'{crop}_factor']), f'{crop}_production'
            ] = reductions_admin[~np.isfinite(reductions_admin[f'{crop}_factor'])][f'{crop}_production_country'] / reductions_admin[~np.isfinite(reductions_admin[f'{crop}_factor'])]['num_admins']
    
    reductions_admin = reductions_admin[admin_cols.tolist()]
    reductions_country = reductions_country[country_cols.tolist()]
    print('admin reductions done')


    transport_admin, transport_country = get_transport_data(reductions_admin)
    print('transport data done')

    # log x variables for both transport and admin reductions
    if log_x:
        for col in ['freight_USD_t', 'transport_USD_t', 'time_h', 'distance_km', 'border_USD_t', 'customs_cost', 'Pref_Applied_AVE']:
            print(f'log {col}')
            transport_admin[col] = transport_admin.apply(lambda row: np.log(row[col]+1), axis=1)
            transport_country[col] = transport_country.apply(lambda row: np.log(row[col]+1), axis=1)

        for col in ['barley_areas', 'maize_areas', 'millet_areas', 'rice_areas', 'sorghum_areas', 'wheat_areas', 'other_cereals_areas', 
                    'barley_production', 'maize_production', 'millet_production', 'rice_production', 'sorghum_production', 'wheat_production', 'other_cereals_production', 
                    'buffaloes_totals', 'cattle_totals', 'chickens_totals', 'ducks_totals', 'goats_totals', 'horses_totals', 'pigs_totals', 'sheep_totals', 'pop_totals',
                    'gdp_totals', 'area_total', 'built_surface', 'built_volume_total', 'built_volume_nres']:
            print(f'log {col}')
            reductions_admin[col] = reductions_admin.apply(lambda row: np.log(row[col]+1), axis=1)
            reductions_country[col] = reductions_country.apply(lambda row: np.log(row[col]+1), axis=1)

    # categorical variable - mode
    transport_country.loc[~transport_country['mode'].isin(['maritime', 'rail', 'road']), 'mode'] = transport_country[
        ~transport_country['mode'].isin(['maritime', 'rail', 'road'])]['mode'].str[0]
    transport_country.loc[transport_country['mode']=='maritime', 'mode'] = 0
    transport_country.loc[transport_country['mode']=='rail', 'mode'] = 1
    transport_country.loc[transport_country['mode']=='road', 'mode'] = 2
    transport_country = transport_country.drop('trade_USD_t', axis=1)
    
    transport_admin.loc[transport_admin['mode']=='maritime', 'mode'] = 0
    transport_admin.loc[transport_admin['mode']=='rail', 'mode'] = 1
    transport_admin.loc[transport_admin['mode']=='road', 'mode'] = 2
    transport_admin = transport_admin.drop('trade_USD_t', axis=1)

    # getting country level data by merging everything (trade, transport, admin reductions)
    df_country = transport_country.merge(trade) # this would remove 'FRO', 'COK', 'KIR', 'TKL'. 'TKL' doesn't exist in admin files itself. The other three have no international connections in the admin level transport connections file [FRO does have domestic connections in transport data, but these would also be removed below so that countries are consistent across spatial scales.]
    # overall left with 195 countries, 3535 from admins, 3534 to admins, 12,414,167 links
    # still some issues in the transport data. several domestic links missing, some international ones also missing (maybe that is not too important?)
    # admins in the 195 countries that don't have incoming links from anywhere (domestic or international) - 'BTN.12_1', 'MUS.10_1', 'NZL.10_1', 'PSE.1_1', 'PYF.3_1', 'STP.1_1'
    # ideally there should be 3540 from and to admins, 12,528,060 links
    df_country = df_country.merge(reductions_country, left_on='from_iso3', right_on='iso3').drop(['iso3'], axis=1).rename(columns={
        'barley_areas': 'from_barley_area', 
        'maize_areas': 'from_maize_area', 
        'millet_areas': 'from_millet_area',
        'rice_areas': 'from_rice_area', 
        'sorghum_areas': 'from_sorghum_area', 
        'wheat_areas': 'from_wheat_area', 
        'other_cereals_areas': 'from_other_cereals_area',
        'barley_production': 'from_barley_production', 
        'maize_production': 'from_maize_production', 
        'millet_production': 'from_millet_production',
        'rice_production': 'from_rice_production', 
        'sorghum_production': 'from_sorghum_production', 
        'wheat_production': 'from_wheat_production', 
        'other_cereals_production': 'from_other_cereals_production',
        'buffaloes_totals': 'from_buffaloes', 
        'cattle_totals': 'from_cattle', 
        'chickens_totals': 'from_chickens',
        'ducks_totals': 'from_ducks', 
        'goats_totals': 'from_goats', 
        'horses_totals': 'from_horses', 
        'pigs_totals': 'from_pigs',
        'sheep_totals': 'from_sheep', 
        'pop_totals': 'from_pop',
        'gdp_totals': 'from_gdp',
        'area_total': 'from_area', 
        'built_surface': 'from_built_surface', 
        'built_volume_total': 'from_built_volume_total', 
        'built_volume_nres': 'from_built_volume_nres'})
    
    df_country = df_country.merge(reductions_country, left_on='to_iso3', right_on='iso3').drop(['iso3'], axis=1).rename(columns={
        'barley_areas': 'to_barley_area', 
        'maize_areas': 'to_maize_area', 
        'millet_areas': 'to_millet_area',
        'rice_areas': 'to_rice_area', 
        'sorghum_areas': 'to_sorghum_area', 
        'wheat_areas': 'to_wheat_area', 
        'other_cereals_areas': 'to_other_cereals_area',
        'barley_production': 'to_barley_production', 
        'maize_production': 'to_maize_production', 
        'millet_production': 'to_millet_production',
        'rice_production': 'to_rice_production', 
        'sorghum_production': 'to_sorghum_production', 
        'wheat_production': 'to_wheat_production', 
        'other_cereals_production': 'to_other_cereals_production', 
        'buffaloes_totals': 'to_buffaloes', 
        'cattle_totals': 'to_cattle', 
        'chickens_totals': 'to_chickens',
        'ducks_totals': 'to_ducks', 
        'goats_totals': 'to_goats', 
        'horses_totals': 'to_horses', 
        'pigs_totals': 'to_pigs',
        'sheep_totals': 'to_sheep', 
        'pop_totals': 'to_pop',
        'gdp_totals': 'to_gdp',
        'area_total': 'to_area', 
        'built_surface': 'to_built_surface', 
        'built_volume_total': 'to_built_volume_total', 
        'built_volume_nres': 'to_built_volume_nres'})
    # remove these 4 area codes ('GLP', 'GUF', 'MTQ', 'REU') from data
    # [wouldn't make a difference, since they all have 0 production and consumption]
    df_country = df_country[(~df_country['from_iso3'].isin(['GLP', 'GUF', 'MTQ', 'REU'])) 
    & (~df_country['to_iso3'].isin(['GLP', 'GUF', 'MTQ', 'REU']))].reset_index(drop=True)
    print('df_country done')

    # list of countries in data (will keep only these in the admin level data)
    iso3_codes = list(set(df_country['from_iso3'].values.tolist() + df_country['to_iso3'].values.tolist()))

    # getting admin level data by merging everything (transport and admin reductions)
    df_admin = transport_admin.merge(reductions_admin, left_on='from_id', right_on='ID').drop(['ID', 'iso3'], axis=1).rename(columns={
        'barley_areas': 'from_barley_area', 
        'maize_areas': 'from_maize_area', 
        'millet_areas': 'from_millet_area',
        'rice_areas': 'from_rice_area', 
        'sorghum_areas': 'from_sorghum_area', 
        'wheat_areas': 'from_wheat_area', 
        'other_cereals_areas': 'from_other_cereals_area',
        'barley_production': 'from_barley_production', 
        'maize_production': 'from_maize_production', 
        'millet_production': 'from_millet_production',
        'rice_production': 'from_rice_production', 
        'sorghum_production': 'from_sorghum_production', 
        'wheat_production': 'from_wheat_production', 
        'other_cereals_production': 'from_other_cereals_production',
        'buffaloes_totals': 'from_buffaloes', 
        'cattle_totals': 'from_cattle', 
        'chickens_totals': 'from_chickens',
        'ducks_totals': 'from_ducks', 
        'goats_totals': 'from_goats', 
        'horses_totals': 'from_horses', 
        'pigs_totals': 'from_pigs',
        'sheep_totals': 'from_sheep', 
        'pop_totals': 'from_pop',
        'gdp_totals': 'from_gdp',
        'area_total': 'from_area', 
        'built_surface': 'from_built_surface', 
        'built_volume_total': 'from_built_volume_total', 
        'built_volume_nres': 'from_built_volume_nres'})
    
    df_admin = df_admin.merge(reductions_admin, left_on='to_id', right_on='ID').drop(['ID', 'iso3'], axis=1).rename(columns={
        'barley_areas': 'to_barley_area', 
        'maize_areas': 'to_maize_area', 
        'millet_areas': 'to_millet_area',
        'rice_areas': 'to_rice_area', 
        'sorghum_areas': 'to_sorghum_area', 
        'wheat_areas': 'to_wheat_area', 
        'other_cereals_areas': 'to_other_cereals_area',
        'barley_production': 'to_barley_production', 
        'maize_production': 'to_maize_production', 
        'millet_production': 'to_millet_production',
        'rice_production': 'to_rice_production', 
        'sorghum_production': 'to_sorghum_production', 
        'wheat_production': 'to_wheat_production', 
        'other_cereals_production': 'to_other_cereals_production', 
        'buffaloes_totals': 'to_buffaloes', 
        'cattle_totals': 'to_cattle', 
        'chickens_totals': 'to_chickens',
        'ducks_totals': 'to_ducks', 
        'goats_totals': 'to_goats', 
        'horses_totals': 'to_horses', 
        'pigs_totals': 'to_pigs',
        'sheep_totals': 'to_sheep', 
        'pop_totals': 'to_pop',
        'gdp_totals': 'to_gdp',
        'area_total': 'to_area', 
        'built_surface': 'to_built_surface', 
        'built_volume_total': 'to_built_volume_total', 
        'built_volume_nres': 'to_built_volume_nres'})

    df_admin = df_admin[(df_admin['from_iso3'].isin(iso3_codes)) & (df_admin['to_iso3'].isin(iso3_codes))].reset_index(drop=True)
    print('df_admin done')

    # add region names
    regions = pd.read_excel('../../OPSIS/Data/Country_group/Country_classification_UNSD.xlsx')
    regions.loc[regions['Sub-region Name'].isin(['Northern America', 'Latin America and the Caribbean']),
                'Region Name'] = regions[regions['Sub-region Name'].isin(['Northern America', 'Latin America and the Caribbean'])]['Sub-region Name']
    regions['Region Name'] = regions['Region Name'].replace(['Africa', 'Asia', 'Europe', 'Latin America and the Caribbean', 
                                                             'Northern America', 'Oceania'], [0, 1, 2, 3, 4, 5])
    df_country = df_country.merge(regions[['iso3', 'Region Name']], how='left', left_on='from_iso3', right_on='iso3').drop('iso3', axis=1).rename(
        columns={'Region Name': 'from_region'})
    df_country = df_country.merge(regions[['iso3', 'Region Name']], how='left', left_on='to_iso3', right_on='iso3').drop('iso3', axis=1).rename(
        columns={'Region Name': 'to_region'})
    
    df_admin = df_admin.merge(regions[['iso3', 'Region Name']], how='left', left_on='from_iso3', right_on='iso3').drop('iso3', axis=1).rename(
        columns={'Region Name': 'from_region'})
    df_admin = df_admin.merge(regions[['iso3', 'Region Name']], how='left', left_on='to_iso3', right_on='iso3').drop('iso3', axis=1).rename(
        columns={'Region Name': 'to_region'})
    
    return df_country, df_admin

def merge_data_ind(log_x=True, log_y=True, trade_thresh=10):
    # trade data, replacing values less than thresh with zero
    ind_wheat = pd.concat([pd.read_excel('../../data/other_data/India domestic/trade_files.xlsx', sheet_name=i) for i in [1,4]], 
                         axis=0, ignore_index=True)
    ind_rice = pd.concat([pd.read_excel('../../data/other_data/India domestic/trade_files.xlsx', sheet_name=i) for i in [2,3]], 
                         axis=0, ignore_index=True)
    # to account for milling
    ind_rice[[col for col in ind_rice.columns if 'Exporter ↓; Importer →' not in col]] = ind_rice[[col for col in ind_rice.columns if 'Exporter ↓; Importer →' not in col]] * 0.7 
    ind_maize = pd.concat([pd.read_excel('../../data/other_data/India domestic/trade_files.xlsx', sheet_name=i) for i in [5]], 
                         axis=0, ignore_index=True)
    ind_other = pd.concat([pd.read_excel('../../data/other_data/India domestic/trade_files.xlsx', sheet_name=i) for i in [6, 7, 8, 9]], 
                         axis=0, ignore_index=True)


    df_list = []
    for df in [ind_wheat, ind_rice, ind_maize, ind_other]:        
        df = df.groupby('Exporter ↓; Importer →').sum().reset_index()
        df = df.melt(id_vars=['Exporter ↓; Importer →'], value_vars=df['Exporter ↓; Importer →'].values.tolist())
        df = df.rename(columns={'Exporter ↓; Importer →': 'from_admin', 'variable': 'to_admin'})
        df.loc[df['from_admin']=='Chattisgarh', 'from_admin'] = 'Chhattisgarh'
        df.loc[df['to_admin']=='Chattisgarh', 'to_admin'] = 'Chhattisgarh'
        df.loc[df['from_admin']=='Delhi', 'from_admin'] = 'NCT of Delhi'
        df.loc[df['to_admin']=='Delhi', 'to_admin'] = 'NCT of Delhi'
        df.loc[df['from_admin']=='Uttaranchal', 'from_admin'] = 'Uttarakhand'
        df.loc[df['to_admin']=='Uttaranchal', 'to_admin'] = 'Uttarakhand'
        df = df[df['from_admin']!=df['to_admin']]
        df.loc[df["value"]<trade_thresh, "value"] = 0
        df["supply_exists"] = 0
        df.loc[df["value"]>0, "supply_exists"] = 1
        df_list.append(df)

    ind_wheat = df_list[0].rename(columns={'value': 'supply_wheat', 'supply_exists': 'supply_wheat_exists'})
    ind_rice = df_list[1].rename(columns={'value': 'supply_rice', 'supply_exists': 'supply_rice_exists'})
    ind_maize = df_list[2].rename(columns={'value': 'supply_maize', 'supply_exists': 'supply_maize_exists'})
    ind_other = df_list[3].rename(columns={'value': 'supply_other_cereals', 'supply_exists': 'supply_other_cereals_exists'})

    trade = ind_wheat.merge(ind_rice).merge(ind_maize).merge(ind_other)
    trade = trade.merge(admin[admin['iso3']=='IND'][['iso3', 'ID', 'admin_name']], left_on='from_admin', right_on='admin_name').drop(
        ['from_admin', 'admin_name'], axis=1).rename(columns={'iso3': 'from_iso3', 'ID': 'from_id'})
    trade = trade.merge(admin[admin['iso3']=='IND'][['iso3', 'ID', 'admin_name']], left_on='to_admin', right_on='admin_name').drop(
        ['to_admin', 'admin_name'], axis=1).rename(columns={'iso3': 'to_iso3', 'ID': 'to_id'})
    
    trade['supply_cereals_all'] = trade[['supply_wheat', 'supply_rice', 'supply_maize', 'supply_other_cereals']].sum(axis=1)
    trade["supply_cereals_all_exists"] = 0
    trade.loc[trade["supply_cereals_all"]>0, "supply_cereals_all_exists"] = 1
    
    if log_y:
        for col in ['supply_wheat', 'supply_rice', 'supply_maize', 'supply_other_cereals', 'supply_cereals_all']:
            trade.loc[trade[f"{col}_exists"]==1, col] = trade[trade[f"{col}_exists"]==1].apply(lambda row: np.log(row[col]), axis=1)

    print('trade data done')
     
    # x variables (admin reductions and transport)
    reductions_admin, reductions_country = get_admin_reductions()
    admin_cols = reductions_admin.columns
    country_cols = reductions_country.columns

    reductions_admin = reductions_admin[reductions_admin['iso3']=='IND']

    # correcting production numbers
    crops = ['Wheat', 'Maize (corn)', 'Rye', 'Barley', 'Oats', 'Sorghum', 
             'Rice, paddy (rice milled equivalent)', 'Buckwheat', 
             'Millet', 'Quinoa', 'Canary seed', 'Fonio', 'Mixed grain', 'Triticale', 'Cereals n.e.c.']
    df_list = []
    for crop in crops:
        df = get_fao_prod(crop, years=[2011, 2012])
        df_list.append(df)
    dfs = [df.set_index('iso3') for df in df_list]
    prod_df = pd.concat(dfs, axis=1).reset_index()
    prod_df = prod_df.rename(columns={
        'Maize (corn)_prod': 'Maize_prod',
        'Rice, paddy (rice milled equivalent)_prod': 'Rice_prod',
        'Cereals n.e.c._prod': 'other_cereals_prod'
    })

    reductions_country = reductions_country.merge(prod_df)
    # barley
    reductions_country['barley_factor'] = reductions_country['Barley_prod'] / (reductions_country['barley_production'])
    reductions_country['barley_production'] = reductions_country['Barley_prod']
    
    # maize
    reductions_country['maize_factor'] = reductions_country['Maize_prod'] / (reductions_country['maize_production'])
    reductions_country['maize_production'] = reductions_country['Maize_prod']
    
    # millet
    reductions_country['millet_factor'] = reductions_country['Millet_prod'] / (reductions_country['millet_production'])
    reductions_country['millet_production'] = reductions_country['Millet_prod']
    
    # rice
    reductions_country['rice_factor'] = reductions_country['Rice_prod'] / (reductions_country['rice_production'])
    reductions_country['rice_production'] = reductions_country['Rice_prod']
    
    # sorghum
    reductions_country['sorghum_factor'] = reductions_country['Sorghum_prod'] / (reductions_country['sorghum_production'])
    reductions_country['sorghum_production'] = reductions_country['Sorghum_prod']
    
    # wheat
    reductions_country['wheat_factor'] = reductions_country['Wheat_prod'] / (reductions_country['wheat_production'])
    reductions_country['wheat_production'] = reductions_country['Wheat_prod']
    
    # other
    reductions_country['other_cereals_factor'] = (reductions_country['Rye_prod'] + reductions_country['Oats_prod'] + reductions_country['Buckwheat_prod'] + reductions_country['Quinoa_prod']  \
                                     + reductions_country['Canary seed_prod'] + reductions_country['Fonio_prod'] + reductions_country['Mixed grain_prod'] + reductions_country['Triticale_prod']  \
                                     + reductions_country['other_cereals_prod']) / (reductions_country['other_cereals_production'])
    reductions_country['other_cereals_production'] = reductions_country['Rye_prod'] + reductions_country['Oats_prod'] + reductions_country['Buckwheat_prod'] \
    + reductions_country['Canary seed_prod'] + reductions_country['Fonio_prod'] + reductions_country['Mixed grain_prod'] + reductions_country['Triticale_prod']  \
    + reductions_country['Quinoa_prod'] + reductions_country['other_cereals_prod']
    
    reductions_country = reductions_country.fillna(0)
    
    reductions_admin = reductions_admin.merge(reductions_country[['iso3', 'barley_factor', 'maize_factor', 'millet_factor', 
                                                'rice_factor', 'sorghum_factor', 'wheat_factor', 'other_cereals_factor',
                                                'barley_production', 'maize_production', 'millet_production',
                                                'rice_production', 'sorghum_production', 'wheat_production',
                                                'other_cereals_production']].rename(columns={
        'barley_production': 'barley_production_country', 'maize_production': 'maize_production_country', 'millet_production': 'millet_production_country',
        'rice_production': 'rice_production_country', 'sorghum_production': 'sorghum_production_country', 'wheat_production': 'wheat_production_country',
        'other_cereals_production': 'other_cereals_production_country'}))
    reductions_admin = reductions_admin.merge(reductions_admin.groupby('iso3')['ID'].count().reset_index().rename(columns={'ID': 'num_admins'}))
    
    for crop in ['barley', 'maize', 'millet', 'rice', 'sorghum', 'wheat', 'other_cereals']:
        reductions_admin[f'{crop}_production'] = (reductions_admin[f'{crop}_production']) * reductions_admin[f'{crop}_factor']
        reductions_admin.loc[~np.isfinite(reductions_admin[f'{crop}_factor']), f'{crop}_production'
            ] = reductions_admin[~np.isfinite(reductions_admin[f'{crop}_factor'])][f'{crop}_production_country'] / reductions_admin[~np.isfinite(reductions_admin[f'{crop}_factor'])]['num_admins']
    
    reductions_admin = reductions_admin[admin_cols.tolist()]
    print('admin reductions done')

    transport_admin, transport_country = get_transport_data(reductions_admin)
    transport_admin = transport_admin[(transport_admin['from_iso3']==transport_admin['to_iso3']) & (transport_admin['from_iso3']=='IND')]
    print('transport data done')

    # categorical variable - mode
    transport_admin.loc[transport_admin['mode']=='maritime', 'mode'] = 0
    transport_admin.loc[transport_admin['mode']=='rail', 'mode'] = 1
    transport_admin.loc[transport_admin['mode']=='road', 'mode'] = 2
    transport_admin = transport_admin.drop('trade_USD_t', axis=1)

    # merging andhra and telangana
    reductions_admin.loc[reductions_admin['ID']==admin[admin['admin_name']=='Telangana']['ID'].values[0], 
        'ID'] = admin[admin['admin_name']=='Andhra Pradesh']['ID'].values[0]
    reductions_admin = reductions_admin.groupby(['ID', 'iso3']).sum().reset_index()
    transport_admin.loc[transport_admin['from_id']==admin[admin['admin_name']=='Telangana']['ID'].values[0], 
        'from_id'] = admin[admin['admin_name']=='Andhra Pradesh']['ID'].values[0]
    transport_admin.loc[transport_admin['to_id']==admin[admin['admin_name']=='Telangana']['ID'].values[0], 
        'to_id'] = admin[admin['admin_name']=='Andhra Pradesh']['ID'].values[0]
    transport_admin = transport_admin[transport_admin['from_id']!=transport_admin['to_id']]
    transport_admin = transport_admin.groupby(['from_id', 'to_id']).agg({'freight_USD_t': 'mean',
                                                                         'transport_USD_t': 'mean',
                                                                         'time_h': 'mean',
                                                                         'distance_km': 'mean',
                                                                         'border_USD_t': 'mean',
                                                                         'customs_cost': 'mean',
                                                                         'Pref_Applied_AVE': 'mean',
                                                                         'mode': lambda x: pd.Series.mode(x).iloc[0]}).reset_index()

    # log x variables for both transport and admin reductions
    if log_x:
        for col in ['freight_USD_t', 'transport_USD_t', 'time_h', 'distance_km', 'border_USD_t', 'customs_cost', 'Pref_Applied_AVE']:
            print(f'log {col}')
            transport_admin[col] = transport_admin.apply(lambda row: np.log(row[col]+1), axis=1)
            
        for col in ['barley_areas', 'maize_areas', 'millet_areas', 'rice_areas', 'sorghum_areas', 'wheat_areas', 'other_cereals_areas', 
                    'barley_production', 'maize_production', 'millet_production', 'rice_production', 
                    'sorghum_production', 'wheat_production', 'other_cereals_production', 
                    'buffaloes_totals', 'cattle_totals', 'chickens_totals', 'ducks_totals', 'goats_totals', 
                    'horses_totals', 'pigs_totals', 'sheep_totals', 'pop_totals',
                    'gdp_totals', 'area_total', 'built_surface', 'built_volume_total', 'built_volume_nres']:
            print(f'log {col}')
            reductions_admin[col] = reductions_admin.apply(lambda row: np.log(row[col]+1), axis=1)

    # getting india admin level data by merging everything (trade, transport, admin reductions)
    df_admin = transport_admin.merge(trade)
    
    # getting admin level data by merging everything (transport and admin reductions)
    df_admin = df_admin.merge(reductions_admin, left_on='from_id', right_on='ID').drop(['ID', 'iso3'], axis=1).rename(columns={
        'barley_areas': 'from_barley_area', 
        'maize_areas': 'from_maize_area', 
        'millet_areas': 'from_millet_area',
        'rice_areas': 'from_rice_area', 
        'sorghum_areas': 'from_sorghum_area', 
        'wheat_areas': 'from_wheat_area', 
        'other_cereals_areas': 'from_other_cereals_area',
        'barley_production': 'from_barley_production', 
        'maize_production': 'from_maize_production', 
        'millet_production': 'from_millet_production',
        'rice_production': 'from_rice_production', 
        'sorghum_production': 'from_sorghum_production', 
        'wheat_production': 'from_wheat_production', 
        'other_cereals_production': 'from_other_cereals_production',
        'buffaloes_totals': 'from_buffaloes', 
        'cattle_totals': 'from_cattle', 
        'chickens_totals': 'from_chickens',
        'ducks_totals': 'from_ducks', 
        'goats_totals': 'from_goats', 
        'horses_totals': 'from_horses', 
        'pigs_totals': 'from_pigs',
        'sheep_totals': 'from_sheep', 
        'pop_totals': 'from_pop',
        'gdp_totals': 'from_gdp',
        'area_total': 'from_area', 
        'built_surface': 'from_built_surface', 
        'built_volume_total': 'from_built_volume_total', 
        'built_volume_nres': 'from_built_volume_nres'})
    
    df_admin = df_admin.merge(reductions_admin, left_on='to_id', right_on='ID').drop(['ID', 'iso3'], axis=1).rename(columns={
        'barley_areas': 'to_barley_area', 
        'maize_areas': 'to_maize_area', 
        'millet_areas': 'to_millet_area',
        'rice_areas': 'to_rice_area', 
        'sorghum_areas': 'to_sorghum_area', 
        'wheat_areas': 'to_wheat_area', 
        'other_cereals_areas': 'to_other_cereals_area',
        'barley_production': 'to_barley_production', 
        'maize_production': 'to_maize_production', 
        'millet_production': 'to_millet_production',
        'rice_production': 'to_rice_production', 
        'sorghum_production': 'to_sorghum_production', 
        'wheat_production': 'to_wheat_production', 
        'other_cereals_production': 'to_other_cereals_production', 
        'buffaloes_totals': 'to_buffaloes', 
        'cattle_totals': 'to_cattle', 
        'chickens_totals': 'to_chickens',
        'ducks_totals': 'to_ducks', 
        'goats_totals': 'to_goats', 
        'horses_totals': 'to_horses', 
        'pigs_totals': 'to_pigs',
        'sheep_totals': 'to_sheep', 
        'pop_totals': 'to_pop',
        'gdp_totals': 'to_gdp',
        'area_total': 'to_area', 
        'built_surface': 'to_built_surface', 
        'built_volume_total': 'to_built_volume_total', 
        'built_volume_nres': 'to_built_volume_nres'})

    print('df_admin done')

    # add region names
    regions = pd.read_excel('../../OPSIS/Data/Country_group/Country_classification_UNSD.xlsx')
    regions.loc[regions['Sub-region Name'].isin(['Northern America', 'Latin America and the Caribbean']),
                'Region Name'] = regions[regions['Sub-region Name'].isin(['Northern America', 'Latin America and the Caribbean'])]['Sub-region Name']
    regions['Region Name'] = regions['Region Name'].replace(['Africa', 'Asia', 'Europe', 'Latin America and the Caribbean', 
                                                             'Northern America', 'Oceania'], [0, 1, 2, 3, 4, 5])
    df_admin = df_admin.merge(regions[['iso3', 'Region Name']], how='left', left_on='from_iso3', right_on='iso3').drop('iso3', axis=1).rename(
        columns={'Region Name': 'from_region'})
    df_admin = df_admin.merge(regions[['iso3', 'Region Name']], how='left', left_on='to_iso3', right_on='iso3').drop('iso3', axis=1).rename(
        columns={'Region Name': 'to_region'})
    
    return df_admin


def merge_data_us_china(log_x=True, log_y=True, trade_thresh=10):
    # trade data, replacing values less than thresh with zero
    df = pd.read_csv('../../data/other_data/Pandit_us_china/us-china_bilateral_trade_data/bilateral_link_level_state.csv')

    df = df[df['SCTG2_num']=='2'][['us', 'china', 'mass', 'us_exporter']]
    df.loc[df['china']=='Neimeng', 'china'] = 'Nei Mongol'
    df.loc[df['china']=='Ningxia', 'china'] = 'Ningxia Hui'
    df.loc[df['china']=='Tibet', 'china'] = 'Xizang'
    df.loc[df['china']=='Xinjiang', 'china'] = 'Xinjiang Uygur'
    df.loc[df['us']=='Dist of Columbia', 'us'] = 'District of Columbia'
    df['mass'] = df['mass'] /1e3
    df.loc[df["mass"]<trade_thresh, "mass"] = 0
    df["supply_cereals_all_exists"] = 0
    df.loc[df["mass"]>0, "supply_cereals_all_exists"] = 1
    
    df_china_us = df[df['us_exporter']==0].drop('us_exporter', axis=1)
    df_china_us = df_china_us.rename(columns={'us': 'to_admin', 'china': 'from_admin', 'mass': 'supply_cereals_all'})
    
    df_us_china = df[df['us_exporter']==1].drop('us_exporter', axis=1)
    df_us_china = df_us_china.rename(columns={'us': 'from_admin', 'china': 'to_admin', 'mass': 'supply_cereals_all'})

    trade = pd.concat([df_china_us, df_us_china], axis=0, ignore_index=True)
    trade = trade.merge(admin[admin['iso3'].isin(['USA', 'CHN'])][['iso3', 'ID', 'admin_name']], left_on='from_admin', right_on='admin_name').drop(
        ['from_admin', 'admin_name'], axis=1).rename(columns={'iso3': 'from_iso3', 'ID': 'from_id'})
    trade = trade.merge(admin[admin['iso3'].isin(['USA', 'CHN'])][['iso3', 'ID', 'admin_name']], left_on='to_admin', right_on='admin_name').drop(
        ['to_admin', 'admin_name'], axis=1).rename(columns={'iso3': 'to_iso3', 'ID': 'to_id'})
    
    if log_y:
        trade.loc[trade["supply_cereals_all_exists"]==1, 'supply_cereals_all'] = trade[trade["supply_cereals_all_exists"]==1].apply(
            lambda row: np.log(row['supply_cereals_all']), axis=1)

    print('trade data done')
     
    # x variables (admin reductions and transport)
    reductions_admin, reductions_country = get_admin_reductions()
    admin_cols = reductions_admin.columns
    country_cols = reductions_country.columns

    reductions_admin = reductions_admin[reductions_admin['iso3'].isin(['USA', 'CHN'])]

    # correcting production numbers
    crops = ['Wheat', 'Maize (corn)', 'Rye', 'Barley', 'Oats', 'Sorghum', 
             'Rice, paddy (rice milled equivalent)', 'Buckwheat', 
             'Millet', 'Quinoa', 'Canary seed', 'Fonio', 'Mixed grain', 'Triticale', 'Cereals n.e.c.']
    df_list = []
    for crop in crops:
        df = get_fao_prod(crop, years=[2017])
        df_list.append(df)
    dfs = [df.set_index('iso3') for df in df_list]
    prod_df = pd.concat(dfs, axis=1).reset_index()
    prod_df = prod_df.rename(columns={
        'Maize (corn)_prod': 'Maize_prod',
        'Rice, paddy (rice milled equivalent)_prod': 'Rice_prod',
        'Cereals n.e.c._prod': 'other_cereals_prod'
    })

    reductions_country = reductions_country.merge(prod_df)
    # barley
    reductions_country['barley_factor'] = reductions_country['Barley_prod'] / (reductions_country['barley_production'])
    reductions_country['barley_production'] = reductions_country['Barley_prod']
    
    # maize
    reductions_country['maize_factor'] = reductions_country['Maize_prod'] / (reductions_country['maize_production'])
    reductions_country['maize_production'] = reductions_country['Maize_prod']
    
    # millet
    reductions_country['millet_factor'] = reductions_country['Millet_prod'] / (reductions_country['millet_production'])
    reductions_country['millet_production'] = reductions_country['Millet_prod']
    
    # rice
    reductions_country['rice_factor'] = reductions_country['Rice_prod'] / (reductions_country['rice_production'])
    reductions_country['rice_production'] = reductions_country['Rice_prod']
    
    # sorghum
    reductions_country['sorghum_factor'] = reductions_country['Sorghum_prod'] / (reductions_country['sorghum_production'])
    reductions_country['sorghum_production'] = reductions_country['Sorghum_prod']
    
    # wheat
    reductions_country['wheat_factor'] = reductions_country['Wheat_prod'] / (reductions_country['wheat_production'])
    reductions_country['wheat_production'] = reductions_country['Wheat_prod']
    
    # other
    reductions_country['other_cereals_factor'] = (reductions_country['Rye_prod'] + reductions_country['Oats_prod'] + reductions_country['Buckwheat_prod'] + reductions_country['Quinoa_prod']  \
                                     + reductions_country['Canary seed_prod'] + reductions_country['Fonio_prod'] + reductions_country['Mixed grain_prod'] + reductions_country['Triticale_prod']  \
                                     + reductions_country['other_cereals_prod']) / (reductions_country['other_cereals_production'])
    reductions_country['other_cereals_production'] = reductions_country['Rye_prod'] + reductions_country['Oats_prod'] + reductions_country['Buckwheat_prod'] \
    + reductions_country['Canary seed_prod'] + reductions_country['Fonio_prod'] + reductions_country['Mixed grain_prod'] + reductions_country['Triticale_prod']  \
    + reductions_country['Quinoa_prod'] + reductions_country['other_cereals_prod']
    
    reductions_country = reductions_country.fillna(0)
    
    reductions_admin = reductions_admin.merge(reductions_country[['iso3', 'barley_factor', 'maize_factor', 'millet_factor', 
                                                'rice_factor', 'sorghum_factor', 'wheat_factor', 'other_cereals_factor',
                                                'barley_production', 'maize_production', 'millet_production',
                                                'rice_production', 'sorghum_production', 'wheat_production',
                                                'other_cereals_production']].rename(columns={
        'barley_production': 'barley_production_country', 'maize_production': 'maize_production_country', 'millet_production': 'millet_production_country',
        'rice_production': 'rice_production_country', 'sorghum_production': 'sorghum_production_country', 'wheat_production': 'wheat_production_country',
        'other_cereals_production': 'other_cereals_production_country'}))
    reductions_admin = reductions_admin.merge(reductions_admin.groupby('iso3')['ID'].count().reset_index().rename(columns={'ID': 'num_admins'}))
    
    for crop in ['barley', 'maize', 'millet', 'rice', 'sorghum', 'wheat', 'other_cereals']:
        reductions_admin[f'{crop}_production'] = (reductions_admin[f'{crop}_production']) * reductions_admin[f'{crop}_factor']
        reductions_admin.loc[~np.isfinite(reductions_admin[f'{crop}_factor']), f'{crop}_production'
            ] = reductions_admin[~np.isfinite(reductions_admin[f'{crop}_factor'])][f'{crop}_production_country'] / reductions_admin[~np.isfinite(reductions_admin[f'{crop}_factor'])]['num_admins']
    
    reductions_admin = reductions_admin[admin_cols.tolist()]
    print('admin reductions done')

    transport_admin, transport_country = get_transport_data(reductions_admin)
    transport_admin = transport_admin[(transport_admin['from_iso3']!=transport_admin['to_iso3']) 
                                      & (transport_admin['from_iso3'].isin(['USA', 'CHN']))
                                      & (transport_admin['to_iso3'].isin(['USA', 'CHN']))]
    print('transport data done')

    # categorical variable - mode
    transport_admin.loc[transport_admin['mode']=='maritime', 'mode'] = 0
    transport_admin.loc[transport_admin['mode']=='rail', 'mode'] = 1
    transport_admin.loc[transport_admin['mode']=='road', 'mode'] = 2
    transport_admin = transport_admin.drop('trade_USD_t', axis=1)

    # log x variables for both transport and admin reductions
    if log_x:
        for col in ['freight_USD_t', 'transport_USD_t', 'time_h', 'distance_km', 'border_USD_t', 'customs_cost', 'Pref_Applied_AVE']:
            print(f'log {col}')
            transport_admin[col] = transport_admin.apply(lambda row: np.log(row[col]+1), axis=1)
            
        for col in ['barley_areas', 'maize_areas', 'millet_areas', 'rice_areas', 'sorghum_areas', 'wheat_areas', 'other_cereals_areas', 
                    'barley_production', 'maize_production', 'millet_production', 'rice_production', 
                    'sorghum_production', 'wheat_production', 'other_cereals_production', 
                    'buffaloes_totals', 'cattle_totals', 'chickens_totals', 'ducks_totals', 'goats_totals', 
                    'horses_totals', 'pigs_totals', 'sheep_totals', 'pop_totals',
                    'gdp_totals', 'area_total', 'built_surface', 'built_volume_total', 'built_volume_nres']:
            print(f'log {col}')
            reductions_admin[col] = reductions_admin.apply(lambda row: np.log(row[col]+1), axis=1)

    # getting us china admin level data by merging everything (trade, transport, admin reductions)
    df_admin = transport_admin.merge(trade, how='left')
    df_admin = df_admin.fillna(0)
    
    # getting admin level data by merging everything (transport and admin reductions)
    df_admin = df_admin.merge(reductions_admin, left_on='from_id', right_on='ID').drop(['ID', 'iso3'], axis=1).rename(columns={
        'barley_areas': 'from_barley_area', 
        'maize_areas': 'from_maize_area', 
        'millet_areas': 'from_millet_area',
        'rice_areas': 'from_rice_area', 
        'sorghum_areas': 'from_sorghum_area', 
        'wheat_areas': 'from_wheat_area', 
        'other_cereals_areas': 'from_other_cereals_area',
        'barley_production': 'from_barley_production', 
        'maize_production': 'from_maize_production', 
        'millet_production': 'from_millet_production',
        'rice_production': 'from_rice_production', 
        'sorghum_production': 'from_sorghum_production', 
        'wheat_production': 'from_wheat_production', 
        'other_cereals_production': 'from_other_cereals_production',
        'buffaloes_totals': 'from_buffaloes', 
        'cattle_totals': 'from_cattle', 
        'chickens_totals': 'from_chickens',
        'ducks_totals': 'from_ducks', 
        'goats_totals': 'from_goats', 
        'horses_totals': 'from_horses', 
        'pigs_totals': 'from_pigs',
        'sheep_totals': 'from_sheep', 
        'pop_totals': 'from_pop',
        'gdp_totals': 'from_gdp',
        'area_total': 'from_area', 
        'built_surface': 'from_built_surface', 
        'built_volume_total': 'from_built_volume_total', 
        'built_volume_nres': 'from_built_volume_nres'})
    
    df_admin = df_admin.merge(reductions_admin, left_on='to_id', right_on='ID').drop(['ID', 'iso3'], axis=1).rename(columns={
        'barley_areas': 'to_barley_area', 
        'maize_areas': 'to_maize_area', 
        'millet_areas': 'to_millet_area',
        'rice_areas': 'to_rice_area', 
        'sorghum_areas': 'to_sorghum_area', 
        'wheat_areas': 'to_wheat_area', 
        'other_cereals_areas': 'to_other_cereals_area',
        'barley_production': 'to_barley_production', 
        'maize_production': 'to_maize_production', 
        'millet_production': 'to_millet_production',
        'rice_production': 'to_rice_production', 
        'sorghum_production': 'to_sorghum_production', 
        'wheat_production': 'to_wheat_production', 
        'other_cereals_production': 'to_other_cereals_production', 
        'buffaloes_totals': 'to_buffaloes', 
        'cattle_totals': 'to_cattle', 
        'chickens_totals': 'to_chickens',
        'ducks_totals': 'to_ducks', 
        'goats_totals': 'to_goats', 
        'horses_totals': 'to_horses', 
        'pigs_totals': 'to_pigs',
        'sheep_totals': 'to_sheep', 
        'pop_totals': 'to_pop',
        'gdp_totals': 'to_gdp',
        'area_total': 'to_area', 
        'built_surface': 'to_built_surface', 
        'built_volume_total': 'to_built_volume_total', 
        'built_volume_nres': 'to_built_volume_nres'})

    print('df_admin done')

    # add region names
    regions = pd.read_excel('../../OPSIS/Data/Country_group/Country_classification_UNSD.xlsx')
    regions.loc[regions['Sub-region Name'].isin(['Northern America', 'Latin America and the Caribbean']),
                'Region Name'] = regions[regions['Sub-region Name'].isin(['Northern America', 'Latin America and the Caribbean'])]['Sub-region Name']
    regions['Region Name'] = regions['Region Name'].replace(['Africa', 'Asia', 'Europe', 'Latin America and the Caribbean', 
                                                             'Northern America', 'Oceania'], [0, 1, 2, 3, 4, 5])
    df_admin = df_admin.merge(regions[['iso3', 'Region Name']], how='left', left_on='from_iso3', right_on='iso3').drop('iso3', axis=1).rename(
        columns={'Region Name': 'from_region'})
    df_admin = df_admin.merge(regions[['iso3', 'Region Name']], how='left', left_on='to_iso3', right_on='iso3').drop('iso3', axis=1).rename(
        columns={'Region Name': 'to_region'})
    
    return df_admin


def merge_data_us_state_int(log_x=True, log_y=True, trade_thresh=10):
    # trade data, replacing values less than thresh with zero
    # exports
    us_state_exp = pd.read_csv('../../data/other_data/US_state_international/State Exports by HS Commodities.csv')
    us_state_exp.loc[us_state_exp['State']=='Dist of Columbia', 'State'] = 'District of Columbia'
    us_state_exp = us_state_exp.merge(admin[admin['iso3']=='USA'][['iso3', 'ID', 'admin_name']], 
                                      left_on='State', right_on='admin_name').drop('admin_name', axis=1).rename(columns={
                                      'iso3': 'from_iso3',
                                      'ID': 'from_id'})
    
    us_state_exp.loc[us_state_exp['Country']=='Burma', 'Country'] = 'Myanmar'
    us_state_exp.loc[us_state_exp['Country']=='Cabo Verde', 'Country'] = 'Cape Verde'
    us_state_exp.loc[us_state_exp['Country']=='Congo (Brazzaville)', 'Country'] = 'Republic of Congo'
    us_state_exp.loc[us_state_exp['Country']=='Congo (Kinshasa)', 'Country'] = 'Democratic Republic of the Congo'
    us_state_exp.loc[us_state_exp['Country']=="Cote d'Ivoire", 'Country'] = "Côte d'Ivoire"
    us_state_exp.loc[us_state_exp['Country']=='Curacao', 'Country'] = 'Curaçao'
    us_state_exp.loc[us_state_exp['Country']=='Korea, South', 'Country'] = 'South Korea'
    us_state_exp.loc[us_state_exp['Country']=='Macau', 'Country'] = 'Macao'
    us_state_exp.loc[us_state_exp['Country']=='Sao Tome and Principe', 'Country'] = 'São Tomé and Príncipe'
    us_state_exp.loc[us_state_exp['Country']=='St Kitts and Nevis', 'Country'] = 'Saint Kitts and Nevis'
    us_state_exp.loc[us_state_exp['Country']=='St Lucia', 'Country'] = 'Saint Lucia'
    us_state_exp.loc[us_state_exp['Country']=='St Vincent and the Grenadines', 'Country'] = 'Saint Vincent and the Grenadines'
    
    us_state_exp = us_state_exp.merge(admin[['iso3', 'country']].drop_duplicates(), 
                                      left_on='Country', right_on='country').drop('country', axis=1).rename(columns={
                                      'iso3': 'to_iso3'})
    
    us_state_exp['Air SWT (kg)'] = us_state_exp['Air SWT (kg)'].fillna('0').str.replace(',', '').astype(int)
    us_state_exp['Vessel SWT (kg)'] = us_state_exp['Vessel SWT (kg)'].fillna('0').str.replace(',', '').astype(int)
    us_state_exp['supply'] = us_state_exp['Air SWT (kg)'] + us_state_exp['Vessel SWT (kg)']
    us_state_exp['supply'] = us_state_exp['supply'] / 1e3
    us_state_exp.loc[us_state_exp['Commodity']=='1006 Rice', 'supply'] = us_state_exp[us_state_exp['Commodity']=='1006 Rice']['supply'] * 0.7 # to account for milling
    
    us_state_exp = us_state_exp.groupby(['State', 'Commodity', 'Country', 'from_iso3', 'from_id', 'to_iso3'])[['supply']].mean().reset_index()
    us_state_exp.loc[us_state_exp['Commodity']=='1001 Wheat And Meslin', 'Commodity'] = 'supply_wheat'
    us_state_exp.loc[us_state_exp['Commodity']=='1006 Rice',  'Commodity'] = 'supply_rice'
    us_state_exp.loc[us_state_exp['Commodity']=='1005 Corn (maize)', 'Commodity'] = 'supply_maize'
    us_state_exp.loc[us_state_exp['Commodity'].isin(['1003 Barley', '1004 Oats', '1007 Grain Sorghum',
                                                     '1008 Buckwheat, Millet And Canary Seeds; Other Cereals',
                                                     '1002 Rye']), 'Commodity'] = 'supply_other_cereals'
    
    us_state_exp = us_state_exp.groupby(['State', 'Commodity', 'Country', 'from_iso3', 'from_id', 'to_iso3'])[['supply']].sum().reset_index()
    us_state_exp = us_state_exp.pivot(index=['State', 'Country', 'from_iso3', 'from_id', 'to_iso3'], 
                                      columns='Commodity', values='supply').reset_index().fillna(0)
    us_state_exp = us_state_exp[~us_state_exp['to_iso3'].isin(['ABW', 'AIA', 'BMU', 'CUW', 'CYM', 
                                                               'GLP', 'GUF', 'MSR', 'MTQ',
                                                               'SXM', 'TCA', 'VGB', 'KIR', 'PLW', 'REU'])]

    us_state_exp['to_id'] = us_state_exp['to_iso3']


    # imports
    us_state_imp = pd.read_csv('../../data/other_data/US_state_international/State Imports by HS Commodities.csv')
    us_state_imp.loc[us_state_imp['State']=='Dist of Columbia', 'State'] = 'District of Columbia'
    us_state_imp = us_state_imp.merge(admin[admin['iso3']=='USA'][['iso3', 'ID', 'admin_name']], 
                                      left_on='State', right_on='admin_name').drop('admin_name', axis=1).rename(columns={
                                      'iso3': 'to_iso3',
                                      'ID': 'to_id'})
    
    us_state_imp.loc[us_state_imp['Country']=='Burma', 'Country'] = 'Myanmar'
    us_state_imp.loc[us_state_imp['Country']=='Cabo Verde', 'Country'] = 'Cape Verde'
    us_state_imp.loc[us_state_imp['Country']=='Congo (Brazzaville)', 'Country'] = 'Republic of Congo'
    us_state_imp.loc[us_state_imp['Country']=='Congo (Kinshasa)', 'Country'] = 'Democratic Republic of the Congo'
    us_state_imp.loc[us_state_imp['Country']=="Cote d'Ivoire", 'Country'] = "Côte d'Ivoire"
    us_state_imp.loc[us_state_imp['Country']=='Curacao', 'Country'] = 'Curaçao'
    us_state_imp.loc[us_state_imp['Country']=='Korea, South', 'Country'] = 'South Korea'
    us_state_imp.loc[us_state_imp['Country']=='Macau', 'Country'] = 'Macao'
    us_state_imp.loc[us_state_imp['Country']=='Sao Tome and Principe', 'Country'] = 'São Tomé and Príncipe'
    us_state_imp.loc[us_state_imp['Country']=='St Kitts and Nevis', 'Country'] = 'Saint Kitts and Nevis'
    us_state_imp.loc[us_state_imp['Country']=='St Lucia', 'Country'] = 'Saint Lucia'
    us_state_imp.loc[us_state_imp['Country']=='St Vincent and the Grenadines', 'Country'] = 'Saint Vincent and the Grenadines'
    
    us_state_imp = us_state_imp.merge(admin[['iso3', 'country']].drop_duplicates(), 
                                      left_on='Country', right_on='country').drop('country', axis=1).rename(columns={
                                      'iso3': 'from_iso3'})
    
    us_state_imp['Air SWT (kg)'] = us_state_imp['Air SWT (kg)'].fillna('0').str.replace(',', '').astype(int)
    us_state_imp['Vessel SWT (kg)'] = us_state_imp['Vessel SWT (kg)'].fillna('0').str.replace(',', '').astype(int)
    us_state_imp['supply'] = us_state_imp['Air SWT (kg)'] + us_state_imp['Vessel SWT (kg)']
    us_state_imp['supply'] = us_state_imp['supply'] / 1e3
    us_state_imp.loc[us_state_imp['Commodity']=='1006 Rice', 'supply'] = us_state_imp[us_state_imp['Commodity']=='1006 Rice']['supply'] * 0.7 # to account for milling
    
    us_state_imp = us_state_imp.groupby(['State', 'Commodity', 'Country', 'from_iso3', 'to_id', 'to_iso3'])[['supply']].mean().reset_index()
    us_state_imp.loc[us_state_imp['Commodity']=='1001 Wheat And Meslin', 'Commodity'] = 'supply_wheat'
    us_state_imp.loc[us_state_imp['Commodity']=='1006 Rice',  'Commodity'] = 'supply_rice'
    us_state_imp.loc[us_state_imp['Commodity']=='1005 Corn (maize)', 'Commodity'] = 'supply_maize'
    us_state_imp.loc[us_state_imp['Commodity'].isin(['1003 Barley', '1004 Oats', '1007 Grain Sorghum',
                                                     '1008 Buckwheat, Millet And Canary Seeds; Other Cereals',
                                                     '1002 Rye']), 'Commodity'] = 'supply_other_cereals'
    
    us_state_imp = us_state_imp.groupby(['State', 'Commodity', 'Country', 'from_iso3', 'to_id', 'to_iso3'])[['supply']].sum().reset_index()
    us_state_imp = us_state_imp.pivot(index=['State', 'Country', 'from_iso3', 'to_id', 'to_iso3'], 
                                      columns='Commodity', values='supply').reset_index().fillna(0)
    us_state_imp = us_state_imp[~us_state_imp['to_iso3'].isin(['ABW', 'AIA', 'BMU', 'CUW', 'CYM', 
                                                               'GLP', 'GUF', 'MSR', 'MTQ',
                                                               'SXM', 'TCA', 'VGB', 'KIR', 'PLW', 'REU'])]
    us_state_imp['from_id'] = us_state_imp['from_iso3']

    trade = pd.concat([us_state_exp, us_state_imp], axis=0, ignore_index=True)
    trade['supply_cereals_all'] = trade[['supply_wheat', 'supply_rice', 'supply_maize', 'supply_other_cereals']].sum(axis=1)
    for crop in ['maize', 'rice', 'wheat', 'other_cereals', 'cereals_all']:
        trade[f"supply_{crop}_exists"] = 0
        trade.loc[trade[f"supply_{crop}"]>0, f"supply_{crop}_exists"] = 1
    
    if log_y:
        for col in ['supply_wheat', 'supply_rice', 'supply_maize', 'supply_other_cereals', 'supply_cereals_all']:
            trade.loc[trade[f"{col}_exists"]==1, col] = trade[trade[f"{col}_exists"]==1].apply(lambda row: np.log(row[col]), axis=1)

    print('trade data done')
     
    # x variables (admin reductions and transport)
    reductions_admin, reductions_country = get_admin_reductions()
    admin_cols = reductions_admin.columns
    country_cols = reductions_country.columns

    reductions_admin = reductions_admin[reductions_admin['iso3']=='USA']

    # correcting production numbers
    crops = ['Wheat', 'Maize (corn)', 'Rye', 'Barley', 'Oats', 'Sorghum', 
             'Rice, paddy (rice milled equivalent)', 'Buckwheat', 
             'Millet', 'Quinoa', 'Canary seed', 'Fonio', 'Mixed grain', 'Triticale', 'Cereals n.e.c.']
    df_list = []
    for crop in crops:
        df = get_fao_prod(crop)
        df_list.append(df)
    dfs = [df.set_index('iso3') for df in df_list]
    prod_df = pd.concat(dfs, axis=1).reset_index()
    prod_df = prod_df.rename(columns={
        'Maize (corn)_prod': 'Maize_prod',
        'Rice, paddy (rice milled equivalent)_prod': 'Rice_prod',
        'Cereals n.e.c._prod': 'other_cereals_prod'
    })

    reductions_country = reductions_country.merge(prod_df)
    # barley
    reductions_country['barley_factor'] = reductions_country['Barley_prod'] / (reductions_country['barley_production'])
    reductions_country['barley_production'] = reductions_country['Barley_prod']
    
    # maize
    reductions_country['maize_factor'] = reductions_country['Maize_prod'] / (reductions_country['maize_production'])
    reductions_country['maize_production'] = reductions_country['Maize_prod']
    
    # millet
    reductions_country['millet_factor'] = reductions_country['Millet_prod'] / (reductions_country['millet_production'])
    reductions_country['millet_production'] = reductions_country['Millet_prod']
    
    # rice
    reductions_country['rice_factor'] = reductions_country['Rice_prod'] / (reductions_country['rice_production'])
    reductions_country['rice_production'] = reductions_country['Rice_prod']
    
    # sorghum
    reductions_country['sorghum_factor'] = reductions_country['Sorghum_prod'] / (reductions_country['sorghum_production'])
    reductions_country['sorghum_production'] = reductions_country['Sorghum_prod']
    
    # wheat
    reductions_country['wheat_factor'] = reductions_country['Wheat_prod'] / (reductions_country['wheat_production'])
    reductions_country['wheat_production'] = reductions_country['Wheat_prod']
    
    # other
    reductions_country['other_cereals_factor'] = (reductions_country['Rye_prod'] + reductions_country['Oats_prod'] + reductions_country['Buckwheat_prod'] + reductions_country['Quinoa_prod']  \
                                     + reductions_country['Canary seed_prod'] + reductions_country['Fonio_prod'] + reductions_country['Mixed grain_prod'] + reductions_country['Triticale_prod']  \
                                     + reductions_country['other_cereals_prod']) / (reductions_country['other_cereals_production'])
    reductions_country['other_cereals_production'] = reductions_country['Rye_prod'] + reductions_country['Oats_prod'] + reductions_country['Buckwheat_prod'] \
    + reductions_country['Canary seed_prod'] + reductions_country['Fonio_prod'] + reductions_country['Mixed grain_prod'] + reductions_country['Triticale_prod']  \
    + reductions_country['Quinoa_prod'] + reductions_country['other_cereals_prod']
    
    reductions_country = reductions_country.fillna(0)
    
    reductions_admin = reductions_admin.merge(reductions_country[['iso3', 'barley_factor', 'maize_factor', 'millet_factor', 
                                                'rice_factor', 'sorghum_factor', 'wheat_factor', 'other_cereals_factor',
                                                'barley_production', 'maize_production', 'millet_production',
                                                'rice_production', 'sorghum_production', 'wheat_production',
                                                'other_cereals_production']].rename(columns={
        'barley_production': 'barley_production_country', 'maize_production': 'maize_production_country', 'millet_production': 'millet_production_country',
        'rice_production': 'rice_production_country', 'sorghum_production': 'sorghum_production_country', 'wheat_production': 'wheat_production_country',
        'other_cereals_production': 'other_cereals_production_country'}))
    reductions_admin = reductions_admin.merge(reductions_admin.groupby('iso3')['ID'].count().reset_index().rename(columns={'ID': 'num_admins'}))
    
    for crop in ['barley', 'maize', 'millet', 'rice', 'sorghum', 'wheat', 'other_cereals']:
        reductions_admin[f'{crop}_production'] = (reductions_admin[f'{crop}_production']) * reductions_admin[f'{crop}_factor']
        reductions_admin.loc[~np.isfinite(reductions_admin[f'{crop}_factor']), f'{crop}_production'
            ] = reductions_admin[~np.isfinite(reductions_admin[f'{crop}_factor'])][f'{crop}_production_country'] / reductions_admin[~np.isfinite(reductions_admin[f'{crop}_factor'])]['num_admins']
    
    reductions_admin = reductions_admin[admin_cols.tolist()]
    reductions_country = reductions_country[country_cols.tolist()]
    print('admin reductions done')
    
    transport_admin, transport_country = get_transport_data(reductions_admin)
    print('transport data done')

    # categorical variable - mode
    transport_admin.loc[transport_admin['mode']=='maritime', 'mode'] = 0
    transport_admin.loc[transport_admin['mode']=='rail', 'mode'] = 1
    transport_admin.loc[transport_admin['mode']=='road', 'mode'] = 2
    transport_admin = transport_admin.drop('trade_USD_t', axis=1)

    # mixing country and admin data since US is subnational and other countries are national
    reductions_country = reductions_country[reductions_country['iso3']!='USA']
    reductions_country['ID'] = reductions_country['iso3']
    reductions_admin = pd.concat([reductions_admin, reductions_country], axis=0, ignore_index=True)

    transport_admin = transport_admin[((transport_admin['from_iso3']=='USA') | (transport_admin['to_iso3']=='USA'))
                                      & (transport_admin['from_iso3']!=transport_admin['to_iso3'])]

    transport_admin.loc[transport_admin['from_iso3']=='USA', 'to_id'] = transport_admin[transport_admin['from_iso3']=='USA']['to_iso3']
    transport_admin.loc[transport_admin['to_iso3']=='USA', 'from_id'] = transport_admin[transport_admin['to_iso3']=='USA']['from_iso3']
    # weighted averages for importing/exporting countries
    transport_admin = transport_admin.merge(reductions_admin[['ID', 'barley_production', 'maize_production', 'millet_production', 'rice_production',
           'sorghum_production', 'wheat_production', 'other_cereals_production']], left_on='from_id', right_on='ID', how='left').drop('ID', axis=1)
    transport_admin = transport_admin.merge(reductions_admin[['ID', 'buffaloes_totals', 'cattle_totals', 'chickens_totals', 'ducks_totals', 'goats_totals',
           'horses_totals', 'pigs_totals', 'sheep_totals', 'pop_totals']], left_on='to_id', right_on='ID', how='left').drop('ID', axis=1)
    
    transport_admin['pop_livestock'] = transport_admin['pop_totals'] + transport_admin['buffaloes_totals'] + transport_admin['cattle_totals'] \
        + transport_admin['chickens_totals'] + transport_admin['ducks_totals'] + transport_admin['goats_totals'] + transport_admin['horses_totals'] \
        + transport_admin['pigs_totals'] + transport_admin['sheep_totals']
    transport_admin['production'] = transport_admin['barley_production'] + transport_admin['maize_production'] + transport_admin['millet_production'] \
        + transport_admin['rice_production'] + transport_admin['sorghum_production'] + transport_admin['wheat_production'] \
        + transport_admin['other_cereals_production'] 
    transport_admin['weight'] = transport_admin['pop_livestock'] * transport_admin['production']
    
    transport_admin = transport_admin.drop(['production', 'barley_production', 'maize_production', 'millet_production', 
                                'rice_production', 'sorghum_production', 'wheat_production', 'other_cereals_production', 'buffaloes_totals', 
                                'cattle_totals', 'chickens_totals', 'ducks_totals', 'goats_totals',
                                'horses_totals', 'pigs_totals', 'sheep_totals', 'pop_totals', 'pop_livestock'], axis=1)
    
    transport_admin.loc[transport_admin['weight']==0, 'weight'] = transport_admin[transport_admin['weight']>0]['weight'].min()
    transport_admin.loc[transport_admin['customs_cost'].isna(), 'customs_cost'] = 0
    
    wm = lambda x: np.average(x, weights=transport_admin.loc[x.index, "weight"])
    transport_admin = transport_admin.groupby(['from_id', 'to_id',
                                               'from_iso3', 'to_iso3']).agg({'freight_USD_t': wm,
                                                                         'transport_USD_t': wm,
                                                                         'time_h': wm,
                                                                         'distance_km': wm,
                                                                         'border_USD_t': wm,
                                                                         'customs_cost': wm,
                                                                         'Pref_Applied_AVE': wm,
                                                                         'mode': lambda x: pd.Series.mode(x).iloc[0]})
    
    
    transport_admin = transport_admin.reset_index()
    transport_admin = transport_admin[transport_admin['transport_USD_t'].notnull()]

    # log x variables for both transport and admin reductions
    if log_x:
        for col in ['freight_USD_t', 'transport_USD_t', 'time_h', 'distance_km', 'border_USD_t', 'customs_cost', 'Pref_Applied_AVE']:
            print(f'log {col}')
            transport_admin[col] = transport_admin.apply(lambda row: np.log(row[col]+1), axis=1)
            
        for col in ['barley_areas', 'maize_areas', 'millet_areas', 'rice_areas', 'sorghum_areas', 'wheat_areas', 'other_cereals_areas', 
                    'barley_production', 'maize_production', 'millet_production', 'rice_production', 
                    'sorghum_production', 'wheat_production', 'other_cereals_production', 
                    'buffaloes_totals', 'cattle_totals', 'chickens_totals', 'ducks_totals', 'goats_totals', 
                    'horses_totals', 'pigs_totals', 'sheep_totals', 'pop_totals',
                    'gdp_totals', 'area_total', 'built_surface', 'built_volume_total', 'built_volume_nres']:
            print(f'log {col}')
            reductions_admin[col] = reductions_admin.apply(lambda row: np.log(row[col]+1), axis=1)

    # getting us china admin level data by merging everything (trade, transport, admin reductions)
    df_admin = transport_admin.merge(trade.drop(['State', 'Country'], axis=1), how='left')
    df_admin = df_admin.fillna(0)
    
    # getting admin level data by merging everything (transport and admin reductions)
    df_admin = df_admin.merge(reductions_admin, left_on='from_id', right_on='ID').drop(['ID', 'iso3'], axis=1).rename(columns={
        'barley_areas': 'from_barley_area', 
        'maize_areas': 'from_maize_area', 
        'millet_areas': 'from_millet_area',
        'rice_areas': 'from_rice_area', 
        'sorghum_areas': 'from_sorghum_area', 
        'wheat_areas': 'from_wheat_area', 
        'other_cereals_areas': 'from_other_cereals_area',
        'barley_production': 'from_barley_production', 
        'maize_production': 'from_maize_production', 
        'millet_production': 'from_millet_production',
        'rice_production': 'from_rice_production', 
        'sorghum_production': 'from_sorghum_production', 
        'wheat_production': 'from_wheat_production', 
        'other_cereals_production': 'from_other_cereals_production',
        'buffaloes_totals': 'from_buffaloes', 
        'cattle_totals': 'from_cattle', 
        'chickens_totals': 'from_chickens',
        'ducks_totals': 'from_ducks', 
        'goats_totals': 'from_goats', 
        'horses_totals': 'from_horses', 
        'pigs_totals': 'from_pigs',
        'sheep_totals': 'from_sheep', 
        'pop_totals': 'from_pop',
        'gdp_totals': 'from_gdp',
        'area_total': 'from_area', 
        'built_surface': 'from_built_surface', 
        'built_volume_total': 'from_built_volume_total', 
        'built_volume_nres': 'from_built_volume_nres'})
    
    df_admin = df_admin.merge(reductions_admin, left_on='to_id', right_on='ID').drop(['ID', 'iso3'], axis=1).rename(columns={
        'barley_areas': 'to_barley_area', 
        'maize_areas': 'to_maize_area', 
        'millet_areas': 'to_millet_area',
        'rice_areas': 'to_rice_area', 
        'sorghum_areas': 'to_sorghum_area', 
        'wheat_areas': 'to_wheat_area', 
        'other_cereals_areas': 'to_other_cereals_area',
        'barley_production': 'to_barley_production', 
        'maize_production': 'to_maize_production', 
        'millet_production': 'to_millet_production',
        'rice_production': 'to_rice_production', 
        'sorghum_production': 'to_sorghum_production', 
        'wheat_production': 'to_wheat_production', 
        'other_cereals_production': 'to_other_cereals_production', 
        'buffaloes_totals': 'to_buffaloes', 
        'cattle_totals': 'to_cattle', 
        'chickens_totals': 'to_chickens',
        'ducks_totals': 'to_ducks', 
        'goats_totals': 'to_goats', 
        'horses_totals': 'to_horses', 
        'pigs_totals': 'to_pigs',
        'sheep_totals': 'to_sheep', 
        'pop_totals': 'to_pop',
        'gdp_totals': 'to_gdp',
        'area_total': 'to_area', 
        'built_surface': 'to_built_surface', 
        'built_volume_total': 'to_built_volume_total', 
        'built_volume_nres': 'to_built_volume_nres'})

    print('df_admin done')

    # add region names
    regions = pd.read_excel('../../OPSIS/Data/Country_group/Country_classification_UNSD.xlsx')
    regions.loc[regions['Sub-region Name'].isin(['Northern America', 'Latin America and the Caribbean']),
                'Region Name'] = regions[regions['Sub-region Name'].isin(['Northern America', 'Latin America and the Caribbean'])]['Sub-region Name']
    regions['Region Name'] = regions['Region Name'].replace(['Africa', 'Asia', 'Europe', 'Latin America and the Caribbean', 
                                                             'Northern America', 'Oceania'], [0, 1, 2, 3, 4, 5])
    df_admin = df_admin.merge(regions[['iso3', 'Region Name']], how='left', left_on='from_iso3', right_on='iso3').drop('iso3', axis=1).rename(
        columns={'Region Name': 'from_region'})
    df_admin = df_admin.merge(regions[['iso3', 'Region Name']], how='left', left_on='to_iso3', right_on='iso3').drop('iso3', axis=1).rename(
        columns={'Region Name': 'to_region'})
    
    return df_admin


if __name__ == '__main__':
   
    # thresh 1
    # df_country, df_admin = merge_data(log_x=False, log_y=False, trade_thresh=1)
    # df_country.to_parquet('../../data/combined_features/df_country_log_false_thresh_1.parquet.gzip', index=False, compression='gzip')
    # df_admin.to_parquet('../../data/combined_features/df_admin_log_false.parquet.gzip', index=False, compression='gzip')
    
    # df_country, df_admin = merge_data(log_x=True, log_y=True, trade_thresh=1)
    # df_country.to_parquet('../../data/combined_features/df_country_log_true_thresh_1.parquet.gzip', index=False, compression='gzip')
    # df_admin.to_parquet('../../data/combined_features/df_admin_log_true.parquet.gzip', index=False, compression='gzip')

    # india data 2012
    df_india = merge_data_ind(log_x=False, log_y=False, trade_thresh=1)
    df_india.to_parquet('../../data/combined_features/df_india_log_false_thresh_1.parquet.gzip', index=False, compression='gzip')
    df_india = merge_data_ind(log_x=True, log_y=True, trade_thresh=1)
    df_india.to_parquet('../../data/combined_features/df_india_log_true_thresh_1.parquet.gzip', index=False, compression='gzip')

    # us china data 2017
    # df_us_china = merge_data_us_china(log_x=False, log_y=False, trade_thresh=1)
    # df_us_china.to_parquet('../../data/combined_features/df_us_china_log_false_thresh_1.parquet.gzip', index=False, compression='gzip')
    # df_us_china = merge_data_us_china(log_x=True, log_y=True, trade_thresh=1)
    # df_us_china.to_parquet('../../data/combined_features/df_us_china_log_true_thresh_1.parquet.gzip', index=False, compression='gzip')

    # us state data 2017-21
    # df_us_state_int = merge_data_us_state_int(log_x=False, log_y=False, trade_thresh=1)
    # df_us_state_int.to_parquet('../../data/combined_features/df_us_state_int_log_false_thresh_1.parquet.gzip', index=False, compression='gzip')
    # df_us_state_int = merge_data_us_state_int(log_x=True, log_y=True, trade_thresh=1)
    # df_us_state_int.to_parquet('../../data/combined_features/df_us_state_int_log_true_thresh_1.parquet.gzip', index=False, compression='gzip')
    