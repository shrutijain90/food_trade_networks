# Usage: python -m food_trade.demand_supply.get_combined_datasets > ../../data/combined_data/output.txt

import os
import geopandas as gpd
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

admin = gpd.read_file('../../data/admin_polygons/admin_polygons.shp')

# trade data
def get_trade_data(trade_thresh=1):

    def get_crop_trade(crop, trade_thresh):
        # re-exported trade matrix (203 x 203)
        trade_2017 = pd.read_csv(f"../../data/FAO_re_export/supply_matrix_{crop}_2017.csv")
        trade_2018 = pd.read_csv(f"../../data/FAO_re_export/supply_matrix_{crop}_2018.csv")
        trade_2019 = pd.read_csv(f"../../data/FAO_re_export/supply_matrix_{crop}_2019.csv")
        trade_2020 = pd.read_csv(f"../../data/FAO_re_export/supply_matrix_{crop}_2020.csv")
        trade_2021 = pd.read_csv(f"../../data/FAO_re_export/supply_matrix_{crop}_2021.csv")
        
        trade = pd.concat([trade_2017, trade_2018, trade_2019, trade_2020, trade_2021]).groupby('iso3').mean().reset_index()
        
        # melt trade data 
        trade = trade.rename(columns={'iso3': 'from_iso3'})
        trade = trade.melt(id_vars=['from_iso3'], value_vars=trade['from_iso3'].values.tolist()).rename(
            columns={'variable': 'to_iso3', 'value': f"supply_{crop}"})

        trade.loc[trade[f"supply_{crop}"]<trade_thresh, f"supply_{crop}"] = 0
        return trade

    crops = ['Wheat', 'Maize (corn)', 'Rye', 'Barley', 'Oats', 'Sorghum', 
             'Rice, paddy (rice milled equivalent)', 'Buckwheat', 
             'Millet', 'Quinoa', 'Cereals n.e.c.', 
             'Canary seed', 'Fonio', 'Mixed grain', 'Triticale', 'cereals_all']
    
    df_list = []
    for crop in crops:
        df = get_crop_trade(crop, trade_thresh)
        df_list.append(df)
        
    dfs = [df.set_index(['from_iso3', 'to_iso3']) for df in df_list]
    trade = pd.concat(dfs, axis=1).reset_index()

    trade = trade.rename(columns={'supply_Wheat': 'supply_wheat',
                                  'supply_Maize (corn)': 'supply_maize',
                                  'supply_Rice, paddy (rice milled equivalent)': 'supply_rice'})
    trade['supply_other_cereals'] = trade['supply_Rye'] + trade['supply_Oats'] + trade['supply_Buckwheat'] + trade['supply_Quinoa']  \
    + trade['supply_Barley'] + trade['supply_Sorghum'] + trade['supply_Canary seed'] + trade['supply_Fonio'] + trade['supply_Mixed grain']  \
    + trade['supply_Triticale'] + trade['supply_Millet'] + trade['supply_Cereals n.e.c.']

    trade = trade[['from_iso3', 'to_iso3', 'supply_wheat', 'supply_maize', 'supply_rice', 'supply_other_cereals', 'supply_cereals_all']]
    
    # remove these 4 area codes ('GLP', 'GUF', 'MTQ', 'REU') from data [wouldn't make a difference, since they all have 0 production and consumption]
    # also remove 'FRO', 'COK', 'KIR', 'TKL'. 'TKL' doesn't exist in admin files itself. The other three have no international connections in the admin level transport connections file [FRO does have domestic connections in transport data, but these would also be removed below so that countries are consistent across spatial scales.]
    # overall left with 195 countries
    trade = trade[(~trade['from_iso3'].isin(['GLP', 'GUF', 'MTQ', 'REU', 'FRO', 'COK', 'KIR', 'TKL'])) 
    & (~trade['to_iso3'].isin(['GLP', 'GUF', 'MTQ', 'REU', 'FRO', 'COK', 'KIR', 'TKL']))].reset_index(drop=True)

    # list of countries in data (will keep only these in the admin level data)
    iso3_codes = list(set(trade['from_iso3'].values.tolist()))
    
    return trade, iso3_codes


# transport data 
def get_transport_data(iso3_codes):
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
        transport_country['customs_cost'] = np.nan
        transport_country['freight_USD_t'] = np.nan
        
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
    maritime_country['customs_cost'] = np.nan
    maritime_country['freight_USD_t'] = np.nan
    transport_admin_country = pd.concat([transport_admin_country, maritime_country], ignore_index=True)
    transport_admin_country = transport_admin_country.sort_values(by=['from_id', 'to_id', 'transport_USD_t']).reset_index(drop=True)
    transport_admin_country = transport_admin_country.drop_duplicates(subset=['from_id', 'to_id'], keep='first')
    
    transport_admin = pd.concat([transport_admin_country, transport_admin], ignore_index=True)
    transport_admin = transport_admin.sort_values(by=['from_id', 'to_id', 'transport_USD_t']).reset_index(drop=True)
    transport_admin = transport_admin.drop_duplicates(subset=['from_id', 'to_id'], keep='first').reset_index(drop=True)

    # only keeping common iso3 codes (those that have both (non-zero) fao and transport data)
    # 3537 from admins, 3536 to admins, 12,415,736 links
    # still some issues in the transport data. several domestic links missing, some international ones also missing (maybe that is not too important?)
    # admins in the 195 countries that don't have incoming links from anywhere (domestic or international) - 'BTN.12_1', 'NZL.10_1', 'PSE.1_1', 'STP.1_1'
    # 'NZL.10_1' is present in from_id but not in to_id
    # ideally there should be 3540 from and to admins, 12,531,600 links 
    transport_admin = transport_admin[(transport_admin['from_iso3'].isin(iso3_codes)) 
    & (transport_admin['to_iso3'].isin(iso3_codes))].reset_index(drop=True)
    
    return transport_admin



# admin reductions
def get_admin_reductions(iso3_codes):
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

    reductions_admin = reductions_admin[(reductions_admin['iso3'].isin(iso3_codes))].reset_index(drop=True)
    reductions_country = reductions_country[(reductions_country['iso3'].isin(iso3_codes))].reset_index(drop=True)

    return  reductions_admin, reductions_country

# production data to scale mapspam aggregates
def get_fao_prod(crop, iso3_codes, years=[2017, 2018, 2019, 2020, 2021]):

    fao_prod = pd.concat([pd.read_csv(f"../../data/FAO_prod_mat/prod_matrix_{crop}_{year}.csv") for year in years], axis=0, ignore_index=True)
    fao_prod = fao_prod.groupby('iso3').mean().reset_index()
    fao_prod = fao_prod[(fao_prod['iso3'].isin(iso3_codes))]

    df = fao_prod[['iso3', 'prod']].rename(columns={'prod': f'{crop}_prod'})

    return df


# fit consumption using country data
def fit_consumption(df_country, crop='cereals_all', cols=None, scatter_plot=False, print_reg=False):
    
    if cols is None:
        cols = ['buffaloes_totals_log', 'cattle_totals_log', 'chickens_totals_log', 
                'ducks_totals_log', 'goats_totals_log', 'horses_totals_log', #'pigs_totals_log', 'sheep_totals_log', 
                'pop_totals_log', 'gdp_totals_log', f'{crop}_prod_log']

    cons = df_country.copy()
    for col in cols:
        cons[col] = np.log(cons[col.split('_log')[0]]+1)

    fml = f"{crop}_cons ~ " + " + ".join(cols) + " - 1 "
    
    cons_mod = sm.GLM.from_formula(fml, family=sm.families.Gamma(link=sm.families.links.Log()), data=cons)
    res = cons_mod.fit()
    
    if print_reg:
        print(res.summary(cons_mod))
    
    cons[f'{crop}_cons_pred'] = res.predict(cons[cols])
    
    print(f"R2: {r2_score(cons[f'{crop}_cons'], cons[f'{crop}_cons_pred'])}")
    print(f"R2 on logs: {r2_score(np.log(cons[f'{crop}_cons']+1), np.log(cons[f'{crop}_cons_pred']+1))}")

    if scatter_plot:
        plt.figure(figsize=(9, 4))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=cons[f'{crop}_cons'], y=cons[f'{crop}_cons_pred'])
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=np.log(cons[f'{crop}_cons']+1), y=np.log(cons[f'{crop}_cons_pred']+1))

    return res

# predict admin consumption
def pred_consumption(df_country, df_admin, scatter_plot=False):
        
    for crop in ['wheat', 'maize', 'rice', 'other_cereals', 'cereals_all']:
        print(crop)
    
        cons = df_admin.copy()
        
        cols = ['buffaloes_totals_log', 'cattle_totals_log', 'chickens_totals_log', 
                'ducks_totals_log', 'goats_totals_log', 'horses_totals_log', 'pigs_totals_log', 'sheep_totals_log', 
                'pop_totals_log', 'gdp_totals_log', f'{crop}_prod_log']
    
        if crop in ['rice', 'other_cereals', 'cereals_all']:
            cols = ['buffaloes_totals_log', 'cattle_totals_log', 'chickens_totals_log', 
                    'ducks_totals_log', 'goats_totals_log', 'horses_totals_log', #'pigs_totals_log', 'sheep_totals_log', 
                    'pop_totals_log', 'gdp_totals_log', f'{crop}_prod_log']

        for col in cols:
            cons[col] = np.log(cons[col.split('_log')[0]]+1)
            
        print('R2 while fitting')
        res = fit_consumption(df_country, crop=crop, cols=cols, scatter_plot=scatter_plot, print_reg=False)
    
        cons[f'{crop}_cons'] = res.predict(cons[cols])
    
        cons_comp = cons.groupby('iso3').sum()[[f'{crop}_cons']].reset_index().rename(columns={
            f'{crop}_cons':f'{crop}_cons_total'}).merge(df_country[['iso3', f'{crop}_cons']].rename(columns={
            f'{crop}_cons': f'{crop}_cons_country'}))

        print('R2 between country level consumption and totalled predicted admin level consumption')
        print(f"R2: {r2_score(cons_comp[f'{crop}_cons_country'], cons_comp[f'{crop}_cons_total'])}")
        print(f"R2 on logs: {r2_score(np.log(cons_comp[f'{crop}_cons_country']+1), np.log(cons_comp[f'{crop}_cons_total']+1))}")

        if scatter_plot:
            plt.figure(figsize=(9, 4))
            plt.subplot(1, 2, 1)
            sns.scatterplot(x=cons_comp[f'{crop}_cons_country'], y=cons_comp[f'{crop}_cons_total'])
            plt.subplot(1, 2, 2)
            sns.scatterplot(x=np.log(cons_comp[f'{crop}_cons_country']+1), y=np.log(cons_comp[f'{crop}_cons_total']+1))
    
        cons = cons.merge(cons_comp)
        cons[f'{crop}_cons'] = cons[f'{crop}_cons'] * cons[f'{crop}_cons_country'] / cons[f'{crop}_cons_total']
        cons = cons[['iso3', 'ID', f'{crop}_cons']]
        
        df_admin = df_admin.merge(cons)
        
    return df_admin


# get country and admin level data
def get_admin_data(trade, iso3_codes):
    
    # admin reductions 
    reductions_admin, reductions_country = get_admin_reductions(iso3_codes)
    admin_cols = reductions_admin.columns
    country_cols = reductions_country.columns

    # correcting production numbers
    crops = ['Wheat', 'Maize (corn)', 'Rye', 'Barley', 'Oats', 'Sorghum', 
             'Rice, paddy (rice milled equivalent)', 'Buckwheat', 
             'Millet', 'Quinoa', 'Canary seed', 'Fonio', 'Mixed grain', 'Triticale', 'Cereals n.e.c.']
    df_list = []
    for crop in crops:
        df = get_fao_prod(crop, iso3_codes)
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
    
    # only keeping columns we need
    reductions_country = reductions_country.rename(columns={'wheat_production': 'wheat_prod', 
                                                            'maize_production': 'maize_prod', 
                                                            'rice_production': 'rice_prod'})
    reductions_country['other_cereals_prod'] = reductions_country['barley_production'] + reductions_country['millet_production'] \
    + reductions_country['sorghum_production'] + reductions_country['other_cereals_production']
    
    reductions_country['cereals_all_prod'] = reductions_country['wheat_prod'] + reductions_country['rice_prod'] \
    + reductions_country['maize_prod'] + reductions_country['other_cereals_prod']
    
    reductions_country = reductions_country[['iso3', 'wheat_prod', 'rice_prod', 'maize_prod', 'other_cereals_prod', 'cereals_all_prod', 
                                             'buffaloes_totals', 'cattle_totals', 'chickens_totals', 'ducks_totals', 'goats_totals', 
                                             'horses_totals', 'pigs_totals', 'sheep_totals', 'pop_totals', 'gdp_totals']]

    df_country = reductions_country.merge(trade.drop('from_iso3', axis=1).groupby('to_iso3').sum().reset_index().rename(columns={
        'to_iso3': 'iso3', 
        'supply_wheat': 'wheat_cons', 
        'supply_maize': 'maize_cons', 
        'supply_rice': 'rice_cons', 
        'supply_other_cereals': 'other_cereals_cons', 
        'supply_cereals_all': 'cereals_all_cons'
    })).merge(trade[trade['from_iso3']!=trade['to_iso3']].drop('from_iso3', axis=1).groupby('to_iso3').sum().reset_index().rename(columns={
        'to_iso3': 'iso3', 
        'supply_wheat': 'wheat_inflows', 
        'supply_maize': 'maize_inflows', 
        'supply_rice': 'rice_inflows', 
        'supply_other_cereals': 'other_cereals_inflows', 
        'supply_cereals_all': 'cereals_all_inflows'
    })).merge(trade[trade['from_iso3']!=trade['to_iso3']].drop('to_iso3', axis=1).groupby('from_iso3').sum().reset_index().rename(columns={
        'from_iso3': 'iso3', 
        'supply_wheat': 'wheat_outflows', 
        'supply_maize': 'maize_outflows', 
        'supply_rice': 'rice_outflows', 
        'supply_other_cereals': 'other_cereals_outflows', 
        'supply_cereals_all': 'cereals_all_outflows'
    }))


    reductions_admin = reductions_admin.rename(columns={'wheat_production': 'wheat_prod', 
                                                        'maize_production': 'maize_prod', 
                                                        'rice_production': 'rice_prod'})
    reductions_admin['other_cereals_prod'] = reductions_admin['barley_production'] + reductions_admin['millet_production'] \
    + reductions_admin['sorghum_production'] + reductions_admin['other_cereals_production']
    
    reductions_admin['cereals_all_prod'] = reductions_admin['wheat_prod'] + reductions_admin['rice_prod'] \
    + reductions_admin['maize_prod'] + reductions_admin['other_cereals_prod']
    
    df_admin = reductions_admin[['ID', 'iso3', 'wheat_prod', 'rice_prod', 'maize_prod', 'other_cereals_prod', 'cereals_all_prod', 
                                 'buffaloes_totals', 'cattle_totals', 'chickens_totals', 'ducks_totals', 'goats_totals', 
                                 'horses_totals', 'pigs_totals', 'sheep_totals', 'pop_totals', 'gdp_totals']]

    df_admin = pred_consumption(df_country, df_admin, scatter_plot=False)
    
    return df_country, df_admin



if __name__ == '__main__':

    # fao trade
    trade, iso3_codes = get_trade_data()
    trade.to_csv('../../data/combined_data/fao_trade.csv', index=False)

    # admin transport
    transport = get_transport_data(iso3_codes)
    transport.to_parquet('../../data/combined_data/transport.parquet.gzip', index=False)

    # admin production and consumption
    df_country, df_admin = get_admin_data(trade, iso3_codes)
    df_country.to_csv('../../data/combined_data/country_data.csv', index=False)
    df_admin.to_csv('../../data/combined_data/admin_data.csv', index=False)
    