import geopandas as gpd
import pandas as pd
import numpy as np
from functools import reduce
import statsmodels.api as sm
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def get_totals(crop, scatter_plot=False):

    excl = ['COK', 'FRO', 'GLP', 'GUF', 'KIR', 'MTQ', 'REU', 'TKL']
    
    supply_2017 = pd.read_csv(f"../../data/FAO_re_export/supply_matrix_{crop}_2017.csv")
    supply_2018 = pd.read_csv(f"../../data/FAO_re_export/supply_matrix_{crop}_2018.csv")
    supply_2019 = pd.read_csv(f"../../data/FAO_re_export/supply_matrix_{crop}_2019.csv")
    supply_2020 = pd.read_csv(f"../../data/FAO_re_export/supply_matrix_{crop}_2020.csv")
    supply_2021 = pd.read_csv(f"../../data/FAO_re_export/supply_matrix_{crop}_2021.csv")
    
    supply = pd.concat([supply_2017, supply_2018, supply_2019, supply_2020, supply_2021]).groupby('iso3').mean().reset_index()
    
    # melt supply data 
    supply = supply.rename(columns={'iso3': 'from_iso3'})
    supply = supply.melt(id_vars=['from_iso3'], value_vars=supply['from_iso3'].values.tolist()).rename(columns={'variable': 'to_iso3', 'value': 'trade'})
    supply = supply[supply['from_iso3']!=supply['to_iso3']]
    supply = supply[(~supply['from_iso3'].isin(excl)) & (~supply['to_iso3'].isin(excl))]

    trade_2017 = pd.read_csv(f"../../data/FAO_bal_trade_mat/trade_matrix_{crop}_2017.csv")
    trade_2018 = pd.read_csv(f"../../data/FAO_bal_trade_mat/trade_matrix_{crop}_2018.csv")
    trade_2019 = pd.read_csv(f"../../data/FAO_bal_trade_mat/trade_matrix_{crop}_2019.csv")
    trade_2020 = pd.read_csv(f"../../data/FAO_bal_trade_mat/trade_matrix_{crop}_2020.csv")
    trade_2021 = pd.read_csv(f"../../data/FAO_bal_trade_mat/trade_matrix_{crop}_2021.csv")
    
    trade = pd.concat([trade_2017, trade_2018, trade_2019, trade_2020, trade_2021]).groupby('iso3').mean().reset_index()
    
    # melt trade data 
    trade = trade.rename(columns={'iso3': 'from_iso3'})
    trade = trade.melt(id_vars=['from_iso3'], value_vars=trade['from_iso3'].values.tolist()).rename(columns={'variable': 'to_iso3', 'value': 'trade'})
    trade = trade[trade['from_iso3']!=trade['to_iso3']]
    trade = trade[(~trade['from_iso3'].isin(excl)) & (~trade['to_iso3'].isin(excl))]

    fao_prod_2017 = pd.read_csv(f"../../data/FAO_prod_mat/prod_matrix_{crop}_2017.csv").rename(columns={'prod': 'prod_2017'})
    fao_prod_2018 = pd.read_csv(f"../../data/FAO_prod_mat/prod_matrix_{crop}_2018.csv").rename(columns={'prod': 'prod_2018'})
    fao_prod_2019 = pd.read_csv(f"../../data/FAO_prod_mat/prod_matrix_{crop}_2019.csv").rename(columns={'prod': 'prod_2019'})
    fao_prod_2020 = pd.read_csv(f"../../data/FAO_prod_mat/prod_matrix_{crop}_2020.csv").rename(columns={'prod': 'prod_2020'})
    fao_prod_2021 = pd.read_csv(f"../../data/FAO_prod_mat/prod_matrix_{crop}_2021.csv").rename(columns={'prod': 'prod_2021'})
    
    fao_prod = fao_prod_2017.merge(fao_prod_2018, on='iso3').merge(fao_prod_2019, on='iso3').merge(
        fao_prod_2020, on='iso3').merge(fao_prod_2021, on='iso3')
    fao_prod['fao_prod'] = (fao_prod['prod_2017'] + fao_prod['prod_2018'] + fao_prod['prod_2019'] \
    + fao_prod['prod_2020'] + fao_prod['prod_2021']) / 5
    fao_prod = fao_prod[(~fao_prod['iso3'].isin(excl))]

    print(f"production: {fao_prod['fao_prod'].sum()}") # production
    print(f"trade: {trade['trade'].sum()}") # trade
    print(f"supply: {supply['trade'].sum()}") # re-export
    print(f"corr: {np.corrcoef(trade['trade'], supply['trade'])[0][1]}")
    print(f"corr log: {np.corrcoef(np.log(trade['trade']+1), np.log(supply['trade']+1))[0][1]}")
    print(f"r2: {r2_score(trade['trade'], supply['trade'])}")
    print(f"r2 log: {r2_score(np.log(trade['trade']+1), np.log(supply['trade']+1))}")

    if scatter_plot:
        plt.figure(figsize=(9, 4))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=trade['trade'], y=supply['trade'])
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=np.log(trade['trade']+1), y=np.log(supply['trade']+1))

    df = supply.groupby('from_iso3')[['trade']].sum().reset_index().rename(columns={'from_iso3': 'iso3', 'trade': f'{crop}_outflows'}).merge(
            supply.groupby('to_iso3')[['trade']].sum().reset_index().rename(columns={'to_iso3': 'iso3', 'trade': f'{crop}_inflows'})
        ).merge(
            trade.groupby('from_iso3')[['trade']].sum().reset_index().rename(columns={'from_iso3': 'iso3', 'trade': f'{crop}_exports'}).merge(
            trade.groupby('to_iso3')[['trade']].sum().reset_index().rename(columns={'to_iso3': 'iso3', 'trade': f'{crop}_imports'})
        )
        ).merge(
            fao_prod[['iso3', 'fao_prod']].rename(columns={'fao_prod': f'{crop}_prod'})
        )

    df[f'{crop}_cons_trade'] = df[f'{crop}_prod'] + df[f'{crop}_imports'] - df[f'{crop}_exports']
    df[f'{crop}_cons_flows'] = df[f'{crop}_prod'] + df[f'{crop}_inflows'] - df[f'{crop}_outflows']

    return df

def country_totals(scatter_plot=False):

    crops = ['Wheat', 'Maize (corn)', 'Rye', 'Barley', 'Oats', 'Sorghum', 
             'Rice, paddy (rice milled equivalent)', 'Buckwheat', 
             'Millet', 'Quinoa', 'Canary seed', 'Fonio', 'Mixed grain', 'Triticale', 
             'Cereals n.e.c.', 'cereals_all']
    
    df_list = []
    
    for crop in crops:
        print(crop)
        df = get_totals(crop, scatter_plot)
        df_list.append(df)
    
    dfs = [df.set_index('iso3') for df in df_list]
    master_df_country = pd.concat(dfs, axis=1).reset_index()
    master_df_country = master_df_country.rename(columns={
        'Maize (corn)_outflows': 'maize_outflows',
        'Maize (corn)_inflows': 'maize_inflows',
        'Maize (corn)_exports': 'maize_exports',
        'Maize (corn)_imports': 'maize_imports',
        'Maize (corn)_prod': 'maize_prod',
        'Maize (corn)_cons_trade': 'maize_cons_trade',
        'Maize (corn)_cons_flows': 'maize_cons_flows',
        'Wheat_outflows': 'wheat_outflows',
        'Wheat_inflows': 'wheat_inflows',
        'Wheat_exports': 'wheat_exports',
        'Wheat_imports': 'wheat_imports',
        'Wheat_prod': 'wheat_prod',
        'Wheat_cons_trade': 'wheat_cons_trade',
        'Wheat_cons_flows': 'wheat_cons_flows',
        'Barley_outflows': 'barley_outflows',
        'Barley_inflows': 'barley_inflows',
        'Barley_exports': 'barley_exports',
        'Barley_imports': 'barley_imports',
        'Barley_prod': 'barley_prod',
        'Barley_cons_trade': 'barley_cons_trade',
        'Barley_cons_flows': 'barley_cons_flows',
        'Rye_outflows': 'rye_outflows',
        'Rye_inflows': 'rye_inflows',
        'Rye_exports': 'rye_exports',
        'Rye_imports': 'rye_imports',
        'Rye_prod': 'rye_prod',
        'Rye_cons_trade': 'rye_cons_trade',
        'Rye_cons_flows': 'rye_cons_flows',
        'Sorghum_outflows': 'sorghum_outflows',
        'Sorghum_inflows': 'sorghum_inflows',
        'Sorhum_exports': 'sorghum_exports',
        'Sorghum_imports': 'sorghum_imports',
        'Sorghum_prod': 'sorghum_prod',
        'Sorghum_cons_trade': 'sorghum_cons_trade',
        'Sorghum_cons_flows': 'sorghum_cons_flows',
        'Oats_outflows': 'oats_outflows',
        'Oats_inflows': 'oats_inflows',
        'Oats_exports': 'oats_exports',
        'Oats_imports': 'oats_imports',
        'Oats_prod': 'oats_prod',
        'Oats_cons_trade': 'oats_cons_trade',
        'Oats_cons_flows': 'oats_cons_flows',
        'Millet_outflows': 'millet_outflows',
        'Millet_inflows': 'millet_inflows',
        'Millet_exports': 'millet_exports',
        'Millet_imports': 'millet_imports',
        'Millet_prod': 'millet_prod',
        'Millet_cons_trade': 'millet_cons_trade',
        'Millet_cons_flows': 'millet_cons_flows',
        'Buckwheat_outflows': 'buckwheat_outflows',
        'Buckwheat_inflows': 'buckwheat_inflows',
        'Buckwheat_exports': 'buckwheat_exports',
        'Buckwheat_imports': 'buckwheat_imports',
        'Buckwheat_prod': 'buckwheat_prod',
        'Buckwheat_cons_trade': 'buckwheat_cons_trade',
        'Buckwheat_cons_flows': 'buckwheat_cons_flows',
        'Quinoa_outflows': 'quinoa_outflows',
        'Quinoa_inflows': 'quinoa_inflows',
        'Quinoa_exports': 'quinoa_exports',
        'Quinoa_imports': 'quinoa_imports',
        'Quinoa_prod': 'quinoa_prod',
        'Quinoa_cons_trade': 'quinoa_cons_trade',
        'Quinoa_cons_flows': 'quinoa_cons_flows',
        'Canary seed_outflows': 'canary_seed_outflows',
        'Canary seed_inflows': 'canary_seed_inflows',
        'Canary seed_exports': 'canary_seed_exports',
        'Canary seed_imports': 'canary_seed_imports',
        'Canary seed_prod': 'canary_seed_prod',
        'Canary seed_cons_trade': 'canary_seed_cons_trade',
        'Canary seed_cons_flows': 'canary_seed_cons_flows',
        'Fonio_outflows': 'fonio_outflows',
        'Fonio_inflows': 'fonio_inflows',
        'Fonio_exports': 'fonio_exports',
        'Fonio_imports': 'fonio_imports',
        'Fonio_prod': 'fonio_prod',
        'Fonio_cons_trade': 'fonio_cons_trade',
        'Fonio_cons_flows': 'fonio_cons_flows',
        'Mixed grain_outflows': 'mixed_grain_outflows',
        'Mixed grain_inflows': 'mixed_grain_inflows',
        'Mixed grain_exports': 'mixed_grain_exports',
        'Mixed grain_imports': 'mixed_grain_imports',
        'Mixed grain_prod': 'mixed_grain_prod',
        'Mixed grain_cons_trade': 'mixed_grain_cons_trade',
        'Mixed grain_cons_flows': 'mixed_grain_cons_flows',
        'Triticale_outflows': 'triticale_outflows',
        'Triticale_inflows': 'triticale_inflows',
        'Triticale_exports': 'triticale_exports',
        'Triticale_imports': 'triticale_imports',
        'Triticale_prod': 'triticale_prod',
        'Triticale_cons_trade': 'triticale_cons_trade',
        'Triticale_cons_flows': 'triticale_cons_flows',       
        'Rice, paddy (rice milled equivalent)_outflows': 'rice_outflows',
        'Rice, paddy (rice milled equivalent)_inflows': 'rice_inflows',
        'Rice, paddy (rice milled equivalent)_exports': 'rice_exports',
        'Rice, paddy (rice milled equivalent)_imports': 'rice_imports',
        'Rice, paddy (rice milled equivalent)_prod': 'rice_prod',
        'Rice, paddy (rice milled equivalent)_cons_trade': 'rice_cons_trade',
        'Rice, paddy (rice milled equivalent)_cons_flows': 'rice_cons_flows',
        'Cereals n.e.c._outflows': 'other_cereals_outflows',
        'Cereals n.e.c._inflows': 'other_cereals_inflows',
        'Cereals n.e.c._exports': 'other_cereals_exports',
        'Cereals n.e.c._imports': 'other_cereals_imports',
        'Cereals n.e.c._prod': 'other_cereals_prod',
        'Cereals n.e.c._cons_trade': 'other_cereals_cons_trade',
        'Cereals n.e.c._cons_flows': 'other_cereals_cons_flows',
    })

    # dropping the trade (i.e. before re-exporting algorithm) columns
    master_df_country = master_df_country.drop([col for col in master_df_country.columns if '_trade' in col], axis=1)
    master_df_country = master_df_country.drop([col for col in master_df_country.columns if 'ports' in col], axis=1)

    # grouping crops other than wheat, rice, maize into other
    master_df_country['other_cereals_prod'] = master_df_country['other_cereals_prod'] + master_df_country['rye_prod'] + master_df_country['barley_prod'] \
    + master_df_country['oats_prod'] + master_df_country['sorghum_prod'] + master_df_country['buckwheat_prod'] + master_df_country['millet_prod'] \
    + master_df_country['canary_seed_prod'] + master_df_country['fonio_prod'] + master_df_country['mixed_grain_prod'] + master_df_country['triticale_prod'] \
    + master_df_country['quinoa_prod']

    master_df_country['other_cereals_inflows'] = master_df_country['other_cereals_inflows'] + master_df_country['rye_inflows'] + master_df_country['barley_inflows'] \
    + master_df_country['oats_inflows'] + master_df_country['sorghum_inflows'] + master_df_country['buckwheat_inflows'] + master_df_country['millet_inflows'] \
    + master_df_country['canary_seed_inflows'] + master_df_country['fonio_inflows'] + master_df_country['mixed_grain_inflows'] + master_df_country['triticale_inflows'] \
    + master_df_country['quinoa_inflows']

    master_df_country['other_cereals_outflows'] = master_df_country['other_cereals_outflows'] + master_df_country['rye_outflows'] + master_df_country['barley_outflows'] \
    + master_df_country['oats_outflows'] + master_df_country['sorghum_outflows'] + master_df_country['buckwheat_outflows'] + master_df_country['millet_outflows'] \
    + master_df_country['canary_seed_outflows'] + master_df_country['fonio_outflows'] + master_df_country['mixed_grain_outflows'] + master_df_country['triticale_outflows'] \
    + master_df_country['quinoa_outflows']

    master_df_country['other_cereals_cons_flows'] = master_df_country['other_cereals_cons_flows'] + master_df_country['rye_cons_flows'] + master_df_country['barley_cons_flows'] \
    + master_df_country['oats_cons_flows'] + master_df_country['sorghum_cons_flows'] + master_df_country['buckwheat_cons_flows'] + master_df_country['millet_cons_flows'] \
    + master_df_country['canary_seed_cons_flows'] + master_df_country['fonio_cons_flows'] + master_df_country['mixed_grain_cons_flows'] + master_df_country['triticale_cons_flows'] \
    + master_df_country['quinoa_cons_flows']

    # dropping the crops grouped into other_cereals
    master_df_country = master_df_country.drop([col for col in master_df_country.columns if any(
        sub in col for sub in ['rye_', 'oats_', 'buckwheat_', 'quinoa_', 'barley_',
                               'canary_seed_', 'fonio_', 'mixed_grain_', 'triticale_', 
                               'sorghum_', 'millet_'])], axis=1)

    master_df_country = master_df_country.rename(columns={
         'wheat_cons_flows': 'wheat_cons',
         'maize_cons_flows': 'maize_cons',
         'rice_cons_flows': 'rice_cons',
         'other_cereals_cons_flows': 'other_cereals_cons',
         'cereals_all_cons_flows': 'cereals_all_cons'
         })

    return master_df_country

def get_bilateral_data(admin_level=True):

    # country level
    df_country_log_false = pd.read_parquet('../../data/combined_features/df_country_log_false_thresh_1.parquet.gzip')
    df_country_log_true = pd.read_parquet('../../data/combined_features/df_country_log_true_thresh_1.parquet.gzip')
    
    cols = ['freight_USD_t', 'transport_USD_t', 'time_h',
           'distance_km', 'border_USD_t', 'customs_cost',
           'Pref_Applied_AVE', 'supply_Wheat', 'trade_Wheat',
           'supply_Maize (corn)', 'trade_Maize (corn)', 'supply_Rye', 'trade_Rye',
           'supply_Barley', 'trade_Barley', 'supply_Oats', 'trade_Oats',
           'supply_Sorghum', 'trade_Sorghum', 'supply_Rice, paddy (rice milled equivalent)',
           'trade_Rice, paddy (rice milled equivalent)', 'supply_Buckwheat', 'trade_Buckwheat',
           'supply_Millet', 'trade_Millet', 'supply_Quinoa', 'trade_Quinoa',
           'supply_Canary seed', 'trade_Canary seed', 'supply_Fonio', 'trade_Fonio',
           'supply_Mixed grain', 'trade_Mixed grain', 'supply_Triticale', 'trade_Triticale',
           'supply_Cereals n.e.c.', 'trade_Cereals n.e.c.', 
           'supply_cereals_all', 'trade_cereals_all', 'from_barley_area',
           'from_maize_area', 'from_millet_area', 'from_rice_area',
           'from_sorghum_area', 'from_wheat_area', 'from_other_cereals_area',
           'from_barley_production', 'from_maize_production',
           'from_millet_production', 'from_rice_production',
           'from_sorghum_production', 'from_wheat_production',
           'from_other_cereals_production', 'from_buffaloes', 'from_cattle',
           'from_chickens', 'from_ducks', 'from_goats', 'from_horses', 'from_pigs',
           'from_sheep', 'from_pop', 'from_gdp', 'from_area', 'from_built_surface',
           'from_built_volume_total', 'from_built_volume_nres', 'to_barley_area', 'to_maize_area',
           'to_millet_area', 'to_rice_area', 'to_sorghum_area', 'to_wheat_area',
           'to_other_cereals_area', 'to_barley_production', 'to_maize_production',
           'to_millet_production', 'to_rice_production', 'to_sorghum_production',
           'to_wheat_production', 'to_other_cereals_production', 'to_buffaloes',
           'to_cattle', 'to_chickens', 'to_ducks', 'to_goats', 'to_horses',
           'to_pigs', 'to_sheep', 'to_pop', 'to_gdp', 'to_area', 'to_built_surface',
           'to_built_volume_total', 'to_built_volume_nres']
    
    log_cols = [i+'_log' for i in cols]
    
    df_country = df_country_log_false.merge(df_country_log_true.rename(columns=dict(zip(cols, log_cols))))
    # dropping the trade (i.e. before re-exporting algorithm) columns
    df_country = df_country.drop([col for col in df_country.columns if 'trade_' in col], axis=1)

    # grouping crops other than wheat, rice, maize into other
    df_country['supply_other_cereals'] = df_country['supply_Rye'] + df_country['supply_Oats'] + df_country['supply_Buckwheat'] \
    + df_country['supply_Quinoa'] + df_country['supply_Barley'] + df_country['supply_Sorghum'] \
    + df_country['supply_Canary seed'] + df_country['supply_Fonio'] + df_country['supply_Mixed grain'] + df_country['supply_Triticale'] \
    + df_country['supply_Millet'] + df_country['supply_Cereals n.e.c.']
    df_country['supply_other_cereals_exists'] = 0
    df_country.loc[df_country['supply_other_cereals']>=1, 'supply_other_cereals_exists'] = 1
    df_country['supply_other_cereals_log'] = np.log(df_country['supply_other_cereals'] + 1)
    
    df_country = df_country.rename(columns={
         'supply_Wheat': 'supply_wheat',
         'supply_Wheat_exists': 'supply_wheat_exists',
         'supply_Wheat_log': 'supply_wheat_log',
         'supply_Maize (corn)': 'supply_maize',
         'supply_Maize (corn)_exists': 'supply_maize_exists',
         'supply_Maize (corn)_log': 'supply_maize_log',
         'supply_Rice, paddy (rice milled equivalent)': 'supply_rice',
         'supply_Rice, paddy (rice milled equivalent)_exists': 'supply_rice_exists',
         'supply_Rice, paddy (rice milled equivalent)_log': 'supply_rice_log'
         })

    # dropping the crops grouped into other_cereals
    df_country = df_country.drop([col for col in df_country.columns if any(
        sub in col for sub in ['_Rye', '_Oats', '_Buckwheat', '_Quinoa', '_Barley', 
                               'Canary seed_', 'Fonio_', 'Mixed grain_', 'Triticale_', 
                               '_Sorghum', '_Millet', '_Cereals n.e.c.'])], axis=1)
    
    if admin_level:
        # admin level
        df_admin_log_false = pd.read_parquet('../../data/combined_features/df_admin_log_false.parquet.gzip')
        df_admin_log_true = pd.read_parquet('../../data/combined_features/df_admin_log_true.parquet.gzip')
        df_admin = df_admin_log_false.merge(df_admin_log_true.rename(columns=dict(zip(cols, log_cols))))
        df_admin = df_admin[(df_admin['from_id']!='NZL.10_1')].reset_index(drop=True)
    
        # some error in domestic transport coding
        df_admin.loc[df_admin['to_id']=='PAK.1_1', 'to_iso3'] = 'PAK'
    else:
        df_admin=None

    return df_country, df_admin


def fit_consumption(master_df_country, df_country, crop='cereals_all', cols=None, scatter_plot=False, print_reg=False):
    
    if cols is None:
        cols = ['buffaloes_log', 'cattle_log', 'chickens_log', 
                'ducks_log', 'goats_log', 'horses_log', #'pigs_log', 'sheep_log', 
                'pop_log', 'gdp_log', f'{crop}_prod_log']

    cons = master_df_country.copy()
    cons['wheat_prod_log'] = np.log(cons['wheat_prod'] + 1)
    cons['maize_prod_log'] = np.log(cons['maize_prod'] + 1)
    cons['rice_prod_log'] = np.log(cons['rice_prod'] + 1)
    cons['other_cereals_prod_log'] = np.log(cons['other_cereals_prod'] + 1)
    cons['cereals_all_prod_log'] = np.log(cons['cereals_all_prod'] + 1)

    cons = cons[cons[f'{crop}_cons']>0][['iso3', f'{crop}_cons', 'wheat_prod_log', 'maize_prod_log', 
                                         'rice_prod_log', 'other_cereals_prod_log', 
                                         'cereals_all_prod_log']].merge(df_country[[
        'from_iso3', 'from_buffaloes_log', 'from_cattle_log', 'from_chickens_log', 'from_ducks_log', 'from_goats_log',
        'from_horses_log', 'from_pigs_log', 'from_sheep_log', 'from_pop_log', 'from_gdp_log']].drop_duplicates(), 
                    left_on='iso3', right_on='from_iso3').drop('from_iso3', axis=1).rename(columns={
        'from_buffaloes_log': 'buffaloes_log', 
        'from_cattle_log': 'cattle_log', 
        'from_chickens_log': 'chickens_log', 
        'from_ducks_log': 'ducks_log', 
        'from_goats_log': 'goats_log', 
        'from_horses_log': 'horses_log', 
        'from_pigs_log': 'pigs_log', 
        'from_sheep_log': 'sheep_log',           
        'from_pop_log': 'pop_log',
        'from_gdp_log': 'gdp_log'
                    })

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

def pred_consumption(master_df_country, df_country, df_admin, scatter_plot=False):

    cons_admin = df_admin[['from_iso3', 'from_id', 'from_buffaloes_log', 'from_cattle_log', 'from_chickens_log', 
                           'from_ducks_log', 'from_goats_log', 'from_horses_log', 'from_pigs_log', 
                           'from_sheep_log', 'from_pop_log', 'from_gdp_log', 'from_barley_production',
                           'from_maize_production', 'from_millet_production', 'from_rice_production', 
                           'from_sorghum_production', 'from_wheat_production', 'from_other_cereals_production'
                           ]].drop_duplicates().reset_index(drop=True).rename(columns={'from_iso3': 'iso3', 'from_id': 'ID',
                                                                                       'from_buffaloes_log': 'buffaloes_log', 
                                                                                       'from_cattle_log': 'cattle_log', 
                                                                                       'from_chickens_log': 'chickens_log', 
                                                                                       'from_ducks_log': 'ducks_log', 
                                                                                       'from_goats_log': 'goats_log', 
                                                                                       'from_horses_log': 'horses_log', 
                                                                                       'from_pigs_log': 'pigs_log', 
                                                                                       'from_sheep_log': 'sheep_log', 
                                                                                       'from_pop_log': 'pop_log', 
                                                                                       'from_gdp_log': 'gdp_log', 
                                                                                       'from_barley_production': 'barley_prod',
                                                                                       'from_maize_production': 'maize_prod', 
                                                                                       'from_millet_production': 'millet_prod', 
                                                                                       'from_rice_production': 'rice_prod', 
                                                                                       'from_sorghum_production': 'sorghum_prod', 
                                                                                       'from_wheat_production': 'wheat_prod', 
                                                                                       'from_other_cereals_production': 'other_cereals_prod'})
    
    cons_admin['other_cereals_prod'] = cons_admin['other_cereals_prod'] + cons_admin['barley_prod'] \
    + cons_admin['millet_prod'] + cons_admin['sorghum_prod']
    cons_admin['cereals_all_prod'] = cons_admin['wheat_prod'] + cons_admin['maize_prod'] \
    + cons_admin['rice_prod'] + cons_admin['other_cereals_prod']

    # dropping the crops grouped into other_cereals
    cons_admin = cons_admin.drop([col for col in cons_admin.columns if any(
        sub in col for sub in ['barley_', 'millet_', 'sorghum_'])], axis=1)
    
    cons_admin['wheat_prod_log'] = np.log(cons_admin['wheat_prod'] + 1)
    cons_admin['maize_prod_log'] = np.log(cons_admin['maize_prod'] + 1)
    cons_admin['rice_prod_log'] = np.log(cons_admin['rice_prod'] + 1)
    cons_admin['other_cereals_prod_log'] = np.log(cons_admin['other_cereals_prod'] + 1)
    cons_admin['cereals_all_prod_log'] = np.log(cons_admin['cereals_all_prod'] + 1)

    master_df_admin = []
        
    for crop in ['wheat', 'maize', 'rice', 'other_cereals', 'cereals_all']:
        print(crop)
    
        cons = cons_admin.copy()
        
        cols = ['buffaloes_log', 'cattle_log', 'chickens_log', 
                'ducks_log', 'goats_log', 'horses_log', 'pigs_log', 'sheep_log', 
                'pop_log', 'gdp_log', f'{crop}_prod_log']
    
        if crop in ['rice', 'other_cereals', 'cereals_all']:
            cols = ['buffaloes_log', 'cattle_log', 'chickens_log', 
                    'ducks_log', 'goats_log', 'horses_log', #'pigs_log', 'sheep_log', 
                    'pop_log', 'gdp_log', f'{crop}_prod_log']

        print('R2 while fitting')
        res = fit_consumption(master_df_country, df_country, crop=crop, cols=cols, scatter_plot=scatter_plot, print_reg=False)
    
        cons[f'{crop}_cons'] = res.predict(cons[cols])
    
        cons_comp = cons.groupby('iso3').sum()[[f'{crop}_cons']].reset_index().rename(columns={
            f'{crop}_cons':f'{crop}_cons_total'}).merge(master_df_country[['iso3', f'{crop}_cons']].rename(columns={
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
        cons = cons[['iso3', 'ID', f'{crop}_prod', f'{crop}_cons']]
        cons[f'{crop}_exp_max'] = cons[f'{crop}_prod'] - cons[f'{crop}_cons']
        
        master_df_admin.append(cons)
        
    master_df_admin = reduce(lambda left, right: pd.merge(left, right, on=['ID', 'iso3'], how='inner'), master_df_admin)

    return master_df_admin

def read_other_data():

    cols = ['Pref_Applied_AVE', 'border_USD_t', 'customs_cost',
            'distance_km', 'freight_USD_t', 'from_area',
            'from_barley_area', 'from_barley_production',
            'from_buffaloes', 'from_built_surface',
            'from_built_volume_nres', 'from_built_volume_total',
            'from_cattle', 'from_chickens', 'from_ducks',
            'from_gdp', 'from_goats', 'from_horses',
            'from_maize_area', 'from_maize_production',
            'from_millet_area', 'from_millet_production',
            'from_other_cereals_area', 'from_other_cereals_production',
            'from_pigs', 'from_pop', 'from_rice_area',
            'from_rice_production', 'from_sheep', 'from_sorghum_area',
            'from_sorghum_production', 'from_wheat_area',
            'from_wheat_production', 'supply_cereals_all',
            'supply_maize', 'supply_other_cereals', 'supply_rice',
            'supply_wheat', 'time_h', 'to_area', 'to_barley_area',
            'to_barley_production', 'to_buffaloes', 'to_built_surface',
            'to_built_volume_nres', 'to_built_volume_total',
            'to_cattle', 'to_chickens', 'to_ducks', 'to_gdp',
            'to_goats', 'to_horses', 'to_maize_area',
            'to_maize_production', 'to_millet_area',
            'to_millet_production', 'to_other_cereals_area',
            'to_other_cereals_production', 'to_pigs', 'to_pop',
            'to_rice_area', 'to_rice_production', 'to_sheep',
            'to_sorghum_area', 'to_sorghum_production', 'to_wheat_area',
            'to_wheat_production', 'transport_USD_t']
    
    log_cols = [i+'_log' for i in cols]

    df_india_log_false = pd.read_parquet('../../data/combined_features/df_india_log_false_thresh_1.parquet.gzip')
    df_india_log_true = pd.read_parquet('../../data/combined_features/df_india_log_true_thresh_1.parquet.gzip')
    df_india = df_india_log_false.merge(df_india_log_true.rename(columns=dict(zip(cols, log_cols))))

    df_us_china_log_false = pd.read_parquet('../../data/combined_features/df_us_china_log_false_thresh_1.parquet.gzip')
    df_us_china_log_true = pd.read_parquet('../../data/combined_features/df_us_china_log_true_thresh_1.parquet.gzip')
    df_us_china = df_us_china_log_false.merge(df_us_china_log_true.rename(columns=dict(zip(cols, log_cols))))

    df_us_state_int_log_false = pd.read_parquet('../../data/combined_features/df_us_state_int_log_false_thresh_1.parquet.gzip')
    df_us_state_int_log_true = pd.read_parquet('../../data/combined_features/df_us_state_int_log_true_thresh_1.parquet.gzip')
    df_us_state_int = df_us_state_int_log_false.merge(df_us_state_int_log_true.rename(columns=dict(zip(cols, log_cols))))

    return df_india, df_us_china, df_us_state_int
 
    