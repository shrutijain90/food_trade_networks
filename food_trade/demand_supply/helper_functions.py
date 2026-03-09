import geopandas as gpd
import pandas as pd
import numpy as np
from functools import reduce
import statsmodels.api as sm
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
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



def calculate_sensitivity_metrics(df_base, df_test, description="Beta 1.5 vs Beta 1.2"):
    """
    Compares two network flow dataframes to test sensitivity to parameters.
    Expects dataframes with 'from_idx', 'to_idx', and 'flow' columns.
    """
    print(f"\n{'='*60}")
    print(f"SENSITIVITY ANALYSIS: {description}")
    print(f"{'='*60}")

    # 1. Merge the two networks on Origin-Destination pairs
    # Use an outer join so we capture links that exist in one but not the other
    merged = pd.merge(
        df_base[['from_idx', 'to_idx', 'flow']], 
        df_test[['from_idx', 'to_idx', 'flow']], 
        on=['from_idx', 'to_idx'], 
        how='outer', 
        suffixes=('_base', '_test')
    )

    # Fill NaNs with 0 (a link active in one scenario might be dead in the other)
    merged['flow_base'] = merged['flow_base'].fillna(0)
    merged['flow_test'] = merged['flow_test'].fillna(0)

    # 2. Calculate Common Part of Flows (CPF)
    # How much literal tonnage perfectly overlaps between the two networks?
    min_flows = np.minimum(merged['flow_base'], merged['flow_test'])
    total_base = merged['flow_base'].sum()
    total_test = merged['flow_test'].sum()
    
    cpf = (2 * min_flows.sum()) / (total_base + total_test)

    # 3. Calculate Spearman Rank Correlation
    # Do the most important links remain the most important?
    # We use Spearman (rank) rather than Pearson (linear) because gravity flows follow a power law
    spearman_corr, p_value = spearmanr(merged['flow_base'], merged['flow_test'])

    # 4. Top 5% Link Stability (The "Heavy Hitters")
    # What percentage of the top 5% of links in the base model are also in the top 5% of the test model?
    top_n = max(1, int(len(df_base) * 0.05))
    
    top_base_links = set(
        zip(df_base.nlargest(top_n, 'flow')['from_idx'], df_base.nlargest(top_n, 'flow')['to_idx'])
    )
    top_test_links = set(
        zip(df_test.nlargest(top_n, 'flow')['from_idx'], df_test.nlargest(top_n, 'flow')['to_idx'])
    )
    
    overlap = len(top_base_links.intersection(top_test_links))
    top_stability = overlap / top_n

    # Print Results
    print(f"  Total Links Base: {len(df_base):,}")
    print(f"  Total Links Test: {len(df_test):,}")
    print(f"  Union of Links:   {len(merged):,}")
    print(f"\n  --- Metrics ---")
    print(f"  Spearman Rank Correlation: {spearman_corr:.4f} (p-val: {p_value:.2e})")
    print(f"  Common Part of Flows (CPF): {cpf:.4f}")
    print(f"  Top 5% Heavy Hitter Overlap: {top_stability * 100:.1f}%")

    return merged, spearman_corr, cpf


def process_faf_net_flows(admin):
    # Converts gross FAF freight movements into net supply-demand flows.

    cols_to_keep = [
        'dms_origst', 'dms_destst', 'sctg2', 
        'tons_2018', 'tons_2019', 'tons_2020', 'tons_2021', 'tons_2022'
    ]
    
    # Read the CSV 
    faf_raw = pd.read_csv('../../data/other_data/US_FAF_data/FAF5.7.1_State_2018-2024/FAF5.7.1_State_2018-2024.csv', 
                          usecols=cols_to_keep, dtype={'sctg2': str}
                         )
    
    # Filter for Cereal Grains (SCTG Code 02)
    grains_df = faf_raw[faf_raw['sctg2'] == '02'].copy()
    
    # Calculate the 2018-2022 Average Tonnage
    grains_df['tons'] = grains_df[['tons_2018', 'tons_2019', 'tons_2020', 'tons_2021', 'tons_2022']].mean(axis=1)
    grains_df = grains_df[['dms_origst', 'dms_destst', 'sctg2', 'tons']]
    grains_df['tons'] = grains_df['tons'] * 1000 # thousand tons to tons
    
    print(f"Extracted {len(grains_df):,} gross flow records for grains.")
    
    # Aggregate total gross flows between all Origin-Destination pairs
    # We ignore transport mode here, we just want total volume
    gross_flows = grains_df.groupby(['dms_origst', 'dms_destst'])['tons'].sum().reset_index()
    
    # Create a merged dataframe to align A->B with B->A
    net_df = pd.merge(
        gross_flows, 
        gross_flows, 
        left_on=['dms_origst', 'dms_destst'], 
        right_on=['dms_destst', 'dms_origst'], 
        how='left', 
        suffixes=('_out', '_in')
    )
    
    # Fill NaNs with 0 (for one-way relationships)
    net_df['tons_in'] = net_df['tons_in'].fillna(0)
    
    # Calculate Net Flow: max(0, Outbound - Inbound)
    net_df['net_flow'] = np.maximum(0, net_df['tons_out'] - net_df['tons_in'])
    
    # Clean up and format to match your model's output style
    # Filter out the zero-net flows and self-flows (intra-state/intra-zone)
    final_net_flows = net_df[
        (net_df['net_flow'] > 0) & 
        (net_df['dms_origst_out'] != net_df['dms_destst_out'])
    ][['dms_origst_out', 'dms_destst_out', 'net_flow']].rename(
        columns={'dms_origst_out': 'from_id', 'dms_destst_out': 'to_id', 'net_flow': 'flow'}
    )
    print(f"  Extracted {len(final_net_flows):,} active net subnational links.")

    # Map to admin IDs
    state_dict = pd.read_excel('../../data/other_data/US_FAF_data/FAF5.7.1_State_2018-2024/FAF5_metadata.xlsx', 
                      sheet_name='State')
    
    state_dict.loc[state_dict['Description']=='Washington DC', 'Description'] = 'District of Columbia'
    state_dict = state_dict.merge(admin[admin['iso3']=='USA'][['admin_name', 'ID']], left_on='Description', right_on='admin_name')
    state_dict = state_dict[['Numeric Label', 'ID']]

    final_net_flows = final_net_flows.merge(state_dict, left_on='from_id', right_on='Numeric Label').drop(
        ['Numeric Label', 'from_id'], axis=1).rename(columns={'ID': 'from_id'}).merge(
        state_dict, left_on='to_id', right_on='Numeric Label').drop(
        ['Numeric Label', 'to_id'], axis=1).rename(columns={'ID': 'to_id'})
    
    return final_net_flows


def process_harris_data(admin, trade_thresh=1):
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
        df_list.append(df)

    ind_wheat = df_list[0].rename(columns={'value': 'supply_wheat'})
    ind_rice = df_list[1].rename(columns={'value': 'supply_rice'})
    ind_maize = df_list[2].rename(columns={'value': 'supply_maize'})
    ind_other = df_list[3].rename(columns={'value': 'supply_other_cereals'})

    trade = ind_wheat.merge(ind_rice).merge(ind_maize).merge(ind_other)
    trade = trade.merge(admin[admin['iso3']=='IND'][['iso3', 'ID', 'admin_name']], left_on='from_admin', right_on='admin_name').drop(
        ['from_admin', 'admin_name'], axis=1).rename(columns={'iso3': 'from_iso3', 'ID': 'from_id'})
    trade = trade.merge(admin[admin['iso3']=='IND'][['iso3', 'ID', 'admin_name']], left_on='to_admin', right_on='admin_name').drop(
        ['to_admin', 'admin_name'], axis=1).rename(columns={'iso3': 'to_iso3', 'ID': 'to_id'})
    
    trade['supply_combined_crops'] = trade[['supply_wheat', 'supply_rice', 'supply_maize', 'supply_other_cereals']].sum(axis=1)
    
    return trade


