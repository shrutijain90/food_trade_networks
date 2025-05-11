# Usage: python -m food_trade.flow_prediction.run_ml

import geopandas as gpd
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle
import joblib

from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, GammaRegressor, LinearRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold
from sklearn.metrics import make_scorer, accuracy_score, matthews_corrcoef, f1_score, root_mean_squared_error, r2_score, roc_auc_score, precision_score, recall_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model

from food_trade.flow_prediction.helper_functions import *

tf.random.set_seed(15)

X_cols_all = [ 
    'transport_USD_t_log',
    'time_h_log',
    'distance_km_log', 
    'border_USD_t_log', 
    'customs_cost',
    'tariff_log', 
    'from_price_log',
    'from_barley_area_log', 
    'from_maize_area_log', 
    'from_millet_area_log', 
    'from_rice_area_log',
    'from_sorghum_area_log', 
    'from_wheat_area_log', 
    'from_other_cereals_area_log',
    'from_barley_production_log', 
    'from_maize_production_log',
    'from_millet_production_log', 
    'from_rice_production_log',
    'from_sorghum_production_log', 
    'from_wheat_production_log',
    'from_other_cereals_production_log', 
    'from_buffaloes_log', 
    'from_cattle_log',
    'from_chickens_log', 
    'from_ducks_log', 
    'from_goats_log', 
    'from_horses_log', 
    'from_pigs_log',
    'from_sheep_log', 
    'from_pop_log', 
    'from_gdp_log',
    'from_area_log', 
    'from_built_volume_total_log',
    'from_region_0', 
    'from_region_1', 
    'from_region_2', 
    'from_region_3',
    'from_region_4', 
    'from_region_5',
    'to_price_log',
    'to_barley_area_log',   
    'to_maize_area_log',
    'to_millet_area_log', 
    'to_rice_area_log', 
    'to_sorghum_area_log', 
    'to_wheat_area_log',
    'to_other_cereals_area_log', 
    'to_barley_production_log', 
    'to_maize_production_log',
    'to_millet_production_log', 
    'to_rice_production_log', 
    'to_sorghum_production_log',
    'to_wheat_production_log', 
    'to_other_cereals_production_log', 
    'to_buffaloes_log',
    'to_cattle_log', 
    'to_chickens_log', 
    'to_ducks_log', 
    'to_goats_log', 
    'to_horses_log',
    'to_pigs_log', 
    'to_sheep_log', 
    'to_pop_log', 
    'to_gdp_log',
    'to_area_log', 
    'to_built_volume_total_log', 
    'to_region_0', 
    'to_region_1', 
    'to_region_2', 
    'to_region_3',
    'to_region_4', 
    'to_region_5',
    'domestic',
    'subnational'
]

def class_metrics_group(g, y_col):
    accuracy = accuracy_score(g[y_col], g['pred'])
    precision = precision_score(g[y_col], g['pred'])
    recall = recall_score(g[y_col], g['pred'])
    mcc = matthews_corrcoef(g[y_col], g['pred'])
    return pd.Series(dict(accuracy=accuracy, precision=precision, recall=recall, mcc=mcc))

# classification model selection - diff y variables and diff models (hyperparameters chosen on the basis of manual tuning)
# 5 train test splits for each model
def model_runs_class(df, n=5, disagg=True, suffix='all'):

    df_all_crops = df.copy()

    for crop in ['wheat', 'rice', 'maize', 'other_cereals', 'cereals_all']:
        print(crop)
        df_class = pd.DataFrame(columns=['model', 'crop', 'iteration', 'dataset', 'split',
                                         'accuracy', 'precision', 'recall', 'mcc'])
        y_col = f'supply_{crop}_exists'
        df = df_all_crops[df_all_crops[y_col].notnull()]
    
        # add prices to features
        prices = pd.read_csv(f'../../data/FAO_prod_prices/prod_prices_{crop}.csv')
        df = df.merge(prices[['iso3', 'Producer_price']], left_on='from_iso3', right_on='iso3').drop('iso3', axis=1).rename(columns={
            'Producer_price': 'from_price'})
        df = df.merge(prices[['iso3', 'Producer_price']], left_on='to_iso3', right_on='iso3').drop('iso3', axis=1).rename(columns={
            'Producer_price': 'to_price'})
        df['tariff'] = df['Pref_Applied_AVE'] * (df['transport_USD_t'] + df['border_USD_t'] + df['from_price'])
        
        for col in ['from_price', 'to_price', 'tariff']:
            df[f'{col}_log'] = np.log(df[col]+1)

        X_cols = X_cols_all.copy()
        
        if df['dataset'].nunique()<3:
            X_cols.remove('domestic')
            X_cols.remove('subnational')
    
        for clf in ['logistic', 'hist', 'rf', 'dl']:
            print(clf)
                        
            for i in range(n):
                print(f'iteration {i}')
                train, test = train_test_split(df, test_size=0.2, random_state=i)
                X_train = train[X_cols]
                y_train = train[y_col]
                X_test = test[X_cols]
                y_test = test[y_col]
    
                if clf=='logistic':
                    model = LogisticRegression(solver='liblinear', max_iter=500)
                    model.fit(X_train, y_train)
                
                if clf=='hist':
                    model =  HistGradientBoostingClassifier(max_iter=5000, 
                                                            class_weight='balanced', 
                                                            min_samples_leaf=100, 
                                                            max_features=0.8,
                                                            categorical_features=['from_region_0','from_region_1', 'from_region_2', 
                                                                                  'from_region_3', 'from_region_4', 'from_region_5',
                                                                                  'to_region_0', 'to_region_1', 'to_region_2', 
                                                                                  'to_region_3', 'to_region_4', 'to_region_5'],
                                                           random_state=4)
                    model.fit(X_train, y_train)
                
                if clf=='rf':
                    model = RandomForestClassifier(n_estimators=200, 
                                                   class_weight='balanced', 
                                                   max_depth=15, 
                                                   min_samples_leaf=4,
                                                   random_state=4)
                    model.fit(X_train, y_train)
    
                if clf=='dl':
                    lr = 0.001
                    batch_size = 512
                    epochs = 300
                    optimizer = tf.keras.optimizers.RMSprop
                    dropout = 0.2
                    dense_neurons = 512
                    metrics = ['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
                    activation='relu'
    
                    model = keras.Sequential()
                    model.add(layers.Input(shape=(X_train.shape[1],)))
                    model.add(layers.Dense(dense_neurons, activation=activation)) # hidden layer 1
                    model.add(layers.Dropout(dropout))
                    model.add(layers.Dense(dense_neurons, activation=activation)) # hidden layer 2
                    model.add(layers.Dropout(dropout))
                    model.add(layers.Dense(1, activation='sigmoid')) 
                    
                    model.compile(loss='binary_crossentropy', optimizer=optimizer(learning_rate=lr), metrics=metrics)
                    
                    model.fit(X_train, y_train, 
                              epochs=epochs, 
                              batch_size=batch_size, 
                              verbose=0)
                
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
    
                if clf=='dl':
                    y_train_pred[y_train_pred <= 0.5] = 0
                    y_train_pred[y_train_pred > 0.5] = 1
                    y_test_pred[y_test_pred <= 0.5] = 0
                    y_test_pred[y_test_pred > 0.5] = 1

                train['pred'] = y_train_pred
                test['pred'] = y_test_pred
    
                df_class.loc[len(df_class)] = [clf, crop, i, 'overall', 'train',
                                               class_metrics_group(train, y_col)['accuracy'], class_metrics_group(train, y_col)['precision'], 
                                               class_metrics_group(train, y_col)['recall'], class_metrics_group(train, y_col)['mcc']]
                
                df_class.loc[len(df_class)] = [clf, crop, i, 'overall', 'test',
                                               class_metrics_group(test, y_col)['accuracy'], class_metrics_group(test, y_col)['precision'], 
                                               class_metrics_group(test, y_col)['recall'], class_metrics_group(test, y_col)['mcc']]

                if disagg:
                    train = train.groupby('dataset').apply(class_metrics_group, y_col=y_col).reset_index()
                    train['model'] = clf
                    train['crop'] = crop
                    train['iteration'] = i
                    train['split'] = 'train'

                    test = test.groupby('dataset').apply(class_metrics_group, y_col=y_col).reset_index()
                    test['model'] = clf
                    test['crop'] = crop
                    test['iteration'] = i
                    test['split'] = 'test'

                    df_class = pd.concat([df_class, train, test], axis=0, ignore_index=True)

        df_class.to_csv(f'../../data/machine_learning/model_performance/classification_models_{crop}_{suffix}.csv', index=False)

# custom loss function for deep learning model for regression (to optimize on both linear and log scales)
@tf.keras.utils.register_keras_serializable(package="custom_losses")
def combined_loss(q=0.6, alpha=0.5, epsilon=1e-7):
    def loss(y_true, y_pred):
        # Compute the linear loss (similar to quantile loss)
        error = y_true - y_pred
        linear_loss = tf.reduce_mean(tf.maximum(q * error, (q - 1) * error)) /100
        
        # Compute the log loss: add epsilon to avoid log(0)
        log_y_true = tf.math.log(y_true + epsilon)
        log_y_pred = tf.math.log(y_pred + epsilon)
        log_loss = tf.keras.losses.MSE(log_y_true, log_y_pred) * 100
        
        # Combine the two losses
        return alpha * linear_loss + (1 - alpha) * log_loss
    return loss


def reg_metrics_group(g, y_col):
    rmse = root_mean_squared_error(g[y_col], g['pred'])
    r2 = r2_score(g[y_col], g['pred'])
    r2_log = r2_score(np.log(g[y_col]+1), np.log(g['pred']+1))

    obs_corr_wheat = g[y_col].corr(g['supply_wheat'])
    obs_corr_rice = g[y_col].corr(g['supply_rice'])
    obs_corr_maize = g[y_col].corr(g['supply_maize'])
    obs_corr_other = g[y_col].corr(g['supply_other_cereals'])

    pred_corr_wheat = g['pred'].corr(g['supply_wheat'])
    pred_corr_rice = g['pred'].corr(g['supply_rice'])
    pred_corr_maize = g['pred'].corr(g['supply_maize'])
    pred_corr_other = g['pred'].corr(g['supply_other_cereals'])
    
    return pd.Series(dict(rmse=rmse, r2=r2, r2_log=r2_log, 
                          obs_corr_wheat=obs_corr_wheat, obs_corr_rice=obs_corr_rice,
                          obs_corr_maize=obs_corr_maize, obs_corr_other=obs_corr_other, 
                          pred_corr_wheat=pred_corr_wheat, pred_corr_rice=pred_corr_rice,
                          pred_corr_maize=pred_corr_maize, pred_corr_other=pred_corr_other))


# regression model selection - diff y variables and diff models (hyperparameters chosen on the basis of manual tuning)
# 5 train test splits for each model
def model_runs_reg(df, n=5, disagg=True, suffix='all'):

    df_all_crops = df.copy()
    
    # r2 on lin, r2 on log, corrs with diff cereals if y is cereals_all
    for crop in ['wheat', 'rice', 'maize', 'other_cereals', 'cereals_all']:
        print(crop)

        df_reg = pd.DataFrame(columns=['model', 'crop', 'iteration', 'dataset', 'split',
                                       'rmse', 'r2', 'r2_log', 
                                       'obs_corr_wheat', 'obs_corr_rice', 'obs_corr_maize', 'obs_corr_other',
                                       'pred_corr_wheat', 'pred_corr_rice', 'pred_corr_maize', 'pred_corr_other'])
        y_col = f'supply_{crop}'
        df = df_all_crops[df_all_crops[y_col]>=1]
    
        # add prices to features
        prices = pd.read_csv(f'../../data/FAO_prod_prices/prod_prices_{crop}.csv')
        df = df.merge(prices[['iso3', 'Producer_price']], left_on='from_iso3', right_on='iso3').drop('iso3', axis=1).rename(columns={
            'Producer_price': 'from_price'})
        df = df.merge(prices[['iso3', 'Producer_price']], left_on='to_iso3', right_on='iso3').drop('iso3', axis=1).rename(columns={
            'Producer_price': 'to_price'})
        df['tariff'] = df['Pref_Applied_AVE'] * (df['transport_USD_t'] + df['border_USD_t'] + df['from_price'])
        
        for col in ['from_price', 'to_price', 'tariff']:
            df[f'{col}_log'] = np.log(df[col]+1)

        X_cols = X_cols_all.copy()
        
        if df['dataset'].nunique()<3:
            X_cols.remove('domestic')
            X_cols.remove('subnational')
    
        for reg in ['linear', 'hist', 'rf', 'dl']: 
            print(reg)
            
            for i in range(n):
                print(f'iteration {i}')
                train, test = train_test_split(df, test_size=0.2, random_state=i)
                X_train = train[X_cols]
                y_train = train[y_col]
                X_test = test[X_cols]
                y_test = test[y_col]
    
                if reg=='linear':
                    y_col_lin = f'supply_{crop}_log' # using log y for linear regression
                    fml = f"{y_col_lin} ~ " + " + ".join(X_cols) + " - 1"
                    model = sm.OLS.from_formula(fml, data=train)
                    res = model.fit()
                
                if reg=='hist':
                    model = HistGradientBoostingRegressor(l2_regularization=100, 
                                                          loss='gamma',
                                                          max_iter=1000,
                                                          max_depth=8,
                                                          max_features=0.4,
                                                          min_samples_leaf=100,
                                                          categorical_features=['from_region_0','from_region_1', 'from_region_2', 
                                                                                'from_region_3', 'from_region_4', 'from_region_5',
                                                                                'to_region_0', 'to_region_1', 'to_region_2', 
                                                                                'to_region_3', 'to_region_4', 'to_region_5'],
                                                         random_state=15)
                    model.fit(X_train, y_train)
                
                if reg=='rf':
                    y_col_rf = f'supply_{crop}_log' # using log y for random forest
                    model = RandomForestRegressor(n_estimators= 500, 
                                                  max_depth=10, 
                                                  random_state=15)
                    model.fit(X_train, train[y_col_rf])
                
                if reg=='dl':
                    lr = 0.01
                    batch_size = 512
                    epochs = 400
                    optimizer = tf.keras.optimizers.RMSprop
                    dropout = 0.2
                    dense_neurons = 256
                    metrics = [tf.keras.metrics.R2Score()]
                    activation='relu'
    
                    model = keras.Sequential()
                    model.add(layers.Input(shape=(X_train.shape[1],)))
                    model.add(layers.BatchNormalization())
                    model.add(layers.Dense(dense_neurons, activation=activation)) # hidden layer 1
                    model.add(layers.Dropout(dropout))
                    model.add(layers.Dense(dense_neurons, activation=activation)) # hidden layer 2
                    model.add(layers.Dropout(dropout))
                    model.add(layers.Dense(1, activation='softplus')) 
                    
                    model.compile(loss=combined_loss(), optimizer=optimizer(learning_rate=lr), metrics=metrics)
                    
                    model.fit(X_train, y_train, 
                              epochs=epochs, 
                              batch_size=batch_size, 
                              verbose=0)
                
                if reg=='linear':
                    y_train_pred = np.exp(res.predict(train))
                    y_test_pred = np.exp(res.predict(test))
                elif reg=='rf':
                    y_train_pred = np.exp(model.predict(X_train))
                    y_test_pred = np.exp(model.predict(X_test))
                else:
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)

                train['pred'] = y_train_pred
                test['pred'] = y_test_pred
                
                df_reg.loc[len(df_reg)] = [reg, crop, i, 'overall', 'train',
                                           reg_metrics_group(train, y_col)['rmse'], reg_metrics_group(train, y_col)['r2'], reg_metrics_group(train, y_col)['r2_log'], 
                                           reg_metrics_group(train, y_col)['obs_corr_wheat'], reg_metrics_group(train, y_col)['obs_corr_rice'], 
                                           reg_metrics_group(train, y_col)['obs_corr_maize'], reg_metrics_group(train, y_col)['obs_corr_other'],
                                           reg_metrics_group(train, y_col)['pred_corr_wheat'], reg_metrics_group(train, y_col)['pred_corr_rice'], 
                                           reg_metrics_group(train, y_col)['pred_corr_maize'], reg_metrics_group(train, y_col)['pred_corr_other']]
                
                df_reg.loc[len(df_reg)] = [reg, crop, i, 'overall', 'test',
                                           reg_metrics_group(test, y_col)['rmse'], reg_metrics_group(test, y_col)['r2'], reg_metrics_group(test, y_col)['r2_log'], 
                                           reg_metrics_group(test, y_col)['obs_corr_wheat'], reg_metrics_group(test, y_col)['obs_corr_rice'], 
                                           reg_metrics_group(test, y_col)['obs_corr_maize'], reg_metrics_group(test, y_col)['obs_corr_other'],
                                           reg_metrics_group(test, y_col)['pred_corr_wheat'], reg_metrics_group(test, y_col)['pred_corr_rice'], 
                                           reg_metrics_group(test, y_col)['pred_corr_maize'], reg_metrics_group(test, y_col)['pred_corr_other']]

                if disagg:
                    train = train.groupby('dataset').apply(reg_metrics_group, y_col=y_col).reset_index()
                    train['model'] = reg
                    train['crop'] = crop
                    train['iteration'] = i
                    train['split'] = 'train'

                    test = test.groupby('dataset').apply(reg_metrics_group, y_col=y_col).reset_index()
                    test['model'] = reg
                    test['crop'] = crop
                    test['iteration'] = i
                    test['split'] = 'test'

                    df_reg = pd.concat([df_reg, train, test], axis=0, ignore_index=True)

        df_reg.to_csv(f'../../data/machine_learning/model_performance/regression_models_{crop}_{suffix}.csv', index=False)
            


# train classification model (use seed here - choose based on manual tuning, different depending on suffix - 4 for all, ) 
def train_model_class(df, crop='cereals_all', suffix='all'):

    X_cols = X_cols_all.copy()
    y_col = f'supply_{crop}_exists'
    df = df[df[y_col].notnull()]

    # add prices to features
    prices = pd.read_csv(f'../../data/FAO_prod_prices/prod_prices_{crop}.csv')
    df = df.merge(prices[['iso3', 'Producer_price']], left_on='from_iso3', right_on='iso3').drop('iso3', axis=1).rename(columns={
        'Producer_price': 'from_price'})
    df = df.merge(prices[['iso3', 'Producer_price']], left_on='to_iso3', right_on='iso3').drop('iso3', axis=1).rename(columns={
        'Producer_price': 'to_price'})
    df['tariff'] = df['Pref_Applied_AVE'] * (df['transport_USD_t'] + df['border_USD_t'] + df['from_price'])
    
    for col in ['from_price', 'to_price', 'tariff']:
        df[f'{col}_log'] = np.log(df[col]+1)   
    
    if df['dataset'].nunique()<3:
        X_cols.remove('domestic')
        X_cols.remove('subnational')

    X = df[X_cols]
    y = df[y_col]
    
    # define model 
    model =  HistGradientBoostingClassifier(max_iter=5000, 
                                            class_weight='balanced', 
                                            min_samples_leaf=100, 
                                            max_features=0.8,
                                            categorical_features=['from_region_0','from_region_1', 'from_region_2', 
                                                                  'from_region_3', 'from_region_4', 'from_region_5',
                                                                  'to_region_0', 'to_region_1', 'to_region_2', 
                                                                  'to_region_3', 'to_region_4', 'to_region_5'],
                                           random_state=4)
    
    # train model 
    model.fit(X, y)
        
    # save model 
    joblib.dump(model, f'../../data/machine_learning/trained_models/classification_{crop}_{suffix}.joblib') 
        
    # save features 
    with open(f'../../data/machine_learning/trained_models/classification_features_{crop}_{suffix}.pkl', "wb") as fp:   
        pickle.dump(X_cols, fp)

# train regression model (use seed here - choose based on manual tuning, different depending on suffix) 
def train_model_reg(df, crop='cereals_all', suffix='all'):

    X_cols = X_cols_all.copy()
    y_col = f'supply_{crop}'
    df = df[df[y_col]>=1]

    # add prices to features
    prices = pd.read_csv(f'../../data/FAO_prod_prices/prod_prices_{crop}.csv')
    df = df.merge(prices[['iso3', 'Producer_price']], left_on='from_iso3', right_on='iso3').drop('iso3', axis=1).rename(columns={
        'Producer_price': 'from_price'})
    df = df.merge(prices[['iso3', 'Producer_price']], left_on='to_iso3', right_on='iso3').drop('iso3', axis=1).rename(columns={
        'Producer_price': 'to_price'})
    df['tariff'] = df['Pref_Applied_AVE'] * (df['transport_USD_t'] + df['border_USD_t'] + df['from_price'])
    
    for col in ['from_price', 'to_price', 'tariff']:
        df[f'{col}_log'] = np.log(df[col]+1)   
    
    if df['dataset'].nunique()<3:
        X_cols.remove('domestic')
        X_cols.remove('subnational')

    ####
    # also make train test scatter plots for the chosen model 
    train, test = train_test_split(df, test_size=0.2, random_state=4)
    X_train = train[X_cols]
    y_train = train[y_col]
    X_test = test[X_cols]
    y_test = test[y_col]

    # define model 
    lr = 0.01
    batch_size = 512
    epochs = 400
    optimizer = tf.keras.optimizers.RMSprop
    dropout = 0.2
    dense_neurons = 256
    metrics = [tf.keras.metrics.R2Score()]
    activation='relu'

    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(dense_neurons, activation=activation)) # hidden layer 1
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(dense_neurons, activation=activation)) # hidden layer 2
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='softplus')) 
    
    model.compile(loss=combined_loss(), optimizer=optimizer(learning_rate=lr), metrics=metrics)

    # train model
    model.fit(X_train, y_train, 
              epochs=epochs, 
              batch_size=batch_size, 
              verbose=0)

    # predict on train and test
    y_train_pred = model.predict(X_train).reshape(-1)
    y_test_pred = model.predict(X_test).reshape(-1)

    # plot
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    
    ax = axs[0]
    g = sns.scatterplot(ax=ax, x=np.log(y_train+1), y=np.log(y_train_pred+1), s=10)
    ax.set_xlabel("Cereal Flows (log tonnes)", fontsize=10)
    ax.set_ylabel("Predicted Cereal Flows (log tonnes)", fontsize=10)
    ax.set_title("Train", fontsize=10) 
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    ax = axs[1]
    g = sns.scatterplot(ax=ax, x=np.log(y_test), y=np.log(y_test_pred+1), s=10)
    ax.set_xlabel("Cereal Flows (log tonnes)", fontsize=10)
    ax.set_ylabel("Predicted Cereal Flows (log tonnes)", fontsize=10)
    ax.set_title("Test", fontsize=12) 
    ax.tick_params(axis='both', which='major', labelsize=8)
    g.figure.subplots_adjust(wspace=.1)

    plt.savefig(f'../../data/machine_learning/trained_models/regression_{crop}_{suffix}.jpg', bbox_inches='tight', dpi=300)

    #####
    
    
    X = df[X_cols]
    y = df[y_col]

    # define model 
    lr = 0.01
    batch_size = 512
    epochs = 400
    optimizer = tf.keras.optimizers.RMSprop
    dropout = 0.2
    dense_neurons = 256
    metrics = [tf.keras.metrics.R2Score()]
    activation='relu'

    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(dense_neurons, activation=activation)) # hidden layer 1
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(dense_neurons, activation=activation)) # hidden layer 2
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='softplus')) 
    
    model.compile(loss=combined_loss(), optimizer=optimizer(learning_rate=lr), metrics=metrics)

    # train model
    model.fit(X, y, 
              epochs=epochs, 
              batch_size=batch_size, 
              verbose=0)
    
    # save model 
    model.save(f'../../data/machine_learning/trained_models/regression_{crop}_{suffix}.keras')
        
    # save features 
    with open(f'../../data/machine_learning/trained_models/regression_features_{crop}_{suffix}.pkl', "wb") as f:   
        pickle.dump(X_cols, f) 


def process_predictions(df_admin, df_country, crop='cereals_all'):

    def _add_all_admins(m):
        m = m.merge(df_ids, left_on='to_id', right_on='id', how='right')
        m = m.drop(['to_id', 'to_iso3', 'from_id', 'from_iso3'], axis=1).rename(columns={'id': 'to_id', 'iso3': 'to_iso3'})
        m = m.sort_values(by='to_id')
        return m

    # merge with country data to remove trade between non-trading countries (as we are sure these should be 0)
    df_admin = df_admin.merge(df_country[['from_iso3', 'to_iso3', f'supply_{crop}_exists']], how='left')
    df_admin.loc[(df_admin['domestic']==0) & (df_admin[f'supply_{crop}_exists']==0), f'supply_{crop}_exists_pred'] = 0
    df_admin.loc[(df_admin['domestic']==0) & (df_admin[f'supply_{crop}_exists']==0), f'supply_{crop}_pred'] = 0

    # all ids
    df_ids = df_admin[['from_id', 'from_iso3']].drop_duplicates().sort_values(by='from_id').reset_index(drop=True)
    df_ids = df_ids.rename(columns={'from_id': 'id', 'from_iso3': 'iso3'})

    # creating a dataframe with all from and to ids
    # add all admins to from and to columns
    df_mat = df_admin[['from_id', 'from_iso3', 'to_id', 'to_iso3']].merge(df_ids, left_on='from_id', right_on='id', how='right')
    df_mat = df_mat.drop(['from_id', 'from_iso3'], axis=1).rename(columns={'id': 'from_id', 'iso3': 'from_iso3'})
    df_mat = df_mat.sort_values(by='from_id')
    
    df_mat = df_mat.groupby(['from_id', 'from_iso3']).apply(lambda g: _add_all_admins(g)).reset_index()
    
    # merge with trade exists and trade (both domestic and international) - all possible trades, for domestic all trades, whatever the probability
    df_mat = df_mat.drop('level_2', axis=1)
    df_mat = df_mat.merge(df_admin[['from_id', 'to_id', f'supply_{crop}_exists_pred', f'supply_{crop}_pred']], how='left')
    df_mat = df_mat.fillna(0)
    
    return df_mat
        
def predict(df_admin, df_country, crop='cereals_all', suffix='all'):

    # convert regions to dummy vars
    df_admin = pd.concat([df_admin, pd.get_dummies(df_admin['from_region'].astype('int32').astype('category'), prefix = 'from_region')], axis=1)
    df_admin = pd.concat([df_admin, pd.get_dummies(df_admin['to_region'].astype('int32').astype('category'), prefix = 'to_region')], axis=1)      
    
    # add prices to features
    prices = pd.read_csv(f'../../data/FAO_prod_prices/prod_prices_{crop}.csv')
    df_admin = df_admin.merge(prices[['iso3', 'Producer_price']], left_on='from_iso3', right_on='iso3').drop('iso3', axis=1).rename(columns={
        'Producer_price': 'from_price'})
    df_admin = df_admin.merge(prices[['iso3', 'Producer_price']], left_on='to_iso3', right_on='iso3').drop('iso3', axis=1).rename(columns={
        'Producer_price': 'to_price'})
    df_admin['tariff'] = df_admin['Pref_Applied_AVE'] * (df_admin['transport_USD_t'] + df_admin['border_USD_t'] + df_admin['from_price'])
    for col in ['from_price', 'to_price', 'tariff']:
        df_admin[f'{col}_log'] = np.log(df_admin[col]+1)

    df_admin['subnational'] = 1
    df_admin['domestic'] = 0
    df_admin.loc[df_admin['from_iso3']==df_admin['to_iso3'], 'domestic'] = 1

    # classification
    with open(f'../../data/machine_learning/trained_models/classification_features_{crop}_{suffix}.pkl', 'rb') as f:
        X_cols = pickle.load(f)
    model = joblib.load(f'../../data/machine_learning/trained_models/classification_{crop}_{suffix}.joblib')
    df_admin[f'supply_{crop}_exists_pred'] = model.predict_proba(df_admin[X_cols])[:,1]
    
    # regression
    with open(f'../../data/machine_learning/trained_models/regression_features_{crop}_{suffix}.pkl', 'rb') as f:
        X_cols = pickle.load(f)
    model = load_model(f'../../data/machine_learning/trained_models/regression_{crop}_{suffix}.keras',
                       custom_objects={"loss": combined_loss()})
    df_admin[f'supply_{crop}_pred'] = model.predict(df_admin[X_cols])
    df_mat = process_predictions(df_admin, df_country, crop)
    df_mat.to_parquet(f'../../data/machine_learning/admin_predictions/df_mat_{crop}_{suffix}.parquet.gzip', index=False, compression='gzip')

if __name__ == '__main__':
    
    master_df_country = country_totals()
    df_country, df_admin = get_bilateral_data(admin_level=False)
    df_india, df_us_china, df_us_state_int = read_other_data()

    df_country['from_id'] = df_country['from_iso3']
    df_country['to_id'] = df_country['to_iso3']
    df_country['domestic'] = 0
    df_country['dataset'] = 'faostat'
    df_country['subnational'] = 0
    
    df_india['domestic'] = 1
    df_india['dataset'] = 'harris et al 2020'
    df_india['subnational'] = 1
    
    df_us_china['domestic'] = 0
    df_us_china['dataset'] = 'pandit et al 2023'
    df_us_china['subnational'] = 1
    
    df_us_state_int['domestic'] = 0
    df_us_state_int['dataset'] = 'us census'
    df_us_state_int['subnational'] = 1

    df_all = pd.concat([df_country, df_india, df_us_china], ignore_index=True) # not using us census data 
    # convert regions to dummy vars
    df_all = pd.concat([df_all, pd.get_dummies(df_all['from_region'].astype('int32').astype('category'), prefix='from_region')], axis=1)
    df_all = pd.concat([df_all, pd.get_dummies(df_all['to_region'].astype('int32').astype('category'), prefix='to_region')], axis=1)   

    ### MODEL SELECTION ###
    # for both class and reg - first show train and test results for 4-5 models each (perf broken down by dataset also)
    # also show these results for the different cereals - they dont do so well - use that to explain why we are not splitting 
    # for regression models - also show correlation of observed and predicted cereal flows with observed disagregated cereal flows 
        # to show we are capturing these variations (the best model should do this best) 
    
    # performance metrics using all data (except us census)
    print('performance metrics using all data')
    df= df_all.copy()
    # print('classification')
    # model_runs_class(df)
    print('regression')
    model_runs_reg(df)

    # performance metrics using fao data only (not sure these are useful) 
    print('performance metrics using fao data only')
    df= df[df['dataset']=='faostat']
    # print('classification')
    # model_runs_class(df, disagg=False, suffix='fao')
    print('regression')
    model_runs_reg(df, disagg=False, suffix='fao')

    ### MODEL TRAINING ###
    
    # # take the best (cereals_all) model for each - train them on entire dataset and store the models
    # print('training using all data')
    # df= df_all.copy()
    # print('classification')
    # train_model_class(df, crop='cereals_all')
    # print('regression')
    # train_model_reg(df, crop='cereals_all')

    # # also train (cereals_all) model only on fao data
    # print('training using fao data only')
    # df= df[df['dataset']=='faostat']
    # print('classification')
    # train_model_class(df, crop='cereals_all', suffix='fao')
    # print('regression')
    # train_model_reg(df, crop='cereals_all', suffix='fao')

    ### PREDICTION ###
    # master_df_admin = pred_consumption(master_df_country, df_country, df_admin)

    # # with models for class and reg , make predictions on admin data 
    # predict(df_admin, df_country, crop='cereals_all')
    
    # # save the predictions using models on fao data only
    # these models dont do well on training data from india and us-china - but maybe they do in the end after harmonizing (exporting just to check later)
    # predict(df_admin, df_country, crop='cereals_all', suffix='fao')

    

    
    
    
    

    