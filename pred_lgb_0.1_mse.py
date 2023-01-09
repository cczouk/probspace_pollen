
lgb_params = { 
    'objective': 'regression', # huber, fair, regression_l1, tweedie, poisson
    'min_data_in_leaf': 40,
    'verbosity': -1,
    'random_state' : seed,
    } 

ind_params = []
for i, c in enumerate(target_columns):
    params_base = lgb_params.copy()
    if c in [target_columns[n] for n in [0,]]:
        # params_base['learning_rate'] = 0.01 #0.4
        # params_base['num_leaves']=127, #511
        params_base['max_bin']=127,
        params_base['min_data_in_leaf']=40,
        ind_params.append(params_base)
    elif c in [target_columns[n] for n in [1]]:
        # params_base['learning_rate'] = 0.02
        # params_base['objective'] = 'tweedie', # huber, fair, regression_l1, tweedie, poisson
        # params_base['num_leaves']=127, #511
        # params_base['max_bin']=256,
        # params_base['feature_fraction']=0.8,
        params_base['min_data_in_leaf']=40,
        ind_params.append(params_base)
    elif c in [target_columns[n] for n in [2]]:
        # params_base['learning_rate'] = 0.1
        # params_base['num_leaves']=255, #511
        params_base['max_bin']=127,
        params_base['min_data_in_leaf']=40,
        ind_params.append(params_base)


n_itr_list = [2, 1, 1] #140, 2, 2
