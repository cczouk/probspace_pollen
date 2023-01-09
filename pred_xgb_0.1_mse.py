
xgb_params = {
    'learning_rate': 0.2,
    'objective': 'reg:squarederror', # huber, fair, regression_l1, tweedie, poisson
    'tree_method': 'gpu_hist', #exact, gpu_hist
    'grow_policy': 'lossguide', #
    'min_child_weight': 40,
    'verbosity': 0,
    'random_state' : seed,
    } 


ind_params = []
for i, c in enumerate(target_columns):
    params_base = xgb_params.copy()
    if c in [target_columns[n] for n in [0,]]:
        params_base['learning_rate'] = 0.2 #0.2
        # params_base['objective'] = 'reg:pseudohubererror'
        # params_base['huber_slope'] = 5
        # params_base['tweedie_variance_power'] = 1.9
        # params_base['max_depth']=6
        params_base['max_bin']=256
        params_base['min_child_weight']=20 #20
        ind_params.append(params_base)
    elif c in [target_columns[n] for n in [1]]:
        params_base['learning_rate'] = 0.01 #0.02
        # params_base['objective'] = 'reg:pseudohubererror'
        # params_base['huber_slope'] = 5
        # params_base['tweedie_variance_power'] = 1.8
        # params_base['max_depth']=5 #511
        params_base['max_bin']=256
        params_base['min_child_weight']=10
        ind_params.append(params_base)
    elif c in [target_columns[n] for n in [2]]:
        params_base['learning_rate'] = 0.2
        # params_base['objective'] = 'reg:pseudohubererror'
        # params_base['huber_slope'] = 5
        # params_base['tweedie_variance_power'] = 1.9       
        # params_base['max_depth']=7 #511
        params_base['max_bin']=255
        params_base['min_child_weight']=40
        ind_params.append(params_base)


n_itr_list = [0.01, 0.01, 0.01]
