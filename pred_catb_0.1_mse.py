
params = {
    'loss_function': 'Tweedie:variance_power=1.99',
    # 'loss_function': 'Huber:delta=5',
    'random_state': seed,
    'od_type': 'IncToDec',
    'max_ctr_complexity':2,
    'od_wait': 400,
    'min_data_in_leaf': 20, # Lossguide
    # "task_type": "GPU",
    'grow_policy': 'Lossguide',
    "num_boost_round": 50_000,
   } 


for i, c in enumerate(target_columns):
    params_base = params.copy()
    if c in [target_columns[n] for n in [0,]]:
        params_base['learning_rate'] = 0.3 #0.4
        params_base['l2_leaf_reg'] = 2
        params_base['loss_function'] = 'RMSE'
        # params_base['loss_function'] = 'Huber:delta=5' # LogCosh, Huber:delta=5
        params_base['bootstrap_type']='Bernoulli' #MVS, Bayesian, Bernoulli   
        # params_base['max_depth']=16
        params_base['rsm']=0.8
        params_base['border_count']=128
        params_base['min_data_in_leaf']=20 #100
        ind_params.append(params_base)
    elif c in [target_columns[n] for n in [1]]:
        # params_base['learning_rate'] = 0.1 #0.4
        # params_base['l2_leaf_reg'] = 2
        params_base['loss_function'] = 'RMSE'
        # params_base['loss_function'] = 'Huber:delta=5' # LogCosh, Huber:delta=5
        params_base['bootstrap_type']='Bayesian' #MVS, Bayesian, Bernoulli
        # params_base['bagging_temperature']=2  
        # params_base['max_depth']=14
        # params_base['rsm']=0.
        params_base['border_count']=254
        params_base['min_data_in_leaf']=40
        ind_params.append(params_base)
    elif c in [target_columns[n] for n in [2]]:
        # params_base['learning_rate'] = 0.1 #0.4
        # params_base['l2_leaf_reg'] = 2
        params_base['loss_function'] = 'RMSE'
        # params_base['loss_function'] = 'LogCosh' # LogCosh, Huber:delta=5
        params_base['bootstrap_type']='MVS' #MVS, Bayesian, Bernoulli
        # params_base['subsample']=0.66   
        # params_base['max_depth']=14
        # params_base['rsm']=0.8 
        params_base['border_count']=511
        params_base['min_data_in_leaf']=40
        ind_params.append(params_base)


n_itr_list = [0.5, 1, 1] 
