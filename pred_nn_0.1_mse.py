
ind_params = []
optimizer_lr_l = []
for i, c in enumerate(target_columns):
    params_base = params.copy()
    if c in [target_columns[n] for n in [0,]]:
        params_base['dropout_0'] = 0.0 #0.4
        params_base['hidden_0'] = 512
        params_base['hidden_1']=2048 #511
        params_base['hidden_3']=2048
        params_base['dropout_3']=0.0
        params_base['layer']=3
        ind_params.append(params_base)
        optimizer_lr_l.append(5e-7)
    elif c in [target_columns[n] for n in [1]]:
        params_base['dropout_0'] = 0.0 #0.4
        params_base['hidden_0'] = 512
        params_base['hidden_1']=2048 #511
        params_base['hidden_3']=2048
        params_base['dropout_3']=0.0
        params_base['layer']=3
        ind_params.append(params_base)
        optimizer_lr_l.append(1e-7)
    elif c in [target_columns[n] for n in [2]]:
        params_base['dropout_0'] = 0.0 #0.4
        params_base['hidden_0'] = 512
        params_base['hidden_1']=2048 #511
        params_base['hidden_3']=2048
        params_base['dropout_3']=0.0
        params_base['layer']=3
        ind_params.append(params_base)
        optimizer_lr_l.append(1e-7)


n_itr_list = [51, 65, 28] #51, 65, 28


    dnn_model = NeuralNet(
                        module=tfreg,
                        max_epochs=n_itr_list[i],
                        device = device,
                        batch_size=32,
                        module__params = ind_params[i],
                        optimizer=torch.optim.AdamW, # optim.AdaBelief # madgrad.MADGRAD
                        # optimizer__batas = (0.9, 0.999),
                        # optimizer__eps = 1e-8,
                        optimizer__lr = optimizer_lr_l[i],
                        # optimizer__weight_decay = 0.0,
                        criterion=nn.MSELoss, #TweedieLoss()
                        callbacks = [lrscheduler],
                        train_split=predefined_split(valid_ds),
                        # train_split=None,
                    ) 
