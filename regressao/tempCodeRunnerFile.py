for config in range(1, 6): 
        models = {
            'Regressao Linear': linear_pipeline(preprocessor,config),
            'KNN Regressor': knn_pipeline(preprocessor, config),
            'Arvore de Decisao Regressora': tree_pipeline(preprocessor, config, semente),
            'MLP Regressor': mlp_pipeline(preprocessor, config, semente)
        }
        
        for name, model in models.items():
            mse, rmse, mae, r2, y_pred = model_evaluation(model, X_train, y_train, X_test, y_test)
            cv = KFold(n_splits=10, random_state=42, shuffle=True)
            score = cross_val_score(model, X_train, y_train,scoring='r2', cv=cv)
            #score = cross_val_score(model, X_train, y_train, cv=10)
            add_global_result(name, score.mean())
            print_results(name, mse, rmse, mae, r2, score)
            # plot_r