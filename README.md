# risk_predict

1、采用时序分割策略(TimeSeriesSplit)，使用xgboost模型10折交叉验证，然后取10折交叉特征交集，得出稳定特征

2、利用3个参数抖动的lightgbm模型，LR模型stacking融合

3、AUC模型评估
