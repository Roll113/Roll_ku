
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scorecardpy
import warnings
warnings.filterwarnings("ignore")


# 相关性可视化
def plot_corr(df,col_list,threshold=None,plt_size=None,is_annot=True):
    """
    df:数据集
    col_list:变量list集合
    threshold: 相关性设定的阈值
    plt_size:图纸尺寸
    is_annot:是否显示相关系数值
    
    return :相关性热力图
    """
    corr_df = df.loc[:,col_list].corr()
    plt.figure(figsize=plt_size)
    sns.heatmap(corr_df,annot=is_annot,cmap='rainbow',vmax=1,vmin=-1,mask=np.abs(corr_df)<=threshold)
    return plt.show()


# 相关性剔除
def delete_corr(df,col_list,target, threshold=None):
    """
    df:数据集
    col_list:变量list集合
    threshold: 相关性设定的阈值
    
    return:相关性剔除后的变量
    """   
    iv_df = scorecardpy.iv(df,y= target,x = col_list)
    iv_df = iv_df[iv_df.iloc[:,1]>0.02]
    high_iv_cols = list(iv_df.iloc[:,0])
    high_iv = iv_df.set_index('variable').T.reset_index()
    for col in high_iv_cols:
        corr = df.loc[:,high_iv_cols].corr()[col]
        corr_index= [x for x in corr.index if x!=col]
        corr_values  = [x for x in corr.values if x!=1]
        for i,j in zip(corr_index,corr_values):
            if abs(j)>=threshold:
                if high_iv.loc[0,col]>high_iv.loc[0,i]:
                    high_iv_cols.remove(i)
                else:
                    high_iv_cols.remove(col)
    print('remain  {} features'.format(len(high_iv_cols)))
    return high_iv_cols


#方差膨胀因子VIF，去除特征共线性
def vif(df, thres=10.0):
    X_m = np.matrix(df)
    VIF_list = [variance_inflation_factor(X_m, i) for i in range(X_m.shape[1])]
    maxvif=pd.DataFrame(VIF_list,index=df.columns,columns=["vif"])
    col_save=list(maxvif[maxvif.vif<=float(thres)].index)
    col_delete=list(maxvif[maxvif.vif>float(thres)].index)
    print('remain {} features:\n' .format(len(col_save)))
    print('\ndelete {} features,{}'.format(len(col_delete),col_delete))
    return col_save





#基于树模型的特征重要性选择
def imp_feat(model,Xtrain,max_num_features = 15):
    df=pd.DataFrame()
    df['feat'] = Xtrain.columns
    df['imp'] = model.feature_importances_#基于树模型基尼系数，特征重要性
    df= df.sort_values(by = 'imp',ascending = False ).reset_index()
    imp_feat = list(df.head(max_num_features)['feat'])
    return imp_feat



#基于交叉验证的稳定特征选择(取每次交叉的特征重要性的交集)
def cross_validate_feats(model,Xtrain,target,cv,score = 'roc_auc',top_features = 15):
    cv_result = cross_validate(estimator = model, X = Xtrain,y= target,cv= cv,scoring = score,n_jobs = -1,return_estimator = True)
    cv_model = cv_result['estimator']#每次交叉验证的模型
    
    imp_dict = {}
    for model,k in zip(cv_model,range(len(cv_model))):
        imp_dict['cv_'+str(k)] = imp_feat(model,Xtrain,max_num_features = top_features)#提取top15 重要特征

    result = set(imp_dict['cv_0'])
    for k in imp_dict.values():
        result = result.intersection(k)
    return list(result)