import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.model_selection import learning_curve,cross_val_score

from scipy import interp
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")


# 模型评估


# 学习曲线
def plot_learning_curve(estimator,x,y,cv=None,train_size = np.linspace(0.1,1.0,5),plt_size =None,score = 'roc_auc'):
    """
    estimator :画学习曲线的基模型
    x:自变量的数据集
    y:target的数据集
    cv:交叉验证的策略
    train_size:训练集划分的策略
    plt_size:画图尺寸
    
    return:学习曲线
    """
    train_sizes,train_scores,test_scores = learning_curve(estimator=estimator,
                                                          X=x,
                                                          y=y,
                                                          cv=cv,
                                                          n_jobs=-1,
                                                          train_sizes=train_size,scoring = score)
    train_scores_mean = np.mean(train_scores,axis=1)
    train_scores_std = np.std(train_scores,axis=1)
    test_scores_mean = np.mean(test_scores,axis=1)
    test_scores_std = np.std(test_scores,axis=1)
    plt.figure(figsize=plt_size)
    plt.xlabel('Training-example')
    plt.ylabel(score)
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,
                     train_scores_mean+train_scores_std,alpha=0.1,color='r')
    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,
                     test_scores_mean+test_scores_std,alpha=0.1,color='g')
    plt.plot(train_sizes,train_scores_mean,'o-',color='r',label='Training  '+ score)
    plt.plot(train_sizes,test_scores_mean,'o-',color='g',label='cross-val  '+ score)
    plt.legend(loc='best')
    return plt.show()


# 混淆矩阵 /分类报告
def plot_matrix_report(y_label,y_pred): 
    """
    y_label:测试集的y
    y_pred:对测试集预测后的概率
    
    return:混淆矩阵
    """
    matrix_array = metrics.confusion_matrix(y_label,y_pred)
    plt.matshow(matrix_array, cmap=plt.cm.summer_r)
    plt.colorbar()

    for x in range(len(matrix_array)): 
        for y in range(len(matrix_array)):
            plt.annotate(matrix_array[x,y], xy =(x,y), ha='center',va='center')

    plt.xlabel('True label')
    plt.ylabel('Predict label')
    print(metrics.classification_report(y_label,y_pred))
    return plt.show()



#K折交叉验证ROC图
def plot_roc_cv(classifier,X ,y,cv,cv_TimeSeries = True,title =''):#默认时序交叉验证
    """
    classifier:分类器
    X:训练集
    y:标签
    cv: 交叉验证次数
    cv_TimeSeries：是否采用时序交叉验证策略，默认True
    title：图表标题
    
    """
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 10000)

    fig, ax = plt.subplots(figsize=(8,5))
    
    if cv_TimeSeries is True:
        cv_split= cv.split(X)
    else:
        cv_split= cv.split(X,y)
        
    for i, (train_index, test_index) in enumerate(cv_split):
        classifier.fit(X.iloc[train_index,:], y.iloc[train_index])
        viz = metrics.plot_roc_curve(classifier, X.iloc[test_index,:], y.iloc[test_index],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        

        
        interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title= "{}  {} folds cross_validate".format( title , str(cv.n_splits)))
    ax.legend(loc="lower right")
    plt.show()