import numpy as np
#import seaborn as sns
import pandas as pd
import pyswarms as ps
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,cross_val_score,RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler,label_binarize,MinMaxScaler
from sklearn.metrics import classification_report,accuracy_score, roc_curve,roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix
from sklearn.feature_selection import SelectKBest,f_classif,chi2

cancer = pd.read_excel(r'/Users/nurinsyazwinamohdhaniff/Documents/MASTER/THESIS/DATA/Pycharm copy.xlsx')
pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.DataFrame(cancer)
X = cancer.drop(columns=['Stage'], axis=1)
y = cancer['Stage']

'''df = pd.DataFrame(X)
df['labels'] = pd.Series(y)

sns.pairplot(X, hue='Stage')
plt.show()'''

X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=10)
'''std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)
scaler = MinMaxScaler(feature_range=(-2,1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)'''

svc = SVC(probability=True)
svc.fit(X_train,y_train)
y_train_predict = svc.predict(X_train)
print (classification_report(y_train,y_train_predict))
print(X_train.shape)

y_predict = svc.predict_proba(X_test)[:,1]
y_score_test = svc.decision_function(X_test)
#print (classification_report(y_test, y_predict))
#print("Accuracy without feature selection :%.3f" % accuracy_score(y_test,y_predict))

"SelectKbest"
selector = SelectKBest(f_classif,k=662)
selector.fit_transform(X_train,y_train)
#print(selector.shape)
columns = selector._get_support_mask()
X_train_new = X_train.iloc[:,columns]
X_test_new = X_test.iloc[:,columns]
print(X_test_new.shape,X_train_new.shape)

svc.fit(X_test_new,y_test)
y_predict_kbest = svc.predict_proba(X_test_new)[:,1]
y_score_kbest = svc.decision_function(X_test_new)
#print (classification_report(y_test, y_predict_kbest))
#print("Accuracy for selectKBest:%.3f" % accuracy_score(y_test,y_predict_kbest))


#print(X_test.shape)


clf = SVC(max_iter=10000)
def f_per_particle(m, alpha):
    total_features = X_train.shape[1]
    if np.count_nonzero(m == 0):
        X_subset_train = X_train
    else:
        X_subset_train = X_train[:,m == 1]
    scores = cross_val_score(clf,X_subset_train,y_train,cv= 3)
    P = scores.mean()
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset_train.shape[1] / total_features)))
    return j
def f(x, alpha=0.9):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i],alpha) for i in range(n_particles)]
    return np.array(j)

#options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 30, 'p': 2}
dimensions = X_train.shape[1]
#optimizer.reset()
optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options= {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 30, 'p': 2} )

cost, pos = optimizer.optimize(f, iters=10, verbose= True)

#print(y[pos == 1])
X_selected_features = X_test.iloc[:,pos == 1]
print(X_selected_features.shape)

svc.fit(X_selected_features,y_test)
y_predict_SF = svc.predict_proba(X_selected_features)[:,1]
y_score_PSO = svc.decision_function(X_selected_features)
#print (classification_report(y_test,y_predict_SF))
#print("Accuracy for selectKBest & PSO :%.3f" % accuracy_score(y_test,y_predict_SF))

'''#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train,y_train)

#DECISION TREE
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(X_train,y_train)

#k-NN
from sklearn.neighbors import KNeighborsClassifier
kNN = KNeighborsClassifier()
kNN.fit(X_train,y_train)

#NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_train,y_train)'''

'''X_new = SelectKBest(f_classif,k=150).fit_transform(X_selected_features,y_test)
print(X_new.shape)
svc.fit(X_new,y_test)
y_predict_newSF = svc.predict(X_new)
y_score_PSOkBest = svc.decision_function(X_new)
print (classification_report(y_test,y_predict_newSF))
#print("Precision for PSO & SelectKBest :%.3f" % recall_score(y_test,y_predict_newSF))
print("Accuracy for PSO & selectkBest :%.3f" % accuracy_score(y_test,y_predict_newSF))'''

"Confusion Matrix"
'''sns.heatmap(confusion_matrix(y_test, y_predict_newSF), annot = True)
plt.title ('Confusion matrix SVM (test set)', size = 20)
plt.ylabel('Real Values', size =20)
plt.xlabel('Predicted Values', size = 20)
plt.show()'''


"AUC-ROC"
#plt.style.use('seaborn')
test_auc = roc_auc_score(y_test,y_predict)
kBEST_auc = roc_auc_score(y_test,y_predict_kbest)
PSO_auc = roc_auc_score(y_test,y_predict_SF)
#PSOkBest_auc = roc_auc_score(y_test,y_predict_newSF)
#print('SVC: ROC AUC=%.3f' % (PSOCHI_auc))
#print('AUC for without feature :%.3f'%test_auc + '\nAUC for without feature :%.3f'%PSO_auc + '\nAUC for without feature :%.3f'%PSOkBest_auc)
test_fpr, test_tpr,threshold = roc_curve(y_test,y_predict)
kBEST_fpr,kBEST_tpr, threshold = roc_curve(y_test,y_predict_kbest)
PSO_fpr,PSO_tpr, threshold = roc_curve(y_test,y_predict_SF)
#PSOkBest_fpr,PSOkBest_tpr, threshold = roc_curve(y_test,y_predict_newSF)
plt.figure()
plt.plot(test_fpr,test_tpr,linestyle=':', color='navy',label='No feature selection, AUC :%.3f' % test_auc)
#plt.plot(kBEST_fpr,kBEST_tpr,linestyle='-', color='pink',label='SelectKBest, AUC :%.3f' % kBEST_auc)
plt.plot(PSO_fpr,PSO_tpr,linestyle='-', color='red',label='+BPSO, AUC :%.3f' % PSO_auc)
#plt.plot(PSOkBest_fpr,PSOkBest_tpr, linestyle='--', color='darkorange', label='LR+F-test+BPSO, AUC :%.3f' % PSOkBest_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('AUC-ROC Curve of SVM')
plt.show()

"Precision-Recall"
average_precision = average_precision_score(y_test,y_score_test)
ave_pre_kBest = average_precision_score(y_test,y_score_kbest)
ave_pre_PSO = average_precision_score(y_test,y_score_PSO)
#ave_pre_PSOkBEST = average_precision_score(y_test,y_score_PSOkBest)
print('Average Precision-Recall Curve without Feature :%.3f'% average_precision)
print('Average Precision-Recall Curve with PSO :%.3f' % ave_pre_kBest)
print('Average Precision-Recall Curve with PSO :%.3f' % ave_pre_PSO)

#print('Average Precision-Recall Curve with PSO-SelectKBest :%.3f'% ave_pre_PSOkBEST)
test_precision,test_recall,threshold = precision_recall_curve(y_test,y_score_test)
kBest_precision,kBest_recall,_ = precision_recall_curve(y_test,y_score_PSO)
PSO_precision,PSO_recall,_ = precision_recall_curve(y_test,y_score_PSO)

no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Baseline')
#PSOkBest_precision,PSOkBest_recall,_ = precision_recall_curve(y_test,y_score_PSOkBest)
plt.plot(test_recall,test_precision,linestyle=':', color='navy', label='LR')
plt.plot(kBest_recall,kBest_precision,linestyle=':', color='pink', label='LR+kBest')
plt.plot(PSO_recall,PSO_precision,linestyle='-', color='red', label='LR+BPSO')
#plt.plot(PSOkBest_recall,PSOkBest_precision,linestyle='--', color='darkorange', label='LR+F-test+BPSO')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.title('Precision-Recall Curve')
plt.show()