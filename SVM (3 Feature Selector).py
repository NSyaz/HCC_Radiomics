import numpy as np
import seaborn as sns
import pandas as pd
import pyswarms as ps
import matplotlib.pyplot as plt
import scikitplot as skplt
import sklearn.metrics
from tabulate import tabulate

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,cross_val_score,RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler,label_binarize,MinMaxScaler
from sklearn.metrics import classification_report,RocCurveDisplay, accuracy_score, recall_score ,precision_score,f1_score, roc_curve,roc_auc_score, average_precision_score, precision_recall_curve,PrecisionRecallDisplay, confusion_matrix
from sklearn.feature_selection import SelectKBest,f_classif,chi2

cancer = pd.read_excel(r'/Users/nurinsyazwinamohdhaniff/Documents/MASTER/THESIS/DATA/Pycharm copy.xlsx')
pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.DataFrame(cancer)
X = cancer.drop(columns=['Stage'], axis=1)
y = cancer['Stage']

X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=10)

feature_selector = []
accuracies = []
recalls = []
precisions = []
specificities = []
F1_Scores = []
aucS = []

svc = SVC(max_iter=10000, probability=True)

# SVM only
svc.fit(X_train,y_train)
y_predict_WFS = svc.predict(X_test)

y_score_WFS = svc.decision_function(X_test)
y_proba_WFS = svc.predict_proba(X_test)[:,1]

accuracy_WFS = accuracy_score(y_test, y_predict_WFS)
recall_WFS = recall_score(y_test, y_predict_WFS)
precision_WFS = precision_score(y_test,y_predict_WFS)
specificity_WFS = recall_score(np.logical_not(y_test), np.logical_not(y_predict_WFS))
f1_WFS = f1_score(y_test, y_predict_WFS)
auc_WFS = roc_auc_score(y_test, y_predict_WFS)

feature_selector.append("SVM only (Without Feature Selection)")
accuracies.append(accuracy_WFS)
precisions.append(precision_WFS)
recalls.append(recall_WFS)
specificities.append(specificity_WFS)
aucS.append(auc_WFS)
F1_Scores.append(f1_WFS)

# SVM (BPSO)
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)

def f_per_particle(m, alpha):
    total_features = X_train.shape[1]
    if np.count_nonzero(m == 0):
        X_subset_train = X_train
    else:
        X_subset_train = X_train[:,m == 1]
    scores = cross_val_score(svc,X_subset_train,y_train,cv= 3)
    P = scores.mean()
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset_train.shape[1] / total_features)))
    return j
def f(x, alpha=0.9):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i],alpha) for i in range(n_particles)]
    return np.array(j)

dimensions = X_train.shape[1]
optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options= {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 30, 'p': 2} )
cost, pos = optimizer.optimize(f, iters=10, verbose= True)
X_BPSO = X_test[:,pos == 1]
print("The shape for SVM (BPSO) is ",X_BPSO.shape)

svc.fit(X_BPSO,y_test)
y_predict_BPSO = svc.predict(X_BPSO)

y_score_BPSO = svc.decision_function(X_BPSO)
y_proba_BPSO = svc.predict_proba(X_BPSO)[:,1]

accuracy_BPSO = accuracy_score(y_test,y_predict_BPSO)
recall_BPSO = recall_score(y_test,y_predict_BPSO)
precision_BPSO = precision_score(y_test,y_predict_BPSO)
specificity_BPSO = recall_score(np.logical_not(y_test), np.logical_not(y_predict_BPSO))
f1_BPSO = f1_score(y_test,y_predict_BPSO)
auc_BPSO = roc_auc_score(y_test,y_predict_BPSO)

feature_selector.append("BPSO")
accuracies.append(accuracy_BPSO)
precisions.append(precision_BPSO)
recalls.append(recall_BPSO)
specificities.append(specificity_BPSO)
aucS.append(auc_BPSO)
F1_Scores.append(f1_BPSO)

# SVM (ANOVA F-Test + BPSO)
'''X_FtBPSO = SelectKBest(f_classif,k=150).fit_transform(X_BPSO,y_test)
svc.fit(X_FtBPSO,y_test)
y_predict_FtBPSO = svc.predict(X_FtBPSO)'''

# SVM (ANOVA + BPSO) METHOD 2
# Filter Method
def select_features(X_train, y_train, X_test):
    fS = SelectKBest(f_classif, k=332)
    fS.fit(X_train, y_train)
    X_train_k = fS.transform(X_train)
    X_test_k = fS.transform(X_test)
    return X_train_k, X_test_k, fS
X_train_k, X_test_k, fS = select_features(X_train, y_train, X_test)

selected_featuresName = X.columns[fS.get_support(indices=True)].tolist() #GET LIST OF SELECTED FEATURES
print(selected_featuresName)

# BPSO Algorithm
def f_per_particles(m, alpha):
    total_features = 332
    if np.count_nonzero(m == 0):
        x_subset = X_train_k
    else:
        x_subset = X_train_k[:, m == 1]
    svc.fit(x_subset, y_train)
    performance = (svc.predict(x_subset) == y_train).mean()
    j = (alpha * (1.0 - performance) + (1.0 - alpha) * (1 - (x_subset.shape[1] / total_features)))
    return j

def f(x, alpha=0.9):
    n_particles = x.shape[0]
    j = [f_per_particles(x[i], alpha) for i in range(n_particles)]
    return np.array(j)

options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k': 30, 'p': 2}
dimensions = 332
# optimizer.reset()
optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)
cost_kBPSO, pos = optimizer.optimize(f, iters=10, verbose=True)

X_train_kBPSO = X_train_k[:, pos == 1]
X_FtBPSO = X_test_k[:, pos == 1]
#print(type(X_train_kBPSO))
#print(X_train_kBPSO.shape)
shape = X_FtBPSO.shape
print("The shape for SVM (ANOVA F-Test + BPSO) is ",shape)

svc.fit(X_train_kBPSO, y_train)
y_predict_FtBPSO = svc.predict(X_FtBPSO)
#######
y_score_FtBPSO = svc.decision_function(X_FtBPSO)
y_proba_FtBPSO = svc.predict_proba(X_FtBPSO)[:,1]

accuracy_FtBPSO = accuracy_score(y_test,y_predict_FtBPSO)
recall_FtBPSO = recall_score(y_test,y_predict_FtBPSO)
precision_FtBPSO = precision_score(y_test,y_predict_FtBPSO)
specificity_FtBPSO = recall_score(np.logical_not(y_test), np.logical_not(y_predict_FtBPSO))
f1_FtBPSO = f1_score(y_test,y_predict_FtBPSO)
auc_FtBPSO = roc_auc_score(y_test,y_proba_FtBPSO)

feature_selector.append("ANOVA F-Test + BPSO")
accuracies.append(accuracy_FtBPSO)
precisions.append(precision_FtBPSO)
recalls.append(recall_FtBPSO)
specificities.append(specificity_FtBPSO)
aucS.append(auc_FtBPSO)
F1_Scores.append(f1_FtBPSO)

data = {"Classification Algorithms": feature_selector, "Accuracy": accuracies, "Precision": precisions, "Recall": recalls,
        "Specificity": specificities, "AUC": aucS, "F1 Scores": F1_Scores}
table = pd.DataFrame(data)
print(tabulate(table, headers='keys', tablefmt='psql'))

# Graph AUC-ROC SVM
fpr_WFS, tpr_WFS, threshold_WFS = roc_curve(y_test, y_proba_WFS)
fpr_BPSO, tpr_BPSO, threshold_BPSO = roc_curve(y_test, y_proba_BPSO)
fpr_FtBPSO, tpr_FtBPSO, threshold_FtBPSO = roc_curve(y_test, y_proba_FtBPSO)

plt.plot(fpr_WFS, tpr_WFS, linestyle='--', label='No feature selection -AUC:%.3f' % auc_WFS)
plt.plot(fpr_BPSO, tpr_BPSO, linestyle='--', label='BPSO -AUC:%.3f' % auc_BPSO)
plt.plot(fpr_FtBPSO, tpr_FtBPSO, linestyle='--', label='ANOVA F-test + BPSO -AUC:%.3f' % auc_FtBPSO)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.title('ROC-AUC Curve of SVM')
plt.show()

#Graph PRC SVM
fig,ax = plt.subplots()
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Baseline')

ave_pre_WFS = average_precision_score(y_test,y_score_WFS)
ave_pre_BPSO = average_precision_score(y_test,y_score_BPSO)
ave_pre_FtBPSO = average_precision_score(y_test,y_score_FtBPSO)

precision_WFS,recall_WFS,thresholdWFS = precision_recall_curve(y_test,y_score_WFS)
#precision_BPSO,recall_BPSO,thresholdBPSO = precision_recall_curve(y_test,y_score_BPSO)
precision_FtBPSO,recall_FtBPSO,thresholdFtBPSO = precision_recall_curve(y_test,y_score_FtBPSO)

plt.plot(recall_WFS,precision_WFS,linestyle='-', label="No feature selection (AP = %.3f)" % ave_pre_WFS)
#plt.plot(recall_BPSO,precision_BPSO,linestyle='-', label="BPSO (AP = %.3f)" % ave_pre_BPSO)
PrecisionRecallDisplay.from_predictions(y_test, y_predict_FtBPSO, ax=ax, name='BPSO')
plt.plot(recall_FtBPSO,precision_FtBPSO,linestyle='-', label="ANOVA F-Test + BPSO (AP = %.3f)" % ave_pre_FtBPSO)

plt.title('Precision-Recall Curve of SVM')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower right')
plt.show()










