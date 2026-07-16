from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps
import seaborn as sns
import plotly.express as px
from tabulate import tabulate
import time

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,cross_val_score,RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler,label_binarize,MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, recall_score ,precision_score,f1_score, roc_curve,roc_auc_score, average_precision_score, precision_recall_curve,PrecisionRecallDisplay, confusion_matrix
from sklearn.feature_selection import SelectKBest,f_classif

cancer = pd.read_excel(r'/Users/nurinsyazwinamohdhaniff/Documents/MASTER/THESIS/DATA/Pycharm copy.xlsx')
pd.set_option("display.max_rows",None,"display.max_columns", None)
df = pd.DataFrame(cancer)
X = cancer.drop(columns=['Stage'], axis=1)
y = cancer['Stage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
#X_train, X_test, y_train, y_test = train_test_split(cancer.drop('Stage', axis = 1), cancer['Stage'], test_size = 0.25, random_state = 42)
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats = 3, random_state= 1)

# SVM (ANOVA F-Test + BPSO)
start_SVM = time.time()   #START TIME

std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)

svc = SVC(max_iter=10000, probability=True)

def f_per_particle(m, alpha):
    total_features = X_train.shape[1]
    if np.count_nonzero(m == 0):
        X_subset_train = X_train
    else:
        X_subset_train = X_train[:,m == 1]
    '''clf.fit(X_subset_train, y_train)
    P = (clf.predict(X_subset_train) == y_train).mean()
    j = (alpha * (1.0-P) + (1.0 - alpha) * (1-(X_subset_train.shape[1]/total_features)))'''
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
X_selected_features = X_test[:,pos == 1]
print(X_selected_features.shape)

svc.fit(X_selected_features,y_test)

X_new = SelectKBest(f_classif,k=150).fit_transform(X_selected_features,y_test)
print( X_new.shape)
svc.fit(X_new,y_test)

y_predict_newSF = svc.predict(X_new)

y_score_SVM = svc.decision_function(X_new)
y_proba_SVM = svc.predict_proba(X_new)[:,1]

accuracy_SVM = accuracy_score(y_test,y_predict_newSF)
recall_SVM = recall_score(y_test,y_predict_newSF)
precision_SVM = precision_score(y_test,y_predict_newSF)
specificity_SVM = recall_score(np.logical_not(y_test), np.logical_not(y_predict_newSF))
f1_SVM = f1_score(y_test,y_predict_newSF)
auc_SVM = roc_auc_score(y_test,y_proba_SVM)

end_SVM = time.time() # END TIME
total_SVM = end_SVM - start_SVM # CALCULATE TIME

time_SVM = time.strftime("%H:%M:%S", time.gmtime(total_SVM))
print(time_SVM)

#################
clf_name = []
accuracies = []
recalls = []
precisions = []
specificities = []
F1_Scores = []
aucS = []
score = []
timeS = []

# TPOT DEFAULT
start_TPOT = time.time()  # START TIME

tpot = TPOTClassifier(generations=5, population_size=2, verbosity=2)
tpot.fit(X_train, y_train)
y_predict_TPOT = tpot.predict(X_test)
y_proba_TPOT = tpot.predict_proba(X_test)[:,1]

accuracy_TPOT = accuracy_score(y_test,y_predict_TPOT)
recall_TPOT = recall_score(y_test, y_predict_TPOT)
precision_TPOT = precision_score(y_test,y_predict_TPOT)
specificity_TPOT = recall_score(np.logical_not(y_test), np.logical_not(y_predict_TPOT))
auc_TPOT = roc_auc_score(y_test,y_predict_TPOT)
f1_TPOT = f1_score(y_test, y_predict_TPOT)
score_TPOT = tpot.score(X_test,y_test)

end_TPOT = time.time()  # END TIME
total_TPOT = end_TPOT - start_TPOT  # CALCULATE TIME
time_TPOT = time.strftime("%H:%M:%S", time.gmtime(total_TPOT))

best_model_TPOT = tpot.fitted_pipeline_.steps[-1][1]
print('The best model is', best_model_TPOT)
best_model_TPOT.fit(X_train, y_train)

#selectedFeatures_TPOT = best_model_TPOT.feature_importances_
#print('Selected Features', selectedFeatures_TPOT)

selected_featuresName = X.columns[best_model_TPOT.get_support(indices=True)].tolist() #GET LIST OF SELECTED FEATURES
print(selected_featuresName)

clf_name.append(tpot)
accuracies.append(accuracy_TPOT)
precisions.append(precision_TPOT)
recalls.append(recall_TPOT)
specificities.append(specificity_TPOT)
aucS.append(auc_TPOT)
F1_Scores.append(f1_TPOT)
score.append(score_TPOT)
timeS.append(time_TPOT)

# SVM
start_TPOTSVM = time.time()  # START TIME

svm_config = {'sklearn.svm.SVC':{'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
                    'kernel':["linear","poly","rbf","sigmoid","precomputed"],
                    'degree': range(0,101),'gamma': ["scale","auto"],
                    'shrinking':[True,False], 'probability': [True],'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                    'verbose': [True, False], 'max_iter': range(0, 100000),
                    'decision_function_shape': ["ovo", "ovr"], 'break_ties': [True, False]}}

tpot_SVM = TPOTClassifier(generations = 5, population_size = 2, verbosity= 2,config_dict=svm_config,max_eval_time_mins=60)
tpot_SVM.fit(X_train,y_train)

y_predict_TPOTSVM = tpot_SVM.predict(X_test)
y_proba_TPOTSVM = tpot_SVM.predict_proba(X_test)[:,1]

accuracy_TPOTSVM = accuracy_score(y_test,y_predict_TPOTSVM)
recall_TPOTSVM = recall_score(y_test, y_predict_TPOTSVM)
precision_TPOTSVM = precision_score(y_test,y_predict_TPOTSVM)
specificity_TPOTSVM = recall_score(np.logical_not(y_test), np.logical_not(y_predict_TPOTSVM))
auc_TPOTSVM = roc_auc_score(y_test,y_predict_TPOTSVM)
f1_TPOTSVM = f1_score(y_test, y_predict_TPOTSVM)
score_TPOTSVM = tpot_SVM.score(X_test,y_test)

end_TPOTSVM = time.time()  # END TIME
total_TPOTSVM = end_TPOTSVM - start_TPOTSVM  # CALCULATE TIME
time_TPOTSVM = time.strftime("%H:%M:%S", time.gmtime(total_TPOTSVM))

clf_name.append(tpot_SVM)
accuracies.append(accuracy_TPOTSVM)
precisions.append(precision_TPOTSVM)
recalls.append(recall_TPOTSVM)
specificities.append(specificity_TPOTSVM)
aucS.append(auc_TPOTSVM)
F1_Scores.append(f1_TPOTSVM)
score.append(score_TPOTSVM)
timeS.append(time_TPOTSVM)

# TABULATION DATA
data = {"Classification Algorithms": clf_name, "Accuracy": accuracies, "Precision": precisions, "Recall": recalls,
        "Specificity": specificities, "AUC": aucS, "F1 Scores": F1_Scores, 'Computational Time': timeS}
data["Classification Algorithms"].append(svc)
data["Accuracy"].append(accuracy_SVM)
data["Precision"].append(precision_SVM)
data["Recall"].append(recall_SVM)
data["Specificity"].append(specificity_SVM)
data["AUC"].append(auc_SVM)
data["F1 Scores"].append(f1_SVM)
data["Computational Time"].append(time_SVM)

table = pd.DataFrame(data)
print(tabulate(table, headers='keys', tablefmt='psql'))

# AUC-ROC
plt.figure()
#PSOkBest_auc = roc_auc_score(y_test,y_predict_SVMcm)
#PSOkBest_fpr,PSOkBest_tpr, threshold_PSOkBest = roc_curve(y_test,y_predict_SVMcm)

fpr_SVM, tpr_SVM, threshold_SVM = roc_curve(y_test, y_proba_SVM)
fpr_, tpr_, threshold_default = roc_curve(y_test, y_proba_TPOT)
fpr_TPOTSVM, tpr_TPOTSVM, threshold_TPOTSVM = roc_curve(y_test, y_proba_TPOTSVM)

plt.plot(fpr_SVM, tpr_SVM, color='green', linestyle='--', label='SVM -AUC:%.3f' % auc_SVM)
plt.plot(fpr_, tpr_, linestyle='--', color='navy', label='TPOT-Default -AUC:%.3f' % auc_TPOT)
plt.plot(fpr_TPOTSVM, tpr_TPOTSVM, linestyle='--', color='darkorange', label='TPOT-SVM -AUC:%.3f' % auc_TPOTSVM)

#plt.plot(PSOkBest_fpr,PSOkBest_tpr, linestyle=':', color='red', label='SVM -AUC :%.3f' % auc_SVMcm)

plt.title('ROC-AUC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# PRC
fig,ax = plt.subplots()

ave_pre_SVM = average_precision_score(y_test,y_score_SVM)
SVM_precision,SVM_recall,threshold = precision_recall_curve(y_test,y_score_SVM)
plt.plot(SVM_recall,SVM_precision,color='green',linestyle='-', label="SVM (AP = %.3f)" % ave_pre_SVM)

PrecisionRecallDisplay.from_predictions(y_test, y_predict_TPOT, ax=ax, name='TPOT-Default')
PrecisionRecallDisplay.from_predictions(y_test, y_predict_TPOTSVM, ax=ax, name='TPOT-SVM')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower right')
plt.show()


# PLOTLY
# AUC-ROC
#fpr, tpr, threshold = roc_curve(y_test,y_proba)
'''fig1 = px.area(
    x =fpr_, y=tpr_,
    title = f'ROC Curve using PLOTLY (AUC={auc(fpr_, tpr_):.4f})',
    labels = dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig1.update_layout(
    title={
        'y':0.9,
        'x':0.5,
        'xanchor':'center',
        'yanchor':'top'}
)
fig1.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig1.update_yaxes(scaleanchor="x", scaleratio=1)
fig1.update_xaxes(constrain='domain')
fig1.show()

# PRC
precision,recall, thresholds = precision_recall_curve(y_test,y_proba)
fig = px.area(
    x= recall, y= precision,
    title = f'Precision-Recall Curve (AUC={auc(fpr_, tpr_):.4f}',
    labels= dict(x='recall', y='Precision'),
    width= 700, height=500
)
fig.update_layout(
    title={
        'y':0.9,
        'x':0.5,
        'xanchor':'center',
        'yanchor':'top'}
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0,x1=1,y0=1,y1=0
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.show()'''

#ave_pre_PSOkBEST = average_precision_score(y_test,y_score_SVMcm)
#PSOkBest_precision,PSOkBest_recall,_ = precision_recall_curve(y_test,y_score_SVMcm)
#plt.plot(PSOkBest_recall,PSOkBest_precision,linestyle=':', color='red', label='SVM (AP = %.3f'% ave_pre_PSOkBEST +')')

