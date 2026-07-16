'''import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn import decomposition
from sklearn.model_selection import train_test_split,RepeatedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import precision_score,accuracy_score,roc_auc_score,roc_curve,confusion_matrix,auc,precision_recall_curve
from sklearn.feature_selection import SelectFromModel

cancer = pd.read_excel(r'/Users/nurinsyazwinamohdhaniff/Documents/MASTER/THESIS/DATA/Pycharm copy.xlsx')
pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.DataFrame(cancer)
X = cancer.drop(columns = ['Stage'], axis = 1)
y = cancer['Stage']
features = X.columns
print(features)
#print(df.columns)

#std = StandardScaler()
#X_std = std.fit_transform(X)

#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=42)

# LASSO FOR FEATURE SELECTOR
# Hyper-parameter Tuning for LASSO
#param = {'model__alpha':[.00001, 0.0001,0.001,0.01], 'model__fit_intercept':[True, False], 'model__normalize':[True, False],
         #'model__positive':[True, False], 'model__selection':["cyclic","random"],}

pipe = Pipeline([ ('model',Lasso())])
search = GridSearchCV(pipe, {'model__alpha':np.arange(0.1,10,0.1),'model__fit_intercept':[True, False]},
                              scoring="neg_mean_absolute_error", cv=5, verbose=3)
#, 'model__fit_intercept':[True, False],'model__positive':[True, False], 'model__selection':["cyclic","random"]
#'model__alpha':[0.00001, 0.0001,0.001,0.01]
search.fit(X_train,y_train)

print('Best score for LASSO: %s'% search.best_score_)
print('Best Hyperparameters: %s'% search.best_params_)

coefficients = search.best_estimator_.named_steps['model'].coef_
importance = np.abs(coefficients)
selected_features = features[importance > 0]
removed_features = features[importance == 0]
number_of_features = len(selected_features)

new_df = df.drop(columns = removed_features, axis = 1)
print(new_df.shape)
#print(selected_features)
#print("Removed features%s" % removed_features)
#print('Shape of selected features after using LASSO: %s'% selected_features.shape)
print('Number of features selected using LASSO: %s'% number_of_features)'''

######

'''model_lasso = SelectFromModel(Lasso(alpha=0.1,fit_intercept=True))
model_lasso.fit(X_train,y_train)
array_f_s= model_lasso.get_support()
f_s = X_train.columns[(model_lasso.get_support())]
print(f_s)
print('Features selected from LASSO: %s'% len(f_s))

s_features = X_train.columns[(model_lasso.estimator_.coef_!=0).ravel().tolist()]

#new_df = df.drop([removed_features],axis=1)
#print('Features selected from LASSO:'% s_features.dtype)

model_lasso = Lasso(alpha=0.1,fit_intercept=True)
model_lasso.fit(X_train,y_train)

y_predict = model_lasso.predict(X_test)
#y_proba = model_lasso.predict_proba(X_test)
#precision = precision_score(y_test,y_predict)
#auc_score = roc_auc_score(y_test,y_predict)
#accuracy = accuracy_score(y_test,y_predict)

print(y_predict)'''

######### BERJAYA LASSO
'''import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso

cancer = pd.read_excel(r'/Users/nurinsyazwinamohdhaniff/Documents/MASTER/THESIS/DATA/Pycharm copy.xlsx')
pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.DataFrame(cancer)
X = cancer.drop(columns = ['Stage'], axis = 1)
y = cancer['Stage']

#X,y = load_diabetes(return_X_y=True)
#features = load_diabetes()['feature_names']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=42)

pipeline = Pipeline([('scaler', StandardScaler()), ('model',Lasso())])

search = GridSearchCV(pipeline, {'model__alpha': np.arange(0.1,10,0.1)},cv = 5, scoring ="neg_mean_squared_error", verbose=3)
search.fit(X_train, y_train)

print(search.best_params_)
coefficients = search.best_estimator_.named_steps['model'].coef_
importance = np.abs(coefficients)
print(importance)
#np.array()'''

# GRADIENT DESCENT as FEATURE SELECTOR
'''import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score,accuracy_score,roc_auc_score,roc_curve,confusion_matrix,auc,precision_recall_curve

# Define SGD algorithm
def sgd_feature_selection(X_train, y_train, alpha, num_epochs, num_features):
    #sgd = SGDClassifier(loss=best_loss, alpha=best_alpha, max_iter=best_max_iter, tol=1e-3)
    #sgd = SGDClassifier(loss=log_loss, alpha=alpha, max_iter=num_epochs, tol=1e-3)
    selected_features = np.zeros(X_train.shape[1], dtype=bool)
    for i in range(num_features):
        best_score = 0
        best_feature = 0
        for j in range(X_train.shape[1]):
            if selected_features[j] == False:
                features = np.logical_or(selected_features, np.zeros(X_train.shape[1], dtype=bool))
                features[j] = True
                X_train_selected = X_train[:, features]
                sgd.fit(X_train_selected, y_train)
                score = sgd.score(X_train_selected, y_train)
                if score > best_score:
                    best_score = score
                    best_feature = j
        selected_features[best_feature] = True
    return selected_features

# Load and preprocess the data
cancer = pd.read_excel(r'/Users/nurinsyazwinamohdhaniff/Documents/MASTER/THESIS/DATA/Pycharm copy.xlsx')
pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.DataFrame(cancer)
X = cancer.drop(columns = ['Stage'], axis = 1)
y = cancer['Stage']

#scaler = StandardScaler()
#X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=42)

# Hyperparameter tuning for SGD
pipeline = Pipeline([('scaler', StandardScaler()), ('model',SGDClassifier())])
search = GridSearchCV(pipeline, {"model__loss" : ["hinge", "log", "squared_hinge", "modified_huber", "perceptron"],
    "model__alpha" : [0.0001, 0.001, 0.01, 0.1],
    "model__penalty" : ["l2", "l1", "elasticnet", "none"]},cv = 5, scoring ="neg_mean_squared_error", verbose=3)
search.fit(X_train,y_train)
#print(search.best_params_)

# Run SGD algorithm to find the best subset of features
#alpha = 0.0001
#num_epochs = 100
#num_features = 10
selected_features = sgd_feature_selection(X_train, y_train, alpha, num_epochs, num_features)

#{‘hinge’, ‘log_loss’, ‘log’, ‘modified_huber’, ‘squared_hinge’, ‘perceptron’}
#params = {"loss" : ["hinge", "log", "squared_hinge", "modified_huber", "perceptron"],
# "alpha" : [0.0001, 0.001, 0.01, 0.1],"penalty" : ["l2", "l1", "elasticnet", "none"],"max_iter" : [10, 100, 1000], }

# Print selected features
name_selected_features = cancer.features_names[selected_features]
print("Selected features: ", name_selected_features)
print(len(name_selected_features))



# Evaluate the performance of the selected features on the testing set
X_train_selected = X_train[:, selected_features]
sgd = search()
#sgd = SGDClassifier(loss='log', alpha=alpha, max_iter=num_epochs, tol=1e-3)
sgd.fit(X_train_selected, y_train)
X_test_selected = X_test[:, selected_features]
score = sgd.score(X_test_selected, y_test)
print('Accuracy on testing set:', score)'''

'''### SGD
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

# Load and preprocess the data
cancer = pd.read_excel(r'/Users/nurinsyazwinamohdhaniff/Documents/MASTER/THESIS/DATA/Pycharm copy.xlsx')
pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.DataFrame(cancer)
X = cancer.drop(columns = ['Stage'], axis = 1)
y = cancer['Stage']
features_name = X.columns

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=42)

# Define the hyperparameters to tune & Use GridSearchCV to tune the hyperparameters
pipeline = Pipeline([('scaler', StandardScaler()), ('model',SGDClassifier())])
search = GridSearchCV(pipeline, {"model__loss" : ["hinge", "log", "squared_hinge", "modified_huber", "perceptron"],
    "model__alpha" : [0.0001, 0.001, 0.01, 0.1],
    "model__penalty" : ["l2", "l1", "elasticnet", "none"],"model__max_iter" : [10, 100, 1000]}
                        ,cv = 5)
search.fit(X_train,y_train)

# Get the best hyperparameters and the selected features
best_loss = search.best_params_['model__loss']
best_alpha = search.best_params_['model__alpha']
best_penalty = search.best_params_['model__penalty']
best_num_epochs = search.best_params_['model__max_iter']
best_num_features = 6  #best_num_features = 662 CHECK BALIK

sgd = SGDClassifier(loss=best_loss, alpha=best_alpha, penalty = best_penalty,max_iter=best_num_epochs, tol=1e-3)
selected_features = np.zeros(X_train.shape[1], dtype=bool)

for i in range(best_num_features):
    best_score = 0
    best_feature = 0
    for j in range(X_train.shape[1]):
        if selected_features[j] == False:
            features = np.logical_or(selected_features, np.zeros(X_train.shape[1], dtype=bool))
            features[j] = True
            X_train_selected = X_train.iloc[:, features]
            sgd.fit(X_train_selected, y_train)
            score = sgd.score(X_train_selected, y_train)
            if score > best_score:
                best_score = score
                best_feature = j
    selected_features[best_feature] = True

# Print the selected features
name_selected_features = features_name[selected_features]
print("Selected features: ", name_selected_features)
print(len(name_selected_features))'''

# HEATMAP FOR CORRELATION
'''import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

cancer = pd.read_excel(r'/Users/nurinsyazwinamohdhaniff/Documents/MASTER/THESIS/DATA/shapeFeatures.xlsx')
pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.DataFrame(cancer)
features = cancer.drop(columns=['Stage'], axis=1)

plt.figure(figsize=(20, 9))

mask = np.triu(np.ones_like(features.corr(), dtype=np.bool))
#hMap= sns.heatmap(features.corr(), mask=mask,vmin=-1, vmax=1,cmap= 'YlGnBu', annot=True, linewidths=.5) PEARSON
hMap= sns.heatmap(features.corr(method ='spearman'), mask=mask,vmin=-1, vmax=1,cmap= 'viridis_r', annot=True) #SPEARMAN
#hMap.set_xticklabels(hMap.get_xticklabels(),rotation=40)
plt.show()'''

# SCATTER PLOT
'''import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

cancer = pd.read_excel(r'/Users/nurinsyazwinamohdhaniff/Documents/MASTER/THESIS/DATA/Pycharm copy.xlsx')
pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.DataFrame(cancer)
#print(cancer.head())

X = cancer['Original-MajorAxisLength']
y = cancer['Original-LongRunHighGrayLevelEmphasis']

"""
'Original-LongRunHighGrayLevelEmphasis' 'Original-LargeDependenceEmphasis' 'Original-LargeDependenceLowGrayLevelEmphasis'
'Original-Idmn' 'Original-JointEnergy' 'Original-10Percentile' 'Original-LongRunHighGrayLevelEmphasis' 'HLL-SmallDependenceHighGrayLevelEmphasis'
'HLL-LowGrayLevelEmphasis' 'HLL-SumAverage' 'HLL-JointEntropy'
"""

label_0 = cancer[cancer['Stage'] == 0]
label_1 = cancer[cancer['Stage'] == 1]

len_id_0 = np.where(cancer.Stage == 0)
len_id_1 = np.where(cancer.Stage == 1)

X1_label_0 = label_0['Original-MeshVolume']
X2_label_0 = label_0['Original-Elongation']

X1_label_1 = label_1['Original-MeshVolume']
X2_label_1 = label_1['Original-Elongation']

plt.scatter(X1_label_0,X2_label_0, color= 'navy', label='Group I')
plt.scatter(X1_label_1,X2_label_1, marker = 'D',color= 'orange', label='Group II')
plt.title('Scatter Plot')
plt.xlabel('Original-MeshVolume')
plt.ylabel('Original-Elongation')
plt.legend(loc='lower right')
plt.show()'''

# SCIPY Pearsons & Spearman
'''import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, bootstrap
import numpy as np

cancer = pd.read_excel(r'/Users/nurinsyazwinamohdhaniff/Documents/MASTER/THESIS/DATA/shapeFeatures.xlsx')
pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.DataFrame(cancer)
X = cancer.drop(columns=['Stage'], axis=1)
y = df['Stage']

results_Spearman = []
results_Pearson = []

# Calculate and print p-values
print("Spearman P-values:\n")

for feature in X.columns:

    correlation_Spearman, p_value_Spearman = spearmanr(X[feature], y)
    # Bootstrap confidence interval
    ci_spearman = bootstrap(
        (X[feature], y),
        lambda x, y: spearmanr(x, y)[0],
        paired=True,
        confidence_level=0.95,
        n_resamples=10000,
        method='percentile'
    )
    results_Spearman.append({'Feature': feature, 'Correlation Spearman': correlation_Spearman, 'p_value': p_value_Spearman,
                             'CI Lower': ci_spearman.confidence_interval.low, 'CI Upper': ci_spearman.confidence_interval.high})

    correlation_Pearson, p_value_Pearson = pearsonr(X[feature], y)
    # Bootstrap confidence interval
    ci_pearson = bootstrap(
        (X[feature], y),
        lambda x, y: spearmanr(x, y)[0],
        paired=True,
        confidence_level=0.95,
        n_resamples=10000,
        method='percentile'
    )
    results_Pearson.append({'Feature': feature, 'Correlation Pearson': correlation_Pearson, 'p_value': p_value_Pearson,
                            'CI Lower': ci_pearson.confidence_interval.low, 'CI Upper': ci_pearson.confidence_interval.high})


# Convert to table
results_df_Spearman = pd.DataFrame(results_Spearman)
results_df_Pearson = pd.DataFrame(results_Pearson)

# Print table
print(results_df_Spearman)
print(results_df_Pearson)
'''

# NETWORK CONNECTIVITY PLOT (FEATURES)
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 1. Load and prepare data
cancer = pd.read_excel(r'/Users/nurinsyazwinamohdhaniff/Documents/MASTER/THESIS/DATA/ICC Paper.xlsx',sheet_name='Features',header=1)

# Configure pandas display settings
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Separate features (X) from the clinical outcome label (y)
X = cancer.drop(columns=['Stage'], axis=1)
y = cancer['Stage']

# 2. Calculate the Spearman correlation matrix (ONLY on features, excluding Stage)
corr_matrix = X.corr(method='spearman').abs()

# 3. Create long-form list of pairwise correlations
threshold = 0.85
links = corr_matrix.stack().reset_index()
links.columns = ['var1', 'var2', 'value']

# Filter out self-correlation and connections weaker than the threshold
links_filtered = links[(links['value'] > threshold) & (links['var1'] != links['var2'])]

# 4. Build the mathematical network graph
G = nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')

# Add any isolated features (nodes with 0 correlations) so they still show on the map
G.add_nodes_from(X.columns)

# 5. Automatically categorize features into families based on common prefixes
def assign_color(feature_name):
    name_lower = feature_name.lower()
    if 'shape' in name_lower:
        return '#FF5733'      # Orange/Coral for Shape
    elif 'firstorder' in name_lower:
        return '#33FF57'      # Green for First-Order
    elif any(t in name_lower for t in ['glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']):
        return '#3357FF'      # Blue for Texture
    elif any(f in name_lower for f in ['wavelet', 'log', 'gradient', 'lbp']):
        return '#E333FF'      # Purple for Filters/Wavelets
    else:
        return '#909090'      # Grey for any other types

node_colors = [assign_color(node) for node in G.nodes()]

# 6. Render the Global 662-Feature Visualization
plt.figure(figsize=(16, 16))

# Use an explicit spring layout setting tailored for massive datasets
pos = nx.spring_layout(G, k=0.08, iterations=50, seed=42)

# Draw connections softly in the background
nx.draw_networkx_edges(G, pos, width=0.2, edge_color="gray", alpha=0.2)

# Draw the 662 features color-coded by their family
nx.draw_networkx_nodes(G, pos, node_size=25, node_color=node_colors, alpha=0.8)

# -- NOTE: We do not draw text labels here because 662 labels will break the visual layout --

# Create a clean visual legend for the reviewer
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF5733', label='Shape Features'),
    Patch(facecolor='#33FF57', label='First-Order Statistics'),
    Patch(facecolor='#3357FF', label='Texture (GLCM, GLRLM, etc.)'),
    Patch(facecolor='#E333FF', label='Wavelet / Filtered Features'),
    Patch(facecolor='#909090', label='Other Attributes')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=True)

plt.title(
    f"Global Radiomic Feature Correlation Network Map (P = {len(X.columns)} features, |Rho| > {threshold})\n"
    f"Massive colorful clusters visually demonstrate severe systematic redundancy across different feature classes",
    fontsize=14,
    pad=20,
    fontweight='bold'
)
plt.axis('off')
plt.tight_layout()
plt.show()

# 7. Print the exact text summary to your console for your thesis writing
print("\n--- TEXT DATA SUMMARY FOR YOUR THESIS REGARDING REDUNDANT CLUSTERS ---")
clusters = list(nx.connected_components(G))
large_clusters = [c for c in clusters if len(c) > 1]
isolated_nodes = [c for c in clusters if len(c) == 1]

print(f"Total Features Analyzed: {len(X.columns)}")
print(f"Number of Independent Features (No correlations above 0.85): {len(isolated_nodes)}")
print(f"Number of Redundant Clusters Found: {len(large_clusters)}")

# Print out details of the 5 largest redundant blocks
print("\nTop 5 Largest Clusters of Redundant Features:")
sorted_clusters = sorted(large_clusters, key=len, reverse=True)
for i, cluster in enumerate(sorted_clusters[:5]):
    print(f"\nCluster {i+1} contains {len(cluster)} identical/correlated features:")
    # Print just the first 8 features as an example so the console doesn't overflow
    sample_features = list(cluster)[:8]
    print(f"   Sample features in this web: {', '.join(sample_features)} ...")

# VIOLIN PLOT FOR ICC
'''import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your SPSS data containing only Method A's ICC results
# Columns: ['Feature_Name', 'Feature_Class', 'ICC']
df = pd.read_excel(r'/Users/nurinsyazwinamohdhaniff/Documents/MASTER/THESIS/DATA/ICC Paper.xlsx', sheet_name="ICC", header=2)

plt.figure(figsize=(9, 6), dpi=300)
sns.set_theme(style="whitegrid")

# =====================================================================
# 1. STANDARDIZE INTO YOUR 4 EXACT TYPES
# =====================================================================
def assign_core_type(row):
    feature_name = str(row['Feature_Name']).lower()
    feature_class = str(row['Feature_Class']).lower()

    # Check for wavelet filter prefix first
    if 'wavelet' in feature_name or 'wavelet' in feature_class:
        return 'Wavelet'
    elif 'shape' in feature_class:
        return 'Shape'
    elif 'firstorder' in feature_class or 'first order' in feature_class or 'intensity' in feature_class:
        return 'First Order Statistics'
    else:
        return 'Texture'  # Captures GLCM, GLRLM, GLSZM, NGTDM, etc.


# Apply the clean categorization mapping
df['Core_Type'] = df.apply(assign_core_type, axis=1)

# Set the explicit presentation order for your 4 types on the X-axis
target_order = ['Shape', 'First Order Statistics', 'Texture', 'Wavelet']

# =====================================================================
# 2. GENERATE THE DYNAMIC AXIS LABELS (NAME + FEATURE COUNT)
# =====================================================================
label_mapping = {}
for cls in target_order:
    # Safely count features matching this core category
    count = len(df[df['Core_Type'] == cls])
    label_mapping[cls] = f"{cls}\n(N={count})"

# Map new text string markers into a dedicated plotting column
df['Plot_Labels'] = df['Core_Type'].map(label_mapping)
ordered_plot_labels = [label_mapping[cls] for cls in target_order]

# =====================================================================
# 3. DEFINE THE 4 DISTINCT HEX COLORS
# =====================================================================
custom_colors = {
    label_mapping['Shape']:                  '#264653',  # Deep Navy Blue
    label_mapping['First Order Statistics']: '#2a9d8f',  # Slate Purple
    label_mapping['Texture']:                '#e9c46a',  # Orchid Pink
    label_mapping['Wavelet']:                '#e76f51',  # Amber Yellow
}

# Matching colors for the individual data points (slightly darker variants)
dot_colors = {
    label_mapping['Shape']:                  '#14252e',
    label_mapping['First Order Statistics']: '#16534c',
    label_mapping['Texture']:                '#a68127',
    label_mapping['Wavelet']:                '#a63d21',
}

# =====================================================================
# 4. GENERATE THE VISUALIZATION
# =====================================================================
ax = sns.violinplot(
    data=df,
    x='Plot_Labels',
    y='ICC',
    order=ordered_plot_labels,  # Guarantees Shape -> First Order -> Texture -> Wavelet order
    hue='Plot_Labels',
    palette=custom_colors,
    cut=0,
    alpha=0.8,
    inner='quartiles',
    legend=False
)

# --- CHANGED: Individual Data Dots are now color-matched to their specific violin ---
sns.stripplot(
    data=df,
    x='Plot_Labels',
    y='ICC',
    order=ordered_plot_labels,
    hue='Plot_Labels',     # Tells Seaborn to apply different colors to the dots too
    palette=dot_colors,    # Uses the corresponding darker dot palette
    alpha=0.25,            # Semi-transparent so overlapping points don't look muddy
    size=3.5,
    jitter=0.25,
    legend=False           # Prevents a duplicate legend box
)

# Draw the crucial stability threshold line
plt.axhline(y=0.75, color='crimson', linestyle='--', linewidth=1.5, label='Stability Threshold (ICC = 0.75)')

# Formatting layout text
plt.xlabel('Types of Radiomic Features', fontsize=11, labelpad=12)
plt.ylabel('Intraclass Correlation Coefficient (ICC)', fontsize=11, labelpad=10)
plt.ylim(-0.05, 1.05)

# Minimal rotation needed since we limited it to 4 balanced columns
plt.xticks(rotation=0, ha='center')
plt.legend(loc='lower left', frameon=True, facecolor='white')
plt.tight_layout()

# Save figure for manuscript assembly
plt.savefig("ICC_Violin.png", dpi=300, bbox_inches='tight')
plt.show()'''

#wavelet asing
'''import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your SPSS data containing only Method A's ICC results
df = pd.read_excel(r'/Users/nurinsyazwinamohdhaniff/Documents/MASTER/THESIS/DATA/ICC Paper.xlsx', sheet_name="ICC", header=2)

plt.figure(figsize=(12, 6), dpi=300) # Increased width slightly to accommodate extra columns
sns.set_theme(style="whitegrid")

# =====================================================================
# 1. STANDARDIZE INTO EXACT TYPES (WITH WAVELET SUB-CLASSES)
# =====================================================================
def assign_core_type(row):
    feature_name = str(row['Feature_Name']).lower()
    feature_class = str(row['Feature_Class']).lower()
    combined_text = f"{feature_name} {feature_class}"

    # Check for wavelet filter prefix first
    if 'wavelet' in combined_text:
        if 'glcm' in combined_text:
            return 'Wavelet (GLCM)'
        elif 'glrlm' in combined_text:
            return 'Wavelet (GLRLM)'
        elif 'gldm' in combined_text:
            return 'Wavelet (GLDM)'
        elif 'glszm' in combined_text:
            return 'Wavelet (GLSZM)'
        elif 'ngtdm' in combined_text:
            return 'Wavelet (NGTDM)'
        elif 'firstorder' in combined_text or 'first order' in combined_text:
            return 'Wavelet (First Order)'
        else:
            return 'Wavelet (Other)'
            
    # Non-wavelet features
    elif 'shape' in feature_class:
        return 'Shape'
    elif 'firstorder' in feature_class or 'first order' in feature_class or 'intensity' in feature_class:
        return 'First Order Statistics'
    else:
        return 'Texture'  # Original non-wavelet textures (GLCM, GLRLM, etc.)


# Apply the clean categorization mapping
df['Core_Type'] = df.apply(assign_core_type, axis=1)

# Get all unique Wavelet sub-classes present in your data dynamically to avoid plotting empty categories
wavelet_subclasses = sorted([cls for cls in df['Core_Type'].unique() if 'Wavelet' in cls])

# Set the explicit presentation order for your plot layout
target_order = ['Shape', 'First Order Statistics', 'Texture'] + wavelet_subclasses

# =====================================================================
# 2. GENERATE THE DYNAMIC AXIS LABELS (NAME + FEATURE COUNT)
# =====================================================================
label_mapping = {}
for cls in target_order:
    count = len(df[df['Core_Type'] == cls])
    # Reformat long labels with newlines so they stack neatly on the X-axis
    display_name = cls.replace(" (", "\n(")
    label_mapping[cls] = f"{display_name}\n(N={count})"

# Map new text string markers into a dedicated plotting column
df['Plot_Labels'] = df['Core_Type'].map(label_mapping)
ordered_plot_labels = [label_mapping[cls] for cls in target_order]

# =====================================================================
# 3. DEFINE PALETTES (DYNAMICALLY GENERATED FOR NEW EXTENDED CLASSES)
# =====================================================================
# Base colors for the standard features
base_colors = {
    'Shape': '#264653',                  # Deep Navy Blue
    'First Order Statistics': '#2a9d8f', # Slate Green
    'Texture': '#e9c46a',                # Muted Yellow
}

base_dot_colors = {
    'Shape': '#14252e',
    'First Order Statistics': '#16534c',
    'Texture': '#a68127',
}

# Generate gradients/variations of the Amber Orange ('#e76f51') for Wavelet sub-classes
wavelet_palette = sns.color_palette("Oranges", len(wavelet_subclasses)).as_hex()
wavelet_dot_palette = sns.color_palette("Dark2", len(wavelet_subclasses)).as_hex() # Distinct dark dots

custom_colors = {}
dot_colors = {}

# Map base features to main colors
for cls in ['Shape', 'First Order Statistics', 'Texture']:
    custom_colors[label_mapping[cls]] = base_colors[cls]
    dot_colors[label_mapping[cls]] = base_dot_colors[cls]

# Map Wavelet sub-classes to orange gradients
for idx, w_cls in enumerate(wavelet_subclasses):
    custom_colors[label_mapping[w_cls]] = wavelet_palette[idx]
    dot_colors[label_mapping[w_cls]] = wavelet_dot_palette[idx]

# =====================================================================
# 4. GENERATE THE VISUALIZATION
# =====================================================================
ax = sns.violinplot(
    data=df,
    x='Plot_Labels',
    y='ICC',
    order=ordered_plot_labels,  
    hue='Plot_Labels',
    palette=custom_colors,
    cut=0,
    alpha=0.8,
    inner='quartiles',
    legend=False
)

sns.stripplot(
    data=df,
    x='Plot_Labels',
    y='ICC',
    order=ordered_plot_labels,
    hue='Plot_Labels',     
    palette=dot_colors,    
    alpha=0.25,            
    size=3.5,
    jitter=0.25,
    legend=False           
)

# Draw the stability threshold line
plt.axhline(y=0.75, color='crimson', linestyle='--', linewidth=1.5, label='Stability Threshold (ICC = 0.75)')

# Formatting layout text
plt.xlabel('Types of Radiomic Features', fontsize=11, labelpad=12)
plt.ylabel('Intraclass Correlation Coefficient (ICC)', fontsize=11, labelpad=10)
plt.ylim(-0.05, 1.05)

# Slight rotation (15 degrees) added since text blocks are slightly wider now
plt.xticks(rotation=15, ha='right')
plt.legend(loc='lower left', frameon=True, facecolor='white')
plt.tight_layout()

# Save figure for manuscript assembly
plt.savefig("ICC_Violin_Split_Wavelet.png", dpi=300, bbox_inches='tight')
plt.show()

'''

# RIDGE PLOT FOR ICC
'''import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Load your SPSS data containing only Method A's ICC results
# Columns: ['Feature_Name', 'Feature_Class', 'ICC']
df = pd.read_excel(r'/Users/nurinsyazwinamohdhaniff/Documents/MASTER/THESIS/DATA/ICC Paper.xlsx', sheet_name="ICC", header=2)

# Ensure the correct plotting environment
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# =====================================================================
# 1. STANDARDIZE INTO YOUR 4 EXACT TYPES
# =====================================================================
def assign_core_type(row):
    feature_name = str(row['Feature_Name']).lower()
    feature_class = str(row['Feature_Class']).lower()

    if 'wavelet' in feature_name or 'wavelet' in feature_class:
        return 'Wavelet'
    elif 'shape' in feature_class:
        return 'Shape'
    elif 'firstorder' in feature_class or 'first order' in feature_class or 'intensity' in feature_class:
        return 'First Order Statistics'
    else:
        return 'Texture'


df['Core_Type'] = df.apply(assign_core_type, axis=1)
target_order = ['Shape', 'First Order Statistics', 'Texture', 'Wavelet']

# =====================================================================
# 2. GENERATE THE DYNAMIC LABELS (NAME + FEATURE COUNT)
# =====================================================================
label_mapping = {}
for cls in target_order:
    count = len(df[df['Core_Type'] == cls])
    label_mapping[cls] = f"{cls}\n(N={count})"

df['Plot_Labels'] = df['Core_Type'].map(label_mapping)
ordered_plot_labels = [label_mapping[cls] for cls in target_order]

# =====================================================================
# 3. DEFINE THE 4 DISTINCT HEX COLORS
# =====================================================================
custom_colors = {
    label_mapping['Shape']: '#003f5c',
    label_mapping['First Order Statistics']: '#58508d',
    label_mapping['Texture']: '#bc5090',
    label_mapping['Wavelet']: '#ffa600',
}

# =====================================================================
# 4. GENERATE THE RIDGE VISUALIZATION (FACETGRID)
# =====================================================================
# Initialize the FacetGrid with row-based stacking
g = sns.FacetGrid(
    data=df,
    row="Plot_Labels",
    hue="Plot_Labels",
    aspect=6,
    height=1.5,
    palette=custom_colors,
    row_order=ordered_plot_labels
)

# Draw the densities (KDE curves) filled and outlined
g.map(sns.kdeplot, "ICC", bw_adjust=0.6, clip=(0, 1), fill=True, alpha=0.7, linewidth=1.5)
g.map(sns.kdeplot, "ICC", bw_adjust=0.6, clip=(0, 1), color="white", linewidth=2)

# Reference line for each facet base
g.map(plt.axhline, y=0, linewidth=2, color="gray", alpha=0.5)

# =====================================================================
# 5. POLISH AND ALIGN SUBPLOTS
# =====================================================================
def label_facets(x, color, label):
    ax = plt.gca()
    # Add text labels on the left side of each ridge
    ax.text(-0.02, 0.2, label,
            ha="right", va="center", transform=ax.transAxes, fontsize=10)

    # Overlap individual data points at the bottom of each ridge
    subset = df[df['Plot_Labels'] == label]
    ax.scatter(subset['ICC'], np.zeros_like(subset['ICC']) - 0.02,
               color=color, alpha=0.15, s=15, zorder=3)

    # Add the crucial 0.75 stability threshold line to every subplot
    ax.axvline(x=0.75, color='crimson', linestyle='--', linewidth=1.2, alpha=0.8)


g.map(label_facets, "ICC")

# Overlap subplots vertically to create the "ridge" effect
g.figure.subplots_adjust(hspace=-0.35)

# Remove explicit titles and side axes elements
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)

# Format the final global X axis at the bottom
plt.xlim(-0.05, 1.05)
plt.xlabel('Intraclass Correlation Coefficient (ICC)', fontsize=12, labelpad=10)

# Add a manual legend entry or label text for the stability line
g.axes[-1][0].text(0.76, 1.5, 'Stability Threshold (ICC = 0.75)',
                   color='crimson', fontsize=9)

# Save and export configuration
plt.savefig("ICC_Ridge.png", dpi=300, bbox_inches='tight')
plt.show()'''

# RAINCLOUD FOR ICC
'''import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Load your SPSS data containing only Method A's ICC results
# Columns: ['Feature_Name', 'Feature_Class', 'ICC']
df = pd.read_excel(r'/Users/nurinsyazwinamohdhaniff/Documents/MASTER/THESIS/DATA/ICC Paper.xlsx', sheet_name="ICC", header=2)

# Set crisp layout theme
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 7), dpi=300)

# =====================================================================
# 1. STANDARDIZE INTO YOUR 4 EXACT TYPES
# =====================================================================
def assign_core_type(row):
    feature_name = str(row['Feature_Name']).lower()
    feature_class = str(row['Feature_Class']).lower()

    if 'wavelet' in feature_name or 'wavelet' in feature_class:
        return 'Wavelet'
    elif 'shape' in feature_class:
        return 'Shape'
    elif 'firstorder' in feature_class or 'first order' in feature_class or 'intensity' in feature_class:
        return 'First Order Statistics'
    else:
        return 'Texture'

df['Core_Type'] = df.apply(assign_core_type, axis=1)
target_order = ['Shape', 'First Order Statistics', 'Texture', 'Wavelet']

# =====================================================================
# 2. GENERATE THE DYNAMIC LABELS (NAME + FEATURE COUNT)
# =====================================================================
label_mapping = {}
for cls in target_order:
    count = len(df[df['Core_Type'] == cls])
    label_mapping[cls] = f"{cls}\n(N={count})"

df['Plot_Labels'] = df['Core_Type'].map(label_mapping)
ordered_plot_labels = [label_mapping[cls] for cls in target_order]

# =====================================================================
# 3. DEFINE THE DISTINCT HEX COLORS
# =====================================================================
custom_colors = {
    label_mapping['Shape']: '#003f5c',
    label_mapping['First Order Statistics']: '#58508d',
    label_mapping['Texture']: '#bc5090',
    label_mapping['Wavelet']: '#ffa600',
}

# =====================================================================
# 4. GENERATE THE RAINCLOUD PLOT ELEMENTS
# =====================================================================
# Configuration parameters for spacing layout
width_val = 0.4
base_positions = np.arange(len(ordered_plot_labels))
axis_label_color = plt.rcParams.get('text.color', '#212121')

ax = plt.gca()

for i, label in enumerate(ordered_plot_labels):
    subset = df[df['Plot_Labels'] == label]
    color_val = custom_colors[label]

    # --- THE CLOUD: Half-Violin / Asymmetric KDE ---
    # Compute kernel density estimation
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(subset['ICC'])
    x_vals = np.linspace(0, 1, 200)
    y_vals = kde(x_vals)

    # Scale density height to fit nicely between columns
    y_vals_scaled = (y_vals / y_vals.max()) * width_val

    # Plot filled shape shifted slightly upwards
    ax.fill_betweenx(x_vals, base_positions[i], base_positions[i] - y_vals_scaled,
                     color=color_val, alpha=0.6, edgecolor=color_val, linewidth=1.5)

    # --- THE RAIN: Jittered Data Points Below ---
    # Add negative jitter offset so it sits opposite to the cloud
    jitter = np.random.uniform(low=0.03, high=0.15, size=len(subset))
    ax.scatter(base_positions[i] + jitter, subset['ICC'],
               color=color_val, alpha=0.3, s=12, zorder=2)

    # --- THE LIGHTNING: Central Boxplot inside ---
    ax.boxplot(subset['ICC'], positions=[base_positions[i] + 0.02],
               widths=0.06, showfliers=False, patch_artist=True,
               medianprops={'color': axis_label_color, 'linewidth': 1.5},
               boxprops={'color': color_val, 'linewidth': 1.5},
               whiskerprops={'color': color_val, 'linewidth': 1.2},
               capprops={'color': color_val, 'linewidth': 1.2})

# =====================================================================
# 5. POLISH AND THRESHOLD FORMATTING
# =====================================================================
# Draw the crucial stability threshold line across the canvas
plt.axhline(y=0.75, color='crimson', linestyle='--', linewidth=1.5, label='Stability Threshold (ICC = 0.75)', zorder=1)

# Format axes and uniform typography colors
plt.xticks(base_positions, ordered_plot_labels, color=axis_label_color, fontsize=10)
plt.yticks(np.arange(0, 1.1, 0.2),color=axis_label_color)
plt.ylabel('Intraclass Correlation Coefficient (ICC)', fontsize=12, labelpad=12, color=axis_label_color)
plt.xlabel('Types of Radiomic Features', fontsize=12, labelpad=12, color=axis_label_color)

plt.ylim(-0.02, 1.02)
plt.xlim(-0.4, len(ordered_plot_labels) - 0.6)

# Grid layout tweaks
ax.xaxis.grid(False)  # Keep horizontal lines, drop cluttering vertical lines
plt.legend(loc='lower left', frameon=True, facecolor='white')
plt.tight_layout()

# Save for publication use
plt.savefig("ICC_Raincloud_scaled.png", dpi=300, bbox_inches='tight')
plt.show()'''
