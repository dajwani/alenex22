# %%
from script.formulation import *
from script.functions import *
import pandas as pd
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# %%
# Prepare for paths
filenames = get_selected_files()
ds_paths = dict(zip(filenames, get_paths(filenames, 'ds')))
df_paths = dict(zip(filenames, get_paths(filenames, 'df')))
log_paths = dict(zip(filenames, get_paths(filenames, 'log')))
test_samples = ['i160-314',
                'i160-245',
                'i160-313',
                'i160-242',
                'i160-241',
                'i160-244',
                'i160-343',
                'i160-344',
                'i160-341',
                'i160-345',
                'i160-342']
# random.seed(0)
# test_samples_rand = random.sample(filenames, 11)

# %%
# Read dataframes
features = {
    0:['LP'],
    1:['Normalized Weight', 'Normalized Weight Std'],
    2:['Local Rank Max', 'Local Rank Min', 'Local Rank Product', 'Edges Connected'],
    3:['Degree Centrality Max', 'Degree Centrality Min', 'Degree Centrality Product'],
    4:['Betweenness Centrality Max', 'Betweenness Centrality Min', 'Betweenness Centrality Product'],
    5:['Eigenvector Centrality Max', 'Eigenvector Centrality Min', 'Eigenvector Centrality Product']
}
# Here we will have our train dataframe and test dataframe for the runtime/objective evaluation
# But we will only use the dataframe (without the test samples) to do the model selection
df, df_eval = get_dfs(test_samples, filenames, ds_paths, log_paths)

# Validation
# x_train, y_train, x_test, y_test = get_xy(df)

# Actual test
x_train, y_train = split_x_y(df)
x_test, y_test = split_x_y(df_eval)

# %%
clfs = {
    "Support Vector Machine" : SVC(
        class_weight='balanced', probability=True, random_state=0, kernel='linear'),
    "Random Forest" : RandomForestClassifier(class_weight='balanced'),
    "Logistic Regression" : LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=0),
    "K-Nearest Neighbour" : KNeighborsClassifier(weights="distance", n_neighbors=10),
    "Gaussian Naive Bayes" : GaussianNB()
}

# %%
# clfs_copy = clfs.copy()
# for clf in clfs_copy:
#     clfs_copy[clf].fit(x_train.drop(['LP_bool'], axis=1), y_train)
# fig, ax = plt.subplots(3, 2)
# ax[0,0].hist(clfs_copy["Support Vector Machine"].predict_proba(x_test.drop(['LP_bool'], axis=1))[:,1], bins=100)
# ax[0,0].set_title("Support Vector Machine")
# ax[0,0].set_xlabel("probability of positive label (Should Not Prune)")

# ax[1,0].hist(clfs_copy["Random Forest"].predict_proba(x_test.drop(['LP_bool'], axis=1))[:,1], bins=100)
# ax[1,0].set_title("Random Forest")
# ax[1,0].set_xlabel("probability of positive label (Should Not Prune)")

# ax[2,0].hist(clfs_copy["Logistic Regression"].predict_proba(x_test.drop(['LP_bool'], axis=1))[:,1], bins=100)
# ax[2,0].set_title("Logistic Regression")
# ax[2,0].set_xlabel("probability of positive label (Should Not Prune)")

# ax[0,1].hist(clfs_copy["K-Nearest Neighbour"].predict_proba(x_test.drop(['LP_bool'], axis=1))[:,1], bins=100)
# ax[0,1].set_title("K-Nearest Neighbour")
# ax[0,1].set_xlabel("probability of positive label (Should Not Prune)")

# ax[1,1].hist(clfs_copy["Gaussian Naive Bayes"].predict_proba(x_test.drop(['LP_bool'], axis=1))[:,1], bins=100)
# ax[1,1].set_title("Gaussian Naive Bayes")
# ax[1,1].set_xlabel("probability of positive label (Should Not Prune)")

# ax[2,1].set_visible(False)

# fig.set_size_inches(10.5, 15.5)
# fig.tight_layout(pad=2)
# plt.show()

# %%
# Ensemble model:
import numpy as np
class Ensemble_clf:
    def __init__(self, major, minors, weights=None):
        self.major = major
        self.minors = minors
        if weights:
            self.weights = weights
        else:
            self.weights = np.ones(len(minors))
            
    def fit(self, X, y):
        # fit major first
        self.major.fit(X, y)
        self.fit_minor(X, y)
        self.fit_weights(X, y)

    def fit_minor(self, X, y):
        for clf in self.minors:
            clf.fit(X, y)
        
    def fit_weights(self, X, y):
        # Find the optimized weight vector for minor clfs
        return None

    def predict_proba(self, x):
        y_major = self.major.predict_proba(x)
        y_minors = [clf.predict_proba(x) for clf in self.minors]
        for i in range(len(y_minors)):
            y_minors[i] = [np.multiply(each, self.weights[i]) for each in y_minors[i]]
        shape = (len(x), 2)
        y_minors_sum = np.zeros(shape=shape)
        for each in y_minors:
            y_minors_sum = np.add(y_minors_sum, each)
        y_minors_mean = y_minors_sum / len(self.minors)
        result = []
        for i in range(len(x)):
            result.append(y_major[i]*0.70 + y_minors_mean[i]*0.30)
        
        return np.array(result)

clfs_copy = clfs.copy()
ensemble_clf_conf = {
    "major"  : clfs_copy["Random Forest"],
    "minor"  : [clfs_copy["Support Vector Machine"], clfs_copy["Logistic Regression"], clfs_copy["Gaussian Naive Bayes"]]
}
ensemble_clf = Ensemble_clf(ensemble_clf_conf['major'], ensemble_clf_conf['minor'])
clfs['Ensemble Classifier'] = ensemble_clf

# %%
# plot_data = {}
# for clf in clfs:
#     print("Model: ", clf)
#     print("Training...")
#     clfs[clf].fit(x_train.drop(['LP_bool'], axis=1), y_train)
#     print("Train Finished")
#     print("Adjust Thresholds by Pruning Percentage:")
#     pruning_rates = np.arange(0,0.975,0.025) + 0.025
#     y_pred_proba = clfs[clf].predict_proba(x_test.drop(['LP_bool'], axis=1))[:,1]
#     length = len(y_pred_proba)
#     plot_data[clf] = []
#     for pr in pruning_rates:
#         threshold = np.sort(y_pred_proba)[int(pr*length):][0]
#         y_pred = (y_pred_proba >= threshold).astype('int')
#         tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
#         print("Threshold:",np.round(threshold, 5), 
#             "FN rate:", fn/(fn+tp),
#             "Pruning Rate:", np.round(100*(fn+tn)/len(y_pred),2), '%')
#         if((fn+tn)/len(y_pred) != 0):
#             plot_data[clf].append((np.round(100*(fn+tn)/len(y_pred),2), np.round(fn/(fn+tp), 2)))

# %%
# fig, ax = plt.subplots(3, 2)
# ax[0,0].plot(*zip(*plot_data["Support Vector Machine"]))
# ax[0,0].set_title("Support Vector Machine")
# ax[0,0].set_xlabel("Pruning Rate (%)")
# ax[0,0].set_ylabel("False Negative Rate")

# ax[1,0].plot(*zip(*plot_data["Random Forest"]))
# ax[1,0].set_title("Random Forest")
# ax[1,0].set_xlabel("Pruning Rate (%)")
# ax[1,0].set_ylabel("False Negative Rate")

# ax[2,0].plot(*zip(*plot_data["Logistic Regression"]))
# ax[2,0].set_title("Logistic Regression")
# ax[2,0].set_xlabel("Pruning Rate (%)")
# ax[2,0].set_ylabel("False Negative Rate")

# ax[0,1].plot(*zip(*plot_data["K-Nearest Neighbour"]))
# ax[0,1].set_title("K-nearest neighbour")
# ax[0,1].set_xlabel("Pruning Rate (%)")
# ax[0,1].set_ylabel("False Negative Rate")

# ax[1,1].plot(*zip(*plot_data["Gaussian Naive Bayes"]))
# ax[1,1].set_title("Gaussian naive bayes")
# ax[1,1].set_xlabel("Pruning Rate (%)")
# ax[1,1].set_ylabel("False Negative Rate")

# ax[2,1].plot(*zip(*plot_data["Support Vector Machine"]), label='Support Vector Machine')
# ax[2,1].plot(*zip(*plot_data["Random Forest"]), label='Random Forest')
# ax[2,1].plot(*zip(*plot_data["Logistic Regression"]), label='Logistic Regression')
# ax[2,1].plot(*zip(*plot_data["K-Nearest Neighbour"]), label='K-Nearest Neighbour')
# ax[2,1].plot(*zip(*plot_data["Gaussian Naive Bayes"]), label='Gaussian Naive Bayes')
# ax[2,1].plot(*zip(*plot_data["Ensemble Classifier"]), label='Ensemble Classifier')
# ax[2,1].set_title("All Classifiers (Pruning rate up to 80%)")
# ax[2,1].set_xlabel("Pruning Rate (%)")
# ax[2,1].set_ylabel("False Negative Rate")
# ax[2,1].set_ylim(0, 0.15)
# ax[2,1].set_xlim(0, 80)
# ax[2,1].legend()

# fig.set_size_inches(10.5, 15.5)
# fig.tight_layout(pad=2)
# plt.show()

# %%
# from sklearn.inspection import permutation_importance

# print("Feature Importance: Random Forest")
# print(x_train.drop(['LP_bool'], axis=1).std()*clfs['Random Forest'].feature_importances_)

# print("Feature Importance: Logistic Regression")
# print(x_train.drop(['LP_bool'], axis=1).std()*clfs['Logistic Regression'].coef_[0])

# print("Feature Importance: Support Vector Machine")
# print(x_train.drop(['LP_bool'], axis=1).std()*clfs['Support Vector Machine'].coef_[0])

# print("Feature Importance: Gaussian Naive Bayes")
# print(x_train.drop(['LP_bool'], axis=1).std()*permutation_importance(clfs['Gaussian Naive Bayes'], x_train, y_train).importances_mean)

# %%
pruning_rates = np.arange(0.30,0.80,0.05)
clf = clfs['Ensemble Classifier']
clf.fit(x_train.drop(['LP_bool'], axis=1), y_train)

print("Classifier: Ensemble Classifier")
for filename in test_samples:
    print("############################ NEW SAMPLE ############################")
    print(f"Sample ID: {filename}")
    ds_path = ds_paths[filename]
    log_path = log_paths[filename]
    log = read_log(log_path)
    print("############################ LP PRUNING ############################")
    obj_LP = solve_LP(ds_path, log_path)
    print(f"obj_LP is: {obj_LP}")
    print("############################ ML PRUNING ############################")
    if obj_LP == log['ilp_c']: 
        print("LP pruned graph remains the optimality")
        continue
    for pruning_rate in pruning_rates:
        print("############################ NEW PRUNING RATE ############################")
        print(f"Current purning percentage: {pruning_rate}")
        obj_ILP = solve_ILP(clf, ds_path, log_path, pruning_rate)
        print(f"obj_ILP is: {obj_ILP}")
        # If the current result return the LP pruning result, break to next
        if obj_ILP >= obj_LP:
            break


