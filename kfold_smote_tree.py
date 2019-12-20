# Import modules
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
import scikitplot as skplt

from sklearn import preprocessing, model_selection, metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTENC
# ML with imbalance data using SMOTE base on http://www.fantasyfutopia.com
# Read csv
shots_df = pd.read_csv('shots_df.csv')

# new columns goal
pd.set_option('mode.chained_assignment', None)
np_shots = shots_df[shots_df['type'] != 'Penalty']
np_shots['goal'] = np.where(np_shots['outcome'] == 'Goal', 1, 0)
attempts = len(np_shots)
goals = sum(np_shots['goal'])
misses = attempts - goals
conv_rate = goals/attempts
print("Average conversion rate:", "{0:.2f}%".format(conv_rate*100))
print("")

# Plot the count of 'goal' and 'no goal' events to show imbalance
sns.set(style="ticks", color_codes=True)
sns.countplot(x="goal", data=np_shots)
# plt.show()

# Feature engineering
np_shots = np_shots.reset_index().drop('level_0', axis=1)
np_shots['assisted'] = np.where(np_shots['key_pass_id'].isna(), 0, 1)
np_shots['x_distance'] = 120 - np_shots['start_location_x']
np_shots['y_distance'] = abs(40 - np_shots['start_location_y'])
np_shots['distance'] = np.sqrt((np_shots['x_distance']**2 + np_shots['y_distance']**2))
np_shots['angle'] = np.degrees(np.arctan((np_shots['y_distance']/np_shots['x_distance'])))
np_shots['body_part'] = np.where((np_shots['body_part'] == 'Right Foot')
                                 | (np_shots['body_part'] == 'Left Foot'), 'foot',
                                 np.where(np_shots['body_part'] == 'Head', 'head', 'other'))

feature_cols = ['play_pattern', 'under_pressure', 'body_part', 'technique', 'first_time',
                'follows_dribble', 'redirect', 'one_on_one', 'open_goal', 'deflected',
                'assisted', 'distance', 'angle']
features = np_shots[feature_cols]
labels = np_shots['goal']
features = features.fillna(0)
labels = labels.fillna(0)

# encode using sklearn
cat_cols = ['play_pattern', 'under_pressure', 'body_part', 'technique', 'first_time',
            'follows_dribble', 'redirect', 'one_on_one', 'open_goal', 'deflected']
cat_features = features[cat_cols]
features = features.drop(cat_cols, axis=1)
le = preprocessing.LabelEncoder()
cat_features = cat_features.apply(le.fit_transform)
features = features.merge(cat_features, left_index=True, right_index=True)

# SMOTE-NC need columns to be index
cat_cols_ind = []
for key in cat_cols:
    ind = features.columns.get_loc(key)
    cat_cols_ind.append(ind)

# fit SMOTE-NC
smote_nc = SMOTENC(categorical_features=cat_cols_ind, random_state=42)
features_resampled, labels_resampled = smote_nc.fit_resample(features, labels)

# scaler between 0 and 1 to speed up processing
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(features_resampled)
y = labels_resampled

# define algorithm Decision Tree and K-Fold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
clf = DecisionTreeClassifier(random_state=42)

# Calculate precision and recall inside a loop using cross validation
# First, create blank arrays to store the results in
precision_0 = np.array([])
recall_0 = np.array([])
precision_1 = np.array([])
recall_1 = np.array([])

for train_index, test_index in cv.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Calculate precision and recall scores
    # average=None means we will get a score for each class separately
    precision_scores = metrics.precision_score(y_test, y_pred, average=None)
    recall_scores = metrics.recall_score(y_test, y_pred, average=None)

    # Add the results to the arrays we created earlier
    precision_0 = np.append(precision_0, np.array(precision_scores[0]))
    precision_1 = np.append(precision_1, np.array(precision_scores[1]))
    recall_0 = np.append(recall_0, np.array(recall_scores[0]))
    recall_1 = np.append(recall_1, np.array(recall_scores[1]))



# save algorithm
import joblib
joblib.dump(clf, 'xG_pred.pkl')

# Print results
print("Precision - no goal:", "{0:.4f}".format(precision_0.mean()))
print("Recall - no goal:", "{0:.4f}".format(recall_0.mean()))
print("Precision - goal:", "{0:.4f}".format(precision_1.mean()))
print("Recall - goal:", "{0:.4f}".format(recall_1.mean()))
print('')
