import pandas as pd
import numpy as np
from collections import OrderedDict, Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from xgboost import XGBClassifier
from sklearn import preprocessing, model_selection, metrics
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import make_scorer, matthews_corrcoef, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from imblearn.metrics import *
from sklearn.datasets import load_breast_cancer
from imblearn.datasets import make_imbalance
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
from dtreeviz.trees import *

np_shots = pd.read_csv('np_shots.csv')
features = np_shots.drop('goal', axis=1)
labels = np_shots['goal']

cat_cols = ['play_pattern', 'under_pressure', 'body_part', 'technique', 'first_time',
            'follows_dribble', 'redirect', 'one_on_one', 'open_goal', 'deflected', 'assisted']
cat_features = features[cat_cols]
features = features.drop(cat_cols, axis=1)

# encode to numeric
le = preprocessing.LabelEncoder()
cat_features = cat_features.apply(le.fit_transform)
features = features.merge(cat_features, left_index=True, right_index=True)
print(features)

m = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X = features
y = labels

m.fit(X, y)
scores = cross_val_score(m, X, y, cv=cv, scoring='brier_score_loss')
print(scores.mean() * -1)

from rfpimp import * # pip install rfpimp first
imp = importances(m, X, y, n_samples=-1, metric='brier_score_loss')
viz = plot_importances(imp)
viz.view()


# eliminated feature lower than threshold
def elim_feats(X, imp=imp, thresh=0.005):
    return X[list(imp[imp.values >= thresh].index.values)]


X_new = elim_feats(X=X)

m.fit(X_new, y)
scores = cross_val_score(m, X_new, y, cv=cv, scoring='brier_score_loss')
print(scores.mean()*-1)

imp = importances(m, X_new, y, n_samples=-1, metric='brier_score_loss')
viz = plot_importances(imp)
viz.view()
