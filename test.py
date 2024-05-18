import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mne.decoding import CSP

# Load EEG data
data = pd.read_csv("filtered_data.csv")
shuffled_index = np.random.permutation(data.index)

# Reindex the DataFrame with the shuffled index
shuffled_data = data.reindex(shuffled_index)# Separate features (EEG data) and labels
X= data.drop(columns=['label'])  # Features
Y = data['label']                  # Labels
print(Y)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, chi2

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=15)
# Preprocess the EEG data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

selector = SelectKBest(score_func=f_classif, k=4)  # feature selection here
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Perform Common Spatial Pattern (CSP) for feature extraction
csp = LDA(n_components=min(X_train_selected.shape[1], len(set(y_train))) - 1)  # LinearDiscriminantAnalysis
X_train_csp = csp.fit_transform(X_train_selected, y_train)
X_test_csp = csp.transform(X_test_selected)
print(1)
print(X_test.shape)

# Train SVM classifier
svm_classifier = SVC(kernel='rbf', C=1.0)  # Radial Basis Function
svm_classifier.fit(X_train, y_train)
print(3)
# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
# Evaluate classifiers
svm_predictions = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(svm_accuracy*100)
rf_predictions = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(rf_accuracy*100)
# Compare classifiers based on accuracy
if svm_accuracy > rf_accuracy:
    print("SVM classifier achieves higher accuracy.")
    joblib.dump(svm_classifier, 'svm_model.pkl')
elif svm_accuracy < rf_accuracy:
    print("Random Forest classifier achieves higher accuracy.")
    joblib.dump(rf_classifier, 'rf_model.pkl')
