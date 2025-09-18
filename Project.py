import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, classification_report,adjusted_rand_score, normalized_mutual_info_score
)

#2.1

# fetch dataset
heart = fetch_ucirepo(id=45)  
X = heart.data.features
y = heart.data.targets
df = X.copy()
df['target'] = y
print(df.shape)
print(df.isna().sum())
# Check class distribution
print("Class distribution in the dataset:")
print(y.value_counts().sort_index())


#Replace_NAN_Values

df.replace('?',np.nan,inplace=True)
df['ca']=df['ca'].fillna(df['ca'].mode()[0])
df['thal']=df['thal'].fillna(df['thal'].mode()[0])
print(df.isna().sum())
print(df.head())
numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_cols = ["cp", "slope", "thal", "ca"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

#Scaling

X = df.drop('target', axis=1)
y = df['target']
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X = pd.DataFrame(X_scaled, columns=X.columns)
# print(X.head())

#Graphs

# Histograms
df.hist(bins=20, figsize=(12, 8))
plt.suptitle("Histograms of features")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Boxplots for a few features vs target
for col in ['age', 'trestbps', 'chol', 'thalach']:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df['target'], y=df[col])
    plt.title(f"{col} by Target")
    plt.show()

#2.2

#Apply_PCA

pca=PCA()
X_temp_scaled = StandardScaler().fit_transform(X)  # Temporary scaling just for PCA demo
X_pca = pca.fit_transform(X_temp_scaled)
# cumulative variance
cum_var = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8,5))
plt.plot(cum_var, marker='o')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance vs Components")
plt.grid(True)
plt.show()
pca = PCA(n_components=13)  # keeps ~90% variance
X_pca = pca.fit_transform(X_temp_scaled)
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - First Two Principal Components")
plt.colorbar(label="Target")
plt.show()

#2.3

#Feature Importance with Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_temp_scaled, y)

importances = rf.feature_importances_
feat_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(10,6))
feat_importance.plot(kind='bar')
plt.title("Feature Importance (Random Forest)")
plt.ylabel("Importance Score")
plt.show()

print("\nTop 10 Important Features (Random Forest):")
print(feat_importance.head(10))


# 2. Recursive Feature Elimination (RFE) with Logistic Regression
log_reg = LogisticRegression(max_iter=1000, solver='liblinear')
rfe = RFE(log_reg, n_features_to_select=10)  # keep top 10 features
rfe.fit(X_temp_scaled, y)

selected_rfe = X.columns[rfe.support_]
print("\nSelected Features via RFE:")
print(selected_rfe.tolist())


# 3. Chi-Square Test (for categorical / discrete features)
chi2_selector = SelectKBest(score_func=chi2, k=10)
X_chi2 = chi2_selector.fit_transform(abs(X_temp_scaled), y)  # abs() to avoid negatives

chi2_features = X.columns[chi2_selector.get_support()]
print("\nTop 10 Features via Chi-Square Test:")
print(chi2_features.tolist())


# 4. Reduced dataset with selected features (intersection of methods or union)
selected_features = list(set(feat_importance.head(10).index) | set(selected_rfe) | set(chi2_features))
X_selected = df[selected_features]
X_selected['target'] = y

print("\nFinal Reduced Dataset Shape:", X_selected.shape)
print(X_selected.head())


#2.4

# ==============================
# 4. Train/Test Split
# ==============================
X_final = X_selected.drop('target', axis=1)  # This already has selected features
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 5. Train Models
# ==============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, solver='saga', multi_class='multinomial'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, class_weight='balanced', random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Probabilities (for ROC/AUC)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # Store evaluation metrics
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='macro'),
        "Recall": recall_score(y_test, y_pred, average='macro'),
        "F1-Score": f1_score(y_test, y_pred, average='macro'),
        "AUC": roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro') if y_proba is not None else None
    }

    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))

# ==============================
# 6. ROC Curve (Macro-Average)
# ==============================
from sklearn.preprocessing import label_binarize

classes = np.unique(y)
y_test_bin = label_binarize(y_test, classes=classes)

plt.figure(figsize=(8,6))

for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        # Compute ROC curve and AUC for each class
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Macro-Average)")
plt.legend()
plt.show()

#2.5

# ==============================
# 1. K-Means Clustering
# ==============================
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_temp_scaled)
    inertia.append(kmeans.inertia_)

# Elbow method plot
plt.figure(figsize=(8,5))
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-Cluster SSE)")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.show()

# Apply KMeans with chosen k (say 4 or 5 depending on elbow curve)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters_kmeans = kmeans.fit_predict(X_temp_scaled)

# Visualize first two PCA components with cluster labels
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters_kmeans, cmap="viridis", edgecolor="k")
plt.title("KMeans Clusters (on PCA-reduced data)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Cluster")
plt.show()


# ==============================
# 2. Hierarchical Clustering
# ==============================
# Perform linkage
linked = linkage(X_temp_scaled[:100], method='ward') # subset for speed if dataset is big

plt.figure(figsize=(10,6))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# Apply Agglomerative Clustering (same k as KMeans for comparison)
agglo = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
clusters_hier = agglo.fit_predict(X_temp_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters_hier, cmap="plasma", edgecolor="k")
plt.title("Hierarchical Clusters (on PCA-reduced data)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Cluster")
plt.show()


# ==============================
# 3. Compare Clusters with True Labels
# ==============================
print("Comparison with actual disease labels:")
print("KMeans Adjusted Rand Index:", adjusted_rand_score(y, clusters_kmeans))
print("KMeans Normalized Mutual Info:", normalized_mutual_info_score(y, clusters_kmeans))
print("Hierarchical Adjusted Rand Index:", adjusted_rand_score(y, clusters_hier))
print("Hierarchical Normalized Mutual Info:", normalized_mutual_info_score(y, clusters_hier))

#2.6

# ==============================
# Baseline model (before tuning)
# ==============================
rf = RandomForestClassifier(random_state=42,class_weight="balanced")
rf.fit(X_train, y_train)
baseline_acc = rf.score(X_test, y_test)

# ==============================
# Hyperparameter Tuning
# ==============================

# Define parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],     # number of trees
    "max_depth": [None, 5, 10, 20],     # depth of each tree
    "min_samples_split": [2, 5, 10],    # min samples to split a node
    "min_samples_leaf": [1, 2, 4],      # min samples per leaf
    "max_features": ["sqrt", "log2"],    # number of features to consider at each split
    "class_weight": [None, "balanced"]
}

# GridSearchCV (exhaustive search)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,             # 5-fold cross validation
    scoring="accuracy",
    n_jobs=-1,        # use all cores
    verbose=2
)

grid_search.fit(X_train, y_train)

# RandomizedSearchCV (faster)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=20,        # number of random combinations to try
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_train, y_train)

# ==============================
# Final Evaluation on Test Data
# ==============================
best_rf = random_search.best_estimator_
final_acc = best_rf.score(X_test, y_test)
print("\nFinal Test Accuracy after tuning:", final_acc)

# 2.7 

selected_feature_names = [col for col in X_selected.columns if col != 'target']

# Create preprocessor for ONLY selected features
selected_numeric = [col for col in selected_feature_names if col in numeric_cols]
selected_categorical = [col for col in selected_feature_names if col in categorical_cols]

final_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), selected_numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), selected_categorical)
    ]
)

final_pipeline = Pipeline([
    ("preprocessor", final_preprocessor),
    ("model", best_rf)
])

# Use ONLY the selected features
final_pipeline.fit(df[selected_feature_names], y)


#  Save pipeline as .pkl
joblib.dump(final_pipeline, "heart_disease_model.pkl")
print("\nModel saved as heart_disease_model.pkl")