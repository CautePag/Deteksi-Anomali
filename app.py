import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Judul aplikasi
st.title('Deteksi Anomali Karyawan PT. XYZ')

# Sidebar untuk parameter
st.sidebar.header('Parameter Model')
nu = st.sidebar.slider('Nu (outlier fraction)', 0.01, 0.5, 0.01)
kernel = st.sidebar.selectbox('Kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
gamma = st.sidebar.selectbox('Gamma', ['scale', 'auto'])

# Generate synthetic dataset
X, y = make_classification(n_samples=100000, n_features=2, n_informative=2,
                          n_redundant=0, n_repeated=0, n_classes=2,
                          n_clusters_per_class=1,
                          weights=[0.995,0.001],
                          class_sep=0.5, random_state=0)

# Convert to DataFrame
df = pd.DataFrame({'feature1': X[:, 0], 'feature2': X[:, 1], 'target': y})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with current parameters
@st.cache_data(ttl=60)  # Cache for 5 minutes
def train_model(nu, kernel, gamma):
    with st.spinner('Training model...'):
        model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        model.fit(X_train)
    return model

# Get predictions
def get_predictions(model):
    predictions = model.predict(X_test)
    return [1 if i==-1 else 0 for i in predictions]

# Train model with current parameters
model = train_model(nu, kernel, gamma)
predictions = get_predictions(model)

# Performance metrics
report = classification_report(y_test, predictions, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Visualisasi
st.header('Visualisasi Data dan Hasil')

# Tab untuk visualisasi
tab1, tab2, tab3 = st.tabs(["Data Distribution", "Model Performance", "Predictions"])

with tab1:
    st.subheader('Distribusi Data')
    
    # Combined visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot of features
    scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_title('Distribusi Fitur')
    
    # Add colorbar for scatter plot
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Kelas')
    
    # Daily access visualization
    days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']
    normal_access = np.random.randint(50, 150, size=7)
    anomaly_access = np.random.randint(0, 50, size=7)
    
    x = np.arange(len(days))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, normal_access, width, label='Normal Access', color='skyblue')
    bars2 = ax2.bar(x + width/2, anomaly_access, width, label='Anomali Access', color='red')
    
    ax2.set_xlabel('Hari')
    ax2.set_ylabel('Jumlah Akses')
    ax2.set_title('Distribusi Akses per Hari')
    ax2.set_xticks(x)
    ax2.set_xticklabels(days)
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader('Performance Metrics')
    st.dataframe(report_df)


with tab3:
    st.subheader('Prediksi Anomali')
    score = model.score_samples(X_test)
    score_threshold = np.percentile(score, 2)
    customized_pred = [1 if i < score_threshold else 0 for i in score]

    # Buat gambar dengan 3 subplot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    # Subplot 1: Data Aktual
    scatter_actual = axes[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', edgecolors='k')
    axes[0].set_title('Data Aktual')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    cbar_actual = plt.colorbar(scatter_actual, ax=axes[0])
    cbar_actual.set_label('Kelas')

    # Subplot 2: Prediksi Model
    scatter_pred = axes[1].scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap='coolwarm', edgecolors='k')
    axes[1].set_title('Prediksi Model')
    axes[1].set_xlabel('Feature 1')
    cbar_pred = plt.colorbar(scatter_pred, ax=axes[1])
    cbar_pred.set_label('Prediksi')

    # Subplot 3: Distribusi Akses per Hari
    days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']
    normal_access = np.random.randint(50, 150, size=7)
    anomaly_access = np.random.randint(0, 50, size=7)

    x = np.arange(len(days))
    width = 0.35

    axes[2].bar(x - width/2, normal_access, width, label='Normal Access', color='skyblue')
    axes[2].bar(x + width/2, anomaly_access, width, label='Anomali Access', color='red')
    axes[2].set_xlabel('Hari')
    axes[2].set_ylabel('Jumlah Akses')
    axes[2].set_title('Distribusi Akses per Hari')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(days)
    axes[2].legend()

    # Atur tata letak dan tampilkan plot
    plt.tight_layout()
    st.pyplot(fig)
