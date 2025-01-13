import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# Title for the app
st.title('Deteksi Anomali Karyawan PT. XYZ')

# Sidebar for model parameters
st.sidebar.header('Parameter Model')
nu = st.sidebar.slider('Nu (outlier fraction)', 0.01, 0.5, 0.01)
kernel = st.sidebar.selectbox('Kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
gamma = st.sidebar.selectbox('Gamma', ['scale', 'auto'])

# Generate synthetic dataset for anomaly detection
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                          n_redundant=0, n_repeated=0, n_classes=2,
                          n_clusters_per_class=1,
                          weights=[0.995,0.001],
                          class_sep=0.5, random_state=0)

# Generate synthetic data for anomaly detection
normal_login = np.random.randint(20, 100, size=7)
anomaly_login = np.random.randint(0, 30, size=7)

# Convert to DataFrame
df = pd.DataFrame({'feature1': X[:, 0], 'feature2': X[:, 1], 'target': y})

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with current parameters
@st.cache_data(ttl=60)  # Cache for 1 minute
def train_model(nu, kernel, gamma):
    with st.spinner('Training model...'):
        model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        model.fit(X_train)
    return model

# Get predictions for anomaly detection
def get_predictions(model):
    predictions = model.predict(X_test)
    return [1 if i == -1 else 0 for i in predictions]

# Train the model
model = train_model(nu, kernel, gamma)
predictions = get_predictions(model)

# Performance metrics
report = classification_report(y_test, predictions, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Visualizations
st.header('Visualisasi Data dan Hasil')

# Tab for visualizations
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

    # Access data visualization (normal vs anomaly access)
    days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']
    normal_access = np.random.randint(50, 150, size=7)
    anomaly_access = np.random.randint(0, 50, size=7)
    normal_login = np.random.randint(30, 100, size=7)
    anomaly_login = np.random.randint(0, 30, size=7)

    x = np.arange(len(days))
    width = 0.2

    # Bar chart for access and login counts
    bars1 = ax2.bar(x - 1.5*width, normal_access, width, label='Normal Access', color='skyblue')
    bars2 = ax2.bar(x - 0.5*width, anomaly_access, width, label='Anomaly Access', color='red')
    bars3 = ax2.bar(x + 0.5*width, normal_login, width, label='Normal Login', color='lightgreen')
    bars4 = ax2.bar(x + 1.5*width, anomaly_login, width, label='Anomaly Login', color='orange')

    ax2.set_xlabel('Hari')
    ax2.set_ylabel('Jumlah Akses')
    ax2.set_title('Normal vs Anomaly Access and Login per Day')
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

    # Score and thresholding to detect anomalies
    score = model.score_samples(X_test)
    score_threshold = np.percentile(score, 2)
    customized_pred = [1 if i < score_threshold else 0 for i in score]

    # Create plots with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    # Subplot 1: Actual Data
    scatter_actual = axes[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', edgecolors='k')
    axes[0].set_title('Data Aktual')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    cbar_actual = plt.colorbar(scatter_actual, ax=axes[0])
    cbar_actual.set_label('Kelas')

    # Subplot 2: Model Predictions
    scatter_pred = axes[1].scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap='coolwarm', edgecolors='k')
    axes[1].set_title('Prediksi Model')
    axes[1].set_xlabel('Feature 1')
    cbar_pred = plt.colorbar(scatter_pred, ax=axes[1])
    cbar_pred.set_label('Prediksi')

    # Subplot 3: Access and Login Distribution
    axes[2].bar(x - 1.5*width, normal_access, width, label='Normal Access', color='skyblue')
    axes[2].bar(x - 0.5*width, anomaly_access, width, label='Anomaly Access', color='red')
    axes[2].bar(x + 0.5*width, normal_login, width, label='Normal Login', color='lightgreen')
    axes[2].bar(x + 1.5*width, anomaly_login, width, label='Anomaly Login', color='orange')
    
    axes[2].set_xlabel('Hari')
    axes[2].set_ylabel('Jumlah Akses')
    axes[2].set_title('Distribusi Akses per Hari')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(days)
    axes[2].legend()

    # Adjust layout and display the plot
    plt.tight_layout()
    st.pyplot(fig)

# Generate example access data
ip_addresses = [f"192.168.1.{random.randint(1, 255)}" for _ in range(7)]
start_date = datetime(2025, 1, 1)
timestamps = [start_date + timedelta(days=i) for i in range(7)]

# Create a DataFrame for access data with IP, Status, and Label
access_data = pd.DataFrame({
    'Date': timestamps,
    'IP Address': ip_addresses,
    'Normal Login': normal_login,
    'Anomaly Login': anomaly_login,
    'Login Status': ['Normal' if login == 0 else 'Anomaly' for login in anomaly_login]
})

# Display access data
st.subheader('Data Akses')
st.dataframe(access_data)
