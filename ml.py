import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE



np.random.seed(42)
num_samples = 3000
data = {
    'Machine_ID': np.arange(1, num_samples + 1),
    'Temperature': np.random.normal(loc=75, scale=10, size=num_samples),  # Mean temperature 75 with some variance
    'Run_Time': np.random.normal(loc=120, scale=30, size=num_samples),    # Mean run time 120 with some variance
    'Downtime_Flag': np.random.choice([0, 1], size=num_samples, p=[0.83, 0.17]),  # 80% No downtime, 20% downtime
    'Machine_Age': np.random.normal(loc=5, scale=2, size=num_samples) # Mean machine age 5 years with some variance 
}


defect_types = ['No Defect', 'Overheating', 'Mechanical Failure', 'Electrical Issue', 'Wear and Tear']


data['Defect_Type'] = np.random.choice(defect_types, size=num_samples)

df = pd.DataFrame(data)


df.to_csv('synthetic_manufacturing_data_with_defects.csv', index=False)

def train(csvFile):


    df = pd.read_csv(r"C:\Users\SRIRAM ADDANKI\synthetic_manufacturing_data_with_defects.csv")




    df = df.drop(columns=['Machine_ID', 'Defect_Type'])


    df.head()


    X = df[['Temperature', 'Run_Time', 'Machine_Age']]
    y = df['Downtime_Flag']


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)


    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=101)


    model = RandomForestClassifier(n_estimators=50, random_state=101)
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)


