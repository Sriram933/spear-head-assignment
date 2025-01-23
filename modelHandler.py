from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from fastapi import HTTPException

class ModelHandler:

    def __init__(self):
        self.model: RandomForestClassifier = RandomForestClassifier()
        self.scaler = StandardScaler() 
        self.acuracy=0
    
    def get_model(self):
        return self.model
    
    def set_csv(self,csvFile):
        self.csvFileDf=pd.read_csv(csvFile)
    
        return {"message":"CSV file set"}
    
    def predict(self, json: dict):
        # Ensure all required columns are present
        expected_columns = ['Temperature', 'Run_Time']
        for col in expected_columns:
            if col not in json:
                json[col] = 0  # Fill missing columns with default value

        # Convert JSON to DataFrame
        data_matrix = self.json_to_df(json)

        # Scale the data (use pre-fitted scaler)
        data_scaled = self.scaler.transform(data_matrix[['Temperature', 'Run_Time']])

        # Make prediction
        prediction = self.model.predict(data_scaled)
        ans = "No" if prediction.tolist()[0] == 0 else "Yes"
        return {"Downtime":ans,"confidence":f"{float(self.acuracy)*100}%"}
    
    def train(self):
        df = self.csvFileDf
        df = df.drop(columns=['Machine_ID', 'Defect_Type','Machine_Age'])
        X = df[['Temperature', 'Run_Time']]
        y = df['Downtime_Flag']
        scaler = self.scaler
        X_scaled = scaler.fit_transform(X)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=101)
        self.model.fit(X_train, y_train)

        y_pred=self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)


        formatted_accuracy = f"{accuracy:.3f}"
        formatted_f1 = f"{f1:.3f}"
        self.acuracy=formatted_accuracy
        return {
            "accuracy": formatted_accuracy,
            "f1": formatted_f1
        }
    
    def json_to_df(self,json):
        return pd.DataFrame([json])

