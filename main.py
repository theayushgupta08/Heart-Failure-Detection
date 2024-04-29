import kivy
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.app import App
from kivy.properties import ObjectProperty
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv("HeartFailureDataset.csv")
#print(dataset.duplicated().any())
#print(dataset.head())
st = StandardScaler()
#dataset=dataset.dropna()
#dataset=dataset.drop_duplicates()
x=dataset.drop("DEATH_EVENT",axis=1)
y=dataset["DEATH_EVENT"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
x_train = st.fit_transform(x_train)
x_test = st.transform(x_test)

model= RandomForestClassifier()
model.fit(x_train,y_train)
pred_model = model.predict(x_test)

class HeartFailurePredictionGrid(Widget):
    age=ObjectProperty(None)
    anaemia=ObjectProperty(None)
    creatinine_phosphokinase=ObjectProperty(None)
    diabetes=ObjectProperty(None)
    ejection_fraction=ObjectProperty(None)
    high_blood_pressure=ObjectProperty(None)
    platelets=ObjectProperty(None)
    serum_creatinine=ObjectProperty(None)
    serum_sodium=ObjectProperty(None)
    sex=ObjectProperty(None)
    smoking=ObjectProperty(None)
    time=ObjectProperty(None)
    cholesterol=ObjectProperty(None)
    body_mass_index=ObjectProperty(None)
    heart_rate=ObjectProperty(None)
    exercise_angina=ObjectProperty(None)
    st_slope=ObjectProperty(None)
    num_vessels=ObjectProperty(None)
    thalassemia=ObjectProperty(None)

    def press(self):
        data = {'age': self.age.text, 
        'anaemia': self.anaemia.text,
        'creatinine_phosphokinase': self.creatinine_phosphokinase.text,
        'diabetes': self.diabetes.text,
        'ejection_fraction': self.ejection_fraction.text,
        'high_blood_pressure': self.high_blood_pressure.text,
        'platelets': self.platelets.text,
        'serum_creatinine': self.serum_creatinine.text,
        'serum_sodium': self.serum_sodium.text,
        'sex': self.sex.text,
        'smoking': self.smoking.text,
        'time':self.time.text,
        'cholesterol': self.cholesterol.text,
        'body_mass_index': self.body_mass_index.text,
        'heart_rate': self.heart_rate.text,
        'exercise_angina': self.exercise_angina.text,
        'st_slope': self.st_slope.text,
        'num_vessels': self.num_vessels.text,
        'thalassemia': self.thalassemia.text
        }

        # Check for missing data
        if any(value == '' for value in data.values()):
            self.ids.result.text = "Incomplete Data!"
            return
        
        #change input data to a numpy array
        input_data_as_numpy_array = np.asarray(tuple(data.values()))

        #Reshapping the numpy array as we are prediction for 1 data not 2000

        input_data_reshapped = input_data_as_numpy_array.reshape(1,-1)

        prediction = model.predict(input_data_reshapped)  #0 represesnts the person is healthy and Vice Versa

        if (prediction[0]==0):
            self.ids.result.text = "No Heart Failure Detected!"
        else:
            self.ids.result.text = "Heart Failure Detected!"

    def reset_values(self):
        self.ids.age.text = ""
        self.ids.anaemia.text = ""
        self.ids.creatinine_phosphokinase.text = ""
        self.ids.diabetes.text = ""
        self.ids.ejection_fraction.text = ""
        self.ids.high_blood_pressure.text = ""
        self.ids.platelets.text = ""
        self.ids.serum_creatinine.text = ""
        self.ids.serum_sodium.text = ""
        self.ids.sex.text = ""
        self.ids.smoking.text = ""
        self.ids.time.text = ""
        self.ids.cholesterol.text = ""
        self.ids.body_mass_index.text = ""
        self.ids.heart_rate.text = ""
        self.ids.exercise_angina.text = ""
        self.ids.st_slope.text = ""
        self.ids.num_vessels.text = ""
        self.ids.thalassemia.text = ""
        self.ids.result.text= "Your result will be displayed here!"
        

class HeartFailurePredictionApp(App):
    def build(self):
        return HeartFailurePredictionGrid()

if __name__=="__main__":
    HeartFailurePredictionApp().run()