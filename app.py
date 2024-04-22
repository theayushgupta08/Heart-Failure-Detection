from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

dataset = pd.read_csv("Heart Failure Dataset.csv")
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


app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        # Get data from the form
        age = request.form['age']
        anaemia = request.form['anaemia']
        creatinine_phosphokinase = request.form['creatinine_phosphokinase']
        diabetes = request.form['diabetes']
        ejection_fraction = request.form['ejection_fraction']
        high_blood_pressure = request.form['high_blood_pressure']
        platelets = request.form['platelets']
        serum_creatinine = request.form['serum_creatinine']
        serum_sodium = request.form['serum_sodium']
        sex = request.form['sex']
        smoking = request.form['smoking']
        time = request.form['time']
        cholesterol = request.form['cholesterol']
        body_mass_index = request.form['body_mass_index']
        heart_rate = request.form['heart_rate']
        exercise_angina = request.form['exercise_angina']
        st_slope = request.form['st_slope']
        num_vessels = request.form['num_vessels']
        thalassemia = request.form['thalassemia']

        # Create a dictionary with the collected data
        data = {'age': age, 
        'anaemia': anaemia,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': diabetes,
        'ejection_fraction': ejection_fraction,
        'high_blood_pressure': high_blood_pressure,
        'platelets': platelets,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'sex': sex,
        'smoking': smoking,
        'time': time,
        'cholesterol': cholesterol,
        'body_mass_index': body_mass_index,
        'heart_rate': heart_rate,
        'exercise_angina': exercise_angina,
        'st_slope': st_slope,
        'num_vessels': num_vessels,
        'thalassemia': thalassemia,
        }
        #change input data to a numpy array
        input_data_as_numpy_array = np.asarray(tuple(data.values()))

        #Reshapping the numpy array as we are prediction for 1 data not 2000

        input_data_reshapped = input_data_as_numpy_array.reshape(1,-1)

        prediction = model.predict(input_data_reshapped)  #0 represesnts the person is healthy and Vice Versa


        if (prediction[0]==0):
            output = "Healthy"
        else:
            output = "Un-Healthy"


        # Optionally, you can return a response to the user
        return output

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
    


