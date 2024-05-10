from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the data
data = pd.read_csv("C:\\Users\\arshi\\OneDrive\\Desktop\\stress_detection_through_sleeping_habits_using_ML\\SaYoPillow.csv")

# Preprocess the data
data.rename(columns={'sr': 'snoring rate', 'rr':'respiration rate', 't': 'body temperature', 'lm':'limb movement',
                            'bo':'blood oxygen', 'rem':'eye movement', 'sr.1':'sleeping hours', 'hr':'heart rate',
                            'sl':'stress level'}, inplace=True)

# Split the data into features and target
X = data.drop('stress level', axis=1)
y = data['stress level']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define the models
models = [DecisionTreeClassifier(), SVC(), RandomForestClassifier()]

# Train the models
for model in models:
    model.fit(X_train, y_train)

# Create data visualizations
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=14, titlepad=10)

# Heatmap of correlation matrix
corr_matrix = data.corr()
plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, cmap="Blues")
plt.title('Absolute Pearson Correlation between Different Features')
plt.savefig('static/corr_heatmap.png')

# Pairplot of features
sns.pairplot(data, hue='stress level')
plt.savefig('static/pairplot.png')

# Countplot of stress levels
sns.countplot(data=data, x='stress level')
plt.title('Distribution of Stress Levels')
plt.savefig('static/stress_levels.png')

# Lineplots of features vs stress level
plt.figure(figsize=(20, 5))
sns.lineplot(x='snoring rate', y='stress level', data=data)
plt.xlabel("Snoring Rate")
plt.ylabel('Stress Level')
plt.title('Snoring Rate vs Stress Level')
plt.xticks(rotation=0)
plt.savefig('static/snoring_rate.png')

plt.figure(figsize=(20, 5))
sns.lineplot(x='respiration rate', y='stress level', data=data)
plt.xlabel("Respiration Rate")
plt.ylabel('Stress Level')
plt.title('Respiration Rate vs Stress Level')
plt.xticks(rotation=0)
plt.savefig('static/respiration_rate.png')

plt.figure(figsize=(20, 5))
sns.lineplot(x='body temperature', y='stress level', data=data)
plt.xlabel("Body Temperature")
plt.ylabel('Stress Level')
plt.title('Body Temperature vs Stress Level')
plt.xticks(rotation=0)
plt.savefig('static/body_temperature.png')

plt.figure(figsize=(20, 5))
sns.scatterplot(x='blood oxygen', y='stress level', data=data)
plt.xlabel("Blood Oxygen")
plt.ylabel('Stress Level')
plt.title('Blood Oxygen vs Stress Level')
plt.xticks(rotation=0)
plt.savefig('static/blood_oxygen.png')

plt.figure(figsize=(20, 5))
sns.scatterplot(x='eye movement', y='stress level', data=data)
plt.xlabel("Eye Movement")
plt.ylabel('Stress Level')
plt.title('Eye Movement vs Stress Level')
plt.xticks(rotation=0)
plt.savefig('static/eye_movement.png')

plt.figure(figsize=(20, 5))
sns.scatterplot(x='sleeping hours', y='stress level', data=data)
plt.xlabel("Sleeping Hours")
plt.ylabel('Stress Level')
plt.title('Sleeping Hours vs Stress Level')
plt.xticks(rotation=0)
plt.savefig('static/sleeping_hours.png')

plt.figure(figsize=(20, 5))
sns.scatterplot(x='heart rate', y='stress level', data=data)
plt.xlabel("Heart Rate")
plt.ylabel('Stress Level')
plt.title('Heart Rate vs Stress Level')
plt.xticks(rotation=0)
plt.savefig('static/heart_rate.png')

# Prediction and confusion matrix for Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
confusion_matrix_RF = confusion_matrix(y_test, y_pred_rf)
cm_dataframe_RF = pd.DataFrame(confusion_matrix_RF, index=['0', '1', '2', '3', '4'], columns=['0', '1', '2', '3', '4'])

# Plot confusion matrix with Heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm_dataframe_RF, annot=True, annot_kws={"size": 18}, fmt="d")
plt.title("Random Forest Confusion Matrix")
plt.ylabel('Actual Classes')
plt.xlabel('Predicted Classes')
plt.savefig('static/confusion_matrix_RF.png')

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/result", methods=["POST"])
def result():
    # Get the form data
    snoring_rate = float(request.form["snoring_rate"])
    respiration_rate = float(request.form["respiration_rate"])
    body_temperature = float(request.form["body_temperature"])
    limb_movement = float(request.form["limb_movement"])
    blood_oxygen = float(request.form["blood_oxygen"])
    eye_movement = float(request.form["eye_movement"])
    sleeping_hours = float(request.form["sleeping_hours"])
    heart_rate = float(request.form["heart_rate"])

    # Make prediction
    input_data = [[snoring_rate, respiration_rate, body_temperature, limb_movement, blood_oxygen, eye_movement, sleeping_hours, heart_rate]]
    predicted_stress_level = rf.predict(input_data)[0]

    # Render the result template
    return render_template("result.html", stress_level=predicted_stress_level)

if __name__ == "__main__":
    app.run(debug=True)
