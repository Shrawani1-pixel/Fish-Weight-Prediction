# Fish-Weight-Prediction
This project predicts the weight of a fish based on its length using **Linear Regression**
#Objective
The objective of this project is to build a machine learning model that predicts the weight of fish based on their physical measurements such as length, height, and width.
This project helps in fisheries, aquaculture, and seafood industries to:
Estimate fish weight without manual weighing.
Automate quality control and inventory management.

Make data-driven decisions in fish farming and sales.

🛠️ Tools & Libraries Used
Python

pandas, numpy → Data handling

matplotlib, seaborn → Data visualization

scikit-learn → Machine Learning models

joblib → Model persistence

Streamlit → Interactive web app

Dataset
We used the Fish Market dataset containing species and physical dimensions of fish.
Features include:

Species (Fish type: Bream, Roach, Perch, etc.)

Length1, Length2, Length3 (Different length measurements in cm)

Height (in cm)

Width (in cm)

Weight (in grams → Target variable)

🔎 Step by Step Implementation
1. Import Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

2.Create Dataset (Length vs Weight)

# Fish Length (cm)
length = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]).reshape(-1, 1)

# Fish Weight (grams) - hypothetical data
weight = np.array([5, 50, 120, 200, 350, 500, 700, 1000, 1200, 1500])

 3 Split Data (Training & Testing)
X_train, X_test, y_train, y_test = train_test_split(length, weight, test_size=0.2, random_state=42)



4.Train Model
model = LinearRegression()
model.fit(X_train, y_train)

<img width="583" height="91" alt="image" src="https://github.com/user-attachments/assets/34c44b44-1597-433d-97f3-e095c3c06543" />


5. Test Predictions
pred = model.predict(X_test)

print("📊 Actual Weights:", y_test)
print("🤖 Predicted Weights:", pred)

6.Predict for New Lengths
fish_length = [[12], [28], [40]]  # test inputs
predicted_weight = model.predict(fish_length)

print(f"🐟 Predicted weight of 12 cm fish = {predicted_weight[0]:.2f} g")
print(f"🐟 Predicted weight of 28 cm fish = {predicted_weight[1]:.2f} g")
print(f"🐟 Predicted weight of 40 cm fish = {predicted_weight[2]:.2f} g")

<img width="655" height="135" alt="image" src="https://github.com/user-attachments/assets/a2172759-fe76-408b-be2e-c0655ea7c65e" />




7. Visualization

  plt.scatter(length, weight, color="blue", label="Actual Data")
plt.plot(length, model.predict(length), color="red", label="Regression Line")
plt.xlabel("Fish Length (cm)")
plt.ylabel("Fish Weight (g)")
plt.title("🐟 Fish Weight Prediction")
plt.legend()
plt.show()

 <img width="816" height="567" alt="image" src="https://github.com/user-attachments/assets/ef2e9360-dcde-46f6-9c67-c6f95a1cdb68" />



