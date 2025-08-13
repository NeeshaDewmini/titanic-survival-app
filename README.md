#  Titanic Survival Prediction App

An interactive Streamlit web app that allows users to explore the **Titanic dataset**, visualize key patterns, and predict passenger survival using a trained **machine learning model**.  
The project combines **data analysis**, **visualization**, and **ML prediction** in a user-friendly interface.

---

##  Features
- **Dataset Exploration** – View the Titanic passenger dataset in an interactive table.
- **Data Visualization** – Beautiful charts to analyze patterns such as survival by age, gender, class, and more.
- **Survival Prediction** – Enter passenger details and get predictions from a trained ML model.
- **Responsive Navigation** – Easily switch between sections via the navigation bar.

---

##  How to Run Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/NeeshaDewmini/titanic-survival-app.git
   cd titanic-streamlit-app
   ```

2. **Install Dependencies**
   Make sure you have **Python 3.9+** installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**
   ```bash
   streamlit run app.py
   ```

4. **Open in Browser**
   [Open the app in your browser](https://titanic-survival-appgit-cybb8nafnmne9shwfcbbax.streamlit.app/)

---

##  Data Description
This app uses the **Titanic dataset** from Kaggle.  
The dataset contains information about passengers, including:
- **PassengerId** – Unique ID for each passenger
- **Pclass** – Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Name** – Passenger’s name
- **Sex** – Gender
- **Age** – Age in years
- **SibSp** – Number of siblings/spouses aboard
- **Parch** – Number of parents/children aboard
- **Ticket** – Ticket number
- **Fare** – Ticket fare
- **Cabin** – Cabin number (if known)
- **Embarked** – Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- **Survived** – Survival status (0 = No, 1 = Yes)

---

##  Model Information

### **Preprocessing Steps**
1. Handle missing values:
   - Fill missing `Age` with median age.
   - Fill missing `Embarked` with the most common value.
   - Drop `Cabin` due to high missing percentage.
2. Encode categorical variables (`Sex`, `Embarked`) using **Label Encoding** or **One-Hot Encoding**.
3. Scale numerical features (`Age`, `Fare`) using **StandardScaler**.

### **Model Used**
- **Logistic Regression Classifier**
- Hyperparameter tuning via GridSearchCV
- Trained on processed Titanic dataset with a train-test split of 80/20
- Model accuracy: **85.47%** 

---

### Author
#### Janeesha Dewmini
#### Final Year IT Student | Data Science & ML Enthusiast