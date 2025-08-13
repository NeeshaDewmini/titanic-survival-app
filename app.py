import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# -------------------- Load Model and Dataset --------------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

df = pd.read_csv("data/processed_titanic.csv")

# -------------------- Page Config --------------------
st.set_page_config(page_title="Titanic Survival Prediction App", layout="wide")

# -------------------- Tabs Navigation --------------------
tabs = st.tabs([
    "Home", 
    "Data Exploration", 
    "Visualizations", 
    "Model Prediction", 
    "Model Performance"
])

# -------------------- Home Tab --------------------
with tabs[0]:
    st.title("Titanic Survival Prediction App")
    st.markdown("""
    ## 🚢 Welcome to the Titanic Survival Prediction App!  

    Step aboard and explore the story of the Titanic through **data**.  
    This interactive app lets you:  
    - 📊 **Explore** the Titanic dataset to uncover fascinating patterns.  
    - 🎨 **Visualize** important features with beautiful and insightful charts.  
    - 🤖 **Predict** survival chances using a trained machine learning model.  

    Navigate through the tabs above to begin your journey — from exploring raw data  
    to making predictions about who might have survived the Titanic disaster.  

    💡 *Tip: The more you explore, the more you'll discover!*  
    """)
    st.image("data/dataset-card.jpg", caption="The Titanic", use_container_width=True)

# -------------------- Data Exploration Tab --------------------
with tabs[1]:
    st.header(" Data Exploration")

    # Dataset overview
    st.subheader("Dataset Overview")
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())
    st.dataframe(df.head())

    # Summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    # Interactive filtering
    st.subheader("Filter Data")
    filter_column = st.selectbox("Select a column to filter", df.columns)
    if df[filter_column].dtype == "object" or df[filter_column].nunique() < 10:
        selected_values = st.multiselect("Select value(s)", df[filter_column].unique())
        if selected_values:
            filtered_df = df[df[filter_column].isin(selected_values)]
        else:
            filtered_df = df.copy()
    else:
        min_val = float(df[filter_column].min())
        max_val = float(df[filter_column].max())
        selected_range = st.slider("Select range", min_val, max_val, (min_val, max_val))
        filtered_df = df[(df[filter_column] >= selected_range[0]) & (df[filter_column] <= selected_range[1])]
    
    st.dataframe(filtered_df)

# -------------------- Visualizations Tab --------------------
with tabs[2]:
    st.header(" Visualizations")

    # Scatter Plot
    st.subheader("Scatter Plot")
    x_axis = st.selectbox("Select X-axis", df.columns, index=list(df.columns).index("Age"))
    y_axis = st.selectbox("Select Y-axis", df.columns, index=list(df.columns).index("Fare"))
    with st.spinner("Rendering scatter plot..."):
        fig = px.scatter(df, x=x_axis, y=y_axis, color='Survived', title=f"Scatter Plot: {x_axis} vs {y_axis}")
        st.plotly_chart(fig)

    # Histogram
    st.subheader("Histogram")
    hist_column = st.selectbox("Select column for histogram", df.columns, index=list(df.columns).index("Age"))
    bins = st.slider("Number of bins", 5, 50, 20)
    with st.spinner("Rendering histogram..."):
        fig2 = px.histogram(df, x=hist_column, color='Survived', nbins=bins, title=f"Histogram of {hist_column}")
        st.plotly_chart(fig2)

    # Bar chart for categorical features
    st.subheader("Survival Count by Category")
    cat_column = st.selectbox("Select categorical column", ["Pclass", "Sex_male", "Embarked_Q", "Embarked_S", "Title_Mr", "Title_Mrs", "Title_Miss", "Title_Rare"])
    with st.spinner("Rendering bar chart..."):
        fig3 = px.bar(df, x=cat_column, color='Survived', barmode='group', title=f"Survival Count by {cat_column}")
        st.plotly_chart(fig3)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    with st.spinner("Rendering correlation heatmap..."):
        corr = df.corr()
        fig4, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig4)

# -------------------- Model Prediction Tab --------------------
with tabs[3]:
    st.header(" Predict Passenger Survival")
    st.write("Enter passenger details below to predict survival.")

    # User input
    Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1,2,3])
    Age = st.number_input("Age", 0, 100, 30)
    SibSp = st.number_input("Number of Siblings/Spouses aboard", 0, 10, 0)
    Parch = st.number_input("Number of Parents/Children aboard", 0, 10, 0)
    Fare = st.number_input("Ticket Fare", 0.0, 600.0, 32.0)
    Sex = st.selectbox("Sex", ["Female","Male"])
    Embarked = st.selectbox("Port of Embarkation", ["C","Q","S"])
    Title = st.selectbox("Title", ["Master","Miss","Mr","Mrs","Rare"])

    # Encode categorical variables
    Sex_male = 1 if Sex=="Male" else 0
    Embarked_Q = 1 if Embarked=="Q" else 0
    Embarked_S = 1 if Embarked=="S" else 0
    Title_Miss = 1 if Title=="Miss" else 0
    Title_Mr = 1 if Title=="Mr" else 0
    Title_Mrs = 1 if Title=="Mrs" else 0
    Title_Rare = 1 if Title=="Rare" else 0
    FamilySize = SibSp + Parch + 1
    IsAlone = 1 if FamilySize==1 else 0

    # Input array
    input_data = np.array([Pclass, Age, SibSp, Parch, Fare, FamilySize, IsAlone,
                           Sex_male, Embarked_Q, Embarked_S,
                           Title_Miss, Title_Mr, Title_Mrs, Title_Rare]).reshape(1,-1)

    # Apply scaling if required
    if type(model).__name__ in ["LogisticRegression", "SVC"]:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        input_data = scaler.transform(input_data)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0][prediction]
            result = "Survived" if prediction==1 else "Did Not Survive"
            st.success(f"Prediction: {result}")
            st.info(f"Prediction Probability: {prediction_proba:.2f}")

# -------------------- Model Performance Tab --------------------
with tabs[4]:
    st.header(" Model Evaluation")
    st.write("This section shows your model's performance metrics and comparisons.")

    # Separate features and target
    X = df.drop(columns=['Survived'])
    y_true = df['Survived']

    # Apply scaling if required
    if type(model).__name__ in ["LogisticRegression", "SVC"]:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(X)
        X_use = X_scaled
    else:
        X_use = X

    # Predict using the deployed model
    y_pred = model.predict(X_use)

    # Metrics for the deployed model
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    st.subheader("Deployed Model Metrics")
    st.write(f"**Accuracy:** {acc:.2f}")
    st.write(f"**Precision:** {prec:.2f}")
    st.write(f"**Recall:** {rec:.2f}")
    st.write(f"**F1-score:** {f1:.2f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Did Not Survive","Survived"])
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax)
    st.pyplot(fig)

    # Model Comparison
    st.subheader("Model Comparison")
    try:
        with open("results.pkl", "rb") as f:
            comparison_df = pickle.load(f)
    except FileNotFoundError:
        st.error("Model comparison results not found. Please retrain the models and save 'results.pkl'.")
        comparison_df = pd.DataFrame()

    if not comparison_df.empty:
        st.dataframe(comparison_df.sort_values(by="Test Accuracy", ascending=False))
        st.bar_chart(comparison_df.set_index("Model")["Test Accuracy"])


