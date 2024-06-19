import streamlit as st
import numpy as np
import pickle 
import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("pages/follow_on_final.csv")
train = data
train = train.drop(["Organization Name","Equity Only Funding", "Announced Date"], axis = 1 )

le=LabelEncoder()
cols = train.select_dtypes(include=['object']).columns.tolist()

cat_labels = {}
for col in cols:
    train[col] = le.fit_transform(train[col]) 
    cat_labels[col] = list(le.classes_)

def str_to_label(col, str):
    return cat_labels[col].index(str) + 1   


st.write(""" # Follow on Funding Model """)

# Funding type
st.write(""" ## Funding Type """)
funding_type = st.selectbox(
    "What is the Funding type of your startup?",
    cat_labels["Funding Type"] )
st.write("You selected:", funding_type)
funding_type = str_to_label("Funding Type", funding_type)

# Money Raised
st.write(""" ## Money Raised """)
money_raised = float(st.text_input('Enter Amount of money raised during the funding type above', 0))

# Organization Industries
st.write(""" ## Organization Industries """)
Organization_Industries = st.selectbox(
    "What is the Organization Industries of your startup?",
    cat_labels["Organization Industries"] )
st.write("You selected:", Organization_Industries)

Organization_Industries = str_to_label("Organization Industries", Organization_Industries)


# Funding Stage
st.write(""" ## Funding Stage """)
Funding_Stage = st.selectbox(
    "What is the Funding Stage of your startup?",
    cat_labels["Funding Stage"] )
st.write("You selected:", Funding_Stage)
Funding_Stage = str_to_label("Funding Stage", Funding_Stage)


# Region
st.write(""" ## Region """)
Region = st.selectbox(
    "What is the Region of your startup?",
    cat_labels["Region"] )
st.write("You selected:", Region)
Region = str_to_label("Region", Region)


# Country
st.write(""" ## Country """)
Country = st.selectbox(
    "What is the Country of your startup?",
    cat_labels["Country"] )
st.write("You selected:", Country)
Country = str_to_label("Country", Country)



# City
st.write(""" ## City""")
City = st.selectbox(
    "What is the City of your startup?",
    cat_labels["City"] )
st.write("You selected:", City)
City = str_to_label("City", City)


# Company Type
st.write(""" ## Company Type """)
Company_Type = st.selectbox(
    "What is the Company type of your startup?",
    cat_labels["Company Type"] )
st.write("You selected:", Company_Type)
Company_Type = str_to_label("Company Type", Company_Type)


# Money Raised
st.write(""" ## Total Funding """)
total_money_raised = float(st.text_input('Enter Total Amount of money raised ', 0))

# Number of Founders
st.write(""" ## Total Funding """)
Founders = float(st.text_input('Enter Total Number of Founders ', 1))


# Number of Employees
st.write(""" ## Number of Employees """)
Employees = st.selectbox(
    "What is the Number of Employees of your startup?",
    cat_labels["Number of Employees"] )
st.write("You selected:", Employees)
Employees = str_to_label("Number of Employees", Employees)


st.markdown('##')
st.markdown('##')

# Button
predict_bt = st.button('Predict')

# ===========================================================================================================================

# list of all input variables

input_data = np.array([[   funding_type,
                                money_raised,
                                Organization_Industries,
                                Funding_Stage,
                                Region,
                                Country,
                                City,
                                total_money_raised,
                                Company_Type,
                                Founders,
                                Employees
                        ]])


# load the model
model =  pickle.load(open('pages/classifier_model.pkl', 'rb'))

def make_predictions(sample):
    out = {}
    pred = model.predict_proba(sample).max()
    if pred > 0.5:
        out["Prediction"] = "Funded"
        out["confidence"] = round(pred * 100)
    else:
         out["Prediction"] = "Not Funded"
         out["confidence"] = round(pred * 100)
        
    return out

#Animation function
@st.experimental_memo
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

if predict_bt:

    result = make_predictions(input_data)
    st.success(f'## The startup is likely to be {result["Prediction"]} ')
    st.success(f'## Confidence: {result["confidence"]}%')
    st.balloons()
