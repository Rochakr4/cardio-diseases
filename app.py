import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np
from PIL import Image
model = pickle.load(open('model.sav', 'rb'))
scalar = pickle.load(open('scalar.sav', 'rb'))
st.title('Heart diseases Prediction')
st.sidebar.header('Data')

# FUNCTION
def user_report():
  age = st.sidebar.slider('age', 33,100)
  education = st.sidebar.slider('education', 1,4)
  sex = st.sidebar.slider('sex', 0,1,1)
  cigsPerDay = st.sidebar.slider('cigsPerDay', 0,100)
  totChol = st.sidebar.slider('totChol',120,350)
  heartRate = st.sidebar.slider('heartRate	', 48,150)
  glucose = st.sidebar.slider('glucose',51,300)
  PP = st.sidebar.slider('PP',15,200)


  user_report_data = {
      'age':age,
      'education':education,
      'sex':sex,
      'cigsPerDay':cigsPerDay,
      'totChol':totChol,
      'heartRate':heartRate,
      'glucose':glucose,
      'PP':PP
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

user_data = user_report()
st.header('Data')
st.write(user_data)
user_data1=scalar.transform(user_data)
salary = model.predict(user_data1)
if salary==0:
     st.subheader('No chance for heart diseases')
else:
    st.subheader('High chance for heart diseases')
   
