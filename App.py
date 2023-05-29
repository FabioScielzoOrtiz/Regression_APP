import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import uuid
import base64 

from sklearn.linear_model import LinearRegression



###################################################################

st.title("Regression APP")

st.markdown("<br>", unsafe_allow_html=True)

# Cargar archivo CSV
st.markdown('## Load data file')
file = st.file_uploader(" ", type=["csv","xlsx"])


st.markdown("<br>", unsafe_allow_html=True)

if file is not None:

    df_original = pd.read_csv(file)

    # Create state variables for the data-frame
    state = st.session_state
    if 'df' not in state:
        state.df = df_original.copy()
    else :
        pass

    st.sidebar.header('Table of content')

############################################################################    

    if st.sidebar.checkbox('Table with the datae'):
       
        st.markdown('### Table with the original data')

        df = state.df
        st.write(df)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

############################################################################
 

    if st.sidebar.checkbox('Select Predictors (Features)'):

        st.markdown('### Select Predictors/Features')
        Predictors_selected = st.multiselect('Select Predictor/Features', options=df_original.columns, key=1)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

############################################################################

    if st.sidebar.checkbox('Select Response (Target Variable)'):

        st.markdown('### Select Response')
        Response_selected = st.selectbox('Select Response', options=df_original.columns, key=2)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

#####################################################################################################################################

    if st.sidebar.checkbox('Select Model'):

       st.markdown('### Select Model')

       model_name = st.selectbox('Select Model', options=['Linear Regression', 'KNN'], key=3)
 
       st.markdown("<br>", unsafe_allow_html=True)
       st.markdown("<br>", unsafe_allow_html=True)


#####################################################################################################################################

    if st.sidebar.checkbox('Make Predictions with the Model'):

        st.markdown('### Enter new data')
        new_data = []
        for predictor in Predictors_selected:
            value = st.number_input(f"Enter value for {predictor}")
            new_data.append(value)
        # Convert new data to numpy array
        new_data = np.array(new_data).reshape(1, -1)

        if st.button('Make Predictions'):

            X_train = df_original.loc[:, Predictors_selected]
            Y_train = df_original.loc[:, Response_selected]
 
            if model_name == "Linear Regression" :
               model = LinearRegression()
               model.fit(X_train, Y_train)
               Y_pred = model.predict(new_data)
            else :
               st.write('This model is not available yet')
               
            st.write('Predicted output:', Y_pred[0])


    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


#####################################################################################################################################

  


#####################################################################################################################################

footer = f"Made by **Fabio Scielzo Ortiz**. [Personal Website](http://fabioscielzoortiz.com/)"
st.write(footer, unsafe_allow_html=True)



    