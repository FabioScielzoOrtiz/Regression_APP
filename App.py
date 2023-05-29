import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import uuid
import base64 

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

######################################################################################################################################

class RandomSimpleValidation : 
    # D --> have to be a pandas data frame.
    # k --> is the proportion of observation of D that define D_train.
    # response --> have to be a string with the name of the response variable.
    # model --> object containing the initialized model to use.
    # The function has been created thinking that the model to be used will be one from the `sklearn` library.
    # metric --> It's the name of the validation metric.
    # random_seed --> seed to replicate the random process.
        
    def __init__(self, k, metric, model, random_seed):
        self.k = k
        self.metric = metric
        self.model = model
        self.random_seed = random_seed   

    def fit(self, D, response_name):         
        N = len(D)
        self.D_train = D.sample(frac=self.k, replace=False, random_state=self.random_seed)
        self.D_test = D.drop( self.D_train.index , )
        self.X_train = self.D_train.loc[: , self.D_train.columns != response_name]
        self.Y_train = self.D_train.loc[: , response_name]
        self.X_test = self.D_test.loc[: , self.D_test.columns != response_name]
        self.Y_test = self.D_test.loc[: , response_name]
        self.model.fit(self.X_train, self.Y_train)
    
    def predict(self):
        self.Y_predict_test = self.model.predict(self.X_test)
    
    def compute_metric(self):    
        if self.metric == 'MSE':
            self.ECM_test = np.mean((self.Y_predict_test - self.Y_test) ** 2)
            return self.ECM_test
        elif self.metric == 'Accuracy':
            self.TAC_test = np.mean((self.Y_predict_test == self.Y_test))
            return self.TAC_test

######################################################################################################################################       


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

    if st.sidebar.checkbox('Table with the data'):
       
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

    if st.sidebar.checkbox('Model Selection'):
       
       

       st.markdown('### Model Selection by cross validation')

       hyperparameter_tuning = st.selectbox('Do you want hyperparameter tuning ?', options=['Yes', 'Not'], key=3)

       if hyperparameter_tuning == 'Not' :
            
            LinearRegression_Model = LinearRegression()
            KNN_Model = KNeighborsRegressor(n_neighbors=10,  p=2, metric='minkowski')

            Models = [LinearRegression_Model, KNN_Model]
           
            cross_validation = st.selectbox('Choose a cross validation algorithm', options=['Simple-NotRandom', 'Simple-Random'], key=4)

            if cross_validation == 'Simple-Random' :

                MSE_Random_Simple_Validation = []

                for model in Models :
                   
                    Random_Simple_Validation = RandomSimpleValidation(k=0.75, metric='MSE', model=model, random_seed=123)
                    Random_Simple_Validation.fit(D=df_original, response_name=Response_selected)
                    Random_Simple_Validation.predict()
                    MSE_Random_Simple_Validation.append( Random_Simple_Validation.compute_metric() )

                MSE_models_df = pd.DataFrame({'Model' : Models, 'MSE' : MSE_Random_Simple_Validation})

                st.write('MSE models:', MSE_models_df)

 



 
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



    