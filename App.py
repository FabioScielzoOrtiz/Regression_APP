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
            
            ################################################################################

            ## Linear Regression
            LinearRegression_Model = LinearRegression()
 
            ## KNN
            st.markdown('#### KNN Parameters')
            n_neighbors = st.number_input('Enter the number of neighbors', min_value=1, step=1, value=10, key=4)
            metric = st.selectbox('Select the distance metric', options=['euclidean', 'manhattan', 'minkowski'], key=5)
            p = st.number_input('Enter the p value for Minkowski distance', min_value=1, step=1, value=2, key=6)

            KNN_Model = KNeighborsRegressor(n_neighbors=n_neighbors, p=p, metric=metric)

            ################################################################################

            Models = [LinearRegression_Model, KNN_Model]
           
            cross_validation = st.selectbox('Choose a cross validation algorithm', options=['Simple-NotRandom', 'Simple-Random'], key=7)

            if cross_validation == 'Simple-Random' :

                RMSE_Random_Simple_Validation = []

                for model in Models :
                   
                    Random_Simple_Validation = RandomSimpleValidation(k=0.75, metric='MSE', model=model, random_seed=123)
                    Random_Simple_Validation.fit(D=df_original, response_name=Response_selected)
                    Random_Simple_Validation.predict()
                    RMSE_Random_Simple_Validation.append( np.sqrt( Random_Simple_Validation.compute_metric() ) )

                RMSE_models_df = pd.DataFrame({'Model' : Models, 'RMSE' : RMSE_Random_Simple_Validation})
                RMSE_models_df_sort = RMSE_models_df.sort_values(by='RMSE', ascending=True)

                st.write('RMSE models:', RMSE_models_df_sort)

                st.write('The best model according to this validation method is', RMSE_models_df_sort.iloc[0,0][0])


 
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

            Final_Model = RMSE_models_df_sort.iloc[0,0]
            st.write('The model that will be used to predict is:', Final_Model)
            Final_Model.fit(X_train, Y_train)
            Y_pred = Final_Model.predict(new_data)
 
            st.write('Predicted response:', Y_pred[0])


    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


#####################################################################################################################################

  


#####################################################################################################################################

footer = f"Made by **Fabio Scielzo Ortiz**. [Personal Website](http://fabioscielzoortiz.com/)"
st.write(footer, unsafe_allow_html=True)



    