# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:48:52 2022

@author: nathl
"""
import streamlit as st

from PIL import Image

import pickle
import pandas as pd
from joblib import load
import requests

import plotly.graph_objects as go
import plotly.express as px
import shap



# Import data
application = pd.read_pickle('sample.pkl')
application = application.reset_index(drop=True)

# Import data_norm
appli_norm = pd.read_pickle('sample_norm.pkl')
appli_norm = appli_norm.reset_index(drop=True)

# Import dic most important values by index client
col_shap_most_importance_dic = pickle.load(open("col_shap_most_importance_dic.p", 'rb'))

# Import dic descriptions columns
columns_descrition_dic = pickle.load(open("columns_descrition_dic.p", 'rb'))

# Threshold calculate for this model
threshold = 0.18

# List of columns to apply model
list_columns = application.columns.to_list()
list_columns.remove('SK_ID_CURR')
list_columns.remove('TARGET')

# Data to apply model
data = appli_norm[list_columns]

# Import shap model
explainer = load('explainer.bz2')

shap_values = load('shap_values.joblib')

# Page parameter
st.set_page_config(page_title='Scoring credit Prêt à dépenser', layout='wide')

# SIDEBAR
## Import logo
st.sidebar.image(Image.open("logo-pad.jpg"))

st.sidebar.markdown('---')
st.sidebar.subheader('Select client ID')
## client choice 
id_client = st.sidebar.selectbox(label = 'Client ID', 
                                 options = application['SK_ID_CURR'])

index_client = application[application['SK_ID_CURR']==id_client].index.tolist()

client_list = application['SK_ID_CURR'].tolist()


# Interact with FastAPI endpoint
## Using Docker images and containers
backend = "http://backend:8000/predict" 

## Using link FastAPI deploy with Heroku
backend_web = "https://scoring-credit-pad.herokuapp.com/predict"

# function to retrieve fastapi result
def process(id_client):
    data_api = {"id" : id_client}
    r = requests.post(backend_web, json=data_api)
    return r

# retrive each result of fastapi
result = process(id_client)
proba0 = result.json()['probability'][0][0] # Not risked
proba1 = result.json()['probability'][0][1] # Risked
pred = result.json()['prediction'][0]

# global pred
y_pred = []
for i in client_list:
    y = process(i)
    y = y.json()['prediction'][0]
    y_pred.append(y)
    

# BODY
## Final Result
col1, col2 = st.columns([1,2])

### gauge plot
with col1:
    
    ### Client ident
    st.markdown(f'## Client ID n° {id_client}')
    
    ### Answer to the client
    st.subheader('Answer to the client')
    
    if pred==0:
        answer = "Credit granted"
    else:
        answer = "Credit not granted"
    
    st.write(answer)

with col2:
    st.subheader('Probability of the client to be classified in risk')
    
    indicator_plot = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = result.json()['probability'][0][1]*100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probability of client to be classified in risk"},
        gauge = {'axis': {'range': [0, 100]},
                'bar': {'color': "#800080"},
                'steps' : [
                    {'range': [0, threshold*100], 'color': "#008B8B"},
                    {'range': [threshold*100, 100], 'color': "#B22222"}],
                }))
    st.plotly_chart(indicator_plot, True)


## More details

### More information about the client profile
with st.expander("Details about the client profile"):
    # Clients data
    application[application['SK_ID_CURR'] == id_client]

### Details to explain the answer 

with st.expander("Details to explain the answer"):
    
    col1, col2 = st.columns(2)
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    with col1:
        # Visualize the first prediction's explaination for expected value :
            # Pred calculate for the client
        st.subheader('Force plot')
        
        force_plot = shap.force_plot(explainer.expected_value[pred],
                                     shap_values[pred][index_client[0],:], 
                                     data.iloc[index_client[0],:],
                                     matplotlib = True)
        shap.initjs()
        
        st.pyplot(force_plot)  
    
    with col2:    
    # multioutput_decision_plot the 20 variables explaination:
     # for the client to obtain these pred
        st.subheader('Multioutput decision plot')
        
        multioutput_decision_plot = shap.multioutput_decision_plot(
            explainer.expected_value,
            shap_values,
            row_index = index_client[0],
            feature_names=data.columns.tolist(),
            highlight = [pred])
        shap.initjs()
        
    
        st.pyplot(multioutput_decision_plot)
        
        
### Details most impactful variables
with st.expander("Details with the most impactful variables"):
    
    # List of the 20 variables that influence the result 
    #feature_names = data.columns
    
    #rf_resultX = pd.DataFrame(shap_values[pred], columns = feature_names)
    
    #vals = np.abs(rf_resultX.filter(
        #items=[index_client[0]],
        #axis=0).values).mean(0)
    
    #shap_importance = pd.DataFrame(
        #list(zip(feature_names, vals)),
        #columns=['col_name','feature_importance_vals'])
    
    #shap_importance.sort_values(by=['feature_importance_vals'],
                                   #ascending=False, inplace=True)
    #shap_most_importance = shap_importance.head(20)
    #col_shap_most_importance = shap_most_importance['col_name'].tolist()
    
    # Clients data
    application.loc[application['SK_ID_CURR'] == id_client, col_shap_most_importance_dic.get(index_client[0])]
    
# SIDEBAR
# Variables choice
st.sidebar.markdown('---')
st.sidebar.subheader('Choose some variables')
col_most_importance = st.sidebar.multiselect(
    label='Impactful variables', 
    options = col_shap_most_importance_dic.get(index_client[0]))

# Button validation
validation = st.sidebar.button('Variables choice validation')

with st.expander("Graphic representation of selected variables"):


    if validation:
        for v in range(0,len(col_most_importance)):
            data_choose = pd.DataFrame(
                data = {'id_client': client_list,
                        col_most_importance[v]: data[col_most_importance[v]],
                        'Risk classification' :y_pred })
            
            data_choose = pd.pivot_table(data = data_choose, 
                                         values='id_client',
                                         index = col_most_importance[v], 
                                         columns='Risk classification',
                                         aggfunc='count')
            
            client_value = data.at[index_client[0], col_most_importance[v]]
            
            bar_plot = px.bar(
                data_choose, 
                color = 'Risk classification',
                title = (u'Client values: <br>'
                         + col_most_importance[v] + ": "+ str(client_value) 
                         +"<br>Client risk classification: " + str(pred)),
                color_discrete_sequence=["#008B8B", "#B22222"])
            
            st.plotly_chart(bar_plot, True)
            
            column_explaining = col_most_importance[v]
            
            if column_explaining not in columns_descrition_dic.keys():
                for i in range (1,len(column_explaining)):
                    column_explaining = column_explaining[0:-(i+1)]
                    if column_explaining[0:-(i+1)] in columns_descrition_dic.keys():
                        st.write(column_explaining[0:-(i+1)]+": "+str(columns_descrition_dic[column_explaining[0:-(i+1)]][0]))
                        break
            else :
                st.write(col_most_importance[v]+": "+str(columns_descrition_dic[col_most_importance[v]][0]))   