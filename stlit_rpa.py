import streamlit as st
import pandas as pd
import numpy as np
import rpa_uti as ru
import json
from io import BytesIO
from io import StringIO
import uuid

st.title('RPA Viscosity Measurements')


def upload_data(uploaded_file):
    dataframe = pd.read_csv(uploaded_file)
    return dataframe
    
    
upper_temp=130.
lower_temp=80.

uploaded_file = st.file_uploader("Choose a file [.html, .zip(html), .csv]")
if uploaded_file is not None:
    filename = uploaded_file.name
    # To read file as bytes:
    if filename.endswith('.zip'):
        bytes_data = uploaded_file.getvalue()
        st.write(bytes_data)

    # To convert to a string based IO:
    if filename.endswith('.html'):
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        st.write(stringio)

    # To read file as string:
    if filename.endswith('.html'):
        print('hi')
        
        string_data = stringio.read()
        st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    if filename.endswith('.csv'):
        dataframe = pd.read_csv(uploaded_file)
        #dataframe = upload_data(uploaded_file)
    df = ru.resample(dataframe,num=200)
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(dataframe)

    st.subheader('Set Temperature Limits for Model')

    # Some number in the range 0-23
    #lower_temp = st.slider('lower', float(dataframe.tempc.min()),float(upper_temp) , lower_temp)
    #upper_t=upper_temp
    col1,col2 = st.columns(2)
    lower_t = col1.number_input('Lower Limit',value=lower_temp,
                    min_value=dataframe.tempc.min(),
                    max_value=dataframe.tempc.max(),step=1.,format='%.1f')
    #col1.write('The current number is ', lower_t)  
    upper_t = col2.number_input('Upper Limit',value=upper_temp,
                        max_value=dataframe.tempc.max(),
                        min_value=lower_t,step=1.,format='%.1f')                      
    #col2.write('The current number is ', upper_t)  

    # no function
    #tl,tu = st.slider('Temperature Limits for Fitting', value=[lower_t,upper_t],step=1. )
    #st.subheader('Data Measured and Filtered')

    # call to api 
    model = ru.fit_visco(dataframe,lower_t,upper_t)
    
    if st.checkbox('Show Plot (slows down!)'):
        st.subheader('Plot Data')
        fig = ru.plotly_fig(df,lower_t,upper_t,model=model)
        st.plotly_chart(fig,use_container_width=True)
    st.write(model)
    with open('model.txt','w') as fo:
        modelstr=json.dumps(model)
        fo.write(modelstr)
    
    st.download_button('Download Model Values', modelstr)

    
    
        