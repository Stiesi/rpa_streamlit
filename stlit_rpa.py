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
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(dataframe)

    st.subheader('Filter Temperature Limits')

    # Some number in the range 0-23
    #lower_temp = st.slider('lower', float(dataframe.tempc.min()),float(upper_temp) , lower_temp)
    lower_t,upper_t = st.slider('Temperature Limits for Fitting', value=[float(dataframe.tempc.min()),float(dataframe.tempc.max())] )
    #st.subheader('Data Measured and Filtered')

    # call to api 
    model = ru.fit_visco(dataframe,lower_t,upper_t)
    
    fig = ru.plotly_fig(dataframe,lower_t,upper_t,model=model)
    st.plotly_chart(fig,use_container_width=True)
    st.write(model)
    with open('model.txt','w') as fo:
        modelstr=json.dumps(model)
        fo.write(modelstr)
    
    st.download_button('Download Model Values', modelstr)

    
    
        