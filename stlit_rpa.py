import streamlit as st
import pandas as pd
import numpy as np

import rpa_uti_mpl as ru
import json
import os
from io import BytesIO
from io import StringIO
import uuid
import yaml
import zipfile

# from https://github.com/jkanner/streamlit-dataview/blob/master/app.py
#
# Use the non-interactive Agg backend, which is recommended as a
# thread-safe backend.
# See https://matplotlib.org/3.3.2/faq/howto_faq.html#working-with-threads.
import matplotlib as mpl
mpl.use("agg")

##############################################################################
# Workaround for the limited multi-threading support in matplotlib.
# Per the docs, we will avoid using `matplotlib.pyplot` for figures:
# https://matplotlib.org/3.3.2/faq/howto_faq.html#how-to-use-matplotlib-in-a-web-application-server.
# Moreover, we will guard all operations on the figure instances by the
# class-level lock in the Agg backend.
##############################################################################
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock


st.set_page_config(
   page_title="RPA App",
   page_icon="🧊",
   #layout="wide",
   initial_sidebar_state="expanded",
)



def get_config():

    try:
        ddconfig=dict()
        drive=os.environ.get('HOMEDRIVE','')
        home = os.environ.get('HOMEPATH','.')
        if drive :
            home = os.path.join(drive,home)
        rpa_global = os.environ.get('RPAPATH',home)
        filename = 'config.yaml'
        #search locations
        search = [rpa_global,home,os.path.join(home,'.rpakey'), os.path.join('.','.rpakey'),'.']
        for loc in search:
            testconfigfile =  os.path.join(loc,filename)
            if os.path.exists(testconfigfile):
                configfile = testconfigfile
                with open(configfile,'r') as fy:
                    dconf = yaml.safe_load(fy)
                ddconfig.update(dconf)
    except:
        print('use Defaults')
        ddconfig =dict(lower_temp=75.,upper_temp=135.)

    apikey = os.environ.get('RPAKEY',ddconfig.get('apikey',''))
    ddconfig['apikey']=apikey
    return ddconfig

def data_fromfileupload(uploaded_file):
    if uploaded_file is not None:
        filename = uploaded_file.name
        # To read file as bytes:
        if filename.endswith('.zip'):
            bytes_data = uploaded_file.getvalue()
            #st.write(bytes_data)
            with zipfile.ZipFile(BytesIO(bytes_data),'r') as zf:
                print(zf.namelist())
                for file in zf.namelist():
                    print(file)
                    try:
                        base,ext=os.path.splitext(file)
                        if ext.lower()=='.html':
                            string = zf.read(file).decode("utf-8")
                            stringio =  StringIO(string)

                            dataframe = ru.html2df(string)
                    except:
                        dataframe = pd.DataFrame([])
                        print('Error extracting')
            

        # To convert to a string based IO:
        if filename.endswith('.htmlx'):
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            #st.write(stringio)
            
            dataframe = ru.html2df(stringio)
        

        # To read file as string:
        if filename.endswith('.html'):
            print('hi')
            
            string_data = uploaded_file.getvalue().decode("utf-8")
            #string_data = uploaded_file.read()
            #st.write(string_data)
            dataframe = ru.html2df(string_data)
        
        # Can be used wherever a "file-like" object is accepted:
        if filename.endswith('.csv'):
            dataframe = pd.read_csv(uploaded_file)
            #dataframe = upload_data(uploaded_file)
        return dataframe
    else:
        return None


@st.cache_data
def upload_data(uploaded_file):
    dataframe = pd.read_csv(uploaded_file)
    return dataframe


config = get_config()

st.title('RPA Viscosity Measurements')

upper_temp=float(config.get('upper_temp',140.))
lower_temp=float(config.get('lower_temp',80.))

apikey = config.get('apikey','')
if not apikey:
    st.write('No ApiKey found: No identification will be done')
    st.write(' For Access contact: ...')

else:
    uploaded_file=None
    #while uploaded_file is None:
    uploaded_file = st.sidebar.file_uploader("Choose a file [.html, .zip(html), .csv]",key='upload_widget')
    with uploaded_file:
        #dataframe = pd.DataFrame([])
        try:
            dataframe = data_fromfileupload(uploaded_file)
            df = ru.resample(dataframe,num=config.get('samples',120))
        except:
            st.write('Cannot read dataframes from file')
            st.stop()

        if st.checkbox('Show raw data'):
            st.subheader('Raw data')
            st.write(dataframe)

        st.sidebar.subheader('Set Temperature Limits for Model')
        #with st.expander(f'use only data from Temperatures in this range'):
        # Some number in the range 0-23
        #lower_temp = st.slider('lower', float(dataframe.tempc.min()),float(upper_temp) , lower_temp)
        #upper_t=upper_temp
        with st.sidebar.form(key='set_limits'):
            col1,col2 = st.columns((1,1))
            with col1:
                lower_t = st.number_input('Lower Limit',value=lower_temp,
                            min_value=dataframe.tempc.min(),
                            max_value=dataframe.tempc.max(),step=1.,format='%.1f')
            #col1.write('The current number is ', lower_t)  
            with col2:
                upper_t = st.number_input('Upper Limit',value=upper_temp,
                                max_value=dataframe.tempc.max(),
                                min_value=lower_t,step=1.,format='%.1f')                      
            st.form_submit_button('start analysis')
        #col2.write('The current number is ', upper_t)  
        lower_temp = lower_t
        upper_temp = upper_t

        # no function
        #tl,tu = st.slider('Temperature Limits for Fitting', value=[lower_t,upper_t],step=1. )
        #st.subheader('Data Measured and Filtered')

        # call to api 
        #model = ru.fit_visco(dataframe,lower_t,upper_t)
        with st.spinner('run fitting...'):
            model = ru.call_fit_visco(dataframe,lower_t,upper_t,apikey=apikey)
        
        if st.checkbox('Show Plot (slows down!)'):
            st.subheader('Plot Data')
            with _lock:                
                plotfilename,ext = os.path.splitext(uploaded_file.name)
                fig = ru.plot_mpl(df,lower_t,upper_t,model,title=f'Fitting RPA Data for Viscous Model',filename = plotfilename)
            #st.plotly_chart(fig,use_container_width=True)
                st.pyplot(fig)
                with open(f'{plotfilename}.png','rb') as fpict:
                    bpicstr = fpict.read()
                st.download_button('Download Picture',bpicstr,file_name=f'{plotfilename}.png')
        st.sidebar.subheader('model parameters')
        st.sidebar.write(model)
        with open('model.txt','w') as fo:
            modelstr=json.dumps(model)
            fo.write(modelstr)
        
        st.sidebar.download_button('Download Model Values', modelstr)


    
    
        