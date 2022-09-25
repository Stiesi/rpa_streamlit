import json
import os
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
#from scipy.optimize import leastsq
import requests

def plotly_fig(data,lowert=80,uppert=140,model=None):

    data['log_nstar']=np.log(data.nstar)
    data['use']=np.where((data.tempc>=lowert)&(data.tempc<=uppert),0.4,0.05)
    
    #fig = px.scatter(data,x='tempc',y='log_nstar',color='gammap')#,size='hrate')
    #fig.update_traces(marker={'size':100})
    myscale = [[0.0, 'yellow'],
                [0.1, 'yellow'],
                [0.1, "green"],
                [0.2, "green"],
                [0.2, "magenta"],
                [0.4, "magenta"],
                [0.4, "blue"],
                [0.7, "blue"],                
                [0.7, "red"],
                [1.0, "red"]]
    traces=[go.Scatter(name='RPA Data',
            x=data.tempc,y=data.log_nstar,mode='markers',
                text=data.gammap,
                marker=dict(color=data.gammap,
                            colorscale=myscale,
                            opacity=data.use))]
    try:        
        A = model['A']
        C = model['C']
        n = model['n']
        data['x']=np.log(data.gammap)
        data['y']=1./(data.tempc+273.15)

        data['za']= data.apply(lambda x: viscosity_log(x.x,x.y,A,C,n),axis=1)
        trace = go.Scatter(name='Model',
                x=data.tempc,y=data.za,mode='markers',
                    text=data.gammap,
                    marker=dict(color=data.gammap,
                                colorbar=dict(title="Shear Rate [rad/s]",
                                len=0.5),
                                colorscale=myscale,
                                opacity=data.use,
                                symbol='diamond-dot',
                                line=dict(
                                    color='Black',
                                    width=0.9,
                                    )))
        traces.append(trace)
    except:
        print(model)
        #print('No Api Key? : Contact...')

    layout=go.Layout(
    #fig.update_layout(
    title="RPA Viscosity Data",
    xaxis_title="Temperature in degC",
    yaxis_title="log(n*)",
    legend_title="Source",)
    fig =  go.Figure(data=traces,layout=layout)
    #fig = 
    return fig


def test_plotly_fig(show=False):
    filename=os.path.join('testdata','Rheology_M-870-6_Batch_31.csv')
    df = pd.read_csv(filename)
    fig = plotly_fig(df,model ={'A': 835.9979707299541, 'C': 1666.8287301629769, 'n': 0.2841424416704478, 'pp': 1, 'lower T[C]': 80, 'upper T[C]': 140})
    #print(type(fig))
    assert isinstance(fig,go.Figure)
    if show:
        fig.show()

        
def synchronize(sdash,nstar,temp,testid=1,trigger=0):
    # create a data array with [time_s,S',n*,TKinv,gammarate]
    # 
    # sdas, nstar , and temp are list of tuples (time, value)
    # gammarate is a scalar
    #
    sda = np.array(sdash)
    nst = np.array(nstar)
    tem = np.array(temp)
    
    if trigger==0:
        time = sda[:,0]
    else:        
        time = np.linspace(sda[0,0],sda[-1,0],num=trigger)
    
    #heatrate = (tem[-1,1]-tem[0,1])/(tem[-1,0]-tem[0,0])
    # or
    heatrate=(np.diff(tem[:,1])/np.diff(tem[:,0])).mean()

    #sda = sda[:,1]
    ndata=len(time)
    sda = np.interp(time, sda[:,0], sda[:,1])
    
    
    
    if len(nst)!= ndata:
        nda = np.interp(time, nst[:,0], nst[:,1])
    else:
        nda =nst[:,1]
    # inverse temperature K
    tempc = np.interp(time, tem[:,0],tem[:,1])
    #tki = 1./(tempc +273.15)
    
    time*=60 #in sec
    trate = np.ones_like(time)*heatrate/60  # per sec
    #trate = np.ones_like(time)*heatrate  # per min
    #gammap = np.ones_like(time)*gammarate*2*np.pi  # per sec
    testnr = np.ones_like(time)*testid
    dataset =np.vstack((time,sda,nda,tempc,trate,testnr))
    return dataset.reshape(6,-1).T

    
def call_fit_visco(df,lowert=80,uppert=140,apikey=''):

    
    #base_url = f"https://cljbb2ponzjy24nlmvc4ogobiq0xjphl.lambda-url.eu-central-1.on.aws/visco"#?lowert={lowert}&uppert={uppert}"
    #base_url = f"https://localhost:8000/visco"#?lowert={lowert}&uppert={uppert}"
    base_url = f"https://pmwanmnatc.execute-api.eu-central-1.amazonaws.com/dev/visco"
    
    print(base_url)
    params = dict()
    params["lowert"] = lowert
    params["uppert"] = uppert
    body = dict()
    body["gammap"] = df.gammap.values.tolist()
    body["tempc"] = df.tempc.values.tolist()
    body["nstar"] = df.nstar.values.tolist()
    headers=dict()
    headers['x-api-key']=apikey
    
    try:
        r = requests.post(base_url, params=params, json=body,headers=headers) 
        #rr = r.content.decode()    # is binary
        rr = json.loads(r.content.decode())
    except:
        print( 'Error in request response with body:')
        dj = json.dumps(body)
        print (dj)
        rr={}

    return rr



def viscosity_log(loggammap,tki,A,C,n):
    return np.log(A) + C*tki + (n-1)*loggammap


def test_fit_visco():
    filename=os.path.join('testdata','Rheology_M-870-6_Batch_31.csv')
    df = pd.read_csv(filename)
    pset = call_fit_visco(df)
    
    ref={'A': 835.9979707299541, 'C': 1666.8287301629769, 'n': 0.2841424416704478, 'pp': 1, 'lower T[C]': 80, 'upper T[C]': 140}
    assert pset==ref, 'test_fit_visco not matching'
    return df,ref

def resample(df,num=20):
    dfc = pd.DataFrame()

    ix = df.loc[df['test_no'].isin([1,5,9,13])].index
    ndat = len(ix)
    ninc = int(ndat/num/4)
    #df2 = df[ix].iloc[::ninc]
    if ninc >0:
        ixi = ix[::ninc]
    return df.loc[ixi]

if __name__=='__main__':
    #test_plotly_fig(show=True)
    df,ref=test_fit_visco()
    dfsamp=resample(df,num=20)
    print(dfsamp.to_json())
    rd = call_fit_visco(dfsamp)
    
    print(rd)