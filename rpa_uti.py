import os
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import leastsq

def plotly_fig(data,lowert=80,uppert=140,model=None):

    data['log_nstar']=np.log(data.nstar)
    data['use']=np.where((data.tempc>=lowert)&(data.tempc<=uppert),0.3,0.02)
    
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
    if  model is not None:
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
                                    width=0.5,
                                    )))
        traces.append(trace)

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

def fit_visco(df,lowert=80,uppert=140):
    dfs=df.loc[(df.tempc<=uppert)&(df.tempc>=lowert)]

          
    p0=[300,1936,0.297]
    x=np.log(dfs.gammap.values)
    y=1./(dfs.tempc.values+273.15)
    z=np.log(dfs.nstar.values)
    #z=dfs.nstar.values
    
    full_output=0    
    result = leastsq(f_visco,p0,args=(x,y,z),full_output=full_output)
    if full_output!=0:
        print(result)
    A,C,nexp=result[0]
    pp=result[-1]
        
    #print(A,C,n,pp)
    return {'A':A,'C':C,'n':nexp,'pp':pp,'lower T[C]':lowert,'upper T[C]':uppert}

def viscosity(loggammap,tki,A,C,n):
    return A + C*tki + (n-1)*loggammap


def viscosity_log(loggammap,tki,A,C,n):
    return np.log(A) + C*tki + (n-1)*loggammap

def f_visco(p,loggammap,tki,log_nstar):
    A=p[0]
    C=p[1]
    n=p[2]
    lognstar_model = viscosity_log(loggammap,tki,A,C,n)
    diffs = lognstar_model-log_nstar
    return diffs.flatten()

def test_fit_visco():
    filename=os.path.join('testdata','Rheology_M-870-6_Batch_31.csv')
    df = pd.read_csv(filename)
    pset = fit_visco(df)
    
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
    test_plotly_fig(show=True)
    df,ref=test_fit_visco()
    dfsamp=resample(df,num=20)
    print(dfsamp.to_json())