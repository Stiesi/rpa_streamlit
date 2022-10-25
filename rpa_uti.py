import json
import os
import re
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
#from scipy.optimize import leastsq
import requests

def find_tables(txt):#### to do?
    txt=txt.replace('\r','')
    txt=txt.replace('\n','')
    rexh=r'<h2>(.*?)</h2>'
    rext=r'<table.*?/table>'
    headers = re.findall(rexh,txt)#,re.DOTALL)#[:1]
    tables  = re.findall(rext,txt)#[:1]
    rexc=r'Compound: (.*?) Order No: (\d{8}-\d{4}?)(.*?)</td>'
    comps=re.findall(rexc,txt)
    headerss=[sub.replace("'",'dash').replace('*','star') for sub in headers]
    
    return headerss,tables,comps

def get_tdata(t):
    t=t.replace(',','')
    rexd=r'<td>(.*?)</td>'
    tdata = re.findall(rexd,t)
    return tdata

def create_sub(h,t,c):
    filename=h+c[-1].replace('(','').replace(')','').replace(':','_') ##header + (datetime)
    tdata = get_tdata(t)
    meta = dict(name=c[0],order=c[1],date=c[2])
    tdata[::2] = [td+',' for td in tdata[::2]]
    tdata[1::2] = [td+'\n' for td in tdata[1::2]]
    #tdata[1::2] = tdata[::2]+'\n'
    data = [tdata[0],h+'\n']
    data.extend(tdata[2:])
    return filename,data,meta

def save_subs(headers,tables,comps):
    files=[]
    dd={}
    for h,t,c in zip(headers,tables,comps):
        filename,data,meta = create_sub(h,t,c)
        
        dd[filename]=get_tdata(t) # no comma, no \n
        
        txt = f'# {c[0]} order {c[1]}, date {c[2]}\n'
        txt += ''.join(data)
        #if mysystem=='web':
        #    text_file = anvil.BlobMedia('text/plain', txt.encode('utf-8'), name=filename+'.csv')
        #    anvil.media.download(text_file)
        #else:
        with open(filename+'.csv', 'w+') as ff:
            ff.write('%s'%txt)
        
            
        print(h,c)
        #with open(filename,'w+') as f:
        #  f.write('#%s\n'%(h))
        #zf.write(filename)
        #
        #print ('written %s'%filename)
        #shutil.make_archive('mya','zip','/tmp')
        #zf.close()
            
        files.append(filename)
    return files,dd

def html2df(txt):
    '''

    Parameters
    ----------
    txt : str
        content of htmlfile

    Returns
    -------
    headers : list 
        List of headers per dataset (measure)        
    tables : list
        list of str with html table 
    comps : list
        tuples of metadata for measurements 
    '''

    headers,tables,comps=find_tables(txt)  
    df = stack_rpa_data(headers,tables,comps)
    return df
    

def stack_rpa_data(headers,tables,comps,
                   gammarates=[1.25,2.5,5,10],
                   heatrates=[5,10,20,30],
                   trigger=1000):
    #fnames,dd = save_subs(headers,tables,comps)
    # sequence of rates 1/s
    # sorting of sequences is unimportant (see below), program will find
    # gammarates according to nstar at different rates @100°C
    # heatrates from temperature tempc (number values are arbitrary (but must be different))
    #gammarates = np.array(gammarates*12).reshape(12,4).T.ravel()
    ic = 0
    datafield = np.ndarray(shape=(0,6))
    testid=0
    for h,t,c in zip(headers,tables,comps):
        data = get_tdata(t)
        filename,datax,meta = create_sub(h,t,c)
        #print(filename,len(datax))
        #txt = ','.join(data[2:])
        mya = np.array(data[2:],dtype=np.float32).reshape((-1,2))
        if ic==0:
            testid+=1
            
            if 'Sdash' not in  filename:
                print (filename,c)
            sda = mya   
            
            #print(c[-1])
        if ic ==1:
            if 'Temp' not in  filename:
                print (filename,c)
            #if c[-1]!= ordertime:
            #    print (h,c)
            temp = mya
        if ic==2:
            #if c[-1]!= ordertime:
            #    print (h,c)
            if 'nstar' not in  filename:
                print (filename,c)

            nstar = mya
            
            
            newset = synchronize(sda, nstar, temp,testid,trigger=trigger)
            datafield = np.vstack((datafield,newset))
            ic=-1
            
        ic+=1
    
    df = pd.DataFrame(datafield,columns=['time',
                                          'sdash',
                                          'nstar',
                                          'tempc',
                                          #'tki',
                                          #'gammap',
                                          'hrate',
                                          'test_no'],
                      )
    #df['gammap']=pd.cut(df.gammap,4,labels=gammarates_sequence)
    #for test_no in df.test_no.unique():
    #    dfs=df[df.test_no==test_no]
        #dfs[]
    # group test by nstar@100°C
    xx=df[(df.tempc>99)&(df.tempc<101)].groupby(df.test_no).nstar.mean()
    df=df.merge(xx,on='test_no')
    # create increasing rates
    gammarates.sort(reverse=True)
    gammaratesa=np.array(gammarates)*2*np.pi
    # associate gammarates with values of nstar at 100°C
    df['gp100']=pd.qcut(df.nstar_y,4,labels=gammaratesa)
    # habe nur 3 categorien !?
    #df['gammap']=pd.qcut(df.nstar_y,4,labels=gammarates)
    df.rename(columns={'nstar_x':'nstar'},inplace=True)
    # set heatrates to given categories (unique values)
    df['hrate']=pd.qcut(df.hrate,4,labels=heatrates)
    df['hrate']=df['hrate'].astype('float')
    df.nstar*=1e5
    df['gammap']=df['gp100'].values.astype('float')
    df['test_no']=df['test_no'].astype('int')
    df.drop(['gp100','nstar_y'],axis=1,inplace=True)
    df = df.round({'nstar':1,'gammap':2,'tempc':2,'sdash':3,'time':2})
    return df


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