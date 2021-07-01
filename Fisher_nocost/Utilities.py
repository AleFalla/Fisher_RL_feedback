import numpy as np
import math
import Matrices as Mat

def deltasc(A,J,E,B,D,sc):
    L=E-sc@B
    return (A+J)@sc+sc@(A.T+J.T)+D-L@(L.T)

def deltadesc(A,deA,J,E,B,D,sc,desc):
    L=E-sc@B
    return (deA@sc+sc@(deA.T)+(A+J)@(desc)+desc@(A.T+J.T)+desc@B@(L.T)+L@((desc@B).T))

def deltar(A,J,E,B,sc,dw,r,dt):
    L=E-sc@B
    return (A+J)@r+(2**(-0.5))*L@(dw/dt)

def deltader(A,deA,J,E,B,sc,desc,dw,dedw,r,der,dt):
    L=E-sc@B
    return deA@r+(A+J)@der-(2**(-0.5))*desc@B@(dw/dt)+(2**(-0.5))*L@(dedw/dt)

def RK4(var,f,args,dt):
    k1=f(**args)
    x=args['{}'.format(var)]
    args['{}'.format(var)]=x+k1*(dt/2)
    k2=f(**args)
    args['{}'.format(var)]=x+k2*(dt/2)
    k3=f(**args)
    args['{}'.format(var)]=x+k3*(dt)
    k4=f(**args)
    return (dt/6)*(k1+2*(k2+k3)+k4)


def fishsystem_nocost_RK4(r,der,sc,desc,A,deA,D,B,E,dt,dw,dedw,J): #in place of J you should use freq*identity(n)
    dr=RK4(var='r',f=deltar,args={'A':A,'J':J,'E':E,'B':B,'sc':sc,'dw':dw,'r':r,'dt':dt},dt=dt)
    dder=RK4(var='der',f=deltader,args={'A':A,'deA':deA,'J':J,'E':E,'B':B,'sc':sc,'desc':desc,'dw':dw,'dedw':dedw,'r':r,'der':der,'dt':dt},dt=dt)
    dsc=RK4(var='sc',f=deltasc,args={'A':A,'J':J,'E':E,'B':B,'D':D,'sc':sc},dt=dt)
    ddesc=RK4(var='desc',f=deltadesc,args={'A':A,'deA':deA,'J':J,'E':E,'B':B,'D':D,'sc':sc,'desc':desc},dt=dt)
    der=der+dder
    desc=desc+ddesc
    r=r+dr
    sc=sc+dsc
    return r,sc,der,desc


def exc_step(exc,Dyn,L,dt):
    dexc=dt*(Dyn@exc+exc@(Dyn.T)+2*L@(L.T))
    exc=exc+dexc
    return exc

def purity_like_rew(r,sc,exc,pow=0.5):
    su=sc+exc
    d1=np.linalg.det(su)
    d2=np.linalg.det(sc)
    h=d1-d2+1
    h=1/(h**pow)
    return h

def Matrices_Calculator(mode,params):
    if mode=='Optomech':
        A,D,B,E=Mat.Optomech(params)
        return A,D,B,E
    elif mode=='Cavity':
        A,D,B,E=Mat.Cavity(params)
        return A,D,B,E
    elif mode=='Fisher':
        A,deA,D,B,E=Mat.Fisher(params)
        return A,deA,D,B,E
    else:
        print('select mode "Optomech" or "Cavity"')
        
    

def check_param(param,range,pos):
    if param==None:
        if pos==False:
            param=np.random.uniform(-range,range)
        if pos==True:
            param=np.random.uniform(0,range)
        return param
    else:
        return param

    
