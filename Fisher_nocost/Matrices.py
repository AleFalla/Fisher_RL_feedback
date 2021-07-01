import numpy as np
import math
import scipy


def Optomech(params):
    wm=params['wm'] #frequenza oscillatore meccanico presa unitaria
    k=params['k']*wm #accoppiamento luce ambiente 
    y=params['y']*wm #accoppiamento mech ambiente
    eta=params['eta'] #efficienza misura su bagno luce
    g=params['g']*wm #accoppiamento ML
    detuning=params['detuning']*wm#1,-1,0
    ne=params['ne'] #numero di fononi meccanici
    na=params['na'] #numero fononi ottici
    phi=params['phi'] #LO phase
    
    cos=np.around(math.cos(phi),15)
    sin=np.around(math.sin(phi),15)

    A=np.array([[-y/2,-wm,0,0],[wm,-y/2,g,0],[0,0,-k/2,detuning],[g,0,-detuning,-k/2]])#il detuning di hammerer è definito al contrario
    D=np.array([[(1+2*ne)*y,0,0,0],[0,(1+2*ne)*y,0,0],[0,0,(1+2*na)*k,0],[0,0,0,(1+2*na)*k]])
    eta=(eta/(1+2*na*eta))**0.5
    b=-(((k*eta)**0.5))
    B=b*np.array([[0,0,0,0],[0,0,0,0],[0,0,cos**2,cos*sin],[0,0,sin*cos,sin**2]])
    E=(1+2*na)*B

    return A,D,B,E

def Cavity(params): #questo è meno dettagliato dell'optomeccanica
    
    k=params['k'] #loss rate
    eta=params['eta'] #eta della misura
    X=params['X_kunit']*k #accoppiamento hamiltoniana del sistema 
    a=-(X+k/2)
    b=X-k/2
    A=np.array([[a,0],[0,b]])
    D=k*np.identity(2)
    d=-(eta*k)**0.5
    B=d*np.array([[1,0],[0,0]])
    E=B

    return A,D,B,E

def Fisher(params): #questo è meno dettagliato dell'optomeccanica
    theta=params['theta']#parameter to estimate
    k=params['k'] #loss rate
    eta=params['eta'] #eta della misura
    X=params['X_kunit']*k #accoppiamento hamiltoniana del sistema 
    a=-(X+k/2)
    b=X-k/2
    A=np.array([[a,0],[0,b]])
    A=A+theta*np.array([[0,1],[-1,0]])
    deA=np.array([[0,1],[-1,0]])
    D=k*np.identity(2)
    d=-(eta*k)**0.5
    B=d*np.array([[1,0],[0,0]])
    E=B

    return A,deA,D,B,E
    
