import numpy as np
import Utilities as Tools
import pandas as pd

def bootstrap_interval(data,iters):
    means=np.zeros(iters)
    sample=np.random.choice(data,size=(iters,len(data)))
    means=np.mean(sample,axis=1)
    conf_int=np.percentile(means,[2.5,97.5])
    return 0.5*(conf_int[1]-conf_int[0])

#creates 10x2x2 matrices given the 2x2 matrices
def matrix_parallel_Fisher(N,a,dea,d,b,e):
    
    A,deA,D,B,E=[],[],[],[],[]
    for i in range(0,N):
        A.append(a)
        deA.append(dea)
        D.append(d)
        B.append(b)
        E.append(e)
    A=np.array(A)
    deA=np.array(deA)
    D=np.array(D)
    B=np.array(B)
    E=np.array(E)
    return A,deA,D,B,E

#definition of element-wise matrix-matrix product 10x2x2(ijk) * 10x2x2(ikl) = 10x2x2(ijl)
def prod_m(a,b):
    return np.einsum('ijk,ikl->ijl',a,b)

#definition of element-wise matrix-vector product 10x2x2(ijk) * 10x2(ik) = 10x2x2(ij)
def prod_v(a,b):
    return np.einsum('ijk,ik->ij',a,b)

#redefine RK4, transposition is somewhat of a mess code-wise but follows the logic A.T=np.transpose(A,axes=(0,2,1)), that is to say ijk->ikj 
def deltasc(A,J,E,B,D,sc):
    L=E-prod_m(sc,B)
    return prod_m((A+J),sc)+prod_m(sc,np.transpose(A+J,axes=(0,2,1)))+D-prod_m(L,(np.transpose(L,axes=(0,2,1))))

def deltadesc(A,deA,J,E,B,D,sc,desc):
    L=E-prod_m(sc,B)
    return prod_m(deA,sc)+prod_m(sc,(np.transpose(deA,axes=(0,2,1))))+prod_m((A+J),(desc))+prod_m(desc,(np.transpose(A+J,axes=(0,2,1))))+prod_m(desc,prod_m(B,(np.transpose(L,axes=(0,2,1)))))+prod_m(L,np.transpose(prod_m(desc,B),axes=(0,2,1)))

def deltar(A,J,E,B,sc,dw,r,dt):
    L=E-prod_m(sc,B)
    return prod_v((A+J),r)+(2**(-0.5))*prod_v(L,(dw/dt))

def deltader(A,deA,J,E,B,sc,desc,dw,dedw,r,der,dt):
    L=E-prod_m(sc,B)
    return prod_v(deA,r)+prod_v((A+J),der)-(2**(-0.5))*prod_v(desc,prod_v(B,(dw/dt)))+(2**(-0.5))*prod_v(L,(dedw/dt))

#this stays the same as before...
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


#...as does this one
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

#goes from vectors and matrices to observations 
def obs_prep(R,deR,CURR,SC,deSC,N):
    SC=np.reshape(SC,(N,4))
    deSC=np.reshape(deSC,(N,4))
    return np.concatenate((R,deR,CURR,SC,deSC),axis=1)

#defines initial conditions from the single traj format to the multiple trajectory one
def initial_conditions(N,r_0,der_0,sc_0,desc_0):
    R_0,deR_0,SC_0,deSC_0=[],[],[],[]
    for i in range(0,N):
        R_0.append(r_0)
        deR_0.append(der_0)
        SC_0.append(sc_0)
        deSC_0.append(desc_0)
    return R_0,deR_0,SC_0,deSC_0

#defines an appropriate vector of symplectic forms
def symplectic(N):
    sy=np.array([[0,1],[-1,0]])
    SY=[]
    for i in range(0,N):
        SY.append(sy)
    return np.array(SY)

#Classical Fisher increment
def dClassical_Fisher(der,B,dt):
    tmp=prod_v(np.transpose(B,axes=(0,2,1)),der)
    tmp=prod_v(B,tmp)
    tmp=np.einsum('ij,ij->i',der,tmp)
    return 2*tmp*dt

#fast element wise inverse of matrices 10x2x2(ijk) over indeces jk
def fast_inverse(A):
    identity = np.identity(A.shape[2], dtype=A.dtype)
    Ainv = np.zeros_like(A)
    for i in range(A.shape[0]):
        Ainv[i] = np.linalg.solve(A[i], identity)
    return Ainv

#Quantum Fisher
def Quantum_Fisher(N,sc,desc,der):
    invsc=fast_inverse(sc)
    pur=(np.linalg.det(sc))**(-0.5)
    I=0.5*np.matrix.trace(prod_m(prod_m(invsc,desc),prod_m(invsc,desc)),axis1=1,axis2=2)/(np.ones(N)+pur**2)+2*np.einsum('ij,ij->i',der,prod_v(invsc,der))+(np.trace(prod_m(invsc,desc),axis1=1,axis2=2)**2)/(2*np.linalg.det(sc)*(np.ones(N)-pur**4+1e-3*np.ones(N)))
    return I

#function for the 'dumb' action
def benchmark(N,a,b):
    z=np.ones(N)*a
    z=z+np.random.randn(N)*b
    return z

#The main plot data generator
def main_mean(N,timesteps,agent,params,dt,det=False,markevery=1,r_0=np.zeros(2),der_0=np.zeros(2),sc_0=7*np.identity(2),desc_0=0*np.identity(2),bench=-0.1,bench_noise=0.1):
    
    iters=N
    a,dea,d,b,e=Tools.Matrices_Calculator('Fisher',params) #calculation of 2x2 matrices
    A,deA,D,B,E=matrix_parallel_Fisher(N,a,dea,d,b,e) #calculation of 10x2x2 matrices
    
    SY=symplectic(N) #10x2x2 symplectic form
    R,deR,SC,deSC=initial_conditions(N,r_0,der_0,sc_0,desc_0) #fix initial conditions
    
    R2,deR2,SC2,deSC2=R,deR,SC,deSC #fix initial conditions for dumb action
    R3,deR3,SC3,deSC3=R,deR,SC,deSC #fix initial conditions for dumb action
    R4,deR4,SC4,deSC4=R,deR,SC,deSC #fix initial conditions for dumb action
    
    dW=np.random.randn(N,2)*(dt**0.5) #extraction of wiener increments (the same for both network and dumb feedback)
    
    dedW=(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(deR))*dt #dedw
    CURR=-(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(R))*dt+dW  #current calculation
    
    dedW2=(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(deR2))*dt #dedw for dumb action
    CURR2=-(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(R2))*dt+dW #current for dumb action
    
    dedW3=(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(deR3))*dt #dedw for dumb action
    CURR3=-(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(R3))*dt+dW #current for dumb action
    
    dedW4=(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(deR3))*dt #dedw for dumb action
    CURR4=-(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(R3))*dt+dW #current for dumb action
    

    Fisher_Cl=0 #initialize fisher for agent
    Fisher_Cl2=0 #initialize fisher for dumb action
    Fisher_Cl3=0
    Fisher_Cl4=0
    
    
    INFOS=[] #empty list for dicts of infos on agent
    INFOS2=[] #empty dict for dicts of infos on dumb action
    INFOS3=[]
    INFOS4=[]
    
    #create a multiindex for trajectories
    #array=[list(range(0,m)),list(0,timesteps),['r_0','r_1','der_0','der_1','sc_00','sc_01','sc_10','sc_11','desc_00','desc_01','desc_10','desc_11','Fisher_Cl','Fisher_Quantum','dW']]
    #index=pd.MultiIndex.from_product(arrays, names=('traj', 'steps','quantities'))
    #df = pd.DataFrame(index=index)
        

    obs=obs_prep(R,deR,CURR,SC,deSC,N) #prepare observation to match observation shape of the model
    
    m=markevery

    #evolution and information collection for all of trajectories at the same time
    for i in range(0,timesteps):
        
        

        action,_tmp=agent.predict(obs,deterministic=det)
        action=np.array(action[:,0])
        
        J=action[:,np.newaxis,np.newaxis]*SY
        R,SC,deR,deSC=fishsystem_nocost_RK4(R,deR,SC,deSC,A,deA,D,B,E,dt,dW,dedW,J)
        dFisher_Cl=dClassical_Fisher(deR,B,dt)
        Fisher_Cl=Fisher_Cl+dFisher_Cl
        Fisher_Quantum=Quantum_Fisher(N,SC,deSC,deR)

        action2=benchmark(N,bench,bench_noise)
        J2=action2[:,np.newaxis,np.newaxis]*SY
        R2,SC2,deR2,deSC2=fishsystem_nocost_RK4(R2,deR2,SC2,deSC2,A,deA,D,B,E,dt,dW,dedW2,J2)
        dFisher_Cl2=dClassical_Fisher(deR2,B,dt)
        Fisher_Cl2=Fisher_Cl2+dFisher_Cl2
        Fisher_Quantum2=Quantum_Fisher(N,SC2,deSC2,deR2)

        action3=benchmark(N,-0.1,0)
        J3=action3[:,np.newaxis,np.newaxis]*SY
        R3,SC3,deR3,deSC3=fishsystem_nocost_RK4(R3,deR3,SC3,deSC3,A,deA,D,B,E,dt,dW,dedW3,J3)
        dFisher_Cl3=dClassical_Fisher(deR3,B,dt)
        Fisher_Cl3=Fisher_Cl3+dFisher_Cl3
        Fisher_Quantum3=Quantum_Fisher(N,SC3,deSC3,deR3)
        
        action4=benchmark(N,0,0)
        J4=action4[:,np.newaxis,np.newaxis]*SY
        R4,SC4,deR4,deSC4=fishsystem_nocost_RK4(R4,deR4,SC4,deSC4,A,deA,D,B,E,dt,dW,dedW4,J4)
        dFisher_Cl4=dClassical_Fisher(deR4,B,dt)
        Fisher_Cl4=Fisher_Cl4+dFisher_Cl4
        Fisher_Quantum4=Quantum_Fisher(N,SC4,deSC4,deR4)

        if i%m==0:
            infos={} #empty dict for infos on agent
            infos2={} #empty dict for infos on dumb action
            infos3={}
            infos4={}

            #data collection
            R_mean=np.mean(R,axis=0)
            deR_mean=np.mean(deR,axis=0)
            SC_mean=np.mean(SC,axis=0)
            deSC_mean=np.mean(deSC,axis=0)
            Fisher_Cl_mean=np.mean(Fisher_Cl,axis=0)
            Fisher_Quantum_mean=np.mean(Fisher_Quantum,axis=0)
            action_mean=np.mean(action,axis=0)
            
            
            

            R2_mean=np.mean(R2,axis=0)
            deR2_mean=np.mean(deR2,axis=0)
            SC2_mean=np.mean(SC2,axis=0)
            deSC2_mean=np.mean(deSC2,axis=0)
            Fisher_Cl2_mean=np.mean(Fisher_Cl2,axis=0)
            Fisher_Quantum2_mean=np.mean(Fisher_Quantum2,axis=0)
            action2_mean=np.mean(action2,axis=0)
            
            R3_mean=np.mean(R3,axis=0)
            deR3_mean=np.mean(deR3,axis=0)
            SC3_mean=np.mean(SC3,axis=0)
            deSC3_mean=np.mean(deSC3,axis=0)
            Fisher_Cl3_mean=np.mean(Fisher_Cl3,axis=0)
            Fisher_Quantum3_mean=np.mean(Fisher_Quantum3,axis=0)
            action3_mean=np.mean(action3,axis=0)
            

            R4_mean=np.mean(R4,axis=0)
            deR4_mean=np.mean(deR4,axis=0)
            SC4_mean=np.mean(SC4,axis=0)
            deSC4_mean=np.mean(deSC4,axis=0)
            Fisher_Cl4_mean=np.mean(Fisher_Cl4,axis=0)
            Fisher_Quantum4_mean=np.mean(Fisher_Quantum4,axis=0)
            action4_mean=np.mean(action4,axis=0)
            
            
            R_abs=np.mean(np.abs(R),axis=0)
            R2_abs=np.mean(np.abs(R2),axis=0)
            R3_abs=np.mean(np.abs(R3),axis=0)
            R4_abs=np.mean(np.abs(R4),axis=0)
            
            infos['t']=i*dt
            infos['r_0 abs']=R_abs[0]
            infos['r_1 abs']=R_abs[1]
            infos['r_0 mean']=R_mean[0]
            infos['r_1 mean']=R_mean[1]
            infos['der_0 mean']=deR_mean[0]
            infos['der_1 mean']=deR_mean[1]
            infos['sc_00 mean']=SC_mean[0,0]
            infos['sc_01 mean']=SC_mean[0,1]
            infos['sc_10 mean']=SC_mean[1,0]
            infos['sc_11 mean']=SC_mean[1,1]
            infos['desc_00 mean']=deSC_mean[0,0]
            infos['desc_01 mean']=deSC_mean[0,1]
            infos['desc_10 mean']=deSC_mean[1,0]
            infos['desc_11 mean']=deSC_mean[1,1]
            infos['Fisher_Cl mean']=Fisher_Cl_mean
            infos['Fisher_Quantum mean']=Fisher_Quantum_mean
            infos['action mean']=action_mean

            infos['r_0 std']=bootstrap_interval(R[:,0],iters)
            infos['r_1 std']=bootstrap_interval(R[:,1],iters)
            infos['der_0 std']=bootstrap_interval(deR[:,0],iters)
            infos['der_1 std']=bootstrap_interval(deR[:,1],iters)
            infos['sc_00 std']=bootstrap_interval(SC[:,0,0],iters)
            infos['sc_01 std']=bootstrap_interval(SC[:,0,1],iters)
            infos['sc_10 std']=bootstrap_interval(SC[:,1,0],iters)
            infos['sc_11 std']=bootstrap_interval(SC[:,1,1],iters)
            infos['desc_00 std']=bootstrap_interval(deSC[:,0,0],iters)
            infos['desc_01 std']=bootstrap_interval(deSC[:,0,1],iters)
            infos['desc_10 std']=bootstrap_interval(deSC[:,1,0],iters)
            infos['desc_11 std']=bootstrap_interval(deSC[:,1,1],iters)
            infos['Fisher_Cl std']=bootstrap_interval(Fisher_Cl,iters)
            infos['Fisher_Quantum std']=bootstrap_interval(Fisher_Quantum,iters)
            infos['action std']=bootstrap_interval(action,iters)


            INFOS.append(infos)

            infos2['t']=i*dt
            infos2['r_0 abs']=R2_abs[0]
            infos2['r_1 abs']=R2_abs[1]
            infos2['r_0 mean']=R2_mean[0]
            infos2['r_1 mean']=R2_mean[1]
            infos2['der_0 mean']=deR2_mean[0]
            infos2['der_1 mean']=deR2_mean[1]
            infos2['sc_00 mean']=SC2_mean[0,0]
            infos2['sc_01 mean']=SC2_mean[0,1]
            infos2['sc_10 mean']=SC2_mean[1,0]
            infos2['sc_11 mean']=SC2_mean[1,1]
            infos2['desc_00 mean']=deSC2_mean[0,0]
            infos2['desc_01 mean']=deSC2_mean[0,1]
            infos2['desc_10 mean']=deSC2_mean[1,0]
            infos2['desc_11 mean']=deSC2_mean[1,1]
            infos2['Fisher_Cl mean']=Fisher_Cl2_mean
            infos2['Fisher_Quantum mean']=Fisher_Quantum2_mean
            infos2['action mean']=action2_mean

            infos2['r_0 std']=bootstrap_interval(R2[:,0],iters)
            infos2['r_1 std']=bootstrap_interval(R2[:,1],iters)
            infos2['der_0 std']=bootstrap_interval(deR2[:,0],iters)
            infos2['der_1 std']=bootstrap_interval(deR2[:,1],iters)
            infos2['sc_00 std']=bootstrap_interval(SC2[:,0,0],iters)
            infos2['sc_01 std']=bootstrap_interval(SC2[:,0,1],iters)
            infos2['sc_10 std']=bootstrap_interval(SC2[:,1,0],iters)
            infos2['sc_11 std']=bootstrap_interval(SC2[:,1,1],iters)
            infos2['desc_00 std']=bootstrap_interval(deSC2[:,0,0],iters)
            infos2['desc_01 std']=bootstrap_interval(deSC2[:,0,1],iters)
            infos2['desc_10 std']=bootstrap_interval(deSC2[:,1,0],iters)
            infos2['desc_11 std']=bootstrap_interval(deSC2[:,1,1],iters)
            infos2['Fisher_Cl std']=bootstrap_interval(Fisher_Cl2,iters)
            infos2['Fisher_Quantum std']=bootstrap_interval(Fisher_Quantum2,iters)
            infos2['action std']=bootstrap_interval(action2,iters)
            
            INFOS2.append(infos2)
            
            infos3['t']=i*dt
            infos3['r_0 abs']=R3_abs[0]
            infos3['r_1 abs']=R3_abs[1]
            infos3['r_0 mean']=R3_mean[0]
            infos3['r_1 mean']=R3_mean[1]
            infos3['der_0 mean']=deR3_mean[0]
            infos3['der_1 mean']=deR3_mean[1]
            infos3['sc_00 mean']=SC3_mean[0,0]
            infos3['sc_01 mean']=SC3_mean[0,1]
            infos3['sc_10 mean']=SC3_mean[1,0]
            infos3['sc_11 mean']=SC3_mean[1,1]
            infos3['desc_00 mean']=deSC3_mean[0,0]
            infos3['desc_01 mean']=deSC3_mean[0,1]
            infos3['desc_10 mean']=deSC3_mean[1,0]
            infos3['desc_11 mean']=deSC3_mean[1,1]
            infos3['Fisher_Cl mean']=Fisher_Cl3_mean
            infos3['Fisher_Quantum mean']=Fisher_Quantum3_mean
            infos3['action mean']=action3_mean

            infos3['r_0 std']=bootstrap_interval(R3[:,0],iters)
            infos3['r_1 std']=bootstrap_interval(R3[:,1],iters)
            infos3['der_0 std']=bootstrap_interval(deR3[:,0],iters)
            infos3['der_1 std']=bootstrap_interval(deR3[:,1],iters)
            infos3['sc_00 std']=bootstrap_interval(SC3[:,0,0],iters)
            infos3['sc_01 std']=bootstrap_interval(SC3[:,0,1],iters)
            infos3['sc_10 std']=bootstrap_interval(SC3[:,1,0],iters)
            infos3['sc_11 std']=bootstrap_interval(SC3[:,1,1],iters)
            infos3['desc_00 std']=bootstrap_interval(deSC3[:,0,0],iters)
            infos3['desc_01 std']=bootstrap_interval(deSC3[:,0,1],iters)
            infos3['desc_10 std']=bootstrap_interval(deSC3[:,1,0],iters)
            infos3['desc_11 std']=bootstrap_interval(deSC3[:,1,1],iters)
            infos3['Fisher_Cl std']=bootstrap_interval(Fisher_Cl3,iters)
            infos3['Fisher_Quantum std']=bootstrap_interval(Fisher_Quantum3,iters)
            infos3['action std']=bootstrap_interval(action3,iters)
            
            INFOS3.append(infos3)

            infos4['t']=i*dt
            infos4['r_0 abs']=R4_abs[0]
            infos4['r_1 abs']=R4_abs[1]
            infos4['r_0 mean']=R4_mean[0]
            infos4['r_1 mean']=R4_mean[1]
            infos4['der_0 mean']=deR4_mean[0]
            infos4['der_1 mean']=deR4_mean[1]
            infos4['sc_00 mean']=SC4_mean[0,0]
            infos4['sc_01 mean']=SC4_mean[0,1]
            infos4['sc_10 mean']=SC4_mean[1,0]
            infos4['sc_11 mean']=SC4_mean[1,1]
            infos4['desc_00 mean']=deSC4_mean[0,0]
            infos4['desc_01 mean']=deSC4_mean[0,1]
            infos4['desc_10 mean']=deSC4_mean[1,0]
            infos4['desc_11 mean']=deSC4_mean[1,1]
            infos4['Fisher_Cl mean']=Fisher_Cl4_mean
            infos4['Fisher_Quantum mean']=Fisher_Quantum4_mean
            infos4['action mean']=action4_mean

            infos4['r_0 std']=bootstrap_interval(R4[:,0],iters)
            infos4['r_1 std']=bootstrap_interval(R4[:,1],iters)
            infos4['der_0 std']=bootstrap_interval(deR4[:,0],iters)
            infos4['der_1 std']=bootstrap_interval(deR4[:,1],iters)
            infos4['sc_00 std']=bootstrap_interval(SC4[:,0,0],iters)
            infos4['sc_01 std']=bootstrap_interval(SC4[:,0,1],iters)
            infos4['sc_10 std']=bootstrap_interval(SC4[:,1,0],iters)
            infos4['sc_11 std']=bootstrap_interval(SC4[:,1,1],iters)
            infos4['desc_00 std']=bootstrap_interval(deSC4[:,0,0],iters)
            infos4['desc_01 std']=bootstrap_interval(deSC4[:,0,1],iters)
            infos4['desc_10 std']=bootstrap_interval(deSC4[:,1,0],iters)
            infos4['desc_11 std']=bootstrap_interval(deSC4[:,1,1],iters)
            infos4['Fisher_Cl std']=bootstrap_interval(Fisher_Cl4,iters)
            infos4['Fisher_Quantum std']=bootstrap_interval(Fisher_Quantum4,iters)
            infos4['action std']=bootstrap_interval(action4,iters)
            
            INFOS4.append(infos4)
        

        dW=np.random.randn(N,2)*(dt**0.5)
        dedW=(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(deR))*dt
        CURR=-(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(R))*dt+dW 
        dedW2=(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(deR2))*dt
        CURR2=-(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(R2))*dt+dW 
        dedW3=(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(deR3))*dt
        CURR3=-(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(R3))*dt+dW 
        dedW4=(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(deR4))*dt
        CURR4=-(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(R4))*dt+dW 
    
        obs=obs_prep(R,deR,CURR,SC,deSC,N)
        
    
    DATA=pd.DataFrame(INFOS)
    DATA2=pd.DataFrame(INFOS2)
    DATA3=pd.DataFrame(INFOS3)
    DATA4=pd.DataFrame(INFOS4)
    
    return DATA,DATA2,DATA3,DATA4

def main_trajs(N,timesteps,agent,params,dt,det=False,markevery=1,r_0=np.zeros(2),der_0=np.zeros(2),sc_0=11*np.identity(2),desc_0=0*np.identity(2),bench=-0.1,bench_noise=0.3):
    
    a,dea,d,b,e=Tools.Matrices_Calculator('Fisher',params) #calculation of 2x2 matrices
    A,deA,D,B,E=matrix_parallel_Fisher(N,a,dea,d,b,e) #calculation of 10x2x2 matrices
    
    SY=symplectic(N) #10x2x2 symplectic form
    R,deR,SC,deSC=initial_conditions(N,r_0,der_0,sc_0,desc_0) #fix initial conditions
    
    R2,deR2,SC2,deSC2=R,deR,SC,deSC #fix initial conditions for dumb action
    R3,deR3,SC3,deSC3=R,deR,SC,deSC
    R4,deR4,SC4,deSC4=R,deR,SC,deSC
    dW=np.random.randn(N,2)*(dt**0.5) #extraction of wiener increments (the same for both network and dumb feedback)
    dedW=(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(deR))*dt #dedw
    CURR=-(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(R))*dt+dW  #current calculation
    
    dedW2=(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(deR2))*dt #dedw for dumb action
    CURR2=-(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(R2))*dt+dW #current for dumb action
    
    
    dedW3=(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(deR3))*dt #dedw for dumb action
    CURR3=-(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(R3))*dt+dW #current for dumb action
    
    
    dedW4=(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(deR4))*dt #dedw for dumb action
    CURR4=-(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(R4))*dt+dW #current for dumb action
    
    
    Fisher_Cl=0 #initialize fisher for agent
    Fisher_Cl2=0 #initialize fisher for dumb action
    Fisher_Cl3=0
    Fisher_Cl4=0
    
    
    INFOS=[] #empty list for dicts of infos on agent
    INFOS2=[] #empty dict for dicts of infos on dumb action
    INFOS3=[] #empty list for dicts of infos on agent
    INFOS4=[]
    
    obs=obs_prep(R,deR,CURR,SC,deSC,N) #prepare observation to match observation shape of the model
    
    m=markevery

    #evolution and information collection for all of trajectories at the same time
    for i in range(0,timesteps):
        
        if i%int(timesteps*1e-1)==0:
            percentage=(i/(timesteps*1e-2))
            print('{}%'.format(percentage))


        action,_tmp=agent.predict(obs,deterministic=det)
        action=np.array(action[:,0])
        
        J=action[:,np.newaxis,np.newaxis]*SY
        R,SC,deR,deSC=fishsystem_nocost_RK4(R,deR,SC,deSC,A,deA,D,B,E,dt,dW,dedW,J)
        dFisher_Cl=dClassical_Fisher(deR,B,dt)
        Fisher_Cl=Fisher_Cl+dFisher_Cl
        Fisher_Quantum=Quantum_Fisher(N,SC,deSC,deR)

        action2=benchmark(N,bench,bench_noise)
        J2=action2[:,np.newaxis,np.newaxis]*SY
        R2,SC2,deR2,deSC2=fishsystem_nocost_RK4(R2,deR2,SC2,deSC2,A,deA,D,B,E,dt,dW,dedW2,J2)
        dFisher_Cl2=dClassical_Fisher(deR2,B,dt)
        Fisher_Cl2=Fisher_Cl2+dFisher_Cl2
        Fisher_Quantum2=Quantum_Fisher(N,SC2,deSC2,deR2)
        
        action3=benchmark(N,-0.1,0)
        J3=action3[:,np.newaxis,np.newaxis]*SY
        R3,SC3,deR3,deSC3=fishsystem_nocost_RK4(R3,deR3,SC3,deSC3,A,deA,D,B,E,dt,dW,dedW3,J3)
        dFisher_Cl3=dClassical_Fisher(deR3,B,dt)
        Fisher_Cl3=Fisher_Cl3+dFisher_Cl3
        Fisher_Quantum3=Quantum_Fisher(N,SC3,deSC3,deR3)
        
        action4=benchmark(N,0,0)
        J4=action4[:,np.newaxis,np.newaxis]*SY
        R4,SC4,deR4,deSC4=fishsystem_nocost_RK4(R4,deR4,SC4,deSC4,A,deA,D,B,E,dt,dW,dedW4,J4)
        dFisher_Cl4=dClassical_Fisher(deR4,B,dt)
        Fisher_Cl4=Fisher_Cl4+dFisher_Cl4
        Fisher_Quantum4=Quantum_Fisher(N,SC4,deSC4,deR4)

        if i%m==0:
            
            infos={} #empty dict for infos on agent
            infos2={} #empty dict for infos on dumb action
            infos3={} 
            infos4={}

            infos['t']=i*dt
            infos['r_0']=R[:,0]
            infos['r_1']=R[:,1]
            infos['der_0']=deR[:,0]
            infos['der_1']=deR[:,1]
            infos['sc_00']=SC[:,0,0]
            infos['sc_01']=SC[:,0,1]
            infos['sc_10']=SC[:,1,0]
            infos['sc_11']=SC[:,1,1]
            infos['desc_00']=deSC[:,0,0]
            infos['desc_01']=deSC[:,0,1]
            infos['desc_10']=deSC[:,1,0]
            infos['desc_11']=deSC[:,1,1]
            infos['dW_0']=dW[:,0]
            infos['dW_1']=dW[:,1]
            infos['Fisher_Cl']=Fisher_Cl
            infos['Fisher_Quantum']=Fisher_Quantum
            infos['action']=action
            
            infos2['t']=i*dt
            infos2['r_0']=R2[:,0]
            infos2['r_1']=R2[:,1]
            infos2['der_0']=deR2[:,0]
            infos2['der_1']=deR2[:,1]
            infos2['sc_00']=SC2[:,0,0]
            infos2['sc_01']=SC2[:,0,1]
            infos2['sc_10']=SC2[:,1,0]
            infos2['sc_11']=SC2[:,1,1]
            infos2['desc_00']=deSC2[:,0,0]
            infos2['desc_01']=deSC2[:,0,1]
            infos2['desc_10']=deSC2[:,1,0]
            infos2['desc_11']=deSC2[:,1,1]
            infos2['dW_0']=dW[:,0]
            infos2['dW_1']=dW[:,1]
            infos2['Fisher_Cl']=Fisher_Cl2
            infos2['Fisher_Quantum']=Fisher_Quantum2
            infos2['action']=action2
            
            infos3['t']=i*dt
            infos3['r_0']=R3[:,0]
            infos3['r_1']=R3[:,1]
            infos3['der_0']=deR3[:,0]
            infos3['der_1']=deR3[:,1]
            infos3['sc_00']=SC3[:,0,0]
            infos3['sc_01']=SC3[:,0,1]
            infos3['sc_10']=SC3[:,1,0]
            infos3['sc_11']=SC3[:,1,1]
            infos3['desc_00']=deSC3[:,0,0]
            infos3['desc_01']=deSC3[:,0,1]
            infos3['desc_10']=deSC3[:,1,0]
            infos3['desc_11']=deSC3[:,1,1]
            infos3['dW_0']=dW[:,0]
            infos3['dW_1']=dW[:,1]
            infos3['Fisher_Cl']=Fisher_Cl3
            infos3['Fisher_Quantum']=Fisher_Quantum3
            infos3['action']=action3
            
            infos4['t']=i*dt
            infos4['r_0']=R4[:,0]
            infos4['r_1']=R4[:,1]
            infos4['der_0']=deR4[:,0]
            infos4['der_1']=deR4[:,1]
            infos4['sc_00']=SC4[:,0,0]
            infos4['sc_01']=SC4[:,0,1]
            infos4['sc_10']=SC4[:,1,0]
            infos4['sc_11']=SC4[:,1,1]
            infos4['desc_00']=deSC4[:,0,0]
            infos4['desc_01']=deSC4[:,0,1]
            infos4['desc_10']=deSC4[:,1,0]
            infos4['desc_11']=deSC4[:,1,1]
            infos4['dW_0']=dW[:,0]
            infos4['dW_1']=dW[:,1]
            infos4['Fisher_Cl']=Fisher_Cl4
            infos4['Fisher_Quantum']=Fisher_Quantum4
            infos4['action']=action4
            
            INFOS.append(infos)
            INFOS2.append(infos2)
            INFOS3.append(infos3)
            INFOS4.append(infos4)
            

        dW=np.random.randn(N,2)*(dt**0.5)
        dedW=(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(deR))*dt
        CURR=-(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(R))*dt+dW 
        dedW2=(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(deR2))*dt
        CURR2=-(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(R2))*dt+dW 
        dedW3=(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(deR3))*dt
        CURR3=-(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(R3))*dt+dW 
        dedW4=(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(deR4))*dt
        CURR4=-(2**0.5)*prod_v((np.transpose(B,axes=(0,2,1))),(R4))*dt+dW 
    
        obs=obs_prep(R,deR,CURR,SC,deSC,N)
        
    
    DATA=pd.DataFrame(INFOS)
    DATA2=pd.DataFrame(INFOS2)
    DATA3=pd.DataFrame(INFOS3)
    DATA4=pd.DataFrame(INFOS4)
    
    return DATA,DATA2,DATA3,DATA4#,traj
    








    
    

        
