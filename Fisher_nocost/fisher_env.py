import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random as rand
import numpy as np
from scipy.linalg import sqrtm
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import math
import scipy
import Utilities as Tools
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
   
def checkpos(a): #function used to get the positive real parts of the eigenvalues of the Dyn matrix (Hurwitz check)
  for i in range(0,len(a)):
      if a[i]<0:
        a[i]=0
  return a
   
class FisherEnv(gym.Env):
      
      metadata = {'render.modes': ['human']} #non so bene a che serva ma per ora lo tengo
      #provo a definire degli attributi che dovrebbero essere le cose che vanno tenute in memoria in esecuzione
      
      def __init__(self,feedback,dt=1e-3,params={'k':1,'eta':1,'X_kunit':0.499,'theta':1},eval=False):
              
              super(FisherEnv, self).__init__() 
              
              #calcolo le matrici e altre cose
              self.eval=eval
              self.feedback=feedback #define what kind of feedback (Bayes or Markov)
              #check selected parameters, if null -> random choice
              k=Tools.check_param(params['k'],1,True)
              eta=Tools.check_param(params['eta'],1,True)
              X=Tools.check_param(params['X_kunit'],0.5,True)
              theta=Tools.check_param(params['theta'],0.5,True)
              params={'k':k,'eta':eta,'X_kunit':X,'theta':theta}
              self.params=params #save parameters
              self.A,self.deA,self.D,self.B,self.E=Tools.Matrices_Calculator('Fisher',params) #matrices calculation
              A,deA,D,B,E=self.A,self.deA,self.D,self.B,self.E #save matrices
              self.dw=np.random.randn(2)*(dt**0.5) #initialize an extraction of wiener increment
              self.dt=dt #define timestep
              self.time=0 #set time to 0
              self.r=np.zeros(2) #initialize first moments to 0 (reset is the important one)
              self.der=np.copy(self.r)
              self.sc=np.identity(2) #initialize \sigma_c to identity (reset is the important one)
              self.desc=np.identity(2)*0
              self.current=-(2**0.5)*(self.B.T)@(self.r)*dt+self.dw #initialize current (Markovian)
              self.dedw=(0.5**0.5)*(B.T)@(self.der)*dt
              self.reward=0 #initialize reward to 0
              self.Fish=0
              self.Io=0
              #initialize stuff for render
              self.viewer=None  
              self.ell=None
              self.res=[]
              self.rs=[]

              self.freq=0
              if feedback=='Bayes':
                #action is a number (frequency)
                self.action_space = spaces.Box( 
                  low=-np.inf, high=np.inf,shape=(1,), dtype=np.float32)
                self.observation_space = spaces.Box( 
                  low=-np.inf, high=np.inf,shape=(14,), dtype=np.float32)
              if feedback=='Markov':
                #action is a number (frequency)
                self.action_space = spaces.Box( 
                  low=-np.inf, high=np.inf,shape=(1,), dtype=np.float32)
                self.observation_space = spaces.Box( 
                  low=-np.inf, high=np.inf,shape=(2,), dtype=np.float32)


      def step(self, action):

              #save some variables to avoid calling always self.
              
              time=self.time
              feedback=self.feedback
              A,deA,D,B,E=self.A,self.deA,self.D,self.B,self.E
              dt=self.dt
              current=self.current
              r=self.r
              sc=self.sc
              der=self.der
              desc=self.desc
              dw=self.dw
              dedw=self.dedw
              Fish=self.Fish

              freq=self.freq
              #test
              #G=np.real(np.linalg.eigvals(sc+1j*np.array([[0,1],[-1,0]])))
              #print(G)
              #if np.any(G<0):
              #  print('NOGO')
              
              #save tha action 
              #dfreq=(action[0])*0.0001
              #freq=freq+dfreq
              #self.freq=freq
              #save tha action 
              #J=
              freq=(action[0])#*math.pi
              # in case action is the matrix
              J=freq*np.array([[0,1],[-1,0]]) #make a matrix out of the action
                
              #here we calculate the matrices and the u vector, necessary for later calculation
              
              #here we update variables
              r,sc,der,desc=Tools.fishsystem_nocost_RK4(r,der,sc,desc,A,deA,D,B,E,dt,dw,dedw,J) #system variables update
              dFish=2*(der.T)@B@(B.T)@(der)*dt
              Fish=Fish+dFish
              invsc=np.linalg.inv(sc)
              pur=np.linalg.det(sc)**(-0.5)
              #print(pur)
              Io=0.5*np.trace((invsc@desc)@(invsc@desc))/(1+pur**2)
              I1=2*(der.T)@invsc@der
              I2=0.5*((np.trace(invsc@desc)*pur)**2)/(1-pur**4+1e-3)
              I=Io+I1+I2#0.5*np.trace((invsc@desc)@(invsc@desc))/(1+pur**2)+2*(der.T)@invsc@der+(np.trace(invsc@desc)**2)/(2*np.linalg.det(sc)*)
              #robadaplot
              #phi=np.arctan((r[1])/(r[0]))
              #R=np.array([[np.cos(-phi),-np.sin(-phi)],[np.sin(-phi),np.cos(-phi)]])
              #tmp=R@sc@(R.T)
              self.res.append([Io,I1,I2])
              #update
              rew=(Fish+I)/((self.time+1)*dt)
              rew=rew
              self.dw=np.random.randn(2)*(dt**0.5)
              dw=self.dw
              self.current=-(2**0.5)*(self.B.T)@(r)*dt+self.dw
             
              if math.isnan(rew):
                exit()         
              if pur>=1.1:
                print('problemino')
                exit()    

              self.sc=sc 
              self.r=r
              self.der=der
              self.desc=desc
              self.Fish=Fish
              self.dedw=(2**(0.5))*(B.T)@np.copy(self.der)*dt
              
              if feedback=='Markov':
                output=self.current#np.concatenate((self.current,self.r),axis=0)
              else:
                output=np.concatenate((self.r.flatten(),self.der.flatten(),self.current,self.sc.flatten(),self.desc.flatten()),axis=0) #self.r#the observation has to be only r for construction
              
              
              #ends the episode at a given time
              if time==1e5:
                self.Done=True
              time+=1
              self.time=time
              
              #here we have the measurement, extraction of wiener increments and current
              
              
              #here we save parameters, if needed for eventual training with variable parameters, to be put into the info dictionary
              #parametri=np.array(list(self.params.items()))
              #parametri=parametri[:,1].astype(np.float)
              #output=np.tanh(0.01*output)
              
              return output , rew , self.Done ,{'r_0':self.r[0],'r_1':self.r[1],'der_0':self.der[0],'der_1':self.der[1],'rew':rew,'desc_00':self.desc[0,0],'desc_01':self.desc[0,1],'desc_10':self.desc[1,0],'desc_11':self.desc[1,1],'current_0':self.current[0],'current_1':self.current[1],'action':action,'sc_00':self.sc[0,0],'sc_01':self.sc[0,1],'sc_10':self.sc[1,0],'sc_11':self.sc[1,1],'purity':(np.linalg.det(self.sc))**(-0.5),'Qcost':0,'fishercl':Fish,'fisherq':I,'dw':0}#,'params':parametri}
#{'r':self.r,'der':self.der,'rew':rew,'desc':self.desc,'current':self.current,'action':action,'sc':self.sc,'purity':(np.linalg.det(self.sc))**(-0.5),'Qcost':0,'fishercl':Fish,'fisherq':I}#,'params':parametri}

      #
    #

      def reset(self):
              
              #self.q=np.random.uniform(5e-5,5e-4)
              dt=self.dt
              self.time=0 #reset time
              A,deA,D,B,E=self.A,self.deA,self.D,self.B,self.E
              
              #random initial first moment
              r=np.random.uniform(-3,3,2)#np.random.uniform(-0.5,0.5,2)
              self.freq=0
              #save the variables
              n=np.random.uniform(0,5)
              self.r=r
              self.desc=np.identity(2)*0
              self.sc=(2*n+1)*np.identity(2)#np.copy(self.sigmacss)##
              self.der=r*0
              self.Fish=0
              
              if self.eval==True:
                self.r=np.zeros(2)
                self.sc=11*np.identity(2)
              self.dedw=np.zeros(2)
              invsc=np.linalg.inv(self.sc)
              pur=np.linalg.det(self.sc)**(-0.5)
              der=self.der
              desc=self.desc
              sc=self.sc
              #print(pur)
              self.Io=0.5*np.trace((invsc@desc)@(invsc@desc))/(1+pur**2)+2*(der.T)@invsc@der+(np.trace(invsc@desc)**2)/(2*np.linalg.det(sc)*(1-pur**4+1e-3))
              

              #sets Done as False at the beginning of each episode
              self.Done=False
              #extraction of the first wiener increment of the episode, save r and the current 
              dw=np.random.randn(2)*(dt**0.5)
              self.dw=dw
              self.current=-(2**0.5)*(np.copy(self.B).T)@(np.copy(self.r))*dt+np.copy(self.dw)
              invsc=np.linalg.inv(self.sc)
              if self.feedback=='Bayes':
                  output=np.concatenate((self.r.flatten(),self.der.flatten(),self.current,self.sc.flatten(),self.desc.flatten()),axis=0)  #r#np.concatenate((r,self.current,self.sc.flatten()),axis=0)#r#np.concatenate((self.r,self.sc.flatten()),axis=0)#np.concatenate((self.r.flatten(),self.der.flatten()),axis=0)#self.r #recall that u_act True is possible only with Bayesian feedback as for now
              if self.feedback=='Markov':
                  output=self.current#np.concatenate((self.current,self.r),axis=0)#np.concatenate((self.r,self.sc.flatten()),axis=0)#np.concatenate((self.r.flatten(),self.der.flatten()),axis=0)#self.r #recall that u_act True is possible only with Bayesian feedback as for now
              self.viewer=None
              self.ell=None
              self.res=[[0,0,0]]
              self.rs=[]
              #output=np.tanh(0.01*output)

              return output

      
      
      #method necessary for rendering
      def _to_rgbarray(self):
        canvas = FigureCanvas(self.viewer)
        canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))
        return image

      #render function to make gifs of trained agent
      def render(self,mode='human'):
        #we look at a conditional trajectory...
        r=self.r[0:2]
        sc=self.sc[0:2,0:2]
        self.rs.append(r)
        w,v=np.linalg.eig(sc)
        conv=180/math.pi
        theta=conv*0.5*np.arctan(2*sc[0,1]/(1e-10+sc[0,0]**2-sc[1,1]**2)) #stuff for the calculation of the covariance ellipse
        a=0.2 #trasparence
        

        if self.viewer is None:
            # Here we initialize the plot, 'human' is basically useless in our implementation
            if mode == 'human':
                self.viewer = plt.figure(figsize=[5, 5], dpi=72)
                self.ax=self.viewer.gca()
                self.ell=Ellipse((r[0], r[1]), width=w[0], height=w[1], angle=theta,alpha=a)
                self.ax.add_patch(self.ell)
                plt.ylim([-5,5])
                plt.xlim([-5,5])
            
            elif mode == 'rgb_array':
                plt.style.use(['seaborn-whitegrid'])
                #figura=plt.figure(figsize=[4,4], dpi=90)
                fig, (ax1, ax2) = plt.subplots(2, 1,figsize=[4,6], dpi=90, gridspec_kw={'width_ratios': [1],'height_ratios': [2, 1]})
                self.viewer = fig#plt.figure(figsize=[4,4], dpi=90)
                self.ax1,self.ax2=self.viewer.axes
                self.ell=Ellipse((r[0], r[1]), width=w[0], height=w[1],color='#b300ff', angle=theta,alpha=a)
                self.ax1.add_patch(self.ell)
                self.ax1.set_ylim([-10,10])
                self.ax1.set_xlim([-10,10])
                self.ax2.set_ylim([-12,7])
                self.ax2.set_xlim([0,int(2e5)])
                #self.line=self.ax2.plot(res)

        #here we modify the objects initialized in the previous plot
        #ax = self.viewer.gca()

        #self.ax1.plot(r[0],r[1],'r:',linewidth=1,markevery=1,alpha=0.6)
        self.ax1.cla()
        self.ax1.set_ylim([-10,10])
        self.ax1.set_xlim([-10,10])
        rs=np.array(self.rs)
        self.ax1.plot(rs[:,0],rs[:,1],color='#00ff2a',alpha=1,linewidth=.5)
        self.ell=Ellipse((r[0], r[1]), width=w[0], height=w[1],color='#b300ff', angle=theta,alpha=a)
        self.ax1.add_patch(self.ell)
                
        #self.ell.set_alpha(a)
        #self.ell.set_center((r[0],r[1]))
        #self.ell.angle=theta
        #self.ell.set_height(w[1])
        #self.ell.set_width(w[0])

        
        res=self.res
        res=np.array(res)
        self.ax2.set_ylim([-12,7])
        self.ax2.set_xlim([0,int(2e5)])
        self.ax2.cla()
        self.ax2.plot(res[:,0],linestyle='solid',linewidth=.5,label=r'I_0',alpha=1)
        self.ax2.plot(res[:,1],linestyle='solid',linewidth=.5,label=r'I_1',alpha=1)
        self.ax2.plot(res[:,2],linestyle='solid',linewidth=.5,label=r'I_2',alpha=1)
        leg = self.ax2.legend(prop={'size': 10})
        text=leg.get_texts()
        for i in range(0,len(text)):
            text[i].set_color('black')

        #-10*np.log10(res)
        #ax.add_patch(Ellipse((r[0], r[1]),
        #width=w[0],#qualche funzione di su
        #height=w[1],
        #angle=theta,alpha=a))

        if mode == 'human':
            return self.viewer
        elif mode == 'rgb_array':
            return self._to_rgbarray()      
          
      