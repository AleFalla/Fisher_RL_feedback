import numpy as np

def step(model,det,env,obs,r,der,current,sc,rew,azione,Qcost,fisherq,fishercl,opt=False):
    if opt==False:
        action, _states = model.predict(obs,deterministic=det)
    else:
        action = [0]#,0]#env.action_space.sample()#*0
    
    obs, rewards, dones, info = env.step(action)
    if type(info)==list:
        info=info[0]
        rewards=rewards[0]
    r.append(info['r'])
    der.append(info['der'])
    current.append(info['current'])
    sc.append(info['sc'])
    rew.append(rewards)
    azione.append(info['u'])
    Qcost.append(info['Qcost'])
    fishercl.append(info['fishercl'])
    fisherq.append(info['fisherq'])
    return obs

                  
    

def media_cammini(N,steps,env,det,optenv,model,opt_model):#,Optimal_Agent):
    
    #resetto l'ambiente
    obs = env.reset()
    optenv.reset()    
    
    #inizializzo le liste per formare i vettori relativi alle singole realizzazioni
    #qui per l'agente allenato
    r=[]
    der=[]
    current=[]
    sc=[]
    rew=[]
    Qcost=[]
    azione=[]
    fishercl=[]
    fisherq=[]
    
    r2=[]
    der2=[]
    current2=[]
    sc2=[]
    rew2=[]
    Qcost2=[]
    azione2=[]
    fishercl2=[]
    fisherq2=[]

    #inizializzo le tabelle
    R=[]
    deR=[]
    Current=[]
    Sc=[]
    Rew=[]
    QCOST=[]
    Azione=[]
    FC=[]
    FQ=[]
    
    R2=[]
    deR2=[]
    Current2=[]
    Sc2=[]
    Rew2=[]
    QCOST2=[]
    Azione2=[]
    FC2=[]
    FQ2=[]
    
    for j in range(0,N): #cammini
    
      for k in range(0,steps): #steps di integrazione
      
        
        obs=step(model,det,env,obs,r,der,current,sc,rew,azione,Qcost,fisherq,fishercl)
        
        step(opt_model,det,optenv,obs,r2,der2,current2,sc2,rew2,azione2,Qcost2,fisherq2,fishercl2,opt=True)
        
        
      
      #resetto l'ambiente
      
      obs=env.reset()
      optenv.reset()
      opt_model=None#Optimal_Agent(optenv)
    
      #salvo sulle tabelle
      R.append(r)
      deR.append(der)
      Current.append(current)
      Sc.append(sc)
      Rew.append(rew)
      QCOST.append(Qcost)
      Azione.append(azione)
      FC.append(fishercl)
      FQ.append(fisherq)
    
      R2.append(r2)
      deR2.append(der2)
      Current2.append(current2)
      Sc2.append(sc2)
      Rew2.append(rew2)
      QCOST2.append(Qcost2)
      Azione2.append(azione2)
      FC2.append(fishercl2)
      FQ2.append(fisherq2)
    
      #reinizializzo i vettorini delle realizzazioni
      r=[]
      der=[]
      current=[]
      sc=[]
      rew=[]
      Qcost=[]
      azione=[]
      fishercl=[]
      fisherq=[]
        
      r2=[]
      der2=[]
      current2=[]
      sc2=[]
      rew2=[]
      Qcost2=[]
      azione2=[] 
      fishercl2=[]
      fisherq2=[]

      #fine del for su N
    
    
    #faccio la media sulle realizzazioni
    rmean=np.mean(np.array(R),axis=0)
    dermean=np.mean(np.array(deR),axis=0)
    rmean2=np.mean(np.array(R2),axis=0)
    dermean2=np.mean(np.array(deR2),axis=0)
    scmean=np.mean(np.array(Sc),axis=0)
    scmean2=np.mean(np.array(Sc2),axis=0)
    currentmean=np.mean(np.array(Current),axis=0)
    currentmean2=np.mean(np.array(Current2),axis=0)
    rewmean=np.mean(np.array(Rew),axis=0)
    rewmean2=np.mean(np.array(Rew2),axis=0)
    Qcostmean=np.mean(np.array(QCOST),axis=0)
    Qcostmean2=np.mean(np.array(QCOST2),axis=0)
    azionemean=np.mean(np.array(Azione),axis=0)
    azionemean2=np.mean(np.array(Azione2),axis=0)
    FCmean=np.mean(np.array(FC),axis=0)
    FCmean2=np.mean(np.array(FC2),axis=0)
    FQmean=np.mean(np.array(FQ),axis=0)
    FQmean2=np.mean(np.array(FQ2),axis=0)

    
    #salvo le std
    rstd=np.std(np.array(R),axis=0)
    derstd=np.std(np.array(deR),axis=0)
    rstd2=np.std(np.array(R2),axis=0)
    derstd2=np.std(np.array(deR2),axis=0)
    scstd=np.std(np.array(Sc),axis=0)
    scstd2=np.std(np.array(Sc2),axis=0)
    currentstd=np.std(np.array(Current),axis=0)
    currentstd2=np.std(np.array(Current2),axis=0)
    rewstd=np.std(np.array(Rew),axis=0)
    rewstd2=np.std(np.array(Rew2),axis=0)
    Qcoststd=np.std(np.array(QCOST),axis=0)
    Qcoststd2=np.std(np.array(QCOST2),axis=0)
    azionestd=np.std(np.array(Azione),axis=0)
    azionestd2=np.std(np.array(Azione2),axis=0)
    FCstd=np.std(np.array(FC),axis=0)
    FCstd2=np.std(np.array(FC2),axis=0)
    FQstd=np.std(np.array(FQ),axis=0)
    FQstd2=np.std(np.array(FQ2),axis=0)
    
    return {'rmean':rmean,'rmean2':rmean2,'dermean':dermean,'dermean2':dermean2,'scmean':scmean,'scmean2':scmean2,'currentmean':currentmean,'currentmean2':currentmean2,'rewmean':rewmean,'rewmean2':rewmean2,'Qcostmean':Qcostmean,'Qcostmean2':Qcostmean2,'azionemean':azionemean,'azionemean2':azionemean2,'FCmean':FCmean,'FCmean2':FCmean2,'FQmean':FQmean,'FQmean2':FQmean2,'rstd':rstd,'rstd2':rstd2,'derstd':derstd,'derstd2':derstd2,'scstd':scstd,'scstd2':scstd2,'currentstd':currentstd,'currentstd2':currentstd2,'rewstd':rewstd,'rewstd2':rewstd2,'Qcoststd':Qcoststd,'Qcoststd2':Qcoststd2,'azionestd':azionestd,'azionestd2':azionestd2,'FCstd':FCstd,'FCstd2':FCstd2,'FQstd':FQstd,'FQstd2':FQstd2}
