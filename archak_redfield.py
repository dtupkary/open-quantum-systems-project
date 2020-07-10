# Started writing on 22nd December 2019. Code to implement RK 4 on Archak's system using the derivation of the general redfield implementaiton (with interactions)
# Does RK 4 evolution, for BOSONS 

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy import integrate



#declaring parameters
w0=1
beta=0.8
mu=-2.5
epsilon=0.1
mu1=mu
mu2=mu
beta1=beta
beta2=beta
Gamma1=1
Gamma2=4
tb=1


#Dimension of Bosonic operators. Higher is more accurate
N=7



def spectral_bath(omega,Gamma, tb):  #spectral bath function
    if (omega <= -2*tb):
        return 0
    if (omega >= 2*tb):
        return 0
    return Gamma*np.sqrt(1-(omega*omega)/(4*tb*tb))


def nbar(omega, beta, mu): #Bosonic occupation numbers
    return 1/(np.exp(beta*(omega-mu))-1)


def intefunc1(omega,Gamma,tb,beta,mu): #spectralbath*nbar (Required later)
    if (omega <= -2*tb):
        return 0
    if (omega >= 2*tb):
        return 0
    return spectral_bath(omega,Gamma,tb)*nbar(omega,beta,mu)



def evolfunc(state,HS,epsilon,c,A,constantjn,constantjn_1): #rk4 evolution. 
    term1=1.0j*commutator(state,HS)

    term2=0

    for l in range(2):
        for alpha in range(2):
            for gamma in range(2):
                term2=term2+c[l,alpha]*c[l,gamma]*constantjn[l,gamma]*commutator(state*A[gamma],A[alpha].dag())
                term2=term2+c[l,alpha]*c[l,gamma]*constantjn_1[l,gamma]*commutator(A[alpha].dag(),A[gamma]*state)
                
    term2=epsilon*epsilon*term2

    return term1-term2-term2.dag()






a1=tensor(destroy(N),qeye(N))
a2=tensor(qeye(N),destroy(N))
A=[]

A.append((a1-a2)/np.sqrt(2))
A.append((a1+a2)/np.sqrt(2)) #A1,A2 are the modes

glist=[0.45]


c=np.empty((2,2),dtype=np.double) #matrix that converts from a1,a2 to A1, A2

c[0,0]=1/np.sqrt(2)
c[0,1]=1/np.sqrt(2)
c[1,0]=-1/np.sqrt(2)
c[1,1]=1/np.sqrt(2)



for g in glist:


    w=[]

    w.append(w0-g)
    w.append(w0+g)

    
    H=w0*(a1.dag()*a1+a2.dag()*a2)+g*(a1.dag()*a2+a1*a2.dag())
    #Hamiltonian of system is created

    #we compute integrals required to compute the constants that appear in Redfield equation. 
    pvaluej=np.empty((2,2),dtype=np.cdouble) 
    pvaluejn=np.empty((2,2),dtype=np.cdouble)

    pvaluejn[0,0]=(-1.0j/(2*np.pi))*integrate.quad(intefunc1,-2*tb,2*tb,args=(Gamma1,tb,beta1,mu1),weight='cauchy',wvar=w[0])[0]  #bath1, w1
    pvaluejn[0,1]=(-1.0j/(2*np.pi))*integrate.quad(intefunc1,-2*tb,2*tb,args=(Gamma1,tb,beta1,mu1),weight='cauchy',wvar=w[1])[0]    #bath1,w2
    pvaluejn[1,0]=(-1.0j/(2*np.pi))*integrate.quad(intefunc1,-2*tb,2*tb,args=(Gamma2,tb,beta2,mu2),weight='cauchy',wvar=w[0])[0] #bath2, w1
    pvaluejn[1,1]=(-1.0j/(2*np.pi))*integrate.quad(intefunc1,-2*tb,2*tb,args=(Gamma2,tb,beta2,mu2),weight='cauchy',wvar=w[1])[0]

    
    pvaluej[0,0]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath,-2*tb,2*tb,args=(Gamma1,tb),weight='cauchy',wvar=w[0])[0]  #bath1, w1
    pvaluej[0,1]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath,-2*tb,2*tb,args=(Gamma1,tb),weight='cauchy',wvar=w[1])[0]    #bath1,w2
    pvaluej[1,0]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath,-2*tb,2*tb,args=(Gamma2,tb),weight='cauchy',wvar=w[0])[0] #bath2, w1
    pvaluej[1,1]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath,-2*tb,2*tb,args=(Gamma2,tb),weight='cauchy',wvar=w[1])[0]
    
    #print("Checking pvaluej values")

   # print(pvaluej[0,0],1.0j*Gamma1*w[0]/(4*tb))
 #   print(pvaluej[0,1],1.0j*Gamma1*w[1]/(4*tb))
  #  print(pvaluej[1,0],1.0j*Gamma2*w[0]/(4*tb))
  #  print(pvaluej[1,1],1.0j*Gamma2*w[1]/(4*tb))    
  #  
  #  print("Checking done")

    constantjn_1=np.empty((2,2),dtype=np.cdouble)
    constantjn=np.empty((2,2),dtype=np.cdouble)


    #compute the full constants (Which consist of delta function real part + imaginary part of cauchy P integral)
    constantjn[0,0]=0.5*intefunc1(w[0],Gamma1,tb,beta1,mu1)+pvaluejn[0,0] #jn constants done
    constantjn[0,1]=0.5*intefunc1(w[1],Gamma1,tb,beta1,mu1)+pvaluejn[0,1]
    constantjn[1,0]=0.5*intefunc1(w[0],Gamma2,tb,beta2,mu2)+pvaluejn[1,0]
    constantjn[1,1]=0.5*intefunc1(w[1],Gamma2,tb,beta2,mu2)+pvaluejn[1,1]

    #j*(n+1) constants needed.

    constantjn_1[0,0]=constantjn[0,0]+pvaluej[0,0]+0.5*spectral_bath(w[0],Gamma1,tb)
    constantjn_1[0,1]=constantjn[0,1]+pvaluej[0,1]+0.5*spectral_bath(w[1],Gamma1,tb)
    constantjn_1[1,0]=constantjn[1,0]+pvaluej[1,0]+0.5*spectral_bath(w[0],Gamma2,tb)
    constantjn_1[1,1]=constantjn[1,1]+pvaluej[1,1]+0.5*spectral_bath(w[1],Gamma2,tb)

    #constants prepared.

    #Next, we implement rk4. We set up initial state first. 
    initialstate=tensor(basis(N,0),basis(N,0))
    initial=initialstate*initialstate.dag()

    states=[] #stores states. 
    states.append(initial)

    h=0.1
    tmax=200
    steps=int(tmax/h +1)

    print("Starting evolution, step size =",h," tmax=",tmax)        
    

    tlist=np.linspace(0,tmax,steps)



    for k in range(steps-1):
        wi=states[k]
        k1=h*evolfunc(wi,H,epsilon,c,A,constantjn,constantjn_1)
        k2=h*evolfunc(wi+0.5*k1,H,epsilon,c,A,constantjn,constantjn_1)
        k3=h*evolfunc(wi+0.5*k2,H,epsilon,c,A,constantjn,constantjn_1)
        k4=h*evolfunc(wi+k3,H,epsilon,c,A,constantjn,constantjn_1)
        wi_1=wi+(k1+2*k2+2*k3+k4)/6
        states.append(wi_1)
        if (k%100==0):
            print ("k=",k,"trace distance betn successive states is",tracedist(wi,wi_1)) #to test convergence.
            



    #print("printing trace")
    #for k in range(steps):
    #    print(k,states[k].tr())



    oper=A[1].dag()*A[1]

    value=[]

    for k in range(steps):
        value.append(expect(oper,states[k]))


    plt.plot(tlist,value)
    plt.show()
    
    
    
    
    
    N_op=a1.dag()*a1+a2.dag()*a2
    theory=(-beta*(H-mu*N_op)).expm()
    ss_theory=theory/theory.tr()
    
    print("Final trace distance is ",tracedist(wi_1,ss_theory))
