# Code to implement the correlation funciton evolution (Redfield) dynamics of the system consisting of fermions



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
Gamma=[]
Gamma.append(Gamma1) #Gamma1=1
Gamma.append(Gamma2) #Gamma2=4
tb=1




def spectral_bath(omega,Gamma, tb): #Define the spectral bath function
    if (omega <= -2*tb):
        return 0
    if (omega >= 2*tb):
        return 0
    return Gamma*np.sqrt(1-(omega*omega)/(4*tb*tb))


def nbar(omega, beta, mu):  # Define the fermionic occupation number
    return 1/(np.exp(beta*(omega-mu))+1)


def intefunc1(omega,Gamma,tb,beta,mu): # Define spectral_bath*nbar, a function required for computing the evolution.
    if (omega <= -2*tb):
        return 0
    if (omega >= 2*tb):
        return 0
    return spectral_bath(omega,Gamma,tb)*nbar(omega,beta,mu)



def evolfunc(state,M,O): #Evolve 
   return M@state+O


def f(omega,c,alpha,gamma,Gamma,tb):     # computes the f_(\alpha,\gamma) defined in the paper.
    f=0
    f=f+0.5*c[0,alpha].conj()*c[0,gamma]*spectral_bath(omega,Gamma[0],tb)
    f=f+0.5*c[1,alpha].conj()*c[1,gamma]*spectral_bath(omega,Gamma[1],tb)
    return f

def F(omega,c,alpha,gamma,Gamma,tb,beta1,beta2,mu1,mu2): # computes the F_(\alpha,\gamma) defined in the paper.
    f=0
    F=0
    F=F+0.5*c[0,alpha].conj()*c[0,gamma]*intefunc1(omega,Gamma[0],tb,beta1,mu1)
    F=F+0.5*c[1,alpha].conj()*c[1,gamma]*intefunc1(omega,Gamma[1],tb,beta2,mu2)
    return F


glist=[0.1]  #Define the g values for which you wish to compute time evolution. 
#glist=np.linspace(0.0,0.5,27)


finallist=[] #optional (See end of code)

c=np.empty((2,2),dtype=np.double) # c matrix as defined in the paper. Converts the eigenmode operators to the original site operators


c[0,0]=1/np.sqrt(2)
c[0,1]=1/np.sqrt(2)
c[1,0]=-1/np.sqrt(2)
c[1,1]=1/np.sqrt(2)



for g in glist:
    print("gvalue is ",g)

    w=[]

    w.append(w0-g)
    w.append(w0+g) #stores eigenfrequencies of the system.

    
    pvaluej=np.empty((2,2),dtype=np.cdouble) #stores Cauchy P value of J 
    pvaluejn=np.empty((2,2),dtype=np.cdouble) #stores Cauchy of J*n

    pvaluejn[0,0]=(-1.0j/(2*np.pi))*integrate.quad(intefunc1,-2*tb,2*tb,args=(Gamma1,tb,beta1,mu1),weight='cauchy',wvar=w[0])[0]  #bath1, w1
    pvaluejn[0,1]=(-1.0j/(2*np.pi))*integrate.quad(intefunc1,-2*tb,2*tb,args=(Gamma1,tb,beta1,mu1),weight='cauchy',wvar=w[1])[0]    #bath1,w2
    pvaluejn[1,0]=(-1.0j/(2*np.pi))*integrate.quad(intefunc1,-2*tb,2*tb,args=(Gamma2,tb,beta2,mu2),weight='cauchy',wvar=w[0])[0] #bath2, w1
    pvaluejn[1,1]=(-1.0j/(2*np.pi))*integrate.quad(intefunc1,-2*tb,2*tb,args=(Gamma2,tb,beta2,mu2),weight='cauchy',wvar=w[1])[0]

    
    pvaluej[0,0]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath,-2*tb,2*tb,args=(Gamma1,tb),weight='cauchy',wvar=w[0])[0]  #bath1, w1
    pvaluej[0,1]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath,-2*tb,2*tb,args=(Gamma1,tb),weight='cauchy',wvar=w[1])[0]    #bath1,w2
    pvaluej[1,0]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath,-2*tb,2*tb,args=(Gamma2,tb),weight='cauchy',wvar=w[0])[0] #bath2, w1
    pvaluej[1,1]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath,-2*tb,2*tb,args=(Gamma2,tb),weight='cauchy',wvar=w[1])[0]
    
    # we have computed the pvalues that are necessary to com
    # Note that we have included the "2" and "-1" factor in integrals above.
 
    
    #first 2 indices are alpha, gamma, last index is w1 or w2. (Some of these entries are not necessary)
    # ie, f_matrix(alpha,gamma,w1)=f_{alpha,gamma} (w1) as defined in the paper.
    
    f_matrix=np.empty((2,2,2),dtype=np.cdouble)     
    F_matrix=np.empty((2,2,2),dtype=np.cdouble)
    f_deltamatrix=np.empty((2,2,2),dtype=np.cdouble)
    F_deltamatrix=np.empty((2,2,2),dtype=np.cdouble)
    
    # We now compute the correlation function evolution. 
    # We let C[0]=C_11 C[1]=C_12, C[2]=C_21, C[3]=C_33
    # Therefore, the correlation function evolution is given by dC/dT=M*C+O. We construct M and O,
    
    for alpha in range(2):
        for gamma in range(2):
            f_matrix[alpha,gamma,0]=f(w[0],c,alpha,gamma,Gamma,tb)
            f_matrix[alpha,gamma,1]=f(w[1],c,alpha,gamma,Gamma,tb)
            F_matrix[alpha,gamma,0]=F(w[0],c,alpha,gamma,Gamma,tb,beta1,beta2,mu1,mu2)
            F_matrix[alpha,gamma,1]=F(w[1],c,alpha,gamma,Gamma,tb,beta1,beta2,mu1,mu2)
            
            f_deltamatrix[alpha,gamma,0]=(-1/np.pi)*integrate.quad(f,-2*tb,2*tb,args=(c,alpha,gamma,Gamma,tb),weight='cauchy',wvar=w[0])[0]
            f_deltamatrix[alpha,gamma,1]=(-1/np.pi)*integrate.quad(f,-2*tb,2*tb,args=(c,alpha,gamma,Gamma,tb),weight="cauchy",wvar=w[1])[0]
            F_deltamatrix[alpha,gamma,0]=(-1/np.pi)*integrate.quad(F,-2*tb,2*tb,args=(c,alpha,gamma,Gamma,tb,beta1,beta2,mu1,mu2),weight="cauchy",wvar=w[0])[0]
            F_deltamatrix[alpha,gamma,1]=(-1/np.pi)*integrate.quad(F,-2*tb,2*tb,args=(c,alpha,gamma,Gamma,tb,beta1,beta2,mu1,mu2),weight="cauchy",wvar=w[1])[0]
            
    
    
    M=np.empty((4,4),dtype=np.cdouble)
    O=np.empty((4,1),dtype=np.cdouble)
    
    #Construction done.

    O[0]=2*epsilon*epsilon*F_matrix[0,0,0]
    M[0,0]=-2*epsilon*epsilon*f_matrix[0,0,0]
    M[0,1]=-epsilon*epsilon*(f_matrix[0,1,1]+1.0j*f_deltamatrix[0,1,1])
    M[0,2]=-epsilon*epsilon*(f_matrix[0,1,1]-1.0j*f_deltamatrix[0,1,1])
    M[0,3]=0
    
    #C11 done.
    
    O[1]=epsilon*epsilon*(F_matrix[1,0,0]+F_matrix[0,1,1]+1.0j*F_deltamatrix[1,0,0]-1.0j*F_deltamatrix[0,1,1])
    M[1,0]=-epsilon*epsilon*(f_matrix[1,0,0]+1.0j*f_deltamatrix[1,0,0])
    M[1,1]=1.0j*w[0]-1.0j*w[1]  -epsilon*epsilon*(f_matrix[0,0,0]+f_matrix[1,1,1]+1.0j*f_deltamatrix[1,1,1]-1.0j*f_deltamatrix[0,0,0])
    M[1,2]=0
    M[1,3]=-epsilon*epsilon*(f_matrix[0,1,1]-1.0j*f_deltamatrix[0,1,1])
        
    #C12 done
    
    O[2]=epsilon*epsilon*(F_matrix[0,1,1]+F_matrix[1,0,0]+1.0j*F_deltamatrix[0,1,1]-1.0j*F_deltamatrix[1,0,0])
    M[2,0]=-epsilon*epsilon*(f_matrix[1,0,0]-1.0j*f_deltamatrix[1,0,0])
    M[2,1]=0
    M[2,2]=1.0j*w[1]-1.0j*w[0]-epsilon*epsilon*(f_matrix[0,0,0]+f_matrix[1,1,1]-1.0j*f_deltamatrix[1,1,1]+1.0j*f_deltamatrix[0,0,0])
    M[2,3]=-epsilon*epsilon*(f_matrix[0,1,1]+1.0j*f_deltamatrix[0,1,1])
    
    #C21 Done
    
    O[3]=2*epsilon*epsilon*F_matrix[1,1,1]
    M[3,0]=0
    M[3,1]=-epsilon*epsilon*(f_matrix[1,0,0]-1.0j*f_deltamatrix[1,0,0])
    M[3,2]=-epsilon*epsilon*(f_matrix[1,0,0]+1.0j*f_deltamatrix[1,0,0])
    M[3,3]=-2*epsilon*epsilon*f_matrix[1,1,1]
    
    #C22 done
    
    
    #implement RK4 evolution.
    initial=np.zeros((4,1),np.cdouble) #Initial values of the correlators.
    store=[] #stores the value of the correlatios.
    
    h=0.2 #step size of RK4
    tmax=500 #max time
    steps=int(tmax/h +1)
    
    store.append(initial)
    N1=[] #Stores occupation number of the first site.
    N1.append(initial[0])
    N2=[] #Stores occupation number of the second site.
    N2.append(initial[0])
    
    
    tlist=np.linspace(0,tmax,steps)
    for k in range(steps-1):
        wi=store[k]
        k1=h*evolfunc(wi,M,O)
        k2=h*evolfunc(wi+0.5*k1,M,O)
        k3=h*evolfunc(wi+0.5*k2,M,O)
        k4=h*evolfunc(wi+k3,M,O)
        wi_1=wi+(k1+2*k2+2*k3+k4)/6
    
        store.append(wi_1)
        #if (k%100==0):
         #   print ("k=",k)
        N1.append(wi_1[0]) #update occupation numbers
        N2.append(wi_1[3])
 
    #plot the occupation numbers.
    plt.plot(tlist,N1,label="N1")
    plt.plot(tlist,N2,label="N2")
    plt.legend()
    plt.show()
    #finallist.append(1.0j*g*(-wi_1[2]+wi_1[1]))
#plt.plot(tlist,blah)
    
#plt.plot(glist,finallist)

    
    
    
    
