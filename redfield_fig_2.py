# Started writing on 22nd December 2019. Code to implement RK 4 on Archak's system using the derivation of the general redfield implementaiton (with interactions)


import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy import integrate



#declaring parameters
w0=1
epsilon=0.1
mu1=-4.5
mu2=-4.5
beta1=0.8
beta2=1.8
Gamma1=1
Gamma2=4
tb=2


N=3



def spectral_bath(omega,Gamma, tb):
	return Gamma*np.sqrt(1-(omega*omega)/(4*tb*tb))


def nbar(omega, beta, mu):
	return 1/(np.exp(beta*(omega-mu))-1)


def intefunc1(omega,Gamma,tb,beta,mu):
	return spectral_bath(omega,Gamma,tb)*nbar(omega,beta,mu)



def evolfunc(state,HS,epsilon,c,A,constantjn,constantjn_1):
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

glist=np.linspace(0,0.5,51)


c=np.empty((2,2),dtype=np.double)

c[0,0]=1/np.sqrt(2)
c[0,1]=1/np.sqrt(2)
c[1,0]=-1/np.sqrt(2)
c[1,1]=1/np.sqrt(2)


value1=[]
value2=[]

for g in glist:


	w=[]

	w.append(w0-g)
	w.append(w0+g)

	
	H=w0*(a1.dag()*a1+a2.dag()*a2)+g*(a1.dag()*a2+a1*a2.dag())


	pvaluej=np.empty((2,2),dtype=np.cdouble) 
	pvaluejn=np.empty((2,2),dtype=np.cdouble)

	pvaluejn[0,0]=(-1.0j/(2*np.pi))*integrate.quad(intefunc1,-2*tb,2*tb,args=(Gamma1,tb,beta1,mu1),weight='cauchy',wvar=w[0])[0]  #bath1, w1
	pvaluejn[0,1]=(-1.0j/(2*np.pi))*integrate.quad(intefunc1,-2*tb,2*tb,args=(Gamma1,tb,beta1,mu1),weight='cauchy',wvar=w[1])[0]	#bath1,w2
	pvaluejn[1,0]=(-1.0j/(2*np.pi))*integrate.quad(intefunc1,-2*tb,2*tb,args=(Gamma2,tb,beta2,mu2),weight='cauchy',wvar=w[0])[0] #bath2, w1
	pvaluejn[1,1]=(-1.0j/(2*np.pi))*integrate.quad(intefunc1,-2*tb,2*tb,args=(Gamma2,tb,beta2,mu2),weight='cauchy',wvar=w[1])[0]

	
	pvaluej[0,0]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath,-2*tb,2*tb,args=(Gamma1,tb),weight='cauchy',wvar=w[0])[0]  #bath1, w1
	pvaluej[0,1]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath,-2*tb,2*tb,args=(Gamma1,tb),weight='cauchy',wvar=w[1])[0]	#bath1,w2
	pvaluej[1,0]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath,-2*tb,2*tb,args=(Gamma2,tb),weight='cauchy',wvar=w[0])[0] #bath2, w1
	pvaluej[1,1]=(-1.0j/(2*np.pi))*integrate.quad(spectral_bath,-2*tb,2*tb,args=(Gamma2,tb),weight='cauchy',wvar=w[1])[0]
	
	print("Checking pvaluej values")

	print(pvaluej[0,0],1.0j*Gamma1*w[0]/(4*tb))
	print(pvaluej[0,1],1.0j*Gamma1*w[1]/(4*tb))
	print(pvaluej[1,0],1.0j*Gamma2*w[0]/(4*tb))
	print(pvaluej[1,1],1.0j*Gamma2*w[1]/(4*tb))	
	
	print("Checking done")

	constantjn_1=np.empty((2,2),dtype=np.cdouble)
	constantjn=np.empty((2,2),dtype=np.cdouble)

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

	
	initialstate=tensor(basis(N,0),basis(N,0))
	initial=initialstate*initialstate.dag()

	states=[]
	states.append(initial)

	h=0.5
	tmax=250
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

	print("value of g is ",g,"\n Trace distance between wi,wi+1 the end of iteration =",tracedist(wi_1,wi), "trace of final state ",wi_1.tr())
	

#	print("printing trace")
#	for k in range(steps):
#		print(k,states[k].tr())

	oper1=a1.dag()*a1
	oper2=1.0j*g*(a1.dag()*a2-a2.dag()*a1)

	value1.append(expect(oper1,states[-1]))
	value2.append(expect(oper2,states[-1]))	
	
	print("value of g is ",g,"\n a1.dag()*a1=",expect(oper1,states[-1]),"   Current =",expect(oper2,states[-1]))


plt.plot(glist,value1)
plt.xlabel("g")
plt.ylabel("<a1.dag()*a1>")
plt.title("Redfield result for fig2_b of Archak's paper")
plt.show()

plt.plot(glist,value2)
plt.xlabel("g")
plt.ylabel("I")
plt.title("Redfield result for fig2_a of Archak's paper")
plt.show()


