"""
Created on Thu May 21 07:58:29 2020
Exact numerics for the Archaks case. 
@author: devashish
"""

#implelemtsn exact numerics for FERMIONS.


import numpy as np
from qutip import *
from scipy.linalg import expm
import matplotlib.pyplot as plt


#Declaring parameters. All variables are same as those in paper.
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
g=0.7
gamma1=np.sqrt(Gamma1*tb/2)
gamma2=np.sqrt(Gamma2*tb/2)




R=511 #size of Bath


#creating Global Hamiltonian
H=np.zeros((2*R+2,2*R+2),dtype=np.complex128)


#index 0 to R-1 is bath 1, R is point 1, R+1 is point 2, 2*R+! is last bath.

for i in range(R-1):
    H[i,i+1]=tb
    H[i+1,i]=tb

for i in range(R+2,2*R+1):
    H[i,i+1]=tb
    H[i+1,i]=tb

#Bath hamltonians done
    
H[R-1,R]=epsilon*gamma1
H[R,R-1]=epsilon*gamma1

H[R+1,R+2]=epsilon*gamma2
H[R+2,R+1]=epsilon*gamma2

H[R,R]=w0
H[R+1,R+1]=w0
H[R,R+1]=g
H[R+1,R]=g

#Full H created.




#Creating bath Hamiltonian (for computing initial correlation matrix)
H_B=np.zeros((R,R),dtype=np.complex128)

for i in range(R-1):
    H_B[i,i+1]=tb
    H_B[i+1,i]=tb
    

modes, matrix =np.linalg.eigh(H_B)


#matrix is actually U (based on your theory)
#bath_matrix (H_jk) crreated.. We must build the correlation matrid
    #D=d *d.dag() now.. 
   
U=matrix
Udag=matrix.conj().T  

N_op=np.zeros((R,R),dtype=np.complex128)
for i in range(R):
    N_op[i,i]=1/(np.exp(beta*(modes[i]-mu))+1)

D_bath=np.identity(R,dtype=np.complex128) - U@N_op@Udag
#See the pdf if this computation is unclear. 


#setting up initial state.
system_state= np.identity(2,dtype=np.complex128)

D=np.zeros((2*R+2,2*R+2),dtype=np.complex128)

D[0:R,0:R]=D_bath
D[R:R+2,R:R+2]=system_state
D[R+2:2*R+2,R+2:2*R+2]=D_bath


# FUll D is ready.


tlist=np.linspace(1,200,10)
value_list=[]
value1_list=[]

for t in tlist:
    V=expm(-1.0j*t*H)
    D_final=V@D@(V.conj().T)


    
    expect=np.zeros((2,2),dtype=np.complex128) #expect[i,j]=<a_i^dagger a_j>
    
    
    #we use commutation relations for fermions to get expect from the correlation matrix.
    expect[0,0]= 1 - D_final[R,R] #a1.dag*a1
    expect[1,1]= 1 - D_final[R+1,R+1]  #a2.dag*a2
    expect[0,1]= -D_final[R+1,R]   #a1.dag()*a2
    expect[1,0]= -D_final[R,R+1] #a2.dag()a2
    
    
    #We can now compute a large number of quantities. We compute the occupation numbers of
    #Eigenmodes below.

    final=0.5*(expect[0,0]+expect[1,1]-expect[0,1]-expect[1,0])  #A1.dag*A1
    final_1=0.5*(expect[0,0]+expect[1,1]+expect[0,1]+expect[1,0]) #A2.dag*A2
    value_list.append(final)
    value1_list.append(final_1)
    print("time = ",t,"A1dagA1",final,"A2dagA2",final_1)


#We plot eigenmodes here.

plt.title("g={}".format(g))
plt.plot(tlist,value_list,label='A1.dag*A1')
plt.plot(tlist,value1_list,label='A2.dag*A2')
plt.xlabel("time")
plt.legend()





