# Code to compute the steady state values of observables via the Langevin Method. Setup is the same as the paper
# The formulae are given in the paper, but are OFF by factors of epsilon. Eq 18 has an epsilon^2 term, Eq 19, has an extra epsilon term



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
mu2=-2.5
beta1=beta
beta2=beta
Gamma1=1
Gamma2=4
tb=1
g=1.1




def spectral_bath(omega,Gamma, tb): #compute spectral bath function
    if (omega <= -2*tb):
        return 0
    if (omega >= 2*tb):
        return 0
    return Gamma*np.sqrt(1-(omega*omega)/(4*tb*tb))


def nbar(omega, beta, mu): #computes bosonic occupation numbers
    return 1/(np.exp(beta*(omega-mu))-1)


def intefunc1(omega,Gamma,tb,beta,mu): #computes J*n 
    if (omega <= -2*tb):
        return 0
    if (omega >= 2*tb):
        return 0
    return spectral_bath(omega,Gamma,tb)*nbar(omega,beta,mu)


def delta(omega_0,Gamma,tb): #Computes the Cauchy P value that appears in the Langevin Calculations. We directly use the analytical form when we can..
    
    if (omega_0 < -2*tb):
        print("Problem in delta evaluation")
        temp=(-1/(2*np.pi))*integrate.quad(spectral_bath,-2*tb,2*tb,args=(Gamma,tb),weight='cauchy',wvar=omega_0)[0]
        return temp
    elif (omega_0 > 2*tb):
        print("Problem in delta evaluation")
        temp=(-1/(2*np.pi))*integrate.quad(spectral_bath,-2*tb,2*tb,args=(Gamma,tb),weight='cauchy',wvar=omega_0)[0]
        return temp
    else:
        return Gamma*omega_0/(4*tb) #The analytic formula as long as omega is inside bath spectrum..
    

def K_mod(omega,omega_0,Gamma,tb,epsilon): # |K|^2 Numerator factor that appears in a1.dag()*a1 calculations
    temp1=omega_0-omega -0.5j*epsilon*epsilon*spectral_bath(omega,Gamma,tb) + epsilon*epsilon*delta(omega,Gamma,tb)
    return temp1*np.conj(temp1)

def B(omega,omega_0,Gamma,tb,epsilon): # Appears in a1dag*a2 calculations, and M
    temp=omega_0-omega-0.5j*epsilon*epsilon*spectral_bath(omega,Gamma,tb) + epsilon*epsilon*delta(omega,Gamma,tb)
    return temp

def M_mod(omega,omega_0,Gamma1,Gamma2,tb,epsilon): # |M|^2 that appears in the denominator 
    temp1=omega_0-omega -0.5j*epsilon*epsilon*spectral_bath(omega,Gamma1,tb)+epsilon*epsilon*delta(omega,Gamma1,tb)
    temp2=omega_0-omega -0.5j*epsilon*epsilon*spectral_bath(omega,Gamma2,tb)+epsilon*epsilon*delta(omega,Gamma2,tb)
    temp=temp1*temp2-g*g
    
    return temp*np.conj(temp)



def n1_integral(omega,omega_0,g,tb,epsilon,Gamma1,Gamma2,beta1,beta2,mu1,mu2): #Needs a factor of epsilon*epsilon/2pi sitting outside
    temp1 = K_mod(omega,omega_0,Gamma2,tb,epsilon)*spectral_bath(omega,Gamma1,tb)*nbar(omega,beta1,mu1)
    temp2 = g*g*spectral_bath(omega,Gamma2,tb)*nbar(omega,beta2,mu2) 
    return (temp1+temp2)/M_mod(omega,omega_0,Gamma1,Gamma2,tb,epsilon)


def n2_integral(omega,omega_0,g,tb,epsilon,Gamma1,Gamma2,beta1,beta2,mu1,mu2):#Needs a factor of epsilon*epsilon/2pi sitting outside
    temp1 = K_mod(omega,omega_0,Gamma1,tb,epsilon)*spectral_bath(omega,Gamma2,tb)*nbar(omega,beta2,mu2)
    temp2 = g*g*spectral_bath(omega,Gamma1,tb)*nbar(omega,beta1,mu1) 
    return (temp1+temp2)/M_mod(omega,omega_0,Gamma1,Gamma2,tb,epsilon)

def a1daga2_integral_real(omega,omega_0,g,tb,epsilon,Gamma1,Gamma2,beta1,beta2,mu1,mu2): #Needs a factor of -g*epsilon*epislon/2pi sitting outside
    temp1=(omega_0-omega+epsilon*epsilon*delta(omega,Gamma1,tb))*spectral_bath(omega,Gamma2,tb)*nbar(omega,beta2,mu2)
    temp2=(omega_0-omega+epsilon*epsilon*delta(omega,Gamma2,tb))*spectral_bath(omega,Gamma1,tb)*nbar(omega,beta1,mu1)
    temp3= (temp1+temp2)/M_mod(omega,omega_0,Gamma1,Gamma2,tb,epsilon)
    return temp3



def a1daga2_integral_imag(omega,omega_0,g,tb,epsilon,Gamma1,Gamma2,beta1,beta2,mu1,mu2): #Needs a factor of -g*epsilon*epislon/2pi sitting outside
    temp1=B(omega,omega_0,Gamma1,tb,epsilon)*spectral_bath(omega,Gamma2,tb)*nbar(omega,beta2,mu2)
    temp2=np.conj(B(omega,omega_0,Gamma2,tb,epsilon))*spectral_bath(omega,Gamma1,tb)*nbar(omega,beta1,mu1)
    temp3= (temp1+temp2)/M_mod(omega,omega_0,Gamma1,Gamma2,tb,epsilon)
    return np.imag(temp3)


def current_integral(omega,omega_0,g,tb,epsilon,Gamma1,Gamma2,beta1,beta2,mu1,mu2): #Needs factor of epsilon^4, g^2/2pi outside
    temp1=spectral_bath(omega,Gamma1,tb)*spectral_bath(omega,Gamma2,tb)*nbar(omega,beta2,mu2)
    temp2=spectral_bath(omega,Gamma1,tb)*spectral_bath(omega,Gamma2,tb)*nbar(omega,beta1,mu1)
    return (temp2-temp1)/M_mod(omega,omega_0,Gamma1,Gamma2,tb,epsilon)
    
    
expect_n1=integrate.quad(n1_integral,-2*tb,2*tb,args=(w0,g,tb,epsilon,Gamma1,Gamma2,beta1,beta2,mu1,mu2))
n1value=(epsilon*epsilon/(2*np.pi))*expect_n1[0]

expect_n2=integrate.quad(n2_integral,-2*tb,2*tb,args=(w0,g,tb,epsilon,Gamma1,Gamma2,beta1,beta2,mu1,mu2))
n2value=(epsilon*epsilon/(2*np.pi))*expect_n2[0]


expect_a1daga2_real=integrate.quad(a1daga2_integral_real,-2*tb,2*tb,args=(w0,g,tb,epsilon,Gamma1,Gamma2,beta1,beta2,mu1,mu2))
expect_a1daga2_imag=integrate.quad(a1daga2_integral_imag,-2*tb,2*tb,args=(w0,g,tb,epsilon,Gamma1,Gamma2,beta1,beta2,mu1,mu2))

a1daga2value=(-g*epsilon*epsilon/(2*np.pi))*(expect_a1daga2_real[0]+1.0j*expect_a1daga2_imag[0])
a2daga1value=np.conj(a1daga2value)

expect_current=integrate.quad(current_integral,-2*tb,2*tb,args=(w0,g,tb,epsilon,Gamma1,Gamma2,beta1,beta2,mu1,mu2))
currentvalue=epsilon*epsilon*epsilon*epsilon*g*g*expect_current[0]/(2*np.pi)

print(1.0j*g*(a1daga2value-a2daga1value))

A1dagA1=0.5*(n1value+n2value-a1daga2value-a2daga1value)
A2dagA2=0.5*(n1value+n2value+a1daga2value+a2daga1value)
   