#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:54:38 2018

@author: MarcoAntonio
"""
import math
import numpy as np
from numpy import linalg as LA
import random



vev=246/math.sqrt(2)
def Hermitian_downYukawa(epsilon,epsilon1,rho,d1,d2,d3,d4,d5):
    return np.array([[d1**2*epsilon1**2*rho**2, -1j*d1*d2*epsilon**2*epsilon1*rho**2, -d1*d3*epsilon*epsilon1*rho**2]\
                     ,[1j*d1*d2*epsilon**2*epsilon1*rho**2 ,d4**2*epsilon**2*rho**2 + d2**2*epsilon**4*rho**2+d1**2*epsilon1**2*rho**2, -d4*d5*epsilon*rho**2-1j*d2*d3*epsilon**3*rho**2]\
                     ,[-d1*d3*epsilon*epsilon1*rho**2  , -d4*d5*epsilon*rho**2 +1j*d2*d3*epsilon**3*rho**2, d5**2*rho**2+d3**2*epsilon**2*rho**2]])


def Hermitian_upYukawa(epsilon,epsilon1,rho,u1,u2,u3,u4,u5):
    return np.array([[u1**2*epsilon1**2*rho**2, -1j*u1*u2*epsilon**2*epsilon1*rho**2, -u1*u3*epsilon*epsilon1*rho**2]\
                     ,[1j*u1*u2*epsilon**2*epsilon1*rho**2 ,u4**2*epsilon**2 + u2**2*epsilon**4*rho**2+u1**2*epsilon1**2*rho**2, -u4*u5*epsilon-1j*u2*u3*epsilon**3*rho**2]\
                     ,[-u1*u3*epsilon*epsilon1*rho**2  , -u4*u5*epsilon +1j*u2*u3*epsilon**3*rho**2, u5**2+u3**2*epsilon**2*rho**2]])



def massesd(epsilon,epsilon1,rho,d1,d2,d3,d4,d5):
    w, v= LA.eigh(Hermitian_downYukawa(epsilon,epsilon1,rho,d1,d2,d3,d4,d5))
    mass=[]
    for i in w:
        mass.append(vev*i**(1/2))
    
    return mass

def massesu(epsilon,epsilon1,rho,d1,d2,d3,d4,d5):
    w, v= LA.eigh(Hermitian_upYukawa(epsilon,epsilon1,rho,d1,d2,d3,d4,d5))
    mass=[]
    for i in w:
        mass.append(vev*i**(1/2))
    
    return mass
    

#print(massesd(0.02,0.004,0.2,1.0,1.0,1.0,1.0,1.0))



def fit_down(epsilon,epsilon1,rho,N):
    fit1=[]
    for i in range(N):
        d1=(-1)**random.choice([0,1])*(0.01 + (10-0.01)*random.random())
        d2=(-1)**random.choice([0,1])*(0.01 + (10-0.01)*random.random())
        d3=(-1)**random.choice([0,1])*(0.01 + (10-0.1)*random.random())
        d4=(-1)**random.choice([0,1])*(0.01 + (10-0.1)*random.random())
        d5=(-1)**random.choice([0,1])*(0.01 + (10-0.1)*random.random())
        if (4.7-3*0.4)*10**(-3)<massesd(epsilon,epsilon1,rho,d1,d2,d3,d4,d5)[0]<(4.7+3*0.5)*10**(-3) \
        and (96-3*4)*10**(-3)<massesd(epsilon,epsilon1,rho,d1,d2,d3,d4,d5)[1]<(96+ 3*8)*10**(-3) \
        and (4.18-3*0.03)<massesd(epsilon,epsilon1,rho,d1,d2,d3,d4,d5)[2]<(4.18+ 3*0.04):
            fit1.append([d1,d2,d3,d4,d5])
    return fit1

      

   
        
def fit_up(epsilon,epsilon1,rho,N):
    fit2=[]
    for i in range(N):  
        u1=((-1)**random.choice([0,1])*(0.01 + (10-0.1)*random.random()))
        u2=((-1)**random.choice([0,1])*(0.01 + (10-0.1)*random.random()))
        u3=((-1)**random.choice([0,1])*(0.01 + (10-0.1)*random.random()))
        u4=((-1)**random.choice([0,1])*(0.01 + (10-0.1)*random.random()))
        u5=((-1)**random.choice([0,1])*(0.9 + (1.1-0.9)*random.random()))
        if (2.2-3*0.4)*10**(-3)<massesu(epsilon,epsilon1,rho,u1,u2,u3,u4,u5)[0]<(2.2+3*0.6)*10**(-3) \
        and (1.27-3*0.03)<massesu(epsilon,epsilon1,rho,u1,u2,u3,u4,u5)[1]<(1.27+ 3*0.03) \
        and (173.21-3*0.71)<massesu(epsilon,epsilon1,rho,u1,u2,u3,u4,u5)[2]<(173.21+ 3*0.71):
            fit2.append([u1,u2,u3,u4,u5])
    return fit2
        
        


def totalfit(u,d,N):
    fit=[]
    for i in range(N):
        epsilon,epsilon1,rho = 0.0001 + (0.1-0.0001)*random.random(), 0.0001 + (0.1-0.0001)*random.random(), 0.0001 + (0.1-0.0001)*random.random()
        if len(fit_up(epsilon,epsilon1,rho,u))!=0 and len(fit_down(epsilon,epsilon1,rho,d))!=0:
            fit.append([epsilon, epsilon1, rho])
    return fit
         
    
    


   

