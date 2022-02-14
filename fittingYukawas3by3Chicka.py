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
mu=2.3*10**(-3)
deltamu=0.6*10**(-3)
mc=1.275
deltamc=0.025
mt=160
deltamt=4.5
md=4.8*10**(-3)
deltamd=0.4*10**(-3)
ms=95*10**(-3)
deltams=5*10**(-3)
mb=4.18
deltamb=0.03
me=5.11*10**(-4)
deltame=5.11*10**(-6)
mmu=0.106
deltammu=0.106*10**(-2)
mtau=1.78
deltamtau=1.78*10**(-2)
Vus=0.225
deltaVus=0.225*10**(-2)
Vub=3.55*10**(-3)
deltaVub=0.15*10**(-3)
Vcb=4.14*10**(-2)
deltaVcb=1.2*10**(-3)



def YYdaggerup(u1,u2,u3,u4,u5,epsilon1,epsilon,rho):
    return np.array([[u1**2*epsilon1**2*rho**2, u1*u2*epsilon*epsilon1*rho**2, u1*u4*epsilon*epsilon1*rho]\
                    ,[u1*u2*epsilon*epsilon1*rho**2, u2**2*epsilon**2*rho**2+u1**2*epsilon1**2*rho**2+u3**2*epsilon**2, u3*u5*epsilon + u2*u4*epsilon**2*rho]\
                    ,[u1*u4*epsilon*epsilon1*rho, u3*u5*epsilon +u2*u4*epsilon**2*rho, u5**2+u4**2*epsilon**2]])

#print(YYdaggerup(1,1,1,1,1,0.004,0.02,0.02))
    
def YYdaggerdown(d1,d2,d3,d4,d5,epsilon1,epsilon,zet):
    return np.array([[d1**2*epsilon1**2*zet**2, d1*d2*epsilon*epsilon1*zet**2 , d1*d4*epsilon*epsilon1*zet**2]\
                    ,[d1*d2*epsilon*epsilon1*zet**2, d1**2*epsilon1**2*zet**2 + d3**2*epsilon**2*zet**2 + d2**2*epsilon**2*zet**2 ,d3*d5*epsilon*zet**2 + d2*d4*epsilon**2*zet**2 ]\
                    ,[d1*d4*epsilon*epsilon1*zet**2, d3*d5*epsilon*zet**2 + d2*d4*epsilon**2*zet**2 ,d5**2*zet**2 + d4**2*epsilon**2*zet**2 ]])

#print(YYdaggerdown(1,1,1,1,1,0.004,0.02,0.02))  

def massesup(u1,u2,u3,u4,u5, epsilon1,epsilon,rho):
    w, v= LA.eigh(YYdaggerup(u1,u2,u3,u4,u5, epsilon1,epsilon,rho))
    mass=[]
    for i in w:
        mass.append(math.sqrt(i)*vev)
    
    return mass

#print(massesup(1,1,1,1,1,0.004,0.02,0.02))


def massesdown(d1,d2,d3,d4,d5, epsilon1,epsilon,zet):
    w, v= LA.eigh(YYdaggerdown(d1,d2,d3,d4,d5, epsilon1,epsilon,zet))
    mass=[]
    for i in w:
        mass.append(math.sqrt(i)*vev)
    
    return mass


#print(massesdown(1,1,1,1,1,0.004,0.02,0.02))
    
def massesleptons(l1,l2,l3,l4,l5, epsilon1,epsilon,zet):
    return massesdown(l1,l2,l3,l4,l5, epsilon1,epsilon,zet)



def Uup(u1,u2,u3,u4,u5,epsilon1,epsilon,rho):
    w, v= LA.eigh(YYdaggerup(u1,u2,u3,u4,u5,epsilon1,epsilon,rho))
    return np.array(v)


#print(Uup(1,1,1,1,1,0.004,0.02,0.02))


def Udown(d1,d2,d3,d4,d5,epsilon1,epsilon,zet):
    w, v= LA.eigh(YYdaggerdown(d1,d2,d3,d4,d5,epsilon1,epsilon,zet))
    return np.array(v)
        


def diagYukawaup(u1,u2,u3,u4,u5,epsilon1,epsilon,rho):
    diag= np.matmul(Uup(u1,u2,u3,u4,u5,epsilon1,epsilon,rho).transpose().conj(),np.matmul(YYdaggerup(u1,u2,u3,u4,u5,epsilon1,epsilon,rho),Uup(u1,u2,u3,u4,u5,epsilon1,epsilon,rho)))
    return diag.diagonal()

#print(diagYukawaup(1,1,1,1,1,0.004,0.02,0.02))
#print(massesup(1,1,1,1,1,0.004,0.02,0.02)[0]**2/vev**2)


def diagYukawadown(d1,d2,d3,d4,d5,epsilon1,epsilon,zet):
    diag= np.matmul(Udown(d1,d2,d3,d4,d5,epsilon1,epsilon,zet).conj().transpose(),np.matmul(YYdaggerdown(d1,d2,d3,d4,d5,epsilon1,epsilon,zet),Udown(d1,d2,d3,d4,d5,epsilon1,epsilon,zet)))
    return diag.diagonal()
    
#print(diagYukawadown(1,1,1,1,1,0.004,0.02,0.02))
#print(massesdown(1,1,1,1,1,0.004,0.02,0.02)[2]**2/vev**2)    


def VCKM(u1,u2,u3,u4,u5,d1,d2,d3,d4,d5,epsilon1,epsilon,rho,zet):
    return np.matmul(Uup(u1,u2,u3,u4,u5,epsilon1,epsilon,rho).transpose().conj(),Udown(d1,d2,d3,d4,d5,epsilon1,epsilon,zet))


#print(VCKM(1,1,1,1,1,1,1,1,1,1,0.004,0.02,0.02,0.02).diagonal())
#print(abs(VCKM(1,1,1,1,1,1,1,1,1,1,0.004,0.02,0.02,0.02).diagonal()[0]))

def chi_square_nolog(u1,u2,u3,u4,u5,d1,d2,d3,d4,d5,l1,l2,l3,l4,l5,epsilon1,epsilon,rho,zet):
    return (massesup(u1,u2,u3,u4,u5,epsilon1,epsilon,rho)[0]-mu)**2/deltamu**2 \
           +(massesup(u1,u2,u3,u4,u5,epsilon1,epsilon,rho)[1]-mc)**2/deltamc**2 \
           +(massesup(u1,u2,u3,u4,u5,epsilon1,epsilon,rho)[2]-mt)**2/deltamt**2 \
           +(massesdown(d1,d2,d3,d4,d5,epsilon1,epsilon,zet)[0]-md)**2/deltamd**2  \
           +(massesdown(d1,d2,d3,d4,d5,epsilon1,epsilon,zet)[1]-ms)**2/deltams**2  \
           +(massesdown(d1,d2,d3,d4,d5,epsilon1,epsilon,zet)[2]-mb)**2/deltamb**2  \
           +(massesleptons(l1,l2,l3,l4,l5,epsilon1,epsilon,zet)[0]-me)**2/deltame**2  \
           +(massesleptons(l1,l2,l3,l4,l5,epsilon1,epsilon,zet)[1]-mmu)**2/deltammu**2  \
           +(massesleptons(l1,l2,l3,l4,l5,epsilon1,epsilon,zet)[2]-mtau)**2/deltamtau**2  \
           +(abs(VCKM(u1,u2,u3,u4,u5,d1,d2,d3,d4,d5,epsilon1,epsilon,rho,zet)[0][1])-Vus)**2/deltaVus**2  \
           +(abs(VCKM(u1,u2,u3,u4,u5,d1,d2,d3,d4,d5,epsilon1,epsilon,rho,zet)[0][2])-Vub)**2/deltaVub**2  \
           +(abs(VCKM(u1,u2,u3,u4,u5,d1,d2,d3,d4,d5,epsilon1,epsilon,rho,zet)[1][2])-Vcb)**2/deltaVcb**2
           
         
  
def chi_square_log(u1,u2,u3,u4,u5,d1,d2,d3,d4,d5,l1,l2,l3,l4,l5,epsilon1,epsilon,rho,zet):
    return (massesup(u1,u2,u3,u4,u5,epsilon1,epsilon,rho)[0]-mu)**2/deltamu**2 \
           +(massesup(u1,u2,u3,u4,u5,epsilon1,epsilon,rho)[1]-mc)**2/deltamc**2 \
           +(massesup(u1,u2,u3,u4,u5,epsilon1,epsilon,rho)[2]-mt)**2/deltamt**2 \
           +(massesdown(d1,d2,d3,d4,d5,epsilon1,epsilon,zet)[0]-md)**2/deltamd**2  \
           +(massesdown(d1,d2,d3,d4,d5,epsilon1,epsilon,zet)[1]-ms)**2/deltams**2  \
           +(massesdown(d1,d2,d3,d4,d5,epsilon1,epsilon,zet)[2]-mb)**2/deltamb**2  \
           +(massesleptons(l1,l2,l3,l4,l5,epsilon1,epsilon,zet)[0]-me)**2/deltame**2  \
           +(massesleptons(l1,l2,l3,l4,l5,epsilon1,epsilon,zet)[1]-mmu)**2/deltammu**2  \
           +(massesleptons(l1,l2,l3,l4,l5,epsilon1,epsilon,zet)[2]-mtau)**2/deltamtau**2  \
           +(abs(VCKM(u1,u2,u3,u4,u5,d1,d2,d3,d4,d5,epsilon1,epsilon,rho,zet)[0][1])-Vus)**2/deltaVus**2  \
           +(abs(VCKM(u1,u2,u3,u4,u5,d1,d2,d3,d4,d5,epsilon1,epsilon,rho,zet)[0][2])-Vub)**2/deltaVub**2  \
           +(abs(VCKM(u1,u2,u3,u4,u5,d1,d2,d3,d4,d5,epsilon1,epsilon,rho,zet)[1][2])-Vcb)**2/deltaVcb**2  \
           +(math.log(abs(u1)))**2/math.log(3)**2 +(math.log(abs(u2)))**2/math.log(3)**2 \
           +(math.log(abs(u3)))**2/math.log(3)**2                                        \
           +(math.log(abs(u5)))**2/math.log(3)**2 +(math.log(abs(d1)))**2/math.log(3)**2 \
           +(math.log(abs(d2)))**2/math.log(3)**2 +(math.log(abs(d3)))**2/math.log(3)**2 \
           +(math.log(abs(d4)))**2/math.log(3)**2 +(math.log(abs(d5)))**2/math.log(3)**2 \
           +(math.log(abs(l1)))**2/math.log(3)**2 +(math.log(abs(l2)))**2/math.log(3)**2 \
           +(math.log(abs(l3)))**2/math.log(3)**2 +(math.log(abs(l4)))**2/math.log(3)**2 \
           +(math.log(abs(l5)))**2/math.log(3)**2 
           
           
         
           
           
          
def chi_square_log1(params):
    u1,u2,u3,u4,u5,d1,d2,d3,d4,d5,l1,l2,l3,l4,l5,epsilon1,epsilon,rho,zet=params
    return (massesup(u1,u2,u3,u4,u5,epsilon1,epsilon,rho)[0]-mu)**2/deltamu**2 \
           +(massesup(u1,u2,u3,u4,u5,epsilon1,epsilon,rho)[1]-mc)**2/deltamc**2 \
           +(massesup(u1,u2,u3,u4,u5,epsilon1,epsilon,rho)[2]-mt)**2/deltamt**2 \
           +(massesdown(d1,d2,d3,d4,d5,epsilon1,epsilon,zet)[0]-md)**2/deltamd**2  \
           +(massesdown(d1,d2,d3,d4,d5,epsilon1,epsilon,zet)[1]-ms)**2/deltams**2  \
           +(massesdown(d1,d2,d3,d4,d5,epsilon1,epsilon,zet)[2]-mb)**2/deltamb**2  \
           +(massesleptons(l1,l2,l3,l4,l5,epsilon1,epsilon,zet)[0]-me)**2/deltame**2  \
           +(massesleptons(l1,l2,l3,l4,l5,epsilon1,epsilon,zet)[1]-mmu)**2/deltammu**2  \
           +(massesleptons(l1,l2,l3,l4,l5,epsilon1,epsilon,zet)[2]-mtau)**2/deltamtau**2  \
           +(abs(VCKM(u1,u2,u3,u4,u5,d1,d2,d3,d4,d5,epsilon1,epsilon,rho,zet)[0][1])-Vus)**2/deltaVus**2  \
           +(abs(VCKM(u1,u2,u3,u4,u5,d1,d2,d3,d4,d5,epsilon1,epsilon,rho,zet)[0][2])-Vub)**2/deltaVub**2  \
           +(abs(VCKM(u1,u2,u3,u4,u5,d1,d2,d3,d4,d5,epsilon1,epsilon,rho,zet)[1][2])-Vcb)**2/deltaVcb**2  \
           +(math.log(abs(u1)))**2/math.log(3)**2 +(math.log(abs(u2)))**2/math.log(3)**2 \
           +(math.log(abs(u3)))**2/math.log(3)**2                                       \
           +(math.log(abs(u5)))**2/math.log(3)**2 +(math.log(abs(d1)))**2/math.log(3)**2 \
           +(math.log(abs(d2)))**2/math.log(3)**2 +(math.log(abs(d3)))**2/math.log(3)**2 \
           +(math.log(abs(d4)))**2/math.log(3)**2 +(math.log(abs(d5)))**2/math.log(3)**2 \
           +(math.log(abs(l1)))**2/math.log(3)**2 +(math.log(abs(l2)))**2/math.log(3)**2 \
           +(math.log(abs(l3)))**2/math.log(3)**2 +(math.log(abs(l4)))**2/math.log(3)**2 \
           +(math.log(abs(l5)))**2/math.log(3)**2 

#print(chi_square_log(1.005,1.01,-0.458,0,0.369,1.005,-0.64,1.024,2.397,-0.676,0.847,-0.633,-1.193,-1.199,-0.847,0.004,0.131,0.02,0.011))

#print(chi_square_log(1.131,0.921,-0.575,0,0.628,1.162,-0.631,1.024,2.375,-0.931,0.651,-0.710,-1.242,-1.244,-0.637,0.005,0.182,0.029,0.014))
           

           