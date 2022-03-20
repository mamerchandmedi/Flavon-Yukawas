#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:21:28 2022

@author: marcoantoniomerchandmedina
"""


import numpy as np
from cosmoTransitions import generic_potential_1
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
import scipy.integrate as integrate
from scipy import interpolate, special


####Some definitions##
v2 = 246.**2 
mh=125
v=246
alpha=1/137 #fine structure constant
sinthw=np.sqrt(0.223) #sine of Weinberg angle
g1=np.sqrt(4*np.pi*alpha/(1-sinthw**2)) ## U(1)_Y gauge coupling constant
g=np.sqrt(4*np.pi*alpha)/sinthw #SU(2)_L gauge coupling constant
Mplanck=2.4*10**18 #reduced planck mass
cs=1/3**0.5 ##Sound speed constant

###Experimental data for fermion masses and mixing angles, taken from past project
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




#nQ=(3,2,0)
#nu=(3,2,0)
#nd=(1,0,0)

nQ=(3,2,0)
nu=(3,2,0)
nd=(2,1,1)

#nQ=(2,1,0)
#nu=(5,2,0)
#nd=(5,4,2)


def Yukawa_mat(yij,nQ,nq,epsilon):
    """Generic Yukawa matrix given numerical coefficients, U(1) charges and 
    ratio of flavon vev to flavor scale.
    --------------
    parameters
    yij: array obj, matrix of coeffcients. Expected to be order 1 as the couplings decend down from the UV
    nQ: tuple, generic charges of left handed quark (or lepton) doublet
    nq: tuple, generic charges of right handed quark (or lepton) doublet
    epsilon: ratio of flavon vev to flavor scale
    -------
    returns
    array, the full Yukawa couplings after EWSB"""
    columns=[]
    for i in range(3):
        rows=[]
        for j in range(3):
            rows.append(yij[i,j]*epsilon**(nQ[i]+nq[j]))
        columns.append(rows)
    return np.array(columns)


def massesup(yij,nQ,nq,epsilon):
    """Returns the masses given numerical coefficients, U(1) charges and 
    ratio of flavon vev to flavor scale
    --------------
    parameters
    yij: array obj, matrix of coeffcients. Expected to be order 1 as the couplings decend down from the UV
    nQ: tuple, generic charges of left handed quark (or lepton) doublet
    nq: tuple, generic charges of right handed quark (or lepton) doublet
    epsilon: ratio of flavon vev to flavor scale
    -------
    returns
    list, the masses"""
    YQ=Yukawa_mat(yij,nQ,nq,epsilon)*v/2**0.5   
    YQ_dagger=YQ.T
    YY_dagger=np.matmul(YQ,YQ_dagger)
    #w, vecs= np.linalg.eigh(YY_dagger)
    w = np.linalg.eigvalsh(YY_dagger)
    
    mass=[]
    for i in w:
        mass.append(np.sqrt(i))
    return mass


def chi_square_log(uparams,nQ,nq,epsilon):
    """Optimization function to find best fit value of the up sector"""
    u1,u2,u3,u4,u5,u6,u7,u8,u9=uparams
    yij=np.array([[u1,u2,u3],[u4,u5,u6],[u7,u8,u9]])
    return (massesup(yij,nQ,nq,epsilon)[0]-mu)**2/deltamu**2 \
           +(massesup(yij,nQ,nq,epsilon)[1]-mc)**2/deltamc**2 \
           +(massesup(yij,nQ,nq,epsilon)[2]-mt)**2/deltamt**2 \
           +(np.log(abs(u1)))**2/np.log(3)**2 \
           +(np.log(abs(u2)))**2/np.log(3)**2 +(np.log(abs(u3)))**2/np.log(3)**2 \
           +(np.log(abs(u4)))**2/np.log(3)**2 +(np.log(abs(u5)))**2/np.log(3)**2 \
           +(np.log(abs(u6)))**2/np.log(3)**2 +(np.log(abs(u7)))**2/np.log(3)**2 \
           +(np.log(abs(u8)))**2/np.log(3)**2 +(np.log(abs(u9)))**2/np.log(3)**2 


##chi-square function minimization
#this loop minimizes the chi-square function for random initial guesses and stops if it finds a good fit
#print("Fittig Yukawa couplings \n ...............")
#fit_sols=[]
#for i in range(100):
#    uparams=np.ones(9)+np.random.uniform(-.3,.3,9)
#    sol_fit=optimize.minimize(chi_square_log,x0=uparams,args=(nQ,nu,0.2))
#    if sol_fit.fun<=10: ###Insert some threshold for the chi-square minima
#        fit_sols.append(sol_fit)
#fit_sols_pd=pd.DataFrame(fit_sols)
#best_fit_loc=fit_sols_pd[fit_sols_pd["fun"]==fit_sols_pd["fun"].min()]
#best_fit=fit_sols[best_fit_loc.index[0]]
best_fit=np.load("./SCANS/Yukawa_fit_up_3.npy")

def fermion_masses_vectorized(X,M_f):
    """Returns the masses squared given position vector in field space"""
    h,s = np.asarray(X)
    Epsilon=s/M_f
    u1,u2,u3,u4,u5,u6,u7,u8,u9=best_fit
    (nQ1,nQ2,nQ3)=nQ
    (nu1,nu2,nu3)=nu
    YYdagger=h**2/2*np.array([[u1**2*Epsilon**(2*nQ1 + 2*nu1) + u2**2*Epsilon**(2*nQ1 + 2*nu2) + u3**2*Epsilon**(2*nQ1 + 2*nu3),
                        u1*u4*Epsilon**(nQ1 + nQ2 + 2*nu1) + u2*u5*Epsilon**(nQ1 + nQ2 + 2*nu2) +  u3*u6*Epsilon**(nQ1 + nQ2 + 2*nu3), 
                        u1*u7*Epsilon**(nQ1 + nQ3 + 2*nu1) + u2*u8*Epsilon**(nQ1 + nQ3 + 2*nu2) + u3*u9*Epsilon**(nQ1 + nQ3 + 2*nu3)], 
                       [u1*u4*Epsilon**(nQ1 + nQ2 + 2*nu1) + u2*u5*Epsilon**(nQ1 + nQ2 + 2*nu2) + u3*u6*Epsilon**(nQ1 + nQ2 + 2*nu3), 
                        u4**2*Epsilon**(2*nQ2 + 2*nu1) + u5**2*Epsilon**(2*nQ2 + 2*nu2) + u6**2*Epsilon**(2*nQ2 + 2*nu3), 
                        u4*u7*Epsilon**(nQ2 + nQ3 + 2*nu1) + u5*u8*Epsilon**(nQ2 + nQ3+2*nu2) +  u6*u9*Epsilon**(nQ2 + nQ3 + 2*nu3)], 
                       [u1*u7*Epsilon**(nQ1 + nQ3 + 2*nu1) + u2*u8*Epsilon**(nQ1 + nQ3 + 2*nu2) +  u3*u9*Epsilon**(nQ1 + nQ3 + 2*nu3), 
                        u4*u7*Epsilon**(nQ2 + nQ3 + 2*nu1) + u5*u8*Epsilon**(nQ2 + nQ3 + 2*nu2) + u6*u9*Epsilon**(nQ2 + nQ3 + 2*nu3), 
                        u7**2*Epsilon**(2*nQ3 + 2*nu1) + u8**2*Epsilon**(2*nQ3 + 2*nu2) + u9**2*Epsilon**(2*nQ3 + 2*nu3)]])
    YYdagger=np.rollaxis(YYdagger,0,len(YYdagger.shape))
    YYdagger=np.rollaxis(YYdagger,0,len(YYdagger.shape))
    evalss=np.linalg.eigvalsh(YYdagger)
    for _ in range(len(evalss.shape)-1):
        evalss=np.rollaxis(evalss,0,len(evalss.shape))
    return evalss

def my_fun(thefit=best_fit):
    class model1(generic_potential_1.generic_potential):
        def init(self, ms, theta , fs ):
            self.Ndim = 2
            self.renormScaleSq = v2
            #independent parameters
            self.ms = ms #flavon tree level renormalized mass
            self.theta = theta ##mixing angle 
            self.f = fs #flavon vvev
            self.Mf=self.f/0.2 #Flavor scale
            
            #dependent parameters
            self.lamh = 1/(4*v**2)*(mh**2 + ms**2 +(mh**2-ms**2)*np.cos(2*self.theta))
            self.lams = 1/(4*self.f**2)*(mh**2 + ms**2 + (ms**2 - mh**2)*np.cos(2*self.theta))
            self.lammix = 1/(self.f*v)*(mh-ms)*(mh+ms)*np.cos(self.theta)*np.sin(self.theta)
            self.muh = v**2*self.lamh + self.lammix*self.f**2/2
            self.mus = -v**2*self.lammix/2 - self.f**2*self.lams
        
        def forbidPhaseCrit(self, X):
            return any([np.array([X])[...,0] < -5.0, np.array([X])[...,1] < -5.0])
        
        def print_couplings(self):
            print("The tree level couplings are given by \n")
            print("mu_h= ",self.muh," mu_s=",self.mus," lam_hs=",self.lammix, " lam_s=",self.lams," lam_h=", self.lamh,"\n")
            print("The parameters of the model are \n")
            print("ms= ",self.ms," theta=",self.theta," f=",self.f,"\n")
            
            
        
        def V0(self, X):
            X = np.asanyarray(X)
            h, s = X[...,0], X[...,1]
            pot = -self.muh*h**2/2 + self.mus*s**2/2 + self.lammix*s**2*h**2/4 + self.lamh*h**4/4 + self.lams*s**4/4
            return pot
        
        def boson_massSq(self, X, T):
            X = np.array(X)
            h, s = X[...,0], X[...,1]
    
            #####Scalar thermal masses##
            Pi_h = T**2*(g1**2/16 + 3*g**2/16 + m.lamh/2 + best_fit[-1]**2/4 + self.lammix/24)
            Pi_s= T**2*(self.lammix/6 + self.lams/4)
         
            ##Scalar mass matrix##
            a= 3*h**2*self.lamh - v**2*self.lamh +  self.lammix/2*(s**2-self.f**2) + Pi_h
            b= self.lammix/2*(h**2-v**2) + self.lams*(3*s**2-self.f**2) + Pi_s
            cc=h*s*self.lammix 
            A=(a+b)/2
            B=1/2*np.sqrt((a-b)**2+4*cc**2)
            m1=A+B
            m2=A-B
            
            ####Gauge boson masses
            mWL = g**2*h**2/4 + 11/6*g**2*T**2
            ag=g**2*h**2/4 + 11/6*g**2*T**2
            bg=1/4*g1**2*h**2 + 11/6*g1**2*T**2
            ccg=-1/4*g1*g*h**2
            Ag=(ag+bg)/2
            Bg=1/2*np.sqrt((ag-bg)**2+4*ccg**2)
            mZL=Ag+Bg
            mPh=Ag-Bg
    
            M = np.array([m1,m2, g**2*h**2/4, h**2/4*(g**2+g1**2)  ,mWL,mZL])
            if self.ms<mh:
                Mphys = np.array([mh**2,self.ms**2,g**2*v**2/4,v**2/4*(g**2+g1**2),g**2*v**2/4,v**2/4*(g**2+g1**2)])
            else:
                Mphys = np.array([self.ms**2,mh**2,g**2*v**2/4,v**2/4*(g**2+g1**2),g**2*v**2/4,v**2/4*(g**2+g1**2)])
    
            # At this point, we have an array of boson masses, but each entry might
            # be an array itself. This happens if the input X is an array of points.
            # The generic_potential class requires that the output of this function
            # have the different masses lie along the last axis, just like the
            # different fields lie along the last axis of X, so we need to reorder
            # the axes. The next line does this, and should probably be included in
            # all subclasses.
            M = np.rollaxis(M, 0, len(M.shape))
            Mphys = np.rollaxis(Mphys, 0, len(Mphys.shape))
    
            # The number of degrees of freedom for the masses. This should be a
            # one-dimensional array with the same number of entries as there are
            # masses.
    
            dof = np.array([1,1,4, 2, 2, 1])
    
    
            # c is a constant for each particle used in the Coleman-Weinberg
            # potential using MS-bar renormalization. It equals 1.5 for all scalars
            # and the longitudinal polarizations of the gauge bosons, and 0.5 for
            # transverse gauge bosons.
            #c = np.array([1.5,1.5,1.5,1.5,1.5,1.5,1.5])
            c = np.array([1.5,1.5,1.5,1.5,1.5,1.5])
            
            return M, dof, c, Mphys
        
        def old_fermion_massSq(self, X):
            X = np.array(X)
            h,s = X[...,0], X[...,1]
            mt=h**2/2
            M = np.array([mt])
            Mphys = np.array([v**2/2])
    
            # At this point, we have an array of boson masses, but each entry might
            # be an array itself. This happens if the input X is an array of points.
            # The generic_potential class requires that the output of this function
            # have the different masses lie along the last axis, just like the
            # different fields lie along the last axis of X, so we need to reorder
            # the axes. The next line does this, and should probably be included in
            # all subclasses.
            M = np.rollaxis(M, 0, len(M.shape))
            Mphys = np.rollaxis(Mphys, 0, len(Mphys.shape))
            
            dof = np.array([12])
            return M, dof, Mphys
    
        def fermion_massSq(self, X):
            X = np.array(X)
            h,s = X[...,0], X[...,1]
            M_f=self.f/0.2
            Epsilon=s/self.Mf
            u1,u2,u3,u4,u5,u6,u7,u8,u9=best_fit
            (nQ1,nQ2,nQ3)=nQ
            (nu1,nu2,nu3)=nu
            YYdagger=h**2/2*np.array([[u1**2*Epsilon**(2*nQ1 + 2*nu1) + u2**2*Epsilon**(2*nQ1 + 2*nu2) + u3**2*Epsilon**(2*nQ1 + 2*nu3),
                            u1*u4*Epsilon**(nQ1 + nQ2 + 2*nu1) + u2*u5*Epsilon**(nQ1 + nQ2 + 2*nu2) +  u3*u6*Epsilon**(nQ1 + nQ2 + 2*nu3), 
                            u1*u7*Epsilon**(nQ1 + nQ3 + 2*nu1) + u2*u8*Epsilon**(nQ1 + nQ3 + 2*nu2) + u3*u9*Epsilon**(nQ1 + nQ3 + 2*nu3)], 
                           [u1*u4*Epsilon**(nQ1 + nQ2 + 2*nu1) + u2*u5*Epsilon**(nQ1 + nQ2 + 2*nu2) + u3*u6*Epsilon**(nQ1 + nQ2 + 2*nu3), 
                            u4**2*Epsilon**(2*nQ2 + 2*nu1) + u5**2*Epsilon**(2*nQ2 + 2*nu2) + u6**2*Epsilon**(2*nQ2 + 2*nu3), 
                            u4*u7*Epsilon**(nQ2 + nQ3 + 2*nu1) + u5*u8*Epsilon**(nQ2 + nQ3+2*nu2) +  u6*u9*Epsilon**(nQ2 + nQ3 + 2*nu3)], 
                           [u1*u7*Epsilon**(nQ1 + nQ3 + 2*nu1) + u2*u8*Epsilon**(nQ1 + nQ3 + 2*nu2) +  u3*u9*Epsilon**(nQ1 + nQ3 + 2*nu3), 
                            u4*u7*Epsilon**(nQ2 + nQ3 + 2*nu1) + u5*u8*Epsilon**(nQ2 + nQ3 + 2*nu2) + u6*u9*Epsilon**(nQ2 + nQ3 + 2*nu3), 
                            u7**2*Epsilon**(2*nQ3 + 2*nu1) + u8**2*Epsilon**(2*nQ3 + 2*nu2) + u9**2*Epsilon**(2*nQ3 + 2*nu3)]])
            YYdagger=np.rollaxis(YYdagger,0,len(YYdagger.shape))
            YYdagger=np.rollaxis(YYdagger,0,len(YYdagger.shape))
            evalss=np.linalg.eigvalsh(YYdagger)
            for _ in range(len(evalss.shape)-1):
                evalss=np.rollaxis(evalss,0,len(evalss.shape))
            
            muth,mcth,mtth = evalss
            M = np.array([muth,mcth,mtth])
            #Mphys = np.array([mu**2,mc**2,mt**2])
            Mphys=fermion_masses_vectorized([v,0.2*self.Mf],self.Mf) 
    
            # At this point, we have an array of boson masses, but each entry might
            # be an array itself. This happens if the input X is an array of points.
            # The generic_potential class requires that the output of this function
            # have the different masses lie along the last axis, just like the
            # different fields lie along the last axis of X, so we need to reorder
            # the axes. The next line does this, and should probably be included in
            # all subclasses.
            M = np.rollaxis(M, 0, len(M.shape))
            Mphys = np.rollaxis(Mphys, 0, len(Mphys.shape))
            
            dof = np.array([12,12,12])
            return M, dof, Mphys
        
    
        def approxZeroTMin(self):
            # There are generically two minima at zero temperature in this model,
            # and we want to include both of them.
            v = v2**.5
            return [np.array([v,self.f]), np.array([-v,-self.f]),np.array([v,-self.f]),np.array([-v,self.f])]
        
        def global_min(self):
            """Finds the global minimum and checks if it agrees with the model implementation
            ----------
            returns: boolean, True if the globl minium is the intended one, False otherwise"""
            global_min=self.findMinimum([v,self.f],0)
            print("Global minimum found at ",global_min)
            if np.sum((global_min-np.array([v,m.f]))**2)>10:
                return False
            else:
                return True
        def theoretical_requirement(self):
            """Check if the potential is bounded from below and if the quartic couplings are perturbative"""
            perturbativity=(self.lams<=2) and (abs(self.lammix)<=2) and (self.lamh<=2)
            positivity= (self.lams>0) and (self.lamh>0) and (self.lammix>-2*(self.lamh*self.lams)**0.5)
            output=perturbativity and positivity
            print("The model is theoretically consistent",output)
            return output
            
        
    
        
    ######MY FUNCTIONS START HERE---------    
    def my_getPhases(m):
        myexps=[(-3,-3),(-5,-4),(-5,-3),(-5,-5)]
        for nord in myexps:
            print("doing",nord)
            try:
                m.getPhases(tracingArgs={"dtstart":10**(nord[0]), "tjump":10**(nord[1])})
                phases_out=m.phases
            except:
                phases_out={}
            finally:
                if len(phases_out)>1:
                    break
        return phases_out
    
    
    def find_nucleation(m):
        """Find min and max temperatures to search for nucleation. IT will be used by bisection method.
        Parameters
            ----------
            m: a model instance. In this case m=model1(kk=1/600**2) for example.
        Returns
            -------
            nuc_dict: a dictionary containing the relevant temperatures and phases indexes. 
                    It will be used by the other methods to find the nucleation and percolation parameters  
        """
        if m.phases is None:
            try:
                phases_dict=my_getPhases(m)
            except:
                print("exception occured")
                return {}
        else:
            phases_dict=m.phases
        if len(phases_dict)<=1:
            return {}
        from cosmoTransitions import transitionFinder as tf
        crit_temps=tf.findCriticalTemperatures(phases_dict, m.Vtot)
        Num_relevant_trans=0
        ###DETERMINE IF THERE COULD BE TWO-STEP FOPTs
        for elem in crit_temps:
            if elem["trantype"]==1 and abs(elem["low_vev"][0]-elem["high_vev"][0])>10 and abs(elem["low_vev"][1]-elem["high_vev"][1])>10:
                print("Tunneling is relevant from phase " + str(elem["high_phase"])+ " to " + str(elem["low_phase"])  )
                Tmax=elem["Tcrit"]
                Tmin=phases_dict[elem["high_phase"]].T[0]
                print("max temperature is", Tmax)
                print("min temperature is", Tmin)
                Num_relevant_trans+=1
                high_phase_key=elem["high_phase"]
                low_phase_key=elem["low_phase"]
            else: 
                continue
        if Num_relevant_trans==0:
            dict_output={}
            return dict_output
        else:
            dict_output= {"Tmin":Tmin, "Tmax":Tmax, "high_phase": high_phase_key,"low_phase": low_phase_key}
        X0=m.phases[dict_output["high_phase"]].X[0]
        T0=m.phases[dict_output["high_phase"]].T[0]
        stable=not np.any(np.linalg.eig(m.d2V(X0,T0))[0]<=0)
        print("DOING WHILE LOOP \n")
        while stable:
            if T0<=0:
                break
            T0-=1e-4
            X0=m.findMinimum(X0,T0)
            if abs(X0[0])>0.1:
                break
            stable=not np.any(np.linalg.eig(m.d2V(X0,T0))[0]<=0)
            print(" ................. \n")
            if stable==False:
                break
        dict_output["Tmin"]=T0
        return dict_output
    
    def my_find_Trans(m):
        """Compute the transition"""
        from cosmoTransitions import transitionFinder as tf
        tntrans=tf.tunnelFromPhase(m.phases, m.phases[nuc_dict["high_phase"]], m.Vtot, m.gradV, nuc_dict["Tmax"], 
                                   Ttol=0.001, maxiter=100, phitol=1e-08, overlapAngle=45.0, 
                                   nuclCriterion=lambda S,T: S/(T+1e-100)-140,
                                   verbose=True, fullTunneling_params={})
        return tntrans
    
    ####This code uses an interpoaltion function for the number of degrees of freedom as function of temperature
    ###Data is obtained from https://member.ipmu.jp/satoshi.shirai/EOS2018
    data = np.loadtxt( 'satoshi_dof.dat' )
    Temperature_d=(data.T)[0][900:3900]
    dof_d=(data.T)[1][900:3900]
    #f = interpolate.interp1d(Temperature_d, dof_d)###"""the function works from T=[10e-4,1000]"""
    g_star = interpolate.interp1d(Temperature_d, dof_d, kind='cubic')
    
    
        
    def alpha_GW(Tnuc,Drho):
        ####This code gives the parameter alpha relevant for stochastic GW spectrum 
     ##AS APPEAR IN FORMULA (8.2) OF 1912.12634
        num_dof=g_star(Tnuc)
        radiationDensity=np.pi**2/30*num_dof*Tnuc**4
        latentHeat=Drho
        return latentHeat/radiationDensity
    
    def S_profile(T):
        """This function calculates the Euclidean action from a model m at temperature T
        after knowing its phase history. If more than one FOPT is found, it uses the last 
        transition to compute the action"""
        profile=tntrans["instanton"].profile1D
        alpha_ode=2
        temp=T
        r, phi, dphi, phivector = profile.R, profile.Phi, profile.dPhi, tntrans["instanton"].Phi
        phi_meta=tntrans["high_vev"]
        # Find the area of an n-sphere (alpha=n):
        d = alpha_ode+1  # Number of dimensions in the integration
        area = r**alpha_ode * 2*np.pi**(d*.5)/special.gamma(d*.5)
        # And integrate the profile
        integrand = 0.5 * dphi**2 + m.Vtot(phivector,temp) - m.Vtot(phi_meta,temp)
        integrand *= area
        S = integrate.simps(integrand, r)
        # Find the bulk term in the bubble interior
        volume = r[0]**d * np.pi**(d*.5)/special.gamma(d*.5 + 1)
        S += volume * (m.Vtot(phivector[0],temp) - m.Vtot(phi_meta,temp))
    
        return S/T      
    
    def gradAT(T,S,eps=0.1,order=4):
        """This function calculates the derivative of S_3/T using finite differences.
        It should only be used for T close to the nucleation tmeperature.
        Parameters
            ----------
            T: (float) is the temperature 
            S: S/T, a function of m, it can be called by S_cosmoT or S_profile
        Returns
            -------
            float: d/dT(S_3/T), minimum action solution divided by temperature"""
        if order==4:
                dT = np.array([-2., -1., 1., 2.])*eps
                coef = np.array([1., -8., 8., -1. ])/(12.*eps)
        elif order==6:    
            dT = np.array([-3., -2., -1., 1., 2.,3.])*eps
            coef = np.array([-1., 9., -45., 45., -9.,1. ])/(60.*eps)
        else: 
            dT = np.array([-4.,-3., -2., -1., 1., 2.,3.,4.])*eps
            coef = np.array([3.,-32., 168.,-672., 672., -168., 32.,-3. ])/(840.*eps)
        action = []
        for i in dT:
            action.append(S(T+i))
        action_output = np.array(action)
        return np.sum(action_output*coef)



    ####This codes the GW signal and SNR given T,alpha, beta and vw. 
    LISA_data = np.loadtxt( 'PLS_ESACallv1-2_04yr.txt' )
    LISA_data=LISA_data[::20]
    LISA_noise=LISA_data[::,0::2]
    LISA_data=LISA_data[::,0::3]
    
    def GW_signal(Temp,alpha,beta,vel):
        """Returns a tuple of (f,Omega_peak)"""
        time=4
        f_redshift=1.65*10**(-5)*(Temp/100)*(g_star(Temp)/100)**(1/6)
        Omega_redshift=1.67*10**(-5)*(100/g_star(Temp))**(1/3)
        kappa_sw=alpha/(0.73+0.083*alpha**0.5+alpha)
        Uf=(3/4*alpha/(1+alpha)*kappa_sw)**0.5
        HR=(8*np.pi)**(1/3)*max(vel,cs)/beta
        HRb=(vel-cs)/vel*HR
        Htau_sw=HR/Uf
        S_fun=lambda s:s**3*(7/(4+3*s**2))**(7/2)
        Omega_sw=3*0.687*Omega_redshift*(1-1/(1+2*HR/Uf)**0.5)*(kappa_sw*alpha/(1+alpha))**2*0.012*HR/cs
        f_sw=f_redshift*(2.6/1.65)*(1/HR)
        GW_tab=[Omega_sw*S_fun(f/f_sw) for f in LISA_noise[::,0]]
        return np.array([LISA_noise[::,0],GW_tab])
    
    def SNR_GW(Temp,alpha,beta,vel):
        """Returns a float, the SNR of the signal"""
        time=4
        f_redshift=1.65*10**(-5)*(Temp/100)*(g_star(Temp)/100)**(1/6)
        Omega_redshift=1.67*10**(-5)*(100/g_star(Temp))**(1/3)
        kappa_sw=alpha/(0.73+0.083*alpha**0.5+alpha)
        Uf=(3/4*alpha/(1+alpha)*kappa_sw)**0.5
        HR=(8*np.pi)**(1/3)*max(vel,cs)/beta
        HRb=(vel-cs)/vel*HR
        Htau_sw=HR/Uf
        S_fun=lambda s:s**3*(7/(4+3*s**2))**(7/2)
        Omega_sw=3*0.687*Omega_redshift*(1-1/(1+2*HR/Uf)**0.5)*(kappa_sw*alpha/(1+alpha))**2*0.012*HR/cs
        f_sw=f_redshift*(2.6/1.65)*(1/HR)
        integral=np.sum([(LISA_noise[i+1][0]-LISA_noise[i][0])/2*(Omega_sw*S_fun(LISA_noise[i][0]/f_sw)/(LISA_noise[i][1]))**2 for i in range(0,len(LISA_noise)-1)])
        return (time*3.15*10**7*integral)**0.5

    
    
    ####End of functions, computations starts here
    #--------------------------------------------
    
    
    np.random.seed()
    M_f=np.random.uniform(2*v,10*v)
    s_vev=0.2*M_f
    ms=np.random.uniform(1,10*v)
    theta_max=np.arcsin(.27)
    theta=np.random.uniform(-theta_max,theta_max)
    m=model1(ms,theta,s_vev)
    isglobal=m.global_min()
    theory_cons=m.theoretical_requirement()
    dict_output={"ms":m.ms,"theta":m.theta,"f":m.f,"Mf":m.Mf,
                     "lamh":m.lamh,"lams":m.lams,"lammix":m.lammix,
                     "muh2":m.muh,"mus2":m.mus}
    if abs(np.sin(m.theta))<=.27 and isglobal and theory_cons:
        dict_output["allowed"]=True
        m.print_couplings()
        nuc_dict=find_nucleation(m)
        if len(nuc_dict)!=0:
            dict_output.update({"Tmin":nuc_dict["Tmin"],"Tmax":nuc_dict["Tmax"]})               
            try:
                tntrans=my_find_Trans(m)
                Tn=tntrans["Tnuc"]
                dict_output["Tnuc"]=tntrans["Tnuc"]
                dict_output["h_low"]=tntrans["low_vev"][0]
                dict_output["h_high"]=tntrans["high_vev"][0]
                dict_output["s_low"]=tntrans["low_vev"][1]
                dict_output["s_high"]=tntrans["high_vev"][1]
                tntrans['Delta_rho'] = m.energyDensity(tntrans["high_vev"],tntrans["Tnuc"]) - m.energyDensity(tntrans["low_vev"],tntrans["Tnuc"])
                dict_output["alpha"]=alpha_GW(tntrans["Tnuc"],tntrans['Delta_rho'])
                dS_TdT=gradAT(Tn,S_profile,eps=0.01,order=6)
                beta=Tn*dS_TdT
                dict_output["beta"]=beta
                dict_output["dT"]=dict_output["Tmax"]-dict_output["Tmin"]
                dict_output["dPhi"]=np.sum((tntrans["low_vev"]-tntrans["high_vev"])**2)**0.5
            except:
                print("Not implemented error")
                     
    return dict_output
    




import multiprocessing as mp
pool = mp.Pool(8)


results = pool.map(my_fun, range(4000))
df_out=pd.DataFrame(results)
df_out.to_csv("./SCANS/transitions.csv")

    
