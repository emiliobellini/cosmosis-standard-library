import os
import numpy as np
from scipy import integrate
from numpy import log, pi, interp, where, loadtxt,dot, append, linalg
from cosmosis.datablock import names as section_names
from cosmosis.datablock import option_section
from cosmosis.gaussian_likelihood import GaussianLikelihood
from scipy.interpolate import interp2d

# dist = section_names.distances
dist = section_names.cosmological_parameters

c_km_per_s =  299792.458


ROOT_dir = os.path.split(os.path.abspath(__file__))[0]


class CORRELATION_PRIORLikelihood(GaussianLikelihood):
    # ???
    # data_type = "elg" 
    like_name = "correlation_prior"    

    def __init__(self, options):
        
        super(CORRELATION_PRIORLikelihood, self).__init__(options)
        # Allow override of these parameters
        self.mode = self.options.get_int("mode", default=0)
        self.feedback = self.options.get_bool("feedback", default=False)
        self.n_points = self.options.get_int("n_points", default=8)
        
    def build_data(self):
        
        self.mode = self.options.get_int("mode", default=0)
        self.n_points = self.options.get_int("n_points", default=8)
        
        print("CORRELATION - additional prior for spline reconstruction")
        if (self.mode == 0):  
            print('Fiducial from theoretical expectations')
            data = np.zeros(self.n_points) #fiducial
        else if (self.mode == 1):
            print('Fiducial from Gaussian Kernel')
            data = np.zeros(self.n_points) #fiducial
        print()
        print('Number of nodes = ', self.n_points)
        print('Fiducial for Delta Omega_X', data)
        print()

        return None, data
    
    def build_covariance(self):
        cov = np.eye(self.n_points) #Initialized to the identity, filled in do_likelihood
        self.inv_cov = linalg.inv(cov)
        return cov
    
    def build_inverse_covariance(self):
        return self.inv_cov

    def extract_theory_points(self,block):   

        omegas=np.zeros(self.n_points)
        for i in range(self.n_points - 1):
            tag='spline_domega_fld__'+str(i+1)
            omegas[i]=block[dist, tag]

        #NB for the time being I don't consider the anchor, but to discuss
        # omegas[0]=block[dist, 'spline_domega_anchor_fld']
        
        return omegas
    
    def do_likelihood(self, block):
        # Run the
        super(CORRELATION_PRIORLikelihood, self).do_likelihood(block)

        # Omegas extracetd in extract_theory_points
        omegas=np.atleast_1d(self.extract_theory_points(block))
        fiducial=np.atleast_1d(self.data_y)

        # Extracting redshifts
        zbin=np.zeros(self.n_points)
        for i in range(self.n_points - 1):
            tag='spline_z_fld__'+str(i+1)
            zbin[i]=block[dist, tag]

        # Defining needed quantitiets 
        Delta=np.round(zbin[:-1]-zbin[1:],2)
        # TO DO
        # if (Delta[0] != Delta.all()):
        #     raise ValueError("Bins MUST be equispaced in redshift")
        # assert math.isclose(Delta, np.any(Delta), abs_tol=1e-3)
        Delta=Delta[0]

        sigma=0.1
        zc=0.3
        zmax=zbin[0]
        xi0=sigma*zmax/zc/np.pi

        # Defyining the integrad 
        # x-> z, y-> z'
        corr = lambda x,y: (zc**2 / (zc**2 + (x-y)**2))

        cov = np.eye(self.n_points) #Initialized to the identity, filled in do_likelihood

        for i in range(self.n_points):
            for j in range(self.n_points):
                cov[i,j] = integrate.dblquad(corr, zbin[i], zbin[i]+Delta, zbin[j], zbin[j]+Delta)[0]*xi0/Delta**2
        self.inv_cov = linalg.inv(cov)

        if self.feedback:
            print()
            print('Delta Omega_X values for the correlation prior:')
            print(omegas)
            print('At redshifts:')
            print(zbin)
            print('with fiducials:')
            print(fiducial)
            print('Used Delta, sigma, zc, xi(0):')
            print(Delta,sigma,zc,np.round(xi0,3))
            print()

        d = omegas-fiducial
        chi2 = np.einsum('i,ij,j', d, self.inv_cov, d)
        chi2 = float(chi2)
        like = -0.5*chi2

        # overwrite the log-likelihood
        block[section_names.likelihoods, self.like_name + "_LIKE"] = like

setup, execute, cleanup = CORRELATION_PRIORLikelihood.build_module()
