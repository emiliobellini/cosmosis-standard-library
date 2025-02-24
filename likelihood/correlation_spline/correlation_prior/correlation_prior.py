import os
import numpy as np
from scipy import integrate
from numpy import log, pi, interp, where, loadtxt,dot, append, linalg
from cosmosis.datablock import names as section_names
from cosmosis.datablock import option_section
from cosmosis.gaussian_likelihood import GaussianLikelihood
from scipy.interpolate import interp2d
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# dist = section_names.distances
dist = section_names.cosmological_parameters

# c_km_per_s =  299792.458


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
        self.n_points = self.options.get_int("n_points", default=10)
        self.delta = self.options.get_double("delta", default=0.1)
        self.amin = self.options.get_double("amin", default=0.1)
        self.amax = self.options.get_double("amax", default=1.)
        self.sigma = self.options.get_double("sigma", default=0.3)
        self.ac = self.options.get_double("ac", default=0.3)
        
    def build_data(self):
        
        # self here is build.data
        self.mode = self.options.get_int("mode", default=0)
        self.n_points = self.options.get_int("n_points", default=8)
        self.delta = self.options.get_double("delta", default=0.1)
        self.amin = self.options.get_double("amin", default=0.1)
        self.amax = self.options.get_double("amax", default=1.)
        self.sigma = self.options.get_double("sigma", default=0.3)
        self.ac = self.options.get_double("ac", default=0.3)
        
        print("CORRELATION - additional prior for spline reconstruction")
        print()
        print('Equispaced bins in scale factor with:  number of points =', self.n_points, '   bin lenght =', self.delta, ' between a ',  self.amin, '  and  ', self.amax)
        if (self.mode == 0):  
            print('Fiducial from theoretical expectations')
            data = np.zeros(self.n_points-1) #fiducial
        elif (self.mode == 1):
            print('Fiducial from Gaussian Kernel - TO DO')
            data = np.zeros(self.n_points-1) #fiducial
        print()
        print('Fiducial for Delta Omega_X', data)

        return None, data
    
    def build_covariance(self):
        self.feedback = self.options.get_bool("feedback", default=False)

        # Extracting scale factor bins
        abin=np.arange(self.amin, self.amax, self.delta)

        if (self.n_points-1 != np.size(abin)):
            raise ValueError("WRONG input: check that amin, amax, n points and delta make sense!")

        # Defining needed quantitiets 
        xi0=self.sigma**2 * (1. - self.amin)/self.ac/np.pi
        print('Considered sigma', self.sigma)
        print('Considered xi(0)', xi0)
        print('-------------------------------')

        # Defyining the integrad 
        # x-> a, y-> a'
        corr = lambda x,y: (self.ac**2 / (self.ac**2 + (x-y)**2))

        cov = np.eye(self.n_points-1) #Initialized to the identity

        for i in range(self.n_points-1):
            for j in range(self.n_points-1):
                cov[i,j] = integrate.dblquad(corr, abin[i], abin[i]+self.delta, abin[j], abin[j]+self.delta)[0]*xi0/self.delta**2

        if self.feedback:
            print()
            print('Covariance matrix:')
            print(cov)
            print()

        self.inv_cov = linalg.inv(cov)
        return cov
    
    def build_inverse_covariance(self):
        return self.inv_cov

    def extract_theory_points(self,block):   
        
        # Computing the mean
        omegas=np.zeros(self.n_points)
        for i in range(self.n_points - 1):
            tag='spline_domega_fld__'+str(i+1)
            omegas[i+1]=block[dist, tag]
        omegas[0]=block[dist, 'spline_domega_anchor_fld']

        abin_z=np.arange(self.amin, self.amax+self.delta, self.delta)
        znodes=1./abin_z - 1.

        omegas_z = self.full_spline(znodes,omegas)

        # Defyining the integrad 
        # x-> z
        mean = lambda x: omegas_z(np.array([x])) / (1.+x)**2

        omegas_means=np.zeros(self.n_points-1)
        for i in range(self.n_points-1):
            omegas_means[i] = integrate.quad(mean,1./(abin_z[i]+self.delta)-1. ,1./abin_z[i]-1.)[0]/self.delta
        
        if self.feedback:
            for i in range(self.n_points - 1):
                tag='spline_z_fld__'+str(i+1)
                znodes=block[dist, tag]
                if (np.abs(znodes-(1./abin_z[i+1]-1.)) > 1.e-4 ):
                    raise ValueError("WRONG input: redshifts in values.ini do not match the selected a bins!")

            name='test_mean.pdf'
            name='/home/users/b/bertim/scratch/correlated_spline/test/'+name

            fig,ax = plt.subplots(1,2)
            zed=np.linspace(0.,9., 100)
            ax[0].plot(zed, omegas_z(zed))
            ax[1].plot(1/(1+zed), omegas_z(zed))
            ax[1].scatter(abin_z[:-1]+self.delta/2., omegas_means)

            for aa in abin_z:
                ax[1].vlines(aa, 0,np.max(omegas_z(zed)), color='black', alpha=0.3)

            ax[0].set_xlabel('z')
            ax[1].set_xlabel('a')
            fig.savefig(name, bbox_inches='tight')
          
        return omegas_means
    
    # Spline implementation
    def full_spline(self,zc,omegas):
        # REDSHIFTS as in code, decreasing, first node must be z anchor (?? check you said z+0, but looking at this is z anchor)
        z_anchor = zc[0]  #---> FIRST redshift in zc must be anchor 
        y_anchor = omegas[0]

        x_nodes = zc[1:]
        y_nodes = omegas[1:]
        spline_xy=CubicSpline(x_nodes[::-1],y_nodes[::-1],bc_type=((2, 0.0), (2, 0.0)))
        
        def final_spline(zz):
            result = np.zeros_like(zz)
            for i, z in enumerate(zz):
                if (z >= z_anchor):
                        result[i] = y_anchor
                elif (z > x_nodes[0]):
                    delta_x = z_anchor-x_nodes[0] #delta between last node and anchor
                    ddy=spline_xy(x_nodes,2)
                    dy0 = spline_xy(x_nodes[0],1)
                        
                    y0 = y_nodes[0]
                    y1 = y_anchor
                        
                    a = -(12.*(y0-y1) + 6.*dy0*delta_x + ddy[0]*pow(delta_x,2.))/2./pow(delta_x, 5.)
                    b = (30.*(y0-y1) + 16.*dy0*delta_x + 3.*ddy[0]*pow(delta_x,2.))/2./pow(delta_x, 4.)
                    c = -(20.*(y0-y1) + 3.*(4.*dy0 + ddy[0]*delta_x)*delta_x)/2./pow(delta_x, 3.)
                    d = ddy[0]
                    e = dy0
                    f = y_nodes[0]
                        
                    dz = z - x_nodes[0]
                        
                    result[i] = a*pow(dz, 5.) + b*pow(dz, 4.) + c*pow(dz, 3.) + d*pow(dz, 2.) + e*dz + f
                        
                elif (z <= x_nodes[0]):
                    result[i] = spline_xy(z)
            return result
        return final_spline
    
    def do_likelihood(self, block):
        # Run the
        super(CORRELATION_PRIORLikelihood, self).do_likelihood(block)

        # Omegas extracetd in extract_theory_points
        omegas=np.atleast_1d(self.extract_theory_points(block)) 
        fiducial=np.atleast_1d(self.data_y)

        if self.feedback:
            print()
            print('Delta Omega_X values for the correlation prior:')
            print(omegas)
            print('with fiducials:')
            print(fiducial)
            print('Inverse covariance matrix:')
            print(self.inv_cov)
            print()

        d = omegas-fiducial
        chi2 = np.einsum('i,ij,j', d, self.inv_cov, d)
        chi2 = float(chi2)
        like = -0.5*chi2

        # overwrite the log-likelihood
        block[section_names.likelihoods, self.like_name + "_LIKE"] = like

setup, execute, cleanup = CORRELATION_PRIORLikelihood.build_module()
