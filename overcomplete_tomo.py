import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import pickle
import pyrr


def find_center_coordinates(min,max,num):
    lin = np.linspace(min,max,num)
    center = (abs(lin[0])-abs(lin[1]))/2
    X,Y = np.meshgrid(lin+center,lin+center)
    return X,Y

def generate_sparse_pixel_model(basis,size,amplitude,random=False):
    ''' Create the pixel model parameters by randomly choosing the x and y locations where the non-zero coefficients will be, and sum them up.
     In total, the pixel model has "size" non-zero coefficients where the maximum amplitude is "ampl" for each. '''
    if random:
        pix_pos = np.random.choice(basis.nbases, replace=False, size=size)
    else:
        # define the specific pixel positions
        pix_pos = [] 
        for i in range(4,7):
            for j in range(6,8):
                pix_pos.append(basis.iconvertindex(i,j))    #block 1
                pix_pos.append(basis.iconvertindex(j,i))    #block 2

        for i in range(6,8):
            pix_pos.append(basis.iconvertindex(i+1,14))     #small horizontal line
            pix_pos.append(basis.iconvertindex(i+4,14))     #long horizontal line
            pix_pos.append(basis.iconvertindex(i+6,14))     #cont. of long line
            pix_pos.append(basis.iconvertindex(12,i+7))     # vertical line
            pix_pos.append(basis.iconvertindex(12,i+9))
        pix_pos = np.array(pix_pos) # convert the list to array

    # create an array of (x, y) coordinates of the non-zero coefficients
    coords = np.array([(basis.xg[basis.convertindex(pix_pos[i])[0]], basis.xg[basis.convertindex(pix_pos[i])[1]]) for i in range(size)])
    
    # evaluate the basis functions using the array of coordinates
    true_pixel = np.zeros(basis.nbases)
    m = np.zeros_like(true_pixel)
    for i in range(size):
        for j in range(basis.nbases):
            m[j] = amplitude*basis.evaluate(j,coords[i])
        true_pixel += m
    return true_pixel

def generate_sparse_cosine_model(upload=True):
    '''This should be completed'''
    if upload:
        with open("synth_files/true_m_c.pickle",'rb') as file:
            true_cosine = pickle.load(file)
    return true_cosine


class pixelbasis2D(object):
    """
    A 2D Model pixel basis object.
    A class that is used by various routines to calculate Jacobians through numerical integration.
    """
    def __init__(self,xg,zg,npoints=[120,120,10000]): # receives linear grid dimension
        self.type = '2Dpixel'
        self.xg = xg           # X-grid for voxel model
        self.zg = zg           # Y-grid for voxel model
        self.nx = len(xg)-1    # Number of cells in X-direction
        self.nz = len(zg)-1    # Number of cells in Y-direction
        self.xmin = xg[0]      # X-min of model
        self.xmax = xg[-1]     # X-max of model
        self.zmin = zg[0]      # Y-min of model
        self.zmax = zg[-1]     # Y-max of model
        self.nbases = (self.nx)*(self.nz) # Total number of basis functions in voxel model

        self.nxi = npoints[0]         # X-integration grid used in each cell of this basis (only used if kernel is 3D and integration mthd is 'simpson')
        self.nzi = npoints[1]         # Y-integration grid used in each cell of this basis (only used if kernel is 3D and integration mthd is 'simpson')
        self.nline = npoints[2]       # Number of integration points along 1D ray within cell (only used if kernel is a 1D and integration method `simpson')

    def evaluate(self,j,pos):
        """ evaluate the ith model basis function at location (x,y) """
        if (j < 0 or j >= self.nbases): raise Error(j)
        x,z = pos
        ix,iz = self.convertindex(j)         # convert j basis to 2D model basis index
        if(np.isscalar(x)):
            b = 0.0                          # pixel basis value
            if((x >= self.xg[ix]) & (x < self.xg[ix+1]) & (z >= self.zg[iz]) & (z < self.zg[iz+1])): b = 1.
            if(((ix == self.nx-1) & (x == self.xg[-1])) or ((iz == self.nz-1) & (z == self.zg[-1]))): b = 1.
        else:
            b = np.zeros_like(x)
            b[(x >= self.xg[ix]) & (x < self.xg[ix+1]) & (z >= self.zg[iz]) & (z < self.zg[iz+1])] = 1.
            b[(ix==self.nx-1 and x == self.xg[-1])] = 1.
            b[(iz==self.nz-1 and z == self.zg[-1])] = 1.
        return b

    def convertindex(self,i):
        """Convert single 1D array index to 2D array index using inner loop over X and outer loop over Y"""
        if (i < 0 or i >= self.nbases): raise Error(i)
        xindex = int(i % self.nx)
        zindex = int((i / self.nx) % self.nz)
        return xindex,zindex

    def iconvertindex(self,ix,iz):
        ''' Convert single 2D array index to 1D array index inner loop over X and outer loop over Y'''
        if (ix < 0 or ix >= self.nx): raise Error(ix)
        if (iz < 0 or iz >= self.nz): raise Error(iz)
        return iz*self.nx + ix

    def intlimits(self,j):
        """ calculates the limits of integration for given basis function   """
        ix,iz = self.convertindex(j)               # convert j basis to 2D model basis index
        # For voxel basis we return the limits of the jth cell (because jth basis is zero elsewhere)
        return [self.xg[ix],self.xg[ix+1]],[self.zg[iz],self.zg[iz+1]]

    def plot(self,plot_all=False):
        '''plot some or all of the basis functions.
        If some, then, the basis functions are choosen as the first three in the X-direction
        and the first three in Z-direction.'''
        Xi,Zi = np.meshgrid(self.xg,self.zg)
        if plot_all:
            plt.figure(figsize=(10,10))
            for j in range(self.nbases):
                plt.subplot(self.nx,self.nz,j+1)
                plt.imshow(self.evaluate(j,(Xi,Zi)),cmap=plt.cm.Blues)
                plt.title(self.convertindex(j))
                plt.xticks([])
                plt.yticks([])
        else:
            bases = [0,1,2,self.nx,self.nx+1,self.nx+2,self.nx*2,self.nx*2+1,self.nx*2+2]
            plt.figure(figsize=(15,10))
            for j in range(np.shape(bases)[0]):
                    plt.subplot(3,3,j+1)
                    plt.imshow(self.evaluate(bases[j],(Xi,Zi)),cmap=plt.cm.Blues)
                    plt.title(self.convertindex(bases[j]))
                    plt.colorbar(shrink=0.6)
        plt.suptitle("Pixel Basis functions")

    def quality_check(self):
        '''do a quality check: if the pixel basis functions are generated correctly.
        First generate the pixel basis functions in every location x,y and
        sum them up to see if each grid has only one value.
        No need to run if the number of unknowns or the gridding is big '''
        Xi,Zi = np.meshgrid(self.xg,self.zg)
        pb = np.zeros([self.nbases,len(self.zg),len(self.xg)])
        for j in range(self.nbases):
            pb[j] = self.evaluate(j,(Xi,Zi))
        print("Check if the number of pixels are correct. \nThe number of pixel basis functions are the same as the sum of all the pixels: ", self.nbases==np.sum(pb))
        print(np.sum(pb))


class cosinebasis2D(object): # Model basis class
    """
    A 2D Model cosine basis object.
    A class that is used by various routines to calculate Jacobians through numerical integration.
    """
    def __init__(self,x0,x1,z0,z1,nb,npoints=[120,120,14400]): # receives linear grid dimension
        self.type = '2D'
        self.xmin = x0      # X-min of model
        self.xmax = x1      # X-max of model
        self.zmin = z0      # Z-min of model
        self.zmax = z1      # Z-max of model
        self.nx = nb        # set number of bases in X-direction
        self.nz = nb        # set number of bases in Z-direction
        self.nbases = self.nx * self.nz # Total number of basis functions in 2D model

        self.nxi = npoints[0]         # X-integration grid used in each cell of this basis (only used if kernel is 3D and integration mthd is 'simpson')
        self.nzi = npoints[1]         # Y-integration grid used in each cell of this basis (only used if kernel is 3D and integration mthd is 'simpson')
        self.nline = npoints[2]       # Number of integration points along 1D ray within cell (only used if kernel is a 1D and integration method `simpson')
        self.Lx = self.xmax-self.xmin # Set maximum X wavelength in model
        self.Lz = self.zmax-self.zmin # Set maximum Z wavelength in model
        self.norm = np.sqrt(self.Lx*self.Lz)
        self.area = self.Lx*self.Lz   # set area of domain

    def evaluate(self,j,pos):
        ''' evaluate the ith data kernel at location (x,z)'''
        x,z = pos
        ix,iz = self.convertindex(j) # convert from single index to pair of ix,iz indices
        #b = np.cos((2*np.pi*ix*x)/self.Lx) * np.cos((2*np.pi*iz*z)/self.Lz) # evaluate jth basis function at input positions
        #b = np.cos(np.pi*(ix+0.5)*x/(self.Lx)) * np.cos(np.pi*(iz+0.5)*z/(self.Lz)) # evaluate jth basis function at input positions
        b = np.cos(np.pi*ix*(x-self.xmin)/(self.Lx)) * np.cos(np.pi*iz*(z-self.zmin)/(self.Lz)) # evaluate jth basis function at input positions
        fx,fz = np.sqrt(2.),np.sqrt(2.)
        if(ix==0): fx = 1.
        if(iz==0): fz = 1.
        return b*fx*fz/self.norm

    def convertindex(self,i):
        '''Convert single 1D array index to 2D array index using inner loop over X and outer loop over Y'''
        if (i < 0 or i >= self.nbases): raise Error(i)
        xindex = int(i % self.nx)
        zindex = int((i / self.nx) % self.nz)
        return xindex,zindex

    def iconvertindex(self,ix,iz):
        '''Convert single 2D array index to 1D array index inner loop over X and outer loop over Y'''
        if (ix < 0 or ix >= self.nx): raise Error(ix)
        if (iz < 0 or iz >= self.nz): raise Error(iz)
        return iz*self.nx + ix # ix is the inner loop and iz the outer loop

    def intlimits(self,j):
        '''calculates the limits of integration for given basis function'''
        return [self.xmin,self.xmax],[self.zmin,self.zmax] # Here we choose whole model

    def check_orhonormality(self,j0,j1):
        ''' Quadratic numerical integration to test ortho-normality of basis functions
        set up integration grid to test ortho-normality of basis functions
        choose a pair of basis functions to integrate over to check they are ortho-normal'''
        xlim,zlim = self.intlimits(0) # get limits of integration for this basis
        x = np.linspace(xlim[0],xlim[1],500)   # X-linear dimension of grid
        z = np.linspace(zlim[0],zlim[1],500)   # Y-linear dimension of grid
        Xi,Zi = np.meshgrid(x,z,indexing='ij') # Create (X,Y) mesh

        print(self.convertindex(j0))
        b0 = self.evaluate(j0,(Xi,Zi))         # Evaluate first basis function on integration mesh
        b1 = self.evaluate(j1,(Xi,Zi))         # Evaluate second basis function on integration mesh
        integrand = b0*b1                        # Evaluate integrand over model domain
        integral = integrate.simps(integrate.simps(integrand, z) ,x) # use 2D Simpson integration over mesh
        return print('Integral of basis function '+str(j0)+' multiplied by basis function '+str(j1)+' over model domain:',integral)

    def plot(self,plot_all=False):
        '''plot some or all of the basis functions.
        If some, then, the frequencies are choosen as the first three frequencies
        along the X-direction and three in Z-direction.'''
        xlim,zlim = self.intlimits(0) # get limits of integration for this basis
        x = np.linspace(xlim[0],xlim[1],500)   # X-linear dimension of grid
        z = np.linspace(zlim[0],zlim[1],500)   # Y-linear dimension of grid
        Xi,Zi = np.meshgrid(x,z,indexing='ij') # Create (X,Y) mesh
        if plot_all:
            plt.figure(figsize=(10,10))
            for j in range(self.nbases):
                plt.subplot(self.nx,self.nz,j+1)
                plt.imshow(self.evaluate(j,(Xi,Zi)),cmap=plt.cm.RdBu,vmin=-0.01,vmax=0.01)
                plt.xticks([])
                plt.yticks([])
        else:
            bases = [0,1,2,self.nx,self.nx+1,self.nx+2,self.nx*2,self.nx*2+1,self.nx*2+2]
            plt.figure(figsize=(15,10))
            for j in range(np.shape(bases)[0]):
                    plt.subplot(3,3,j+1)
                    plt.imshow(self.evaluate(bases[j],(Xi,Zi)),cmap=plt.cm.RdBu)
                    plt.title(self.convertindex(bases[j]))
                    plt.colorbar(shrink=0.6)
        plt.suptitle("Cosine Basis functions")

    def quality_check(self):
        '''Check the ratio between the old and new cosine bases'''
        xlim,zlim = self.intlimits(0) # get limits of integration for this basis
        x = np.linspace(xlim[0],xlim[1],500)   # X-linear dimension of grid
        z = np.linspace(zlim[0],zlim[1],500)   # Y-linear dimension of grid
        Xi,Zi = np.meshgrid(x,z,indexing='ij') # Create (X,Y) mesh
        with open("../overcomplete/outputs/synthetic_outputs/latest_run/D.pickle",'rb') as d:
            D = pickle.load(d)
        a = np.zeros(400)
        for j in range(400):
             a[j] = self.evaluate(j,(Xi,Zi)).max()/D[:,j].reshape(120,120).max()
        plt.figure()
        plt.imshow(a.reshape(20,20), cmap=plt.cm.Blues)
        plt.colorbar()
        plt.title('The ratio between the max value of the new bfs and the previous bfs')
        b = np.zeros(400)
        for j in range(400):
             b[j] = self.evaluate(j,(Xi,Zi)).min()/D[:,j].reshape(120,120).min()
        plt.figure()
        plt.imshow(b.reshape(20,20), cmap=plt.cm.Blues)
        plt.colorbar()
        plt.title('The ratio between the min value of the new bfs and the previous bfs')


class straightraykernel1D(object):
    """
    A linear data kernel object.
    A class that is used by various routines to calculate Jacobians through numerical integration.
    """
    def __init__(self,paths): # receives the end points of rays.
        self.nkernels = len(paths)
        self.paths = paths
        self.type = '1Dray' # This id indicates a straight ray (use `1D' if ray is not straight line)
        self.constant = 1.0 # # kernel is a constant along the ray (typically 1.0)
        d = np.zeros(self.nkernels)
        for k,p in enumerate(paths):
            d[k] = np.linalg.norm(paths[k][1]-paths[k][0])
        self.lengths = d

    def evaluate(self,i): # evaluate the ith data kernel at location (x,y,z)
        if (i < 0 or i >= self.nkernels): raise Error(i)

        return self.constant # Return constant along ray for integration

    def position(self,i,l):
        '''find (x,y,z) position of ith data kernel at length l along ray.
        This example is for a straight seismic ray (self.type = '1Dray').
        For a curved ray (self.type = '1D') this is where one would specify (x,y,z) as a function of length l.'''
        if (i < 0 or i >= self.nkernels): raise Error(i)
        alpha = l/self.lengths[i]
        a0 = self.paths[i][0]
        a1 = self.paths[i][1]
        return (1.0-alpha)*a0[:,np.newaxis] + alpha*a1[:,np.newaxis] # here we use a straight line between endpoints of locations
    
class Error(Exception):
    """ Raised when input id does not exist in the definition    """
    def __init__(self,cset=[]):
        super().__init__('\n id '+str(cset)+' not recognized. \n')

def fwd(Gpixel,Gcosine,m_pixel,m_cosine):
    ''' Linear tomography forward '''
    dp = Gpixel.dot(m_pixel)
    dc = Gcosine.dot(m_cosine)
    return (dp+dc)

def opt_func(m,d,Gp,Gc,Gp_size,Gc_size,noise,mp_shape,alpha,beta,minL1=True,deriv=False):
    ''' optimisation function for overcomplete inversion where 
    phi = 1/(2*N*noise^2) ||d - G_p m_p - G_c m_c||_2^2 +
    alpha/noise * [ beta * ||G_p||_2 * ||m_p||_1 + (1-beta) * ||G_c||_2 * ||m_c||_1 
    when minL1=True. 
    OR
    phi = 1/(2*N*noise^2) ||d - G_p m_p - G_c m_c||_2^2 +
    alpha/noise * [ beta * ||G_p||_2 * ||m_p||_2^2 + (1-beta) * ||G_c||_2 * ||m_c||_12^2 
    where minL1=False'''
    m_pixel = m[:mp_shape]
    m_cosine = m[mp_shape:]
    resid = d - fwd(Gp,Gc,m_pixel,m_cosine)
    data_fit = np.linalg.norm(resid,2)**2/(2*d.shape[0]*noise**2)
    if minL1:
        pixel_size = np.linalg.norm(m_pixel,1)
        cosine_size = np.linalg.norm(m_cosine,1)
    else:
        pixel_size = (np.linalg.norm(m_pixel,2))**2
        cosine_size = (np.linalg.norm(m_cosine,2))**2
    phi = data_fit+(alpha/noise)*((beta*Gp_size)*pixel_size + ((1-beta)*Gc_size)*cosine_size)
    print("f = %.3f       df = %.5f  %.3f  %.3f mf = %.3f"%(phi,data_fit,pixel_size,cosine_size,((beta*Gp_size)*pixel_size + ((1-beta)*Gc_size)*cosine_size)))
    if deriv:
        GTr = np.hstack([Gp.T.dot(resid),Gc.T.dot(resid)])
        if minL1:
            drv = -GTr/(d.shape[0]*noise**2)+(alpha/noise)*(np.hstack([(beta*Gp_size)*np.sign(m_pixel),((1-beta)*Gc_size)*np.sign(m_cosine)]))
        else:
            drv = -GTr/(d.shape[0]*noise**2)+(alpha/noise)*(np.hstack([(beta*Gp_size)*m_pixel,((1-beta)*Gc_size)*m_cosine]))

        return phi, drv
    else:
        return phi


def opf_onebasis(m,d,G,noise,alpha,minL1=True,deriv=False):
    ''' optimisation function for inversion using one set of basis functions
    if minL1 is True:
    phi = 1/(2*N*noise^2) * ||d - G_p m_p||_2^2 + alpha * ||m_p||_1
    else:
    phi = 1/(2*N*noise^2) * ||d - G_p m_p||_2^2 + alpha * ||m_p||_2^2 '''
    resid = d - G.dot(m)
    data_fit = np.linalg.norm(resid,2)**2/(2*d.shape[0]*noise**2)
    if minL1:
        m_size = np.linalg.norm(m,1)
    else:
        m_size = np.linalg.norm(m,2)**2
    phi = data_fit + alpha*m_size
    print("f = %.3f,   datafit = %.3f,  size m = %.3f"%(phi,data_fit,m_size))
    GTr = G.T.dot(resid)
    if deriv:
        if minL1:
            return phi, -GTr/(d.shape[0]*noise**2)+alpha*np.sign(m)
        else:
            return phi, -GTr/(d.shape[0]*noise**2)+alpha*m
    else:
        return phi
    
def discretize_models(coef,basis,XX,YY):
    ''' plot the model on a finer grid'''
    disc_m = np.zeros([XX.shape[0],YY.shape[0]])
    for j in range(basis.nbases):
        disc_m += coef[j]*(basis.evaluate(j,(XX,YY)))
    return disc_m

def find_rays_intersect(Nrec,Nproj,dxy,Np,modelxy,xm0,ym0,xr,yr0,yr1,xs,ys):
    def intersect(i,raystart,rayend): #Routine to calculate entry an exit points from rays to sides of model.
        aabb = np.array([[xm0,ym0,0.0],[xm0+modelxy,ym0+modelxy,0.0]]) # cell defined by extremes
        line = np.array([raystart,rayend])  # raypath
        #print(line)
        #print(aabb)
        p0 = pyrr.geometric_tests.ray_intersect_aabb(pyrr.ray.create_from_line(line),aabb) # intersection point
        if(p0 is None): # line does not intersect cell
            #print('ray does not intersect',i,j)
            return p0,p0
        else:
            line = np.array([rayend,raystart]) # reverse raypath to find exit point from cell
            p1 = pyrr.geometric_tests.ray_intersect_aabb(pyrr.ray.create_from_line(line),aabb) # intersection point
            if(p1 is None): # This case will arise when there is inconsistent results from pyrr package.
                            # forward ray indicates intresection with cell but reversed ray does not.
                            # Must be due to rounding error, so we ignore intersection for this case.
                return p1,p1
        return p0,p1

    # preliminaries
    rad2deg = 180./np.pi

    sl = yr1-yr0                                  # screen length
    dr = sl/(Nrec-1)                              # bin width on screen
    yb = np.linspace(yr0+dr/2.,yr1-dr/2.,Nrec)    # Initial Cartesian y-co-ordinates of bin centres on screen
    xb = xr*np.ones(Nrec)                         # Initial Cartesian x-c-oordinates of bin centres on screen
    rb,ab = np.zeros_like(xb),np.zeros_like(xb)
    for i in range(len(xb)):                      # model bin centres in polar co-ordinates about origin
        rb[i] = np.sqrt(xb[i]*xb[i]+yb[i]*yb[i])  # radii
        ab[i] = np.arctan2(yb[i],xb[i])*rad2deg  # angles to x axis

    da = 360./Nproj           # angle increment in degrees for source clockwise rotation
    r =  np.linalg.norm(xs)   # distance from source to centre of model
    k=0
    rayintersect = np.zeros((Nproj*Nrec), dtype=bool)
    paths = np.zeros((Nproj*Nrec,2,2))
    for i in range(Nproj):    # loop over projections
        angle = i*da      # angle of source relative to initial poistion (clockwise)
        xsr = -r*np.cos(angle*np.pi/180.) # x-position of source
        ysr = r*np.sin(angle*np.pi/180.)  # y-position of source
        xbr = rb*np.cos((ab-angle)/rad2deg)
        ybr = rb*np.sin((ab-angle)/rad2deg)
        raystart = [xsr,ysr,0.]
        #print(i,angle,xsr,ysr,xbr[0],ybr[0])
        for j in range(Nrec): # loop over receivers per projection
            rayend = [xbr[j],ybr[j],0.]
            p0,p1 = intersect(k,raystart,rayend)
            if(p0 is not None):
                paths[k,0] = p0[:2]
                paths[k,1] = p1[:2]
                rayintersect[k] = True
            else:
                rayintersect[k] = False
            k+=1
    return paths, rayintersect
