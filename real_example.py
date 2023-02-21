import numpy as np
import pickle
import time
import scipy.optimize as optimize
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import overcomplete_tomo as octomo
import plotting_functions as plot
from regularised_lsq import solveLSQtikhonov
import pyrr

# Set the random seed to ensure reproducibility
np.random.seed(42)


###### Lotus
data_128 = loadmat("realex_files/LotusData128.mat")['m'].T.flatten() # 128 pixel data set

# Define the max of grid square for pixel and cosine
Np,Nc = 128,128

# Define the discretisation number for each pixel and cosine
Sp,Sc = 128,128

# Compute the number of pixel and cosine basis functions
Mp,Mc = Sp**2,Sc**2

# Number of integration points along the ray to calculate Jacobian (need at least 4 per shortest wavelength)
Nline = 500 

# Pixel size in reconstruction (mm) for this model anf tje number of model pixels.
dxy,Npixels = 0.627,128
# Model size     
modelxy = Npixels*dxy
# Model origin                        
xm0,ym0 = -modelxy/2.,-modelxy/2.
# Location of initial X-ray source for angle 0.             
xs,ys = -540.,0.
# End point of screen            
xr,yr0,yr1 = 90.,-60,60
# Number of projections and number of receivers     
Nproj,Nrec = 120,429       
           
# Create a linear grid of positions in x and y directions
x = np.linspace(xm0,xm0+modelxy,Sp+1)
y = np.linspace(ym0,ym0+modelxy,Sp+1)

# Create the 2D pixel basis functions using the linear grid positions
pbasis = octomo.pixelbasis2D(x,y)
paths2D, rayintersect = octomo.find_rays_intersect(Nproj,Nrec,dxy,Np,modelxy,xm0,ym0,xr,yr0,yr1,xs,ys)

# Create straight ray 1D kernel object from paths
raykernels = octomo.straightraykernel1D(paths2D[rayintersect])   

# Create the 2D cosine basis functions using the grid positions -- min X and Y value
cbasis = octomo.cosinebasis2D(xm0,xm0+modelxy,ym0,ym0+modelxy,Sc,npoints=[1,1,Nline])

print('\nTotal number of rays in data set',len(data_128.flatten()))
print('Number of calculated rays that intersect with model',np.sum(rayintersect))

with open("realex_files/Jpixel.pickle",'rb') as file:
    pixel_jacobian_info = pickle.load(file)
Jpixel = pixel_jacobian_info['Jpixel'].toarray()
with open("realex_files/Jcosine.pickle",'rb') as file:
    cosine_jacobian_info = pickle.load(file)
Jcosine = cosine_jacobian_info['Jcosine']

# Define the grid coordinates for plotting the pixel model
Xp,Yp = octomo.find_center_coordinates(pbasis.xmin,pbasis.xmax,Np)
# Define the grid coordinates for plotting the cosine model
Xc,Yc = octomo.find_center_coordinates(cbasis.xmin,cbasis.xmax,Nc)

data = data_128[pixel_jacobian_info['intersect']]
rows = np.arange(Jpixel.shape[0])  # rows to choose from data & G_pixel rows
np.random.shuffle(rows) # shuffle so that it will be random

Gp_size, Gc_size = np.load('realex_files/lotus_size.npy')
alpha = 1e-6
beta = 0.5
sigma = 0.1 # or smaller!

nd = 3000 # number of data to use
d = data[rows[0:nd]]
G_p = Jpixel[rows[0:nd],:]
G_c = Jcosine[rows[0:nd],:]

############ L1-REGULARIZED INVERSION IN AN OVERCOMPLETE BASIS ############
tp = time.process_time()
l1_solution = optimize.minimize(octomo.opt_func,x0=np.zeros(Mp+Mc),args=(d,G_p,G_c,Gp_size,Gc_size,sigma,Mp,alpha,beta,True),jac=True,method='L-BFGS-B', options={'disp': True})
l1_sol = l1_solution.x
process_time_l1 = (time.process_time()-tp)
############ L2-REGULARIZED INVERSION IN AN OVERCOMPLETE BASIS ############
tp = time.process_time()
l2_solution = optimize.minimize(octomo.opt_func,x0=np.zeros(Mp+Mc),args=(d,G_p,G_c,Gp_size,Gc_size,sigma,Mp,alpha,beta,False,True), jac=True,method='L-BFGS-B', options={'disp': True})
l2_sol = l2_solution.x
process_time_l2 = (time.process_time()-tp)
############ L2 INVERSION IN ONLY PIXEL BASIS ############
# Solve the problem using the code from Valentine & Sambridge (2018) see regularised_lsq.py for more details
tp = time.process_time()
l2_sol_pixel, pCov, alphap,betap = solveLSQtikhonov(G_p.T.dot(G_p),G_p.T.dot(d),fullOutput=True)
# l2_solution = lsqr(G_p,d,damp=damp)
# l2_sol_pixel = l2_solution[0]
process_time_l2_pixel = (time.process_time()-tp)
############ L2 INVERSION IN ONLY COSINE BASIS ############
# Solve the problem using the code from Valentine & Sambridge (2018) see regularised_lsq.py for more details
tp = time.process_time()
l2_sol_cosine, pCov, alphac,betac = solveLSQtikhonov(G_c.T.dot(G_c),G_c.T.dot(d),fullOutput=True)
# l2_solution = lsqr(G_c,d,damp=damp)
# l2_sol_cosine = l2_solution[0]
process_time_l2_cosine = (time.process_time()-tp)


print('Number of data: %d' %(G_p.shape[0]) +' and number of unknowns: %d' %(G_p.shape[1]+G_c.shape[1]))
print('----- Process Time(s) ---- \n L1:    ocb: %.3f' %(process_time_l1))
print(' L2:    ocb: %.3f, pb: %.3f, cb: %.3f' %(process_time_l2,process_time_l2_pixel,process_time_l2_cosine))
print('----- optimisation function for inversions in overcomplete basis---- \n phi = 1/(2*N*sigma^2) * ||d - G_p m_p - G_c m_c||_2^2 + alpha/sigma * [ (beta / ||G_p||_2) * ||m_p||_p + ((1-beta)/||G_c||_2) * ||m_c||_p')
print('----- optimisation function for inversions in one basis ----- \n phi = 1/(2*N*sigma^2) * ||d - G m||_2^2 + alpha * ||m||_p')
print('---- Parameters ----')
print('Gp size: %.3f,   \nGc size: %.3f ' %(Gp_size,Gc_size))
print('alpha: %.3f L2 in pixel basis \alpha: %.3f L2 in cosine basis' %(alphap,alphac))
print('alpha: %.4f and beta: %.3f for inv in overcomplete basis for L1&L2' %(alpha,beta))


### Plot
params = {'axes.labelsize': 28,
          'font.size': 28,
          'legend.fontsize': 28,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'savefig.bbox': 'Tight',
          'figure.titlesize': 20,
          'axes.titlesize': 28}
plt.rcParams.update(params)
colors1 = plt.cm.pink(np.linspace(0.5, 1, 128))
colors2 = plt.cm.bone_r(np.linspace(0, 1, 128))
cmap = colors.LinearSegmentedColormap.from_list('my_colormap', np.vstack((colors1, colors2))) # combine them and build a new colormap
cmin = -0.03
cmax = 0.15

### Plot figure 1 -- 
fig = plt.figure(figsize=(15,15))
gs = gridspec.GridSpec(1,1)
gs.update(left=0.05,right=0.35,top=0.99,bottom=0.69)
ax = plt.subplot(gs[:,:])
plot.one_model(ax,(octomo.discretize_models(l1_sol[:Mp],pbasis,Xp,Yp)+octomo.discretize_models(l1_sol[Mp:],cbasis,Xc,Yc)).T,cmap,cmin,cmax)
ax.text(-15,7, "a)")
gs = gridspec.GridSpec(1,1)
gs.update(left=0.04,right=0.36,top=0.65,bottom=0.35)
ax = plt.subplot(gs[:,:])
plot.one_model(ax,octomo.discretize_models(l2_sol_pixel.T,pbasis,Xp,Yp),cmap,cmin,cmax)
ax.text(-15,7, "c)")

gs = gridspec.GridSpec(1,1)
gs.update(left=0.44,right=0.74,top=0.99,bottom=0.69)
plot.one_model(ax,(octomo.discretize_models(l2_sol[:Mp],pbasis,Xp,Yp)+octomo.discretize_models(l2_sol[Mp:],cbasis,Xc,Yc)).T,cmap,cmin,cmax)
ax.text(-16,7, "b)")
gs = gridspec.GridSpec(1,1)
gs.update(left=0.44,right=0.74,top=0.65,bottom=0.35)
plot.one_model(ax,octomo.discretize_models(l2_sol_cosine.T,cbasis,Xc,Yc),cmap,cmin,cmax)
ax.text(-16,7, "d)")
gs = gridspec.GridSpec(1, 1)
gs.update(left=0.3,right=0.5,top=0.0,bottom=-0.5)
cax = plt.subplot(gs[:, :])
plt.axis('off')
cbar = plt.colorbar(im,ax=cax,orientation='horizontal',fraction=0.9,shrink=0.8,aspect=10)
cbar.set_ticks([cmin,(cmax-abs(cmin))/2,cmax])
cbar.set_label('Amplitude',labelpad=-80)
plt.show()