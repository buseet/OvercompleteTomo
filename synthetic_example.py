'''
This code is a reference implementation of the framework that is described in the paper: 
"Overcomplete Tomography: A novel approach to imaging" by B. Turunctur, A. P. Valentine, and M. Sambridge, 
submitted to RAS Techniques and Instruments.

Buse Turunctur
Research School of Earth Sciences
The Australian National University
buse.turunctur@anu.edu.au
'''
import numpy as np
import pickle
import time
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import overcomplete_tomo as octomo
import plotting_functions as plot
from regularised_lsq import solveLSQtikhonov

# Set the random seed to ensure reproducibility
np.random.seed(42)

# Define the max of grid square for pixel and cosine
Np,Nc = 120,120

# Define the discretisation number for each pixel and cosine
Sp,Sc = 20,20      

# Compute the number of pixel and cosine basis functions
Mp,Mc = Sp**2,Sc**2 

# Create a linear grid of positions in x and y directions
x = np.linspace(1,Np,Sp+1)
y = np.linspace(1,Np,Sp+1)

# Create the 2D pixel basis functions using the linear grid positions
pbasis = octomo.pixelbasis2D(x,y) 

# Define the grid coordinates for plotting the pixel model
Xp,Yp = octomo.find_center_coordinates(2,Np,Np) 

# Set the sparsity level for the pixel model
sparsity_level = 0.055

# Generate a sparse pixel model with the given basis, sparsity level, and the amplitude of non-zero coefficients
true_m_p = octomo.generate_sparse_pixel_model(pbasis,int(Mp*sparsity_level),Sp/Np)
print('The number of pixel model coefficients: ', true_m_p.shape[0])

# Create the 2D cosine basis functions using the grid positions -- min X and Y value
cbasis = octomo.cosinebasis2D(1,Nc,1,Nc,Sc)

# Define the grid coordinates for plotting the cosine model
Xc,Yc = octomo.find_center_coordinates(cbasis.xmin,cbasis.xmax,Nc)

# Generate a sparse cosine model
true_m_c = octomo.generate_sparse_cosine_model()
print('The number of cosine model coefficients: ', true_m_c.shape[0])

# This part should change
with open("synth_files/paths.pickle",'rb') as file:
    paths2D = pickle.load(file)

# Create straight ray 1D kernel object from paths
linekernel2D = octomo.straightraykernel1D(paths2D)   

with open("synth_files/Jpixel.pickle",'rb') as file:
    pixel_jacobian_info = pickle.load(file)
G_p = pixel_jacobian_info['Jpixel'].toarray()
with open("synth_files/Jcosine.pickle",'rb') as file:
    cosine_jacobian_info = pickle.load(file)
G_c = cosine_jacobian_info['Jcosine']

# Forward
data = G_p.dot(true_m_p)+G_c.dot(true_m_c) #Gcombined.dot(mcombined)
sigma = 0.015*abs(data).max()
data+=np.random.normal(0,sigma,size=data.shape)
data.clip(min=0.0)

rows = np.arange(data.shape[0]-1000)  # rows to choose from data & G
np.random.shuffle(rows) # shuffle so that it will be random

##### Define the parameters for inversion
Gp_size = np.linalg.norm(G_p,2)
Gc_size = np.linalg.norm(G_c,2)

nd = 300 # number of data to use
tolr = 1e-20
alpha=1e-4
beta=0.5

############ L1-REGULARIZED INVERSION IN AN OVERCOMPLETE BASIS ############
tp = time.process_time()
l1_solution = optimize.minimize(octomo.opt_func, x0=np.zeros(Mp+Mc),args=(data[rows[0:nd]],G_p[rows[0:nd],:],G_c[rows[0:nd],:],Gp_size,Gc_size,sigma,true_m_p.shape[0],alpha,beta,True,True),jac=True,method='L-BFGS-B', tol=tolr, options={'disp': True})
l1_sol = l1_solution.x
process_time_l1 = (time.process_time()-tp)
############ L2-REGULARIZED INVERSION IN AN OVERCOMPLETE BASIS ############
tp = time.process_time()
l2_solution = optimize.minimize(octomo.opt_func,x0=np.zeros(Mp+Mc),args=(data[rows[0:nd]],G_p[rows[0:nd],:],G_c[rows[0:nd],:],Gp_size,Gc_size,sigma,true_m_p.shape[0],alpha,beta,False,True), jac=True,method='L-BFGS-B', tol=tolr, options={'disp': True})
l2_sol = l2_solution.x
process_time_l2 = (time.process_time()-tp)
############ L2 INVERSION IN ONLY PIXEL BASIS ############
tp = time.process_time()
# Solve the problem using the code from Valentine & Sambridge (2018) see regularised_lsq.py for more details
l2_sol_pixel, pCov, alphap,betap = solveLSQtikhonov(G_p[rows[0:nd],:].T.dot(G_p[rows[0:nd],:]),G_p[rows[0:nd],:].T.dot(data[rows[0:nd]]),fullOutput=True)
process_time_l2_pixel = (time.process_time()-tp)
############ L2 INVERSION IN ONLY COSINE BASIS ############
tp = time.process_time()
# Solve the problem using the code from Valentine & Sambridge (2018) see regularised_lsq.py for more details
l2_sol_cosine, pCov, alphac,betac = solveLSQtikhonov(G_c[rows[0:nd],:].T.dot(G_c[rows[0:nd],:]),G_c[rows[0:nd],:].T.dot(data[rows[0:nd]]),fullOutput=True)
process_time_l2_cosine = (time.process_time()-tp)

print('Number of data: %d' %(G_p[rows[0:nd],:].shape[0]) +' and number of unknowns: %d' %(G_p.shape[1]+G_c.shape[1]))
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
cmin=-0.05
cmax=0.3
### Plot figure 1 -- 
fig = plt.figure(figsize=(15,15))
gs = gridspec.GridSpec(1,1)
gs.update(left=0.05,right=0.35,top=0.99,bottom=0.69)
ax = plt.subplot(gs[:,:])
plot.rays_paths(ax,paths2D[rows[0:nd]])
ax.text(3,115,'%d rays' %data[rows[0:nd]].shape[0], bbox=dict(facecolor='white', edgecolor='black',alpha=0.7))
ax.text(-15,7, "a)")
gs = gridspec.GridSpec(1,1)
gs.update(left=0.04,right=0.36,top=0.65,bottom=0.35)
ax = plt.subplot(gs[:,:])
plot.one_model(ax,octomo.discretize_models(l2_sol_pixel,pbasis,Xp,Yp),cmap,cmin,cmax)
ax.text(-15,7, "c)")
gs = gridspec.GridSpec(1,1)
gs.update(left=0.04,right=0.36,top=0.31,bottom=0.01)
ax = plt.subplot(gs[:,:])
plot.one_model(ax,octomo.discretize_models(l2_sol_cosine,cbasis,Xc,Yc),cmap,cmin,cmax)
ax.text(-15,7, "e)")

gs = gridspec.GridSpec(2,3)
gs.update(left=0.44,right=0.92,top=0.99,bottom=0.69)
ax,im = plot.overcomplete_model(gs,octomo.discretize_models(true_m_p,pbasis,Xp,Yp),octomo.discretize_models(true_m_c,cbasis,Xc,Yc),cmap,cmin,cmax)
ax.text(-16,7, "b)")
gs = gridspec.GridSpec(2,3)
gs.update(left=0.44,right=0.92,top=0.65,bottom=0.35)
ax,im = plot.overcomplete_model(gs,octomo.discretize_models(l2_sol[:true_m_p.shape[0]],pbasis,Xp,Yp),octomo.discretize_models(l2_sol[true_m_p.shape[0]:],cbasis,Xc,Yc),cmap,cmin,cmax)
ax.text(-16,7, "d)")
gs = gridspec.GridSpec(2,3)
gs.update(left=0.44,right=0.92,top=0.31,bottom=0.01)
ax,im = plot.overcomplete_model(gs,octomo.discretize_models(l1_sol[:true_m_p.shape[0]],pbasis,Xp,Yp),octomo.discretize_models(l1_sol[true_m_p.shape[0]:],cbasis,Xc,Yc),cmap,cmin,cmax)
ax.text(-13,7, "f)")

gs = gridspec.GridSpec(1, 1)
gs.update(left=0.3,right=0.5,top=0.0,bottom=-0.5)
cax = plt.subplot(gs[:, :])
plt.axis('off')
cbar = plt.colorbar(im,ax=cax,orientation='horizontal',fraction=0.9,shrink=0.8,aspect=10)
cbar.set_ticks([cmin,(cmax-abs(cmin))/2,cmax])
cbar.set_label('Amplitude',labelpad=-80)
plt.show()
