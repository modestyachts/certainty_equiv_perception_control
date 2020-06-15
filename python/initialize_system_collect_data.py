"""Script for collecting training and evaluation data."""
import sys
import pickle
import numpy as np
import tqdm.autonotebook

import experiments as ex


if len(sys.argv) > 1:
    observer_name = sys.argv[1]
else:
    observer_name = 'dot'
filetag = observer_name
directory = '../data/'

## Defining System Dynamics and Observation

n_dims = 2

x0 = np.array([0, 0] * (n_dims))
x0[2] = 2.
x0[3] = -0.1
plant = ex.double_integrator_plant(ndims=n_dims, x0=x0, sw=0., decay=(1, 1))

if observer_name == 'carla-uav':
    observer = ex.carla_observations(setting='uav')
    su = 0.01
elif observer_name == 'carla-car':
    observer = ex.carla_observations(setting='car')
    su = 0.01
elif observer_name == 'dot':
    observer = ex.dot_observations(ndims=n_dims, dot_width=0.05, im_size=50)
    su = 0.01

## Defining reference trajectory for training data

# these parameters determine how fine the training grid is
f = 0.01
a = 2
amp_range = np.linspace(1.75, 2.25, 5)

yref = []
for a in amp_range:
    for t in range(int(1./f)):
        yref.append(np.array([a*np.sin(f*2*np.pi*t), a*np.cos(f*2*np.pi*t)])) # , 1]))
yref = np.array(yref)
xref = yref @ observer.C
observer.x0 = xref[0]

## Defining the controller

controller_C = observer.C

Q = controller_C.T @ controller_C
R = np.eye(plant.B.shape[1])
K = ex.lqr_inf_horizon(plant.A, plant.B, Q, R)[0]

W = np.eye(controller_C.shape[1])
V = 0.1*np.eye(controller_C.shape[0])
L = ex.lqg_inf_horizon(plant.A, observer.C, W, V)[0]

controller = ex.periodic_tracking_controller(xref, plant.A, plant.B,
                                             controller_C, -K, L, su=su,
                                             x0=xref[0])

## Sampling Interconnection for Training

# this parameter determines the label noise
sv = 0.01
measurement = ex.linear_observations(controller_C, sv)

interconnection = ex.interconnection(plant, observer.observe, controller,
                                     get_observation_for_controller=measurement.observe)
T = 2000
for _ in tqdm.autonotebook.tqdm(range(T)):
    interconnection.step()

# saving training data
interconnection.get_observation = None # remove CARLA dependency for pickle
with open('{}training_interconnection_{}.pkl'.format(directory, filetag),
          'wb') as output:
    pickle.dump(interconnection, output, pickle.HIGHEST_PROTOCOL)
np.savez('{}training_{}.npz'.format(directory, filetag), yref=yref, C=observer.C)
interconnection.get_observation = observer.observe # put CARLA observer back

## Sampling Grid for Error Characterization

size = 2.5
ngrid = 50
grid = np.linspace(-size, size, ngrid)
xs_pos = []
ys_pos = []
ypos_list = np.linspace(-size, size, ngrid)
xpos_list = np.linspace(-size, size, ngrid)

for ypos in ypos_list:
    for xpos in xpos_list:
        ys_pos.append(ypos)
        xs_pos.append(xpos)

xs_grid = []
zs_grid = []

for i in tqdm.autonotebook.tqdm(range(len(xs_pos))):
    xpos = xs_pos[i]
    ypos = ys_pos[i]
    # the values for vx and vy result in nice angles for CARLA
    x = np.array([xpos, ypos, ypos, -xpos])
    xs_grid.append(x)
    zs_grid.append(observer.observe(x, angle_offset=0.))

np.savez('{}grid_{}.npz'.format(directory, filetag), xs_grid=xs_grid,
         zs_grid=zs_grid)

if 'carla' in observer_name:
    observer.kill()
