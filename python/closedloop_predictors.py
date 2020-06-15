"""Script for training then running NW regressor in closedloop."""
import sys
import pickle
import numpy as np

import experiments as ex
import predictors as pr

observer_name = sys.argv[1]
if len(sys.argv) <= 2:
    reference_rad = ''
else:
    reference_rad = sys.argv[2]
filetag = observer_name
directory = '../data/'

with open('{}training_interconnection_{}.pkl'.format(directory, filetag), 'rb') as fileinput:
    interconnection = pickle.load(fileinput)
params = np.load('{}training_{}.npz'.format(directory, filetag))
C = params['C']
controller_C = interconnection.controller.C
yref = params['yref']
ys_label = interconnection.zs_c

# re-initialize observer
if observer_name == 'carla-uav':
    observer = ex.carla_observations(setting='uav')
elif observer_name == 'carla-car':
    observer = ex.carla_observations(setting='car')

interconnection.get_observation = observer.observe

if observer_name == 'carla-uav':
    gamma = 25000.0
elif observer_name == 'carla-car':
    gamma = 16666.666666666668

pred = pr.KernelPredictor(zs=interconnection.zs, ys=ys_label,
                          gamma=gamma, online=True)

yref = []
f = 0.01
a = 2
if reference_rad == '':
    a_x = 0.95*a
else:
    a_x = 0.8*a
for t in range(int(1./f)):
    yref.append(np.array([a_x*np.sin(f*2*np.pi*t), a*np.cos(f*2*np.pi*t)]))
yref = np.array(yref)
xref = yref @ C
xref[0, 1] = xref[0, 2]/100
xref[0, 3] = -xref[0, 0]/100

interconnection.plant.reset(x0=xref[0])


controller_p = ex.PeriodicTrackingController(xref, interconnection.plant.A,
                                               interconnection.plant.B, controller_C,
                                               interconnection.controller.K,
                                               interconnection.controller.L,
                                               perception=pred.pred, x0=xref[0])
interconnection_v = ex.Interconnection(interconnection.plant, observer.observe, controller_p)
T = 110
for i in range(T):
    interconnection_v.step()
    curr_x = interconnection_v.xs[-1]
    print(i, curr_x)
    if curr_x[0]**2 + curr_x[2]**2 > 4**2:
        break

interconnection_v.get_observation = None
interconnection_v.controller.perception = None
with open('{}{}CL_kernel_interconnection_{}.pkl'.format(directory, reference_rad, filetag),
          'wb') as output:
    pickle.dump(interconnection_v, output, pickle.HIGHEST_PROTOCOL)
np.savez('{}{}CL_kernel_{}.npz'.format(directory, reference_rad, filetag), yref=yref)

observer.kill()
