"""Script for training then running ORB-SLAM in closedloop."""
import sys
import time
import pickle
import numpy as np

# need to import ORBSLAM before CARLA
from orb_utils import OrbPredictor
import experiments as ex

observer_name = sys.argv[1]
filetag = observer_name
directory = '../data/'

if observer_name == 'carla-uav':
    scale = ex.CARLA_UAV_SCALE
    height = ex.CARLA_UAV_HEIGHT
else:
    scale = ex.CARLA_CAR_SCALE
    height = ex.CARLA_CAR_HEIGHT

if len(sys.argv) <= 2:
    pred_type = 'slam'
else:
    pred_type = sys.argv[2]

# Loading train data

with open('{}training_interconnection_{}.pkl'.format(directory, filetag), 'rb') as fileinput:
    interconnection = pickle.load(fileinput)
params = np.load('{}training_{}.npz'.format(directory, filetag))
C = params['C']
controller_C = interconnection.controller.C

ys = (interconnection.xs @ C.T)
ys_label = interconnection.zs_c

# Initializing predictor

slam_predictor = OrbPredictor(fps=10., scale=scale, height=height)

num_train_images = 200
slam_predictor.process_train(interconnection.zs, ys_label,
                             cutoff=num_train_images)

slam_predictor.online = True

# Initializing reference

yref = []
f = 0.01
a = 2
for t in range(int(1./f)):
    yref.append(np.array([(0.95*a)*np.sin(f*2*np.pi*t), a*np.cos(f*2*np.pi*t)]))
yref = np.array(yref)
xref = yref @ C

# small initial speed for correct angle
xref[0, 1] = xref[0, 2]/100
xref[0, 3] = -xref[0, 0]/100

# re-initialize observer
if observer_name == 'carla-uav':
    observer = ex.CarlaObservations(setting='uav')
elif observer_name == 'carla-car':
    observer = ex.CarlaObservations(setting='car')

interconnection.get_observation = observer.observe


# VO CLOSED LOOP

if pred_type == 'vo':

    slam_predictor.reset_between = True

    
    interconnection.plant.reset(x0=xref[0])
    controller_p = ex.PeriodicTrackingController(xref, interconnection.plant.A,
                                                   interconnection.plant.B, controller_C,
                                                   interconnection.controller.K,
                                                   interconnection.controller.L,
                                                   perception=slam_predictor.pred, x0=xref[0])


    # running interconnection forward
    interconnection_v = ex.Interconnection(interconnection.plant, observer.observe, controller_p)
    T = 100
    for i in range(T):
        interconnection_v.step()
        curr_x = interconnection_v.xs[-1]
        print(i, curr_x)
        time.sleep(1./10.)
        if curr_x[0]**2 + curr_x[2]**2 > 4**2:
            break

    interconnection_v.get_observation = None
    interconnection_v.controller.perception = None
    with open('{}CL_vo_interconnection_{}.pkl'.format(directory, filetag),
              'wb') as output:
        pickle.dump(interconnection_v, output, pickle.HIGHEST_PROTOCOL)
    np.savez('{}CL_vo_{}.npz'.format(directory, filetag), yref=yref)


# SLAM CLOSED LOOP

if pred_type == 'slam':

    slam_predictor.reset_between = False

    interconnection.plant.reset(x0=xref[0])
    controller_p = ex.PeriodicTrackingController(xref, interconnection.plant.A,
                                                   interconnection.plant.B, controller_C,
                                                   interconnection.controller.K,
                                                   interconnection.controller.L,
                                                   perception=slam_predictor.pred, x0=xref[0])


    # running interconnection forward
    interconnection_v = ex.Interconnection(interconnection.plant, observer.observe, controller_p)
    T = 100
    for i in range(T):
        interconnection_v.step()
        curr_x = interconnection_v.xs[-1]
        print(i, curr_x)
        time.sleep(1./10.)
        if curr_x[0]**2 + curr_x[2]**2 > 4**2:
            break

    interconnection_v.get_observation = None
    interconnection_v.controller.perception = None
    with open('{}CL_slam_interconnection_{}.pkl'.format(directory, filetag),
              'wb') as output:
        pickle.dump(interconnection_v, output, pickle.HIGHEST_PROTOCOL)
    np.savez('{}CL_slam_{}.npz'.format(directory, filetag), yref=yref)

observer.kill()
slam_predictor.shutdown()
