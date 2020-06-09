import orbslam2
import numpy as np
import pickle
import sys

import experiments as ex
from orb_utils import *

observer_name = sys.argv[1] 
filetag = observer_name
directory = '../data/'

if observer_name == 'carla-uav':
    scale = ex.CARLA_UAV_SCALE # multiplier to convert into meters
    height = ex.CARLA_UAV_HEIGHT # fixed height parameter
else:
    scale = ex.CARLA_CAR_SCALE # multiplier to convert into meters
    height = ex.CARLA_CAR_HEIGHT # fixed height parameter

## Loading train data

with open('{}training_interconnection_{}.pkl'.format(directory,filetag), 'rb') as input:
    interconnection = pickle.load(input)
params = np.load('{}training_{}.npz'.format(directory,filetag))
C = params['C']
controller_C = interconnection.controller.C

ys = (interconnection.xs @ C.T)
ys_label = interconnection.zs_c

## Initializing predictor

slam_predictor = OrbPredictor(fps=10., scale=scale, height=height)

num_train_images = 200 
slam_predictor.process_train(interconnection.zs, ys_label,
                             cutoff=num_train_images)

## Loading grid evaluation

grid_params = np.load('{}grid_{}.npz'.format(directory,filetag))
xs_grid = grid_params['xs_grid']
zs_grid = grid_params['zs_grid']

### VO evaluation
pos_est = slam_predictor.process_grid(zs_grid, speed=True, no_mapping=True) 
np.savez('{}pose_est_{}.npz'.format(directory,filetag), pos_est=pos_est)
errs = pos_est[:,:2] - scale * xs_grid[:,[0,2]]

### SLAM evaluation
grid_reorder_inds = []
num_grid = int(np.sqrt(len(xs_grid)))
for i_chunk in range(num_grid):
    if i_chunk % 2 == 0:
        grid_reorder_inds += list(range(num_grid*(i_chunk),num_grid*(i_chunk+1)))
    else:
        grid_reorder_inds += list(range(num_grid*(i_chunk+1)-1,num_grid*(i_chunk)-1, -1))
inverse_reorder_inds = np.argsort(grid_reorder_inds)

pose_est_sequential = slam_predictor.process_grid(zs_grid[grid_reorder_inds], speed=True, 
                                                  no_mapping=False)
pose_est_sequential = pose_est_sequential[inverse_reorder_inds]
np.savez('{}pose_est_sequential_{}.npz'.format(directory,filetag), pos_est=pose_est_sequential)

slam_predictor.shutdown()