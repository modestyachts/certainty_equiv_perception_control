"""Script for evaluating predictors on grid."""
import sys
import pickle
import numpy as np
import tqdm.autonotebook

import predictors as pr


if len(sys.argv) > 1:
    observer_name = sys.argv[1]
else:
    observer_name = 'dot'
filetag = observer_name
directory = '../data/'

## Loading

with open('{}training_interconnection_{}.pkl'.format(directory, filetag), 'rb') as fileinput:
    interconnection = pickle.load(fileinput)
params = np.load('{}training_{}.npz'.format(directory, filetag))
C = params['C']
controller_C = interconnection.controller.C
yref = params['yref']

## Instantiating Predictor

predictors = {}

if observer_name == 'carla-uav':
    param_list = {'kernel': np.linspace(15000, 45000, 10),
                  'krr': np.linspace(0.0001, 1, 10)}
    model_list = ['kernel', 'krr']
    # model_list = ['kernel-gaussian']
elif observer_name == 'carla-car':
    param_list = {'kernel': np.linspace(10000, 40000, 10)}
    model_list = ['kernel']

for i in range(len(model_list)):
    model = model_list[i]
    assert np.all(np.isclose(C, controller_C))
    ys_label = interconnection.zs_c
    if model == 'krr':
        pred = pr.KernelRidgePredictor(zs=interconnection.zs, ys=ys_label)
    elif model == 'kernel':
        pred = pr.KernelPredictor(zs=interconnection.zs, ys=ys_label)
    predictors[model] = pred

## Evaluating on Grid

grid_params = np.load('{}grid_{}.npz'.format(directory, filetag))
xs_grid = grid_params['xs_grid']
zs_grid = grid_params['zs_grid']

ys_true = np.array(xs_grid) @ C.T

all_ys_est = {}
all_errs_est = {}
key_list = list(predictors.keys())
for i in tqdm.autonotebook.tqdm(range(len(key_list))):
    key = key_list[i]
    ys_est, errs_est = predictors[key].compute_pred(zs_grid, param_list=param_list[key])
    for param in ys_est.keys():
        all_ys_est['{}_{}'.format(key, param)] = np.array(ys_est[param])
        if errs_est is not None:
            all_errs_est['{}_{}'.format(key, param)] = np.array(errs_est[param])

for key in all_ys_est.keys():
    if key in all_errs_est.keys():
        errs_save = all_errs_est[key]
    else:
        errs_save = None
    np.savez('{}estimates_{}_{}'.format(directory, key, filetag), all_ys_est=all_ys_est[key],
             all_errs_est=errs_save)
