# Certainty Equivalent Perception-based Control

This repository includes the code necessary for reproducing experiments presented in *Certainty Equivalent Perception-based Control*.

In addition to the packages listed in `requirements.txt`, the code requires the CARLA simulator (https://carla.readthedocs.io/en/latest/start_quickstart/) with python bindings (https://carla.readthedocs.io/en/stable/connecting_the_client/), ORB-SLAM (https://github.com/raulmur/ORB_SLAM2), and ORB-SLAM python bindings (https://github.com/jskinn/ORB_SLAM2-PythonBindings).
The host IP and port of the CARLA simulator should be edited in `observers.py`.

## Reproducing experiments

The following set of commands are sufficient for reproducing experiments.

```
mkdir data/
cd python
python initialize_system_collect_data.py carla-uav
python initialize_system_collect_data.py carla-car
python evaluate_predictors.py carla-uav
python evaluate_predictors.py carla-car
python evaluate_orb.py carla-uav
python evaluate_orb.py carla-car
python closedloop_predictors.py carla-uav
python closedloop_predictors.py carla-uav small
python closedloop_predictors.py carla-car
python closedloop_orb.py carla-car slam
```

The notebook `Plot Results.ipynb` parses the saved data and generates plots.