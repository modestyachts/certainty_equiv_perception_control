# Certainty Equivalent Perception-based Control

This repository includes the code necessary for reproducing experiments presented in *Certainty Equivalent Perception-based Control*.

To run, the code requires the CARLA simulator (https://carla.readthedocs.io/en/latest/start_quickstart/), ORB-SLAM (https://github.com/raulmur/ORB_SLAM2), and python bindings (https://github.com/jskinn/ORB_SLAM2-PythonBindings).

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
python closedloop_orb.py carla-car
```