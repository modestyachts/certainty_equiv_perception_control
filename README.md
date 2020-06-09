# Certainty Equivalent Perception-based Control

This repository includes the code necessary for reproducing ...

TODO: list requirements, including CARLA and ORB-SLAM python installs.

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