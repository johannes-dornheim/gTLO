# gTLO
generalized multi-objective deep reinforcement learning algorithm

## Installation
(1) General:
- Manually install tensorflow-gpu up to v. 1.14 (e.g. `conda install tensorflow-gpu=1.14` with python version <= 1.4.)

(2.a) For Deep-Sea Treasure Environment:
- Install ALE following https://github.com/garlicdevs/Fruit-API
- Install fruitAPI

(2.b) For Deep Drawing Environment:
- Prerequisite: abaqus (Tested on V. 6.14), Student version should be sufficient
- Install gym_fem following https://github.com/johannes-dornheim/Reinforce-FE

(3) General:

- install gTLO (`pip install .` in gTLO root folder) 

## Run presets / reproduce paper results
experiments are managed by agents/morl_agent.py and configured in ini files. To reproduce the results presented within the gTLO paper, the example configurations can be used as follows:

### DST
- gTLQ: `python morl_agent.py --config ./ini_templates/configTMPL_DST_gTLO_250ksteps.ini`
- outer-loop gTLQ: `python morl_agent.py --config ./ini_templates/configTMPL_DST_gTLO_outerloop_25kSteps.ini`
- gLinear: `python morl_agent.py --config ./ini_templates/configTMPL_DST_gLinear_250kSteps.ini`
- dTLO (baseline agent): run `study_starter.py` from the FruitAPI fork https://github.com/johannes-dornheim/Fruit-API

### Deep Drawing