# gTLO
generalized multi-objective deep reinforcement learning algorithm

## Installation
(1) General:
- Manually install tensorflow-gpu up to v. 1.14 (e.g. `conda install tensorflow-gpu=1.14` with python version <= 3.4.)

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
- gTLQ: `python morl_agent.py --config ./preset_configs/DST_gTLO_250ksteps.ini`
- outer-loop gTLQ: `python morl_agent.py --config ./preset_configs/DST_gTLO_outerloop_25kSteps.ini`
- gLinear: `python morl_agent.py --config ./preset_configs/DST_gLinear_250kSteps.ini`
- dTLO (baseline agent): run `study_starter.py` from the FruitAPI fork https://github.com/johannes-dornheim/Fruit-API

### Deep Drawing
- gTLQ: `python morl_agent.py --config ./preset_configs/DeepDrawing_gTLO.ini`
- gLinear: `python morl_agent.py --config ./preset_configs/DeepDrawing_gLinear.ini`

## Cite
```
@misc{https://doi.org/10.48550/arxiv.2204.04988,
  author = {Dornheim, Johannes},
  title = {gTLO: A Generalized and Non-linear Multi-Objective Deep Reinforcement Learning Approach},
  publisher = {arXiv},
  year = {2022},
  doi = {10.48550/ARXIV.2204.04988},
  url = {https://arxiv.org/abs/2204.04988},
  
}
```
