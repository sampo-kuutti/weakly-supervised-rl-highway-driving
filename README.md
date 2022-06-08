# Weakly supervised reinforcement learning for autonomous highway driving via virtual safety cages

This is the repo for paper Weakly supervised reinforcement learning for autonomous highway driving via virtual safety cages. 
Train a DDPG policy to control an autonomous agent for vehicle following. 
Additionally, safety cages can be used to ensure safety and provide additional training signal when the policy
makes a mistake.
Testing can be completed with the IPG CarMaker Simulator, or with adversarial RL policies which aim to learn to cause mistakes in the target policy.


## Installation
Clone the repo

```bash
git clone https://github.com/sampo-kuutti/weakly-supervised-rl-highway-driving/.git
```

install requirements:
```bash
pip install -r requirements.txt
```

## Training the policy


For training the model, run `train_ddpg.py`.

## Citing the Repo

If you find the code useful in your research or wish to cite it, please use the following BibTeX entry.

```text
@article{kuutti2021weakly,
  title={Weakly supervised reinforcement learning for autonomous highway driving via virtual safety cages},
  author={Kuutti, Sampo and Bowden, Richard and Fallah, Saber},
  journal={Sensors},
  volume={21},
  number={6},
  pages={2032},
  year={2021},
  publisher={Multidisciplinary Digital Publishing Institute}
}
}
```
