<h1 align="center"> Constraint embedding for recognizing interactions between constraints in Solving Vehicle Routing Problems </h1>

This is the PyTorch code for the paper **"Constraint embedding for recognizing interactions between constraints in Solving Vehicle Routing Problems"**.
framework implemented on [RL4CO](https://github.com/ai4co/rl4co)

Vehicle Routing Problems (VRPs) are critical challenges in intelligent transportation logistics, often constrained by various complex environments such as capacity and time windows. Traditional neural network (NN)-based approaches address these constraints by masking infeasible actions but fail to capture the underlying interactions between constraints. This leads to inefficient learning and suboptimal generalization in environments with complex constraints.

In this work, we propose a novel methodology to embed action-dependent constraint interactions as learnable representations. By explicitly encoding the causes of constraint violations, our approach preserves the context of these violations and integrates it into the decision-making process. This method is potentially enhancing performance while significantly reducing computational costs during training.

## Training

```shell
python run.py experiment=routing/am env={CVRP,CVRPTW} model.policy_kwargs.constraint_method={'weighted','linear','none'}
```
