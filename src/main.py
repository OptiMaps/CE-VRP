from Environment import TSPEnv
from Model import TSPInitEmbedding, TSPContext, StaticEmbedding
from rl4co.utils.decoding import random_policy, rollout
from rl4co.models.zoo import AttentionModel, AttentionModelPolicy
from rl4co.utils.trainer import RL4COTrainer
import torch
import wandb
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    print("-------torch version check start-----------")
    print(f'torch.__version__: {torch.__version__}')
    print(f'GPU 사용여부: {torch.cuda.is_available()}')
    gpu_count = torch.cuda.device_count()
    print(f'GPU count: {gpu_count}')
    if gpu_count > 0:
        print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print("-------torch version check finish-----------")

    load_dotenv("../.env")
    batch_size = 2

    env = TSPEnv(generator_params=dict(num_loc=20))
    reward, td, actions = rollout(env, env.reset(batch_size=[batch_size]), random_policy)
    # env.render(td, actions)

    # Instantiate our environment
    env = TSPEnv(generator_params=dict(num_loc=20))

    # Instantiate policy with the embeddings we created above
    emb_dim = 128
    policy = AttentionModelPolicy(env_name=env.name, # this is actually not needed since we are initializing the embeddings!
                                embed_dim=emb_dim,
                                init_embedding=TSPInitEmbedding(emb_dim),
                                context_embedding=TSPContext(emb_dim),
                                dynamic_embedding=StaticEmbedding(emb_dim)
    )
    
    # Model: default is AM with REINFORCE and greedy rollout baseline
    model = AttentionModel(env,
                        policy=policy,
                        baseline='rollout',
                        train_data_size=100_000,
                        val_data_size=10_000)
    
    # Greedy rollouts over untrained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td_init = env.reset(batch_size=[3]).to(device)
    policy = model.policy.to(device)
    out = policy(td_init.clone(), env, phase="test", decode_type="greedy", return_actions=True)
    actions_untrained = out['actions'].cpu().detach()
    rewards_untrained = out['reward'].cpu().detach()
    print("------------------")
    print(os.environ.get("WANDB_API_KEY"))
    print("------------------")
    wandb.login(
        key=os.environ.get("WANDB_API_KEY"),
        relogin=True,
        force=True
    )

    wandb.init(
        project="rl4co"
    )

    # We use our own wrapper around Lightning's `Trainer` to make it easier to use
    trainer = RL4COTrainer(max_epochs=3, devices=1)
    trainer.fit(model)


    # Greedy rollouts over trained policy (same states as previous plot)
    policy = model.policy.to(device)
    out = policy(td_init.clone(), env, phase="test", decode_type="greedy", return_actions=True)
    actions_trained = out['actions'].cpu().detach()

    # evaluation
    for i, td in enumerate(td_init):
        print(f"Untrained | Cost = {-rewards_untrained[i].item():.3f}")
        print(r"Trained $\pi_\theta$" + f"| Cost = {-out['reward'][i].item():.3f}")