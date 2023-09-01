import torch
from main import *
import gym
from minerl.herobraine.env_specs.basalt_specs import BasaltBaseEnvSpec
from vima_mc_policy1 import *
from vimamc_dataset import VimaMCDataset


@torch.no_grad()
def main():
    dataset = VimaMCDataset()
    prompt_data, _, actions = dataset[0]
    print("Actions that should the model predict:", actions)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on:", device)

    tokenizer_data = init_tokenizer()

    policy = VIMAMCPolicy(views=views, embed_dim=256, xf_n_layers=1, sattn_n_heads=8, xattn_n_heads=8)
    #policy = VIMAMCPolicy.create_policy_from_ckpt("checkpoint.ckpt")
    policy.to(device)

    inventory = [
        *[{"type": "air", "quantity": 1} for _ in range(19)],
        {
            "type": "oak_planks",
            "quantity": 1
        }
    ]
    obs, env_data = initMineRL(inventory)
    inference_cache = init_cache()
    while True:
        inference_cache = step(policy, obs, env_data, prompt_data, tokenizer_data, inference_cache, device)


def step(policy, obs, env_data, prompt_data, tokenizer_data, inference_cache, device):
    #obs = add_batch_dim(obs)
    obs_data, _ = obs_to_forwarddata(policy, obs, init_cache(), device)
    prompt_data = prompt_to_forwarddata(policy, prompt_data, tokenizer_data, device)
    
    predicted_action_tokens = policy.forward(**obs_data, **prompt_data)
    # forward
    predicted_action_tokens = predicted_action_tokens[-1].unsqueeze(0)
    dist_dict = policy.forward_action_decoder(predicted_action_tokens)
    actions = {k: v.mode() for k, v in dist_dict.items()}
    
    action_tokens = policy.forward_action_token(actions)
    action_tokens = action_tokens.squeeze(0)  # (B, E)
    # add actions to cache
    inference_cache["action_tokens"].append(action_tokens[0])
    
    actions = {k: v.squeeze() for k, v in actions.items()}

    cursor_y = actions["cursor_y"].item()
    cursor_x = actions["cursor_x"].item()
    camera_y = -(cursor_y-50)*env_data["height"]/32/50
    camera_x = (cursor_x-50)*env_data["width"]/53/50
    # executing MineRL step with action
    env = env_data["env"]
    obs, _, _, _ = env.step({"ESC": 0, "camera": [camera_y, camera_x]})
    obs, _, _, _ = env.step({"ESC": 0, "camera": [-camera_y, -camera_x]})
    env.render()
    return inference_cache


def initMineRL(inventory):
    # create MineRL env
    BasaltBaseEnvSpec(
        name="MyTestEnvSpec-v0",
        demo_server_experiment_name="test",
        inventory=inventory,
    ).register()
    print("registered custom environment")

    print("starting environment...")
    env = gym.make("MyTestEnvSpec-v0")
    env.reset()
    # open the inventory
    obs, _, _, _ = env.step({"ESC": 0, "inventory": 1})
    env_data = {
        "env": env,
        "width": obs["pov"].shape[1],
        "height": obs["pov"].shape[0]
    }
    return obs, env_data


if __name__ == "__main__":
    main()