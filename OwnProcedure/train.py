from vima_mc_policy1 import *
import torch
from torch.utils.data import DataLoader, Subset
from vimamc_dataset import VimaMCDataset
from main import *

batch_size = 16

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on:", device)

    tokenizer_data = init_tokenizer()

    policy = VIMAMCPolicy(views=views, embed_dim=256, xf_n_layers=1, sattn_n_heads=8, xattn_n_heads=8)
    #ckpt = torch.load("2M.ckpt", map_location=device)
    #policy = VIMAMCPolicy(views, **ckpt["cfg"])
    #state_dict = ckpt["state_dict"]
    #state_dict = {k.replace("policy.", ""): v for k, v in state_dict.items()}
    #state_dict = {k.replace("top", "pov"): v for k, v in state_dict.items()}
    #state_dict = {k: v for k, v in state_dict.items() if "front" not in k}
    #policy.load_state_dict(state_dict, strict=True)
    #policy.load_state_dict(torch.load("checkpoint_epoch4.ckpt"))
    policy.to(device)
    
    training_loader = DataLoader(VimaMCDataset(), batch_size=batch_size, shuffle=True, collate_fn=lambda batch: batch)

    
    loss_fn = torch.nn.CrossEntropyLoss()

    print("Training:")
    train(policy, training_loader, tokenizer_data, loss_fn, device)


def train(policy, training_loader, tokenizer_data, loss_fn, device):
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=0, patience=0, verbose=True)

    EPOCHS = 10
    for epoch in range(0, EPOCHS):
        policy.train(True)
        avg_loss = 0
        for batch_index, batch in enumerate(training_loader):
            optimizer.zero_grad()
            
            for i, (prompt_data, obs, actions) in enumerate(batch):
                actions = {k: torch.tensor([v], device=device) for k, v in actions.items()}

                actions["cursor_x"] = actions["cursor_x"]/2
                obs_data, _ = obs_to_forwarddata(policy, obs, init_cache(), device)
                prompt_data = prompt_to_forwarddata(policy, prompt_data, tokenizer_data, device)
                
                logits_x = np.zeros(100) # 50 when using VIMAs pose0 action
                logits_y = np.zeros(100)
                logits_x[round(actions["cursor_x"].item())] = 1
                logits_y[round(actions["cursor_y"].item())] = 1

                logits_x = torch.tensor(logits_x, device=device)
                logits_y = torch.tensor(logits_y, device=device)


                predicted_action_tokens = policy.forward(**obs_data, **prompt_data)

                predicted_dist_dict = policy.forward_action_decoder(predicted_action_tokens[-1].unsqueeze(0))
                predicted_logits_x = torch.softmax(predicted_dist_dict["cursor_x"]._dists[0].logits.squeeze(), dim=0)
                predicted_logits_y = torch.softmax(predicted_dist_dict["cursor_y"]._dists[0].logits.squeeze(), dim=0)
                #predicted_logits_x = torch.softmax(predicted_dist_dict["pose0_position"]._dists[0].logits.squeeze(), dim=0)
                #predicted_logits_y = torch.softmax(predicted_dist_dict["pose0_position"]._dists[1].logits.squeeze(), dim=0)

                loss_x = loss_fn(predicted_logits_x, logits_x)
                loss_y = loss_fn(predicted_logits_y, logits_y)
                loss = loss_x + loss_y                
                loss.backward()

                actions = {k: round(v.squeeze().item()) for k, v in actions.items()}
                predicted_actions = {k: v.mode().squeeze() for k, v in predicted_dist_dict.items()}
                print(f"Epoch: {epoch}, Batch: {batch_index}, Datapoint: {i} => lr={optimizer.param_groups[0]['lr']} loss: {loss.item()}")
                print("actions", actions, "vs predicted", predicted_actions)

                avg_loss += loss.item()
                optimizer.step()
        torch.save(policy.state_dict(), f"checkpoint_epoch{epoch}.ckpt")
        
        avg_loss /= batch_size*len(training_loader)
        scheduler.step(avg_loss)


if __name__ == "__main__":
    main()