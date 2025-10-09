from vlarl_launcher.algorithm.dppo.dppo_buffer import DPPOBuffer
import pickle
import tensordict

filename = "rollout_buffer.pkl"

with open(filename, 'rb') as f:
    rollout_buffer: DPPOBuffer = pickle.load(f)
    
import torch
dataloader = torch.utils.data.DataLoader(
    rollout_buffer,
    batch_size=1024,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    num_workers=0,
    collate_fn=rollout_buffer.collate_fn
)
import ipdb; ipdb.set_trace()
for batch in dataloader:
    
    obs, action, oldlogprob, reward, value, advantage, ret = tuple(t.to('cpu') for t in batch)
    print(obs.shape, action.shape, oldlogprob.shape, reward.shape, value.shape, advantage.shape, ret.shape)
    break