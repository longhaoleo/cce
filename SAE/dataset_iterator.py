import webdataset as wds
import os
import torch

class ActivationsDataloader:
    def __init__(self, paths_to_datasets, block_name, batch_size, output_or_diff='diff', uncond_or_cond="cond", timestep="mid", num_in_buffer=100):
        assert output_or_diff in ['diff', 'output'], "Provide 'output' or 'diff'"
        assert uncond_or_cond in ['cond', 'uncond'], "Provide 'cond' or 'uncond'"
        assert timestep in ['first', 'mid', 'last'], "Provide 'first', 'mid', or 'last' for timestep"

        self.dataset = wds.WebDataset(
            [os.path.join(path_to_dataset, f"{block_name}.tar")
            for path_to_dataset in paths_to_datasets]
        ).decode("torch")
        self.iter = iter(self.dataset)
        self.buffer = None
        self.pointer = 0
        self.num_in_buffer = num_in_buffer
        self.output_or_diff = output_or_diff
        self.uncond_or_cond = uncond_or_cond # "cond" or "uncond"
        self.timestep = timestep # "first", "mid", "last"
        self.batch_size = batch_size
        self.one_size = None

    def renew_buffer(self, to_retrieve):
        to_merge = []
        if self.buffer is not None and self.buffer.shape[0] > self.pointer:
            to_merge = [self.buffer[self.pointer:].clone()]
        del self.buffer
        for _ in range(to_retrieve):
            sample = next(self.iter)
            latents = sample['output.pth'] if self.output_or_diff == 'output' else sample['diff.pth'] 
            # Saved feature shape is [N, T, D, H, W], where:
            #   N=2 is uncond/text cond., 
            #   T is number of time steps, in the order first (t=1), middle (t=501), last (t=981) step of reverse diffusion,
            #   D=1280 is number of features, 
            #   H is height, 16 for non-mid block, 8 for mid block
            #   W is width
            
            if latents.shape[0] == 2:
                latents = latents[0] if self.uncond_or_cond == 'uncond' else latents[1]
                
                if self.timestep == 'first':
                    latents = latents[0]
                elif self.timestep == 'mid':
                    latents = latents[1]
                elif self.timestep == 'last':
                    latents = latents[2]
            else: # This is a little bit hacky right now...
                latents = latents[0]
                if self.timestep == 'first':
                    latents = latents[0]
                elif self.timestep == 'mid':
                    latents = latents[1]
            
            latents = latents.permute((1,2,0))
            latents = latents.reshape((-1, latents.shape[-1]))
            
            to_merge.append(latents.to('cuda'))
            self.one_size = latents.shape[0]
        self.buffer = torch.cat(to_merge, dim=0)
        shuffled_indices = torch.randperm(self.buffer.shape[0])
        self.buffer = self.buffer[shuffled_indices]
        self.pointer = 0

    def iterate(self):
        while True:
            if self.buffer == None or self.buffer.shape[0] - self.pointer < self.num_in_buffer * self.one_size * 4 // 5:
                try:
                    to_retrieve = self.num_in_buffer if self.buffer is None else self.num_in_buffer 
                    self.renew_buffer(to_retrieve)
                except StopIteration:
                    break
            batch = self.buffer[self.pointer: self.pointer + self.batch_size]
            self.pointer += self.batch_size
            assert batch.shape[0] == self.batch_size
            yield batch