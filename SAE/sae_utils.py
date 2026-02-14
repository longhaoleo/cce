from dataclasses import dataclass

@dataclass
class SAETrainingConfig:
    d_model: int
    n_dirs: int
    k: int
    block_name: str
    bs: int
    save_path_base: str
    uncond_or_cond: str
    timestep: str
    auxk: int = 256
    lr: float = 1e-4
    eps: float = 6.25e-10
    dead_toks_threshold: int = 10_000_000
    auxk_coef: float = 1/32
    n_epochs: int = 1
    custom_name_suffix: str = ''

    @property
    def sae_name(self):
        return f'{self.block_name}_k{self.k}_hidden{self.n_dirs}_auxk{self.auxk}_bs{self.bs}_lr{self.lr}_{self.uncond_or_cond}_timestep_{self.timestep}_n_epochs_{self.n_epochs}{self.custom_name_suffix}'
    
    @property
    def save_path(self):
        return f'{self.save_path_base}/{self.block_name}_k{self.k}_hidden{self.n_dirs}_auxk{self.auxk}_bs{self.bs}_lr{self.lr}_{self.uncond_or_cond}_timestep_{self.timestep}_n_epochs_{self.n_epochs}{self.custom_name_suffix}'


@dataclass
class Config:
    saes: list[SAETrainingConfig]
    paths_to_latents: list[str]
    log_interval: int
    save_interval: int
    bs: int
    n_epochs: int
    block_name: str
    wandb_project: str = 'sd1.4_sae_train_demo'
    custom_name: str = ''

    def __init__(self, cfg_json):
        self.saes = [SAETrainingConfig(**sae_cfg, block_name=cfg_json['block_name'], bs=cfg_json['bs'], n_epochs=cfg_json['n_epochs'], save_path_base=cfg_json['save_path_base'], uncond_or_cond=cfg_json['uncond_or_cond'], custom_name_suffix = cfg_json['custom_name'], timestep=cfg_json['timestep']) 
                    for sae_cfg in cfg_json['sae_configs']]

        self.save_path_base = cfg_json['save_path_base']
        self.paths_to_latents = cfg_json['paths_to_latents']
        self.log_interval = cfg_json['log_interval']
        self.save_interval = cfg_json['save_interval']
        self.bs = cfg_json['bs']
        self.n_epochs = cfg_json['n_epochs']
        self.block_name = cfg_json['block_name']
        self.uncond_or_cond = cfg_json['uncond_or_cond']
        self.timestep = cfg_json['timestep']