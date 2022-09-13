from utils.config import Config

default_config = Config({
    # global program config
    "n_thread": 1,
    "n_sample": 4096,
    "n_epoch": 2000,
    
    # global policy config
    "activation": "relu",
    "hidden_policy": (64, 64),
    "shared_policy": False, # TODO: fine-tune
    "log_clamp_policy": (-20., 0.),
    "optimizer_lr_policy": 3.e-4,

    "dim_c": 4,
    "hidden_option": (64, 64),
    "optimizer_lr_option": 3.e-4,

    # ppo config
    "hidden_critic": (64, 64),
    "shared_critic": False,
    "train_policy": True,
    "optimizer_lr_critic": 3.e-4,

    "use_gae": True,
    "gamma": 0.99,
    "gae_tau": 0.95,
    "clip_eps": 0.2,
    "mini_batch_size": 256,
    "lambda_entropy_policy": 0., # TODO: fine-tune
    "lambda_entropy_option": 1.e-4, # TODO: probably the most important parameter

    "pretrain_log_interval": 500,
    "log_interval": 5,

    # MHA-related
    "dmodel": 40, # dimension of the embedding
    "mha_nhead": 1, # number of attention head
    "mha_nlayers": 1,
    "mha_nhid": 50,
    "dropout": 0.2,
    "use_MHA_critic": False, # we suggest use false here, so that the algorithm for learning the hier policy would be DAC
    "use_MHA_policy": True,

    # posterior related
    "gru_hid_dim": 64,
    "n_gru_layers": 2,
    "gru_dropout": 0.2,
    "gru_include_action": False,
    "gru_training_iterations": 10, # TODO: fine-tune
    "use_posterior_sampling": False, # we suggest use False

    # il config
    "hidden_discriminator": (256, 256),
    "shared_discriminator": False, # TODO: fine-tune
    "optimizer_lr_discriminator": 3.e-4
})

mujoco_config = default_config.copy()

rlbench_config = default_config.copy()



