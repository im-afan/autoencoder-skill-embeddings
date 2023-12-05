from torch import nn

normalize = True
n_envs = 16
n_timesteps = 2e6
policy = 'MlpPolicy'
batch_size = 128
n_steps = 512
gamma = 0.99
gae_lambda = 0.9
n_epochs = 20
ent_coef = 0.0
sde_sample_freq = 4
max_grad_norm = 0.5
vf_coef = 0.5
learning_rate = 3e-5
use_sde = True
clip_range = 0.4
policy_kwargs = dict(log_std_init=-2,
                    ortho_init=False,
                    activation_fn=nn.ReLU,
                    net_arch=dict(pi=[256, 256], vf=[256, 256])
                    )

AUTOENCODER_LATENT_SIZE_PANDA = 4
AUTOENCODER_LATENT_SIZE_ANT = 3
AUTOENCODER_LATENT_SIZE_HUMANOID = 4
ENV_NAME = "Ant"
DECODER_PATH = "autoencoder_pretrained/ant/decoder.pth"
