"""rsl_rl_ppo_cfg.py - PPO hiperparametri.

Polazne vrijednosti, ne optimirane (§10 kaze da su empirijske). Sto stvarno
treba mjeriti prije podesavanja:

- num_envs krece na 64. Opazanje je state-based (nema kamere u petlji), pa
  je ovo puno jeftinije od vizualnog RL-a, ali fizika hvata s cetiri prsta
  u kontaktu nije besplatna. Mjeri VRAM i FPS pa se penji.
- ako PPO ne konvergira dovoljno brzo, SAC je alternativa (§7) - off-policy
  bolje podnosi skupe env korake, sto je ovdje slucaj.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class DoorPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "door_opening"
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
