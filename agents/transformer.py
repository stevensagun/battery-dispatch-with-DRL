"""Transformer Q-learning agent and reusable temporal feature extractor.

The learning rule is SB3's DQN; only the Q-network architecture changes.
Inputs must be flattened observations stacked by VecFrameStack (oldest first).
"""

import gym
import torch
from torch import nn
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """Attend over a fixed history of frames and return the latest token.

    Each token contains all features from one observation. Learned positional
    embeddings preserve temporal order. All tokens are historical, so attention
    needs no causal mask. Zero padding at episode starts follows VecFrameStack.
    The defaults target the parameter budget of the AB-EA extra_large MLP.
    """

    def __init__(
        self,
        observation_space,
        frame_stack=4,
        d_model=144,
        nhead=4,
        num_layers=2,
        dim_feedforward=640,
        dropout=0.0,
        activation_fn=nn.LeakyReLU,
    ):
        if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 1:
            raise ValueError("Transformer inputs require flattened Box observations.")
        if frame_stack < 1 or observation_space.shape[0] % frame_stack:
            raise ValueError("Observation size must be divisible by a positive frame_stack.")
        if d_model < 1 or nhead < 1 or d_model % nhead:
            raise ValueError("d_model must be positive and divisible by a positive nhead.")
        if num_layers < 1 or dim_feedforward < 1:
            raise ValueError("num_layers and dim_feedforward must be positive.")
        super().__init__(observation_space, features_dim=d_model)
        self.frame_stack = frame_stack
        self.frame_features = observation_space.shape[0] // frame_stack
        self.input_projection = nn.Linear(self.frame_features, d_model)
        self.position_embeddings = nn.Parameter(torch.empty(1, frame_stack, d_model))
        nn.init.normal_(self.position_embeddings, std=0.02)
        # Instantiate separately so layers start with independent weights.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation_fn(),
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, observations):
        frames = observations.reshape(-1, self.frame_stack, self.frame_features)
        tokens = self.input_projection(frames) + self.position_embeddings
        for layer in self.layers:
            tokens = layer(tokens)
        return self.output_norm(tokens[:, -1])


class TransformerDQN(DQN):
    """DQN with a transformer encoder and a linear action-value head.

    Accepts the same arguments as stable_baselines3.DQN. Configure the encoder
    through policy_kwargs['features_extractor_kwargs']; frame_stack must match
    the environment wrapper. The extractor can also be used in other SB3 agents.
    Optimizer, replay, exploration, target updates, and loss stay with SB3 DQN.
    """

    def __init__(self, policy="MlpPolicy", env=None, policy_kwargs=None, **kwargs):
        policy_kwargs = dict(policy_kwargs or {})
        policy_kwargs.setdefault("features_extractor_class", TransformerFeaturesExtractor)
        policy_kwargs.setdefault("net_arch", [])
        super().__init__(policy=policy, env=env, policy_kwargs=policy_kwargs, **kwargs)
