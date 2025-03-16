from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn


class Scale(nn.Module):
    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta * x


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        dim_context: int,
        dim_output: int,
        objective: str,
        hidden_layer_size: tuple[int, ...] = (30, 30, 30),
        activation: str = "elu",
        beta: float = 1.0,
        random_state: int = 12345,
    ) -> None:
        super().__init__()
        self.dim_context = dim_context
        self.dim_output = dim_output
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.beta = beta
        self.random_state = random_state
        self.objective = objective
        self.name = "mlp"

        if self.activation == "tanh":
            self.activation_layer = nn.Tanh
        elif self.activation == "relu":
            self.activation_layer = nn.ReLU
        elif self.activation == "elu":
            self.activation_layer = nn.ELU
        else:
            raise NotImplementedError

        if not self.objective in ["regression", "decision-making"]:
            raise ValueError("objective must be either regression or decision-making")

        torch.manual_seed(self.random_state)

        self.input_size = self.dim_context
        self.layer_list = []
        for i, h in enumerate(self.hidden_layer_size):
            self.layer_list.append(("l{}".format(i), nn.Linear(self.input_size, h)))
            self.layer_list.append(("a{}".format(i), self.activation_layer()))
            self.input_size = h

        self.layer_list.append(("output", nn.Linear(self.input_size, self.dim_output)))

        if self.objective == "regression":
            self.nn_model = nn.Sequential(OrderedDict(self.layer_list))

        elif self.objective == "decision-making":
            self.layer_list.append(("scale", Scale(self.beta)))
            self.layer_list.append(("softmax", nn.Softmax(dim=1)))
            self.nn_model = nn.Sequential(OrderedDict(self.layer_list))

    def forward(
        self, x: torch.Tensor, e_a: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.nn_model(x)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        dim_context: int,
        dim_action_context: int,
        dim_embed: int,
        objective: str,
        hidden_layer_size: tuple[int, ...] = (30, 30, 30),
        activation: str = "elu",
        beta: float = 1.0,
        random_state: int = 12345,
    ) -> None:
        super().__init__()
        self.dim_context = dim_context
        self.dim_action_context = dim_action_context
        self.dim_embed = dim_embed
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.objective = objective
        self.beta = beta
        self.random_state = random_state
        self.name = "two_tower"

        if not self.objective in ["regression", "decision-making"]:
            raise ValueError("objective must be either regression or decision-making")

        self.user_tower = MultiLayerPerceptron(
            dim_context=self.dim_context,
            dim_output=self.dim_embed,
            hidden_layer_size=self.hidden_layer_size,
            activation=self.activation,
            objective="regression",
            random_state=self.random_state,
        )

        self.action_tower = MultiLayerPerceptron(
            dim_context=self.dim_action_context,
            dim_output=self.dim_embed,
            hidden_layer_size=self.hidden_layer_size,
            activation=self.activation,
            objective="regression",
            random_state=self.random_state,
        )

    def forward(self, x: torch.Tensor, e_a: torch.Tensor) -> torch.Tensor:
        user_embed = self.user_tower(x)
        action_embed = self.action_tower(e_a)
        logits = torch.matmul(user_embed, action_embed.T)

        if self.objective == "regression":
            return logits

        pi = torch.softmax(self.beta * logits, dim=1)
        return pi
