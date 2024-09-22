import re
from typing import Optional
from typing import Union

from sklearn.utils import check_random_state
import numpy as np


BEHAVIOR_PATTERNS = {
    "top_k_cascade": re.compile(r"^top_(\d+)_cascade$"),
    "neighbor_k": re.compile(r"^neighbor_(\d+)$"),
    "random_c": re.compile(r"random_(\d+)$"),
}


def parse_behavior(behavior: str) -> Union[tuple[str, int], None]:
    for behavior_name, pattern in BEHAVIOR_PATTERNS.items():
        match = pattern.match(behavior)
        if match:
            k = int(match.group(1))
            return behavior_name, k
    return None


def create_decay_matrix(len_list: int) -> np.ndarray:
    position = np.arange(len_list, dtype=float)
    tiled_position = np.tile(position, reps=(len_list, 1))
    repeat_position = np.repeat(position[:, None], len_list, axis=1)
    decay_matrix = np.abs(tiled_position - repeat_position)
    np.fill_diagonal(decay_matrix, 1)

    return decay_matrix


def create_interaction_params(
    behavior_names: list[str],
    len_list: int,
    interaction_noise: float,
    random_state: Optional[int] = None,
) -> dict[str, np.ndarray]:
    random_ = check_random_state(random_state)
    interaction_params = random_.uniform(
        0, interaction_noise, size=(len_list, len_list)
    )
    np.fill_diagonal(interaction_params, 1)

    decay_matrix = create_decay_matrix(len_list=len_list)
    interaction_params /= decay_matrix

    interaction_params_dict = dict()
    for behavior_name in behavior_names:
        if behavior_name == "standard":
            behavior_matrix = np.ones((len_list, len_list))

        elif behavior_name == "independent":
            behavior_matrix = np.eye(len_list)

        elif behavior_name == "cascade":
            behavior_matrix = np.tri(len_list)

        elif behavior_name == "inverse_cascade":
            behavior_matrix = np.tri(len_list).T

        else:
            parsed_behavior = parse_behavior(behavior_name)
            if parsed_behavior is None:
                raise NotImplementedError

            behavior_name_, int_ = parsed_behavior
            if behavior_name_ == "neighbor_k":
                neighbor_k = int_
                all_one_mat = np.ones((len_list, len_list), dtype=int)
                behavior_matrix = np.triu(all_one_mat, k=-neighbor_k) & np.tril(
                    all_one_mat, k=neighbor_k
                )

            elif behavior_name_ == "top_k_cascade":
                top_k = int_
                behavior_matrix = np.ones((len_list, len_list), dtype=int)
                behavior_matrix[:, :top_k] = 1
                np.fill_diagonal(behavior_matrix, 1)
                behavior_matrix = behavior_matrix & np.tri(len_list, dtype=int)

            elif behavior_name_ == "random_c":
                random_state_ = int_
                behavior_matrix = check_random_state(random_state_).randint(
                    2, size=(len_list, len_list)
                )
                np.fill_diagonal(behavior_matrix, 1)

            else:
                raise NotImplementedError

        interaction_params_dict[behavior_name] = behavior_matrix * interaction_params

    return interaction_params_dict
