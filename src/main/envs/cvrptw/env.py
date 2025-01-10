from typing import Optional

import torch

from tensordict.tensordict import TensorDict

from rl4co.data.utils import (
    load_npz_to_tensordict,
    load_solomon_instance,
    load_solomon_solution,
)
from rl4co.envs.routing.cvrptw.env import CVRPTWEnv
from ..cvrp.env import CVRPCEnv
from rl4co.utils.ops import gather_by_index, get_distance


class CVRPTWCEnv(CVRPTWEnv):
    """Capacitated Vehicle Routing Problem with Time Windows (CVRPTW) environment.
    Inherits from the CVRPEnv class in which customers are considered.
    Additionally considers time windows within which a service has to be started.

    Observations:
        - location of the depot.
        - locations and demand of each customer.
        - current location of the vehicle.
        - the remaining customer of the vehicle.
        - the current time.
        - service durations of each location.
        - time windows of each location.

    Constraints:
        - the tour starts and ends at the depot.
        - each customer must be visited exactly once.
        - the vehicle cannot visit customers exceed the remaining customer.
        - the vehicle can return to the depot to refill the customer.
        - the vehicle must start the service within the time window of each location.

    Finish Condition:
        - the vehicle has visited all customers and returned to the depot.

    Reward:
        - (minus) the negative length of the path.

    Args:
        generator: CVRPTWGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "cvrptw"

    @staticmethod
    def get_action_mask(td: TensorDict) -> TensorDict:
        """In addition to the constraints considered in the CVRPEnv, the time windows are considered.
        The vehicle can only visit a location if it can reach it in time, i.e. before its time window ends.
        """
        # get mask and reasons from masking_dict
        not_masked = CVRPCEnv.get_action_mask(td)

        current_loc = gather_by_index(td["locs"], td["current_node"])
        dist = get_distance(current_loc[..., None, :], td["locs"])
        td.update({"current_loc": current_loc, "distances": dist})
        can_reach_in_time = (
            td["current_time"] + dist <= td["time_windows"][..., 1]
        )  # I only need to start the service before the time window ends, not finish it.
        mask = not_masked & can_reach_in_time # shape: [batch_size, num_loc+1]

        # masking_reasons update time_window constraint
        td["masking_reasons"]["time_window"] = ~can_reach_in_time[..., 1:]  

        return mask


