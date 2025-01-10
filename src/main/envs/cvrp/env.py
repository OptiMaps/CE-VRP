from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.utils.pylogger import get_pylogger

from rl4co.envs.routing.cvrp.env import CVRPEnv


log = get_pylogger(__name__)


class CVRPCEnv(CVRPEnv):
    """Capacitated Vehicle Routing Problem (CVRP) environment.
    At each step, the agent chooses a customer to visit depending on the current location and the remaining capacity.
    When the agent visits a customer, the remaining capacity is updated. If the remaining capacity is not enough to
    visit any customer, the agent must go back to the depot. The reward is 0 unless the agent visits all the cities.
    In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.

    Observations:
        - location of the depot.
        - locations and demand of each customer.
        - current location of the vehicle.
        - the remaining customer of the vehicle,

    Constraints:
        - the tour starts and ends at the depot.
        - each customer must be visited exactly once.
        - the vehicle cannot visit customers exceed the remaining capacity.
        - the vehicle can return to the depot to refill the capacity.

    Finish Condition:
        - the vehicle has visited all customers and returned to the depot.

    Reward:
        - (minus) the negative length of the path.

    Args:
        generator: CVRPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "cvrp"


    @staticmethod
    def get_action_mask(td: TensorDict) -> TensorDict:
        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = td["demand"] + td["used_capacity"] > td["vehicle_capacity"]
        
        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = td["visited"][..., 1:].to(exceeds_cap.dtype) | exceeds_cap
        
        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (td["current_node"] == 0) & ((mask_loc == 0).int().sum(-1) > 0)[:, None]
        
        # mask tensor
        final_mask = ~torch.cat((mask_depot, mask_loc), -1)

        # update mask and reasons
        td.update({
            "masking_reasons": TensorDict({
                "exceeds_capacity": exceeds_cap,
                "already_visited": td["visited"][..., 1:].to(exceeds_cap.dtype),
            }, batch_size=td.batch_size)
        })

        return final_mask
