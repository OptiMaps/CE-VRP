from typing import Optional
import torch

from tensordict.tensordict import TensorDict
# Use torch RL module
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec
)

from rl4co.envs.common import RL4COEnvBase, Generator, get_sampler
from rl4co.utils.ops import gather_by_index, get_tour_length

# Optional[x] -> None or x type
def _reset(self, td: Optional[TensorDict]=None, batch_size=None) -> TensorDict:
    # initalize locations
    init_locs = td['locs'] if td is not None else None

    # batch size init
    if batch_size is None:
        batch_size = self.batch_size if init_locs is None else init_locs.shape[:-2]

    # device init
    device = init_locs.device if init_locs is not None else self.device
    self.to(device)

    # reassign init_locs
    if init_locs is None:
        init_locs = self.generate_data(batch_size=batch_size).to(device)["locs"]

    # create batch size to list
    batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

    # Do not enforce loading from self for flexibility
    num_loc = init_locs.shape[-2]

    # Other variables
    current_node = torch.zeros((batch_size), dtype=torch.int64, device=device)
    available = torch.ones(
        (*batch_size, num_loc), dtype=torch.bool, device=device
    )
    # 1 means not visitied, i.e. action is allowed
    i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

    return TensorDict(
        dict(
            locs=init_locs,
            first_node=current_node,
            current_node=current_node,
            i=i,
            action_mask=available,
            reward=torch.zeros((*batch_size, 1), dtype=torch.float32),
        ),
        batch_size=batch_size
    )

def _step(self, td: TensorDict) -> TensorDict:
    current_node = td["action"]
    first_node = current_node if td["i"].all() == 0 else td["first_node"]

    # Set not visited to 0 (i.e., we visited the node)
    # Note: we may also use a separate function for obtaining the mask for more flexibility
    available = td["action_mask"].scatter(
        -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), 0
    )

    # We are done there are no unvisited locations
    done = torch.sum(available, dim=-1) == 0

    # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
    reward = torch.zeros_like(done)

    td.update(
        {
            "first_node": first_node,
            "current_node": current_node,
            "i": td["i"] + 1,
            "action_mask": available,
            "reward": reward,
            "done": done,
        },
    )
    return td

def get_action_mask(self, td: TensorDict) -> TensorDict:
    # Here: your logic
    return td["action_mask"]

def check_solution_validity(self, td: TensorDict, actions: torch.Tensor):
    """Check that solution is valid: nodes are visited exactly once"""
    assert (
        torch.arange(actions.size(1), out=actions.data.new())
        .view(1, -1)
        .expand_as(actions)
        == actions.data.sort(1)[0]
    ).all(), "Invalid tour"


def _get_reward(self, td, actions) -> TensorDict:
    # Sanity check if enabled
    if self.check_solution:
        self.check_solution_validity(td, actions)

    # Gather locations in order of tour and return distance between them (i.e., -reward)
    locs_ordered = gather_by_index(td["locs"], actions)
    return -get_tour_length(locs_ordered)

def _make_spec(self, generator):
    """Make the observation and action specs from the parameters"""
    self.observation_spec = CompositeSpec(
        locs=BoundedTensorSpec(
            low=self.generator.min_loc,
            high=self.generator.max_loc,
            shape=(self.generator.num_loc, 2),
            dtype=torch.float32,
        ),
        first_node=UnboundedDiscreteTensorSpec(
            shape=(1),
            dtype=torch.int64,
        ),
        current_node=UnboundedDiscreteTensorSpec(
            shape=(1),
            dtype=torch.int64,
        ),
        i=UnboundedDiscreteTensorSpec(
            shape=(1),
            dtype=torch.int64,
        ),
        action_mask=UnboundedDiscreteTensorSpec(
            shape=(self.generator.num_loc),
            dtype=torch.bool,
        ),
        shape=(),
    )
    self.action_spec = BoundedTensorSpec(
        shape=(1,),
        dtype=torch.int64,
        low=0,
        high=self.generator.num_loc,
    )
    self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
    self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)

class TSPGenerator(Generator):
    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.loc_sampler = torch.distributions.Uniform(
            low=min_loc, high=max_loc
        )

    def _generate(self, batch_size) -> TensorDict:
        # Sample locations
        locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))
        return TensorDict({"locs": locs}, batch_size=batch_size)
    

def render(self, td, actions=None, ax=None):
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        # Create a plot of the nodes
        _, ax = plt.subplots()

    td = td.detach().cpu()

    if actions is None:
        actions = td.get("action", None)
    # if batch_size greater than 0 , we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]
        actions = actions[0]

    locs = td["locs"]

    # gather locs in order of action if available
    if actions is None:
        print("No action in TensorDict, rendering unsorted locs")
    else:
        actions = actions.detach().cpu()
        locs = gather_by_index(locs, actions, dim=0)

    # Cat the first node to the end to complete the tour
    locs = torch.cat((locs, locs[0:1]))
    x, y = locs[:, 0], locs[:, 1]

    # Plot the visited nodes
    ax.scatter(x, y, color="tab:blue")

    # Add arrows between visited nodes as a quiver plot
    dx, dy = np.diff(x), np.diff(y)
    ax.quiver(
        x[:-1], y[:-1], dx, dy, scale_units="xy", angles="xy", scale=1, color="k"
    )

    # Setup limits and show
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)


class TSPEnv(RL4COEnvBase):
    """Traveling Salesman Problem (TSP) environment"""

    name = "tsp"

    def __init__(
        self,
        generator = TSPGenerator,
        generator_params = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.generator = generator(**generator_params)
        self._make_spec(self.generator)

    _reset = _reset
    _step = _step
    _get_reward = _get_reward
    check_solution_validity = check_solution_validity
    get_action_mask = get_action_mask
    _make_spec = _make_spec
    render = render