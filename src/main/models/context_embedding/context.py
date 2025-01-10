import torch
import torch.nn as nn

from tensordict import TensorDict

from rl4co.utils.ops import gather_by_index
from main.models.masking_encoder import MaskingEncoder


def env_context_embedding(env_name: str, config: dict) -> nn.Module:
    """Get environment context embedding. The context embedding is used to modify the
    query embedding of the problem node of the current partial solution.
    Usually consists of a projection of gathered node embeddings and features to the embedding space.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "cvrp": VRPContext,
        "cvrptw": VRPTWContext,
    }

    if env_name not in embedding_registry:
        raise ValueError(
            f"Unknown environment name '{env_name}'. Available context embeddings: {embedding_registry.keys()}"
        )

    return embedding_registry[env_name](**config)


class EnvContext(nn.Module):
    """Base class for environment context embeddings. The context embedding is used to modify the
    query embedding of the problem node of the current partial solution.
    Consists of a linear layer that projects the node features to the embedding space."""

    def __init__(self, embed_dim, step_context_dim=None, linear_bias=False, num_reasons=None):
        super(EnvContext, self).__init__()
        self.embed_dim = embed_dim
        step_context_dim = step_context_dim if step_context_dim is not None else embed_dim
        self.project_context = nn.Linear(step_context_dim, embed_dim, bias=linear_bias)
        if num_reasons is not None:
            self.constraint_linear = nn.Linear(num_reasons, embed_dim, bias=False)
        self.masking_encoder = MaskingEncoder()

    def _cur_node_embedding(self, embeddings, td):
        """Get embedding of current node"""
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        """Get state embedding"""
        raise NotImplementedError("Implement for each environment")

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
        return self.project_context(context_embedding)


class VRPContext(EnvContext):
    """Context embedding for the Capacitated Vehicle Routing Problem (CVRP)."""

    def __init__(self, embed_dim, constraint_method='none'):
        # Calculate step_context_dim based on constraint method
        if constraint_method == 'none':
            step_context_dim = embed_dim + 1
        elif constraint_method == 'linear':
            step_context_dim = 2 * embed_dim + 1  # original node embedding + capacity + contraint embedding
        elif constraint_method == 'weighted':
            step_context_dim = embed_dim + 3  # node embedding + capacity + constraint node weights mean
        
        super(VRPContext, self).__init__(
            embed_dim=embed_dim,
            step_context_dim=step_context_dim,
            num_reasons=2 if constraint_method != 'none' else None
        )
        
        self.constraint_method = constraint_method
        if constraint_method == 'weighted':
            self.node_weights = nn.Parameter(torch.ones(1))
    
    def _state_embedding(self, embeddings, td):
        # Get capacity state [batch_size, 1]
        state_embedding = td["vehicle_capacity"] - td["used_capacity"]
        
        if self.constraint_method == 'none':
            return state_embedding
            
        # Get constraint embedding [batch_size, num_loc, num_reasons]
        constraint_embedding = self.masking_encoder.onehot_encode(td["masking_reasons"])
        if constraint_embedding is None:
            raise ValueError("Constraint embedding is None but constraint_method is not 'none'. This might indicate an error in masking_reasons tensor.")
            
        if self.constraint_method == 'linear':
            # linear embedding method: linear projection and mean
            constraint_embedding = self.constraint_linear(constraint_embedding)
            constraint_embedding = constraint_embedding.mean(dim=1)
            
        elif self.constraint_method == 'weighted':
            # weighted method: weighted sum across nodes
            node_weights = self.node_weights.expand(
                constraint_embedding.size(0), 
                constraint_embedding.size(1), 
                1
            )
            constraint_embedding = (constraint_embedding * node_weights).sum(dim=1)

            
        return torch.cat([state_embedding, constraint_embedding], dim=-1)


class VRPTWContext(EnvContext):
    """Context embedding for the VRPTW."""

    def __init__(self, embed_dim, constraint_method='none'):
        # Calculate step_context_dim based on constraint method
        if constraint_method == 'none':
            step_context_dim = embed_dim + 2  # node embedding + capacity + time
        elif constraint_method == 'linear':
            step_context_dim = 2 * embed_dim + 2  # node embedding + constraint embedding + capacity + time
        elif constraint_method == 'weighted':
            step_context_dim = embed_dim + 5  # weighted constraint embedding + capacity + time
        
        super(VRPTWContext, self).__init__(
            embed_dim=embed_dim,
            step_context_dim=step_context_dim,
            num_reasons=3 if constraint_method != 'none' else None
        )
        self.constraint_method = constraint_method
        if constraint_method == 'weighted':
            self.node_weights = nn.Parameter(torch.ones(1))

    def _state_embedding(self, embeddings, td):
        # current time and remaining capacity [batch_size, 2]
        current_time = td["current_time"]
        remaining_capacity = td["vehicle_capacity"] - td["used_capacity"]
        
        # constraint method: none
        if self.constraint_method == 'none':
            return torch.cat([remaining_capacity, current_time], -1)
        
        # onehot encoding
        constraint_embedding = self.masking_encoder.onehot_encode(td["masking_reasons"])
        if constraint_embedding is None:
            raise ValueError("Constraint embedding is None but constraint_method is not 'none'. This might indicate an error in masking_reasons tensor.")
            
        # linear embedding
        if self.constraint_method == 'linear':
            constraint_embedding = self.constraint_linear(constraint_embedding)
            constraint_embedding = constraint_embedding.mean(dim=1)
            
        # weighted embedding
        elif self.constraint_method == 'weighted':
            node_weights = self.node_weights.expand(
                constraint_embedding.size(0), 
                constraint_embedding.size(1), 
                1
            )
            constraint_embedding = (constraint_embedding * node_weights).sum(dim=1)
            
        return torch.cat([remaining_capacity, current_time, constraint_embedding], -1)

