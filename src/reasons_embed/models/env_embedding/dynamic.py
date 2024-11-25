import torch
import torch.nn as nn

from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger
from src.utils.masking_encoder import MaskingEncoder

log = get_pylogger(__name__)


def env_dynamic_embedding(env_name: str, config: dict) -> nn.Module:
    """Get environment dynamic embedding. The dynamic embedding is used to modify query, key and value vectors of the attention mechanism
    based on the current state of the environment (which is changing during the rollout).
    Consists of a linear layer that projects the node features to the embedding space.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "cvrp": ConstraintDynamicEmbedding,
        "cvrptw": ConstraintDynamicEmbedding,
        "pdp": ConstraintDynamicEmbedding,
    }

    if env_name not in embedding_registry:
        log.warning(
            f"Unknown environment name '{env_name}'. Available dynamic embeddings: {embedding_registry.keys()}. Defaulting to StaticEmbedding."
        )
    return embedding_registry.get(env_name, ConstraintDynamicEmbedding)(**config)


class StaticEmbedding(nn.Module):
    """Static embedding for general problems.
    This is used for problems that do not have any dynamic information, except for the
    information regarding the current action (e.g. the current node in TSP). See context embedding for more details.
    """

    def __init__(self, *args, **kwargs):
        super(StaticEmbedding, self).__init__()

    def forward(self, td):
        return 0, 0, 0

class ConstraintDynamicEmbedding(nn.Module):
    """Dynamic embedding for problems with constraints.
    """

    def __init__(self, embed_dim: int, linear_bias: bool = False, **kwargs):
        super(ConstraintDynamicEmbedding, self).__init__()
        self.project_constraint = nn.Linear(1, 3 * embed_dim, bias=linear_bias)
        self.masking_encoder = MaskingEncoder()
    def forward(self, td):
        constraint_emb = self.project_constraint(self.masking_encoder.onehot_encode(td["masking_reasons"]))
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = constraint_emb.chunk(3, dim=-1)
        return glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic
