import torch
import torch.nn as nn

from tensordict.tensordict import TensorDict

from rl4co.models.nn.ops import PositionalEncoding


def env_init_embedding(env_name: str, config: dict) -> nn.Module:
    """Get environment initial embedding. The init embedding is used to initialize the
    general embedding of the problem nodes without any solution information.
    Consists of a linear layer that projects the node features to the embedding space.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "atsp": TSPInitEmbedding,
        "cvrp": VRPInitEmbedding,
        "cvrptw": VRPTWInitEmbedding,
        "svrp": SVRPInitEmbedding,
        "pctsp": PCTSPInitEmbedding,
        "pdp": PDPInitEmbedding,
        "mdcpdp": MDCPDPInitEmbedding,
    }

    if env_name not in embedding_registry:
        raise ValueError(
            f"Unknown environment name '{env_name}'. Available init embeddings: {embedding_registry.keys()}"
        )

    return embedding_registry[env_name](**config)


class VRPInitEmbedding(nn.Module):
    """Initial embedding for the Vehicle Routing Problems (VRP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot and customers separately)
        - demand: demand of the customers
    """

    def __init__(self, embed_dim, linear_bias=True, node_dim: int = 3):
        super(VRPInitEmbedding, self).__init__()
        node_dim = node_dim  # 3: x, y, demand
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)  # depot embedding

    def forward(self, td):
        # [batch, 1, 2]-> [batch, 1, embed_dim]
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        # [batch, n_city, 2, batch, n_city, 1]  -> [batch, n_city, embed_dim]
        node_embeddings = self.init_embed(
            torch.cat((cities, td["demand"][..., None]), -1)
        )
        # [batch, n_city+1, embed_dim]
        out = torch.cat((depot_embedding, node_embeddings), -2)
        return out


class VRPTWInitEmbedding(VRPInitEmbedding):
    def __init__(self, embed_dim, linear_bias=True, node_dim: int = 6):
        # node_dim = 6: x, y, demand, tw start, tw end, service time
        super(VRPTWInitEmbedding, self).__init__(embed_dim, linear_bias, node_dim)

    def forward(self, td):
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        durations = td["durations"][..., 1:]
        time_windows = td["time_windows"][..., 1:, :]
        # embeddings
        depot_embedding = self.init_embed_depot(depot)
        node_embeddings = self.init_embed(
            torch.cat(
                (cities, td["demand"][..., None], time_windows, durations[..., None]), -1
            )
        )
        return torch.cat((depot_embedding, node_embeddings), -2)


class SVRPInitEmbedding(nn.Module):
    def __init__(self, embed_dim, linear_bias=True, node_dim: int = 3):
        super(SVRPInitEmbedding, self).__init__()
        node_dim = node_dim  # 3: x, y, skill
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)  # depot embedding

    def forward(self, td):
        # [batch, 1, 2]-> [batch, 1, embed_dim]
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        # [batch, n_city, 2, batch, n_city, 1]  -> [batch, n_city, embed_dim]
        node_embeddings = self.init_embed(torch.cat((cities, td["skills"]), -1))
        # [batch, n_city+1, embed_dim]
        out = torch.cat((depot_embedding, node_embeddings), -2)
        return out


class PCTSPInitEmbedding(nn.Module):
    """Initial embedding for the Prize Collecting Traveling Salesman Problems (PCTSP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot and customers separately)
        - expected_prize: expected prize for visiting the customers.
            In PCTSP, this is the actual prize. In SPCTSP, this is the expected prize.
        - penalty: penalty for not visiting the customers
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(PCTSPInitEmbedding, self).__init__()
        node_dim = 4  # x, y, prize, penalty
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)

    def forward(self, td):
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        node_embeddings = self.init_embed(
            torch.cat(
                (
                    cities,
                    td["expected_prize"][..., None],
                    td["penalty"][..., 1:, None],
                ),
                -1,
            )
        )
        # batch, n_city+1, embed_dim
        out = torch.cat((depot_embedding, node_embeddings), -2)
        return out


class PDPInitEmbedding(nn.Module):
    """Initial embedding for the Pickup and Delivery Problem (PDP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot, pickups and deliveries separately)
           Note that pickups and deliveries are interleaved in the input.
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(PDPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)
        self.init_embed_pick = nn.Linear(node_dim * 2, embed_dim, linear_bias)
        self.init_embed_delivery = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td):
        depot, locs = td["locs"][..., 0:1, :], td["locs"][..., 1:, :]
        num_locs = locs.size(-2)
        pick_feats = torch.cat(
            [locs[:, : num_locs // 2, :], locs[:, num_locs // 2 :, :]], -1
        )  # [batch_size, graph_size//2, 4]
        delivery_feats = locs[:, num_locs // 2 :, :]  # [batch_size, graph_size//2, 2]
        depot_embeddings = self.init_embed_depot(depot)
        pick_embeddings = self.init_embed_pick(pick_feats)
        delivery_embeddings = self.init_embed_delivery(delivery_feats)
        # concatenate on graph size dimension
        return torch.cat([depot_embeddings, pick_embeddings, delivery_embeddings], -2)


class MDCPDPInitEmbedding(nn.Module):
    """Initial embedding for the MDCPDP environment
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot, pickups and deliveries separately)
           Note that pickups and deliveries are interleaved in the input.
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(MDCPDPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)
        self.init_embed_pick = nn.Linear(node_dim * 2, embed_dim, linear_bias)
        self.init_embed_delivery = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td):
        num_depots = td["capacity"].size(-1)
        depot, locs = td["locs"][..., 0:num_depots, :], td["locs"][..., num_depots:, :]
        num_locs = locs.size(-2)
        pick_feats = torch.cat(
            [locs[:, : num_locs // 2, :], locs[:, num_locs // 2 :, :]], -1
        )  # [batch_size, graph_size//2, 4]
        delivery_feats = locs[:, num_locs // 2 :, :]  # [batch_size, graph_size//2, 2]
        depot_embeddings = self.init_embed_depot(depot)
        pick_embeddings = self.init_embed_pick(pick_feats)
        delivery_embeddings = self.init_embed_delivery(delivery_feats)
        # concatenate on graph size dimension
        return torch.cat([depot_embeddings, pick_embeddings, delivery_embeddings], -2)


class FJSPInitEmbedding(JSSPInitEmbedding):
    def __init__(self, embed_dim, linear_bias=False, scaling_factor: int = 100):
        super().__init__(embed_dim, linear_bias, scaling_factor)
        self.init_ma_embed = nn.Linear(1, self.embed_dim, bias=linear_bias)
        self.edge_embed = nn.Linear(1, embed_dim, bias=linear_bias)

    def forward(self, td: TensorDict):
        ops_emb = self._init_ops_embed(td)
        ma_emb = self._init_machine_embed(td)
        edge_emb = self._init_edge_embed(td)
        # get edges between operations and machines
        # (bs, ops, ma)
        edges = td["ops_ma_adj"].transpose(1, 2)
        return ops_emb, ma_emb, edge_emb, edges

    def _init_edge_embed(self, td: TensorDict):
        proc_times = td["proc_times"].transpose(1, 2) / self.scaling_factor
        edge_embed = self.edge_embed(proc_times.unsqueeze(-1))
        return edge_embed

    def _init_machine_embed(self, td: TensorDict):
        busy_for = (td["busy_until"] - td["time"].unsqueeze(1)) / self.scaling_factor
        ma_embeddings = self.init_ma_embed(busy_for.unsqueeze(2))
        return ma_embeddings
