import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

"""Implementation of the Edge-augmented Graph Transformer
Paper: https://arxiv.org/pdf/2108.03348
Github: https://github.com/shamim-hussain/egt_pytorch
"""


class FFN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), activation(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: Tensor):
        return self.layers(x)


class EGTLayer(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_heads: int,
        clip_logits_min: int = -5,
        clip_logits_max: int = 5,
    ):
        super().__init__()

        assert node_dim % num_heads == 0, f"{node_dim} must be divisible by {num_heads}"

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.clip_logits_min = clip_logits_min
        self.clip_logits_max = clip_logits_max

        self.node_ln = nn.LayerNorm(node_dim)
        self.edge_ln = nn.LayerNorm(edge_dim)

        # Multi-head linear layers
        self.lin_QKV = nn.Linear(node_dim, node_dim * 3)
        self.lin_GE = nn.Linear(edge_dim, num_heads * 2)

        # Out linear layers
        self.lin_O_h = nn.Linear(node_dim, node_dim)
        self.lin_O_e = nn.Linear(num_heads, edge_dim)

        # Node FFN
        self.node_ffn_ln = nn.LayerNorm(node_dim)
        self.node_ffn = FFN(node_dim, node_dim * 2, node_dim)

        # Edge FFN
        self.edge_ffn_ln = nn.LayerNorm(edge_dim)
        self.edge_ffn = FFN(edge_dim, edge_dim, edge_dim)

    # TODO: Add adj masking
    def forward(self, node_embedding: Tensor, edge_embedding: Tensor, adj: Tensor):
        # Step 1: Layernorm Embeddings
        node_emb = self.node_ln(node_embedding)
        edge_emb = self.edge_ln(edge_embedding)

        # Step 2: Apply QKV linear
        QKV = self.lin_QKV(node_emb)
        # Step 3: Apply GE linear
        GE = self.lin_GE(edge_emb)

        # Step 4: Split QKV and GE
        B, N, d_all = QKV.shape
        h = self.num_heads
        d_head = d_all // 3 // h

        # [B, N, d_all] -> [B, N, d_head, h]
        Q, K, V = QKV.chunk(3, dim=-1)
        Q = Q.view(B, N, d_head, h)
        K = K.view(B, N, d_head, h)
        V = V.view(B, N, d_head, h)

        # [B, N, N, h]
        G, E = GE.chunk(2, dim=-1)
        G = G.view(B, N, N, h)
        E = E.view(B, N, N, h)

        # Step 5: Compute H_hat
        A_hat = torch.einsum("bldh, bmdh -> blmh", Q, K)
        A_hat = A_hat * (d_head**-0.5)
        H_hat = A_hat.clamp(self.clip_logits_min, self.clip_logits_max) + E

        # Step 6: Compute A_tild
        gates = torch.sigmoid(G)
        A_tild = F.softmax(H_hat, dim=-1) * gates
        # TODO: Add dropout for A_tild

        # Step 7: Compute V_att
        V_att = torch.einsum("blmh,bmdh->bldh", A_tild, V)
        V_att = V_att.reshape(B, N, d_head * h)

        # Step 8: node MHA + residual
        node_emb_mha = self.lin_O_h(V_att)
        # TODO: Add dropout for node_emb_mha
        node_emb = node_emb_mha + node_emb

        # Step 9: node FFN + residual
        node_emb_ln = self.node_ffn_ln(node_emb)
        node_emb_ffn = self.node_ffn(node_emb_ln)
        # TODO: Add dropout for node_emb_ffn
        node_emb = node_emb_ffn + node_emb

        # Step 10: edge MHA + residual
        edge_emb_mha = self.lin_O_e(H_hat)
        # TODO: Add dropout for edge_emb_mha
        edge_emb = edge_emb_mha + edge_emb

        # Step 11: edge FFN + residual
        edge_emb_ln = self.edge_ffn_ln(edge_emb)
        edge_emb_ffn = self.edge_ffn(edge_emb_ln)
        # TODO: Add dropout for edge_emb_ffn
        edge_emb = edge_emb_ffn + edge_emb

        return node_emb, edge_emb


class EGT(nn.Module):
    def __init__(
        self,
        node_in_dim: int = 5,
        edge_in_dim: int = 5,
        node_hidden_dim: int = 64,
        edge_hidden_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 3,
    ):
        super().__init__()

        # TODO: Position Embedding
        self.node_emb = FFN(node_in_dim, node_hidden_dim, node_hidden_dim)
        self.edge_emb = FFN(edge_in_dim, edge_hidden_dim, edge_hidden_dim)

        self.layers = nn.ModuleList(
            [
                EGTLayer(node_hidden_dim, edge_hidden_dim, num_heads)
                for _ in range(num_layers)
            ]
        )

        self.decoder = FFN(edge_hidden_dim, edge_hidden_dim, 1)

    def forward(self, node_features: Tensor, edge_features: Tensor, adj: Tensor):
        # Embed features
        node_emb = self.node_emb(node_features)
        edge_emb = self.edge_emb(edge_features)

        for layer in self.layers:
            node_emb, edge_emb = layer(node_emb, edge_emb, adj)

        edge_scores = self.decoder(edge_emb).squeeze(-1)
        return edge_scores


if __name__ == "__main__":
    import rootutils
    import time
    from torchinfo import summary

    # Setup Root
    root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    # Import utils
    from sop.utils.graph_torch import generate_sop_graphs, preprocess_graph

    # Example graph preprocess
    start = time.time()
    B = 32
    N = 100
    S = 0
    G = 99
    # Generate Sample Graphs
    graphs = generate_sop_graphs(
        batch_size=B,
        num_nodes=N,
        start_node=S,
        goal_node=G,
        budget=2,
        num_samples=100,
        kappa=0.5,
    )
    print(f"Graph Gen: {time.time() - start}")

    start = time.time()
    node_features, edge_features, adj = preprocess_graph(graphs)
    print(f"Preprocess: {time.time() - start}")

    start = time.time()
    node_dim = node_features.shape[-1]
    edge_dim = edge_features.shape[-1]
    egt = EGT(node_in_dim=node_dim, edge_in_dim=edge_dim)
    print(f"EGT Init: {time.time() - start}")

    summary(egt, [(1, N, node_dim), (1, N, N, edge_dim), (1, N, N)])

    start = time.time()
    x = egt(node_features, edge_features, adj)
    print(f"EGT Forward: {time.time() - start}")
    print(x.shape)
