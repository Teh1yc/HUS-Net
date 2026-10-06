import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import HypergraphConv
from omegaconf import DictConfig
import numpy as np


class LearnableCommunityEnhancer(nn.Module):
    # BM-FA
    def __init__(self, hidden_size, use_gate=False, dropout=0.1, 
                 init_temperature=2.0, min_temperature=0.5,
                 use_adaptive_weight=True, num_communities=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_communities = num_communities
        self.use_gate = use_gate
        
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
        self.min_temperature = min_temperature
        self.community_prototypes = nn.Parameter(
            torch.randn(num_communities, hidden_size) * 0.1
        )
        
        if use_adaptive_weight:
            self.node_adaptive = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            )
        
        self.community_importance = nn.Parameter(
            torch.ones(num_communities) * 0.5
        )
        
        self.community_transform = nn.Sequential(
           # To be supplemented ...
        )
        
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def get_temperature(self, epoch=None):
        if epoch is not None:
            current_temp = max(
                self.min_temperature,
                self.temperature.item() * (0.95 ** epoch)
            )
            return current_temp
        return max(self.min_temperature, self.temperature.item())
    
    def compute_membership(self, node_features, epoch=None):
        node_norm = F.normalize(node_features, dim=-1)
        proto_norm = F.normalize(self.community_prototypes, dim=-1)
        similarity = torch.matmul(node_norm, proto_norm.T)
        temp = self.get_temperature(epoch)
        membership = F.softmax(similarity / temp, dim=-1)
        return membership
    
    def compute_adaptive_weights(self, node_features, membership):
        comm_importance = torch.sigmoid(self.community_importance)
        
        # To be supplemented ...
        
        return weighted_membership, node_weight, comm_importance
    
    def forward(self, node_features, batch, epoch=None):
        num_graphs = batch.max().item() + 1
        
        membership = self.compute_membership(node_features, epoch)
        
        weighted_membership, node_weight, comm_importance = self.compute_adaptive_weights(
            node_features, membership
        )
        
        enhanced_features_list = []
        
        for graph_idx in range(num_graphs):
            mask = batch == graph_idx
            graph_nodes = node_features[mask]
            graph_membership = weighted_membership[mask]
            
            membership_sum = graph_membership.sum(dim=0, keepdim=True)
            community_transformed = self.community_transform(community_raw)
            
            community_contrib = graph_membership.unsqueeze(-1) * community_transformed.unsqueeze(0)
            aggregated_community = community_contrib.sum(dim=1)
            
            if self.use_gate:
                # To be supplemented ...
                enhanced = gate * aggregated_community + (1 - gate) * graph_nodes
            else:
                enhanced = graph_nodes + aggregated_community
            
            enhanced = self.layer_norm(enhanced)
            enhanced_features_list.append(enhanced)
        
        enhanced_features = torch.cat(enhanced_features_list, dim=0)
        
        info = {
            'membership': membership,
            'weighted_membership': weighted_membership,
            'node_weight': node_weight,
            'community_importance': comm_importance,
            'prototypes': self.community_prototypes
        }
        
        return enhanced_features, weighted_membership, info



class EvidentialBernoulliSelection(nn.Module):
      # U-DFS
    def __init__(self, embed_dim):
        super().__init__()
        self.evidence_extractor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, original_features, enhanced_features, temp=1):

        # evidence generation
        evidence = F.softplus(self.evidence_extractor(enhanced_features))
        alpha = evidence[:, 0:1] + 1.0
        beta = evidence[:, 1:2] + 1.0
        
        prob = alpha / (alpha + beta)
        uncertainty = 1 / (alpha + beta)
        evidence_strength = alpha + beta
        
        beta_dist = torch.distributions.Beta(alpha, beta)
        sampled_prob = beta_dist.rsample()
        mask = ((logits + log_random_noise) / temp).sigmoid()
        
        output_emb = original_features * mask + original_features * (1 - mask) * 0.001
        
        evidence_info = {
            'alpha': alpha.squeeze(-1),
            'beta': beta.squeeze(-1),
            'prob': prob.squeeze(-1),
            'uncertainty': uncertainty.squeeze(-1),
            'evidence_strength': evidence_strength.squeeze(-1),
            'mask': mask.squeeze(-1),
            'sampled_prob': sampled_prob.squeeze(-1)
        }
        
        return output_emb, mask, evidence_info



class HUS_Net(torch.nn.Module):
    def __init__(self, cfg: DictConfig):
        super(HUS_Net, self).__init__()
        self.cfg = cfg
        self.num_layers = cfg.model.num_layers
        self.dropout = cfg.model.dropout
        self.hidden_size = cfg.model.hidden_size
        self.node_sz = cfg.dataset.node_sz

        self.r = cfg.model.r
        self.evidence_weight = cfg.model.evidence_weight
        self.num_communities = cfg.model.num_communities

        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = cfg.dataset.node_feature_sz + 1 if i == 0 else self.hidden_size
            self.convs.append(HypergraphConv(in_dim, self.hidden_size))

        #   BM-FA
        self.community_enhancer = LearnableCommunityEnhancer(
            hidden_size=self.hidden_size,
            use_gate=cfg.model.use_community_gate,
            dropout=self.dropout,
            init_temperature=cfg.model.init_temperature,
            use_adaptive_weight=cfg.model.use_adaptive_weight,
            num_communities=self.num_communities,
        )

        #   U-DFS
        self.evidential_selection = EvidentialBernoulliSelection(
            embed_dim=self.hidden_size
        )

        self.readout_lin = nn.Linear(
            self.node_sz * self.hidden_size, self.hidden_size
        )
        self.lin = nn.Linear(self.hidden_size, 1)

    def loss_function(self, logits, labels, evidence_info):
        ce_loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(), labels.float()
        )

        prob = evidence_info['prob']
        kl_loss = (
            prob * torch.log(prob / self.r + 1e-6)
            + (1 - prob) * torch.log((1 - prob) / (1 - self.r + 1e-6) + 1e-6)
        ).mean()

        evidence_strength = evidence_info['evidence_strength']
        uncertainty_reg = torch.log(1 / (evidence_strength + 1e-6)).mean()

        return ce_loss + kl_loss + self.evidence_weight * uncertainty_reg

    def forward(self, data, **kwargs):
        test_phase = kwargs.get('test_phase', False)

        x = data.x
        hyperedge_index = data.edge_index
        hyperedge_weight = data.edge_weight
        batch = data.batch
        labels = data.y

        # 超图卷积
        for i in range(self.num_layers):
            x = self.convs[i](x, hyperedge_index, hyperedge_weight)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        original_features = x

        # ===== USBM + BM-FA =====
        enhanced_features, weighted_membership, info = self.community_enhancer(
            x, batch
        )

        # ===== U-DFS =====
        output_features, mask, evidence_info = self.evidential_selection(
            original_features, enhanced_features
        )

        evidence_info['membership'] = weighted_membership
        evidence_info['community_importance'] = info['community_importance']
        evidence_info['prototypes'] = info['prototypes']

        # Readout
        xs = []
        for graph_idx in batch.unique():
            graph_nodes = output_features[batch == graph_idx].view(-1)
            xs.append(self.readout_lin(graph_nodes))
        output_features = torch.stack(xs)

        output_features = F.leaky_relu(output_features)
        output_features = F.dropout(
            output_features, p=self.dropout, training=self.training
        )
        logits = self.lin(output_features)

        loss = self.loss_function(logits, labels, evidence_info)

        if test_phase:
            return logits, loss, {'evidence_info': evidence_info}
        return logits, loss




 
            