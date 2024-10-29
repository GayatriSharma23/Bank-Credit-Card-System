import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier
from tab_transformer_pytorch import TabTransformer

class CombinedModel(nn.Module):
    def __init__(self, input_dim, output_dim, tabnet_params, tabtransformer_params, hidden_dim=64):
        super(CombinedModel, self).__init__()
        self.tabnet = TabNetClassifier(**tabnet_params)
        self.tabtransformer = TabTransformer(**tabtransformer_params)

        # Combine the output dimensions of both models
        combined_dim = tabnet_params['n_d'] + tabtransformer_params['dim_out']

        # Shared layers for both models
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Final layer
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        tabnet_output = self.tabnet(x)
        tabtransformer_output = self.tabtransformer(x)

        # Concatenate outputs at a higher layer
        combined_output = torch.cat([tabnet_output, tabtransformer_output], dim=1)

        # Pass combined output through shared layers
        shared_output = self.shared_layers(combined_output)

        # Final output
        final_output = self.final_layer(shared_output)
        return final_output
