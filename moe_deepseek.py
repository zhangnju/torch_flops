import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import torch.onnx
import onnx_tool
from typing import Tuple
import math
#from torch_flops import TorchFLOPsByFX

config = {
    "hidden_act": "silu",
    "hidden_size": 2048,
    "intermediate_size": 10944,
    "d_expert": 1408,
    "n_routed_experts": 64,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "scoring_func": "softmax",
}

class Expert(nn.Module):
    def __init__(self, config: Dict, intermediate_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size: int = config["hidden_size"]
        self.intermediate_size: int = (
            config["intermediate_size"] if intermediate_size is None else intermediate_size
        )

        self.gate_projection = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.value_projection = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.output_projection = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.activation = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gated = self.activation(self.gate_projection(inputs)) * self.value_projection(inputs)
        return self.output_projection(gated)


class MoEGate(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.top_k: int = config["num_experts_per_tok"]
        self.num_routed_experts: int = config["n_routed_experts"]
        self.scoring_function = config["scoring_func"]
        self.gating_input_dim: int = config["hidden_size"]

        self.expert_logits_weight = nn.Parameter(torch.empty((self.num_routed_experts, self.gating_input_dim)))
        self.linear_projection = nn.Linear(self.gating_input_dim, self.num_routed_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flattened_states = hidden_states.view(-1, hidden_dim)
        logits = self.linear_projection(flattened_states)
        scores = logits.softmax(dim=-1)
        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        return topk_indices, topk_scores


class MoE(nn.Module):
    """
    Mixture of Experts (MoE) module with both shared and routed experts.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_experts_per_token: int = config["num_experts_per_tok"]

        # Routed experts
        self.routed_experts = nn.ModuleList([
            Expert(config, intermediate_size=config["d_expert"])
            for _ in range(config["n_routed_experts"])
        ])

        # Gating mechanism
        self.gating_network = MoEGate(config)

        # Shared expert
        shared_expert_dim = config["d_expert"] * config["n_shared_experts"]
        self.shared_expert = Expert(config=config, intermediate_size=shared_expert_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MoE block.
        Args:
            hidden_states (Tensor): Shape (batch_size, seq_len, hidden_dim)
        Returns:
            Tensor: Combined expert output of shape (batch_size, seq_len, hidden_dim)
        """
        residual = hidden_states
        topk_indices, topk_weights = self.gating_network(hidden_states)

        batch_size, seq_len, _ = hidden_states.shape
        routed_output = self.route_through_experts(
            token_embeddings=hidden_states,
            expert_indices=topk_indices.view(batch_size, seq_len, -1),
            expert_weights=topk_weights.view(batch_size, seq_len, -1)
        )

        shared_output = self.shared_expert(residual)
        return routed_output + shared_output

    @torch.no_grad()
    def route_through_experts(
        self,
        token_embeddings: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Route tokens to their top-k experts and aggregate outputs.
        Args:
            token_embeddings (Tensor): (batch_size, seq_len, hidden_dim)
            expert_indices (Tensor): (batch_size, seq_len, k)
            expert_weights (Tensor): (batch_size, seq_len, k)
        Returns:
            Tensor: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = token_embeddings.shape
        aggregated_output = torch.zeros_like(token_embeddings)

        for b in range(batch_size):
            for t in range(seq_len):
                token = token_embeddings[b, t].unsqueeze(0)  # (1, hidden_dim)
                for k in range(self.num_experts_per_token):
                    expert_id = int(expert_indices[b, t, k])
                    weight = expert_weights[b, t, k]
                    expert = self.routed_experts[expert_id]

                    expert_output = expert(token).squeeze(0)  # (hidden_dim,)
                    aggregated_output[b, t] += expert_output * weight

        return aggregated_output
"""
def torch_flops():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dummy_input = torch.randn(2, 10, 2048,device=device)
    moe_model = MoE(config).to(device)
    output = moe_model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    model = moe_model
    model.to(device)
    x=dummy_input.to(device)
    # Run a forward pass.
    with torch.no_grad():
        output=model(x)
    # Output
    # Build the graph of the model. You can specify the operations (listed in `MODULE_FLOPs_MAPPING`, `FUNCTION_FLOPs_MAPPING` and `METHOD_FLOPs_MAPPING` in 'flops_ops.py') to ignore.
    flops_counter = TorchFLOPsByFX(model)
    # Print the grath (not essential)
    print('*' * 120)
    flops_counter.graph_model.graph.print_tabular()
    # Feed the input tensor
    with torch.no_grad():
        flops_counter.propagate(x)
    # Print the flops of each node in the graph. Note that if there are unsupported operations, the "flops" of these ops will be marked as 'not recognized'.
    print('*' * 120)
    result_table = flops_counter.print_result_table()
    # Print the total FLOPs
    total_flops = flops_counter.print_total_flops()
    total_time = flops_counter.print_total_time()
    max_memory = flops_counter.print_max_memory()
    flops_counter.save_result_to_csv("./result.csv", 'w')   
"""
def Convert_ONNX(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    moe_model = MoE(config).to(device).half()
    # set the model to inference mode 
    moe_model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(2, 10, 2048,device=device).half()

    # Export the model   
    torch.onnx.export(moe_model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "moe.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=11,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         #dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
         #                       'modelOutput' : {0 : 'batch_size'}}
         ) 
    print(" ") 
    print('Model has been converted to ONNX') 
    onnx_tool.model_profile("moe.onnx")

if __name__ == "__main__":
    #torch_flops()
    Convert_ONNX()