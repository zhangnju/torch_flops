# Adapted from https://huggingface.co/bird-of-paradise/deepseek-mla/blob/main/src/mla.py



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math
from torch_flops import TorchFLOPsByFX

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  
    freqs = torch.outer(t, freqs).float()  
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    q_len = xq.shape[1]
    k_len = xk.shape[1]
    q_freqs = freqs_cis[:q_len]
    k_freqs = freqs_cis[:k_len]
    
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    q_freqs = reshape_for_broadcast(q_freqs, xq_)
    k_freqs = reshape_for_broadcast(k_freqs, xk_)
    
    xq_out = torch.view_as_real(xq_ * q_freqs).flatten(xq.ndim-1) 
    xk_out = torch.view_as_real(xk_ * k_freqs).flatten(xk.ndim-1)

    return xq_out.type_as(xq), xk_out.type_as(xk)




class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self, 
        d_model,             
        num_head, 
        d_embed, 
        d_c, 
        d_c1, 
        d_rotate, 
        bias=True,
        max_batch_size=256,   
        max_seq_len=8192    
        ):
        super().__init__()
        
        self.d_model = d_model
        self.num_head = num_head
        self.d_head=d_model//num_head
        self.d_embed = d_embed
        self.d_c = d_c
        self.d_c1 = d_c1
        self.d_rotate = d_rotate

        # Linear down-projection(compression) transformations
        self.DKV_proj = nn.Linear(d_embed, d_c, bias=bias)
        self.DQ_proj = nn.Linear(d_embed, d_c1, bias=bias)
        
        # linear up-projection transformations
        self.UQ_proj = nn.Linear(d_c1, d_model, bias=bias)
        self.UK_proj = nn.Linear(d_c, d_model, bias=bias)
        self.UV_proj = nn.Linear(d_c, d_model, bias=bias)

        # Linear RoPE-projection
        self.RQ_proj = nn.Linear(d_c1, num_head*d_rotate, bias=bias)
        self.RK_proj = nn.Linear(d_embed, d_rotate, bias=bias)
        
        # linear output transformations
        self.output_proj = nn.Linear( d_model, d_model, bias=bias)

        # Initiialize scaler
        self.scaler = float(1.0 / math.sqrt(self.d_head + d_rotate)) # Store as float in initialization

        # Initialize freqs_cis for RoPE
        self.freqs_cis = precompute_freqs_cis(
            d_rotate, max_seq_len * 2
        )
    

    def forward(
        self, 
        sequence, 
        start_pos: int = 0
    ):

    
        batch_size, seq_len, model_dim = sequence.size()
        # prepare for RoPE
        self.freqs_cis = self.freqs_cis.to(sequence.device)
        freqs_cis = self.freqs_cis[start_pos : ]
        kv_seq_len =  seq_len
        # Down and up projection for query
        C_Q = self.DQ_proj(sequence)     #[batch_size, seq_len, d_c1]
        Q_state = self.UQ_proj(C_Q)      #[batch_size, seq_len, d_model]
        # Linear projection for query RoPE pathway
        Q_rotate = self.RQ_proj(C_Q)      #[batch_size, seq_len, num_head*d_rotate]
        C_KV = self.DKV_proj(sequence) 
        K_rotate = self.RK_proj(sequence)

        K_state = self.UK_proj(C_KV)               #[batch_size, kv_seq_len/cached_len, d_model]
        V_state = self.UV_proj(C_KV)               #[batch_size, kv_seq_len/cached_len, d_model]

        
        Q_state = Q_state.view(batch_size, seq_len, self.num_head, self.d_head)

        actual_kv_len = K_state.size(1)    # kv_seq_len or start_pos + kv_seq_len
        # Use actual_kv_len instead of kv_seq_len for reshaping
        K_state = K_state.view(batch_size, actual_kv_len, self.num_head, self.d_head) 
        V_state = V_state.view(batch_size, actual_kv_len, self.num_head, self.d_head)

        #Apply RoPE to query and shared key
        Q_rotate = Q_rotate.view(batch_size, seq_len, self.num_head, self.d_rotate)
        K_rotate = K_rotate.unsqueeze(2).expand(-1, -1, self.num_head, -1)  # [batch, cached_len, num_head, d_rotate]
        #Q_rotate, K_rotate = apply_rotary_emb(Q_rotate, K_rotate, freqs_cis=freqs_cis)

        # Concatenate along head dimension
        Q_state = torch.cat([Q_state, Q_rotate], dim=-1)  # [batch_size, seq_len, num_head, d_head + d_rotate]
        K_state = torch.cat([K_state, K_rotate], dim=-1)  # [batch_size, actual_kv_len, num_head, d_head + d_rotate]

        # Scale Q by 1/sqrt(d_k)
        Q_state = Q_state * self.scaler
        Q_state = Q_state.transpose(1, 2)  # [batch_size, num_head, seq_len, head_dim]
        K_state = K_state.transpose(1, 2)  # [batch_size, num_head, actual_kv_len, head_dim]
        V_state = V_state.transpose(1, 2)  # [batch_size, num_head, actual_kv_len, head_dim]

        # Compute attention matrix: QK^T
        self.att_matrix = torch.matmul(Q_state, K_state.transpose(-1,-2)) 
        
        # apply softmax to the last dimension to get the attention score: softmax(QK^T)
        att_score = F.softmax(self.att_matrix, dim = -1)
    
        # get final output: softmax(QK^T)V
        att_output = torch.matmul(att_score, V_state)
        
        # concatinate all attention heads
        att_output = att_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_head*self.d_head) 

        # final linear transformation to the concatenated output
        att_output = self.output_proj(att_output)

        return att_output

device = 'cuda:0'
        
mla = MultiHeadLatentAttention(512, 8, 512, 64, 64, 64) # (model_dim, num_head, d_embed, d_c, d_c1, d_rotate)
attention_input = torch.randn(2, 16, 512) # (batch_size, seq_len, model_dim)
output = mla(attention_input)
print(output.shape)

model = mla
model.to(device)
x=attention_input.to(device)
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