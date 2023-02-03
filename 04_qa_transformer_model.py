import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        )
        output = nn.Linear(n_heads * d_v, d_model)(context)

        return (
            nn.LayerNorm(d_model)(output + residual),
            attn,
        )


# q: what does this MultiHeadAttention class do?
# a: it takes in a query, key, and value and returns an attention mask

# q: What a attention mask is used for?
# a: it is used to mask out certain values in the query, key, and value

# q: what is the purpose of the residual connection?
# a: it is used to help the model learn the identity function

# q: what is the purpose of the layer normalization?
# a: it is used to normalize the output of the model

# q: what is normalization?
# a: it is used to scale the output of the model to a certain range

# q: what is the purpose of the linear layer?
# a: it is used to transform the output of the model
