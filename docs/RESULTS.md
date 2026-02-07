# Memory Access Counting Methodology

I measure **total memory accesses**, assuming a flattened memory hierarchy (treating L2, SRAM, and HBM uniformly). Each tensor operation is counted as: (1) **reading input activations**, (2) **reading parameters/weights**, and (3) **writing output activations**. Reshape, permute, and transpose operations count as a single touch of all elements. For attention, we assume the full T×T score matrix is materialized (not fused), which introduces the dominant O(BHT²) term. 

For example X_new = X @ W would be batch_size * token_length * hidden_dim to read in X, it would be hidden_dim * hidden_dim to read in W, and it would be batch_size * token_length * hidden_dim to write X_new to memory. So, assuming everything is in bf16, then this would have total memory access of 2 * (2* batch_size * token_length * hidden_dim + hidden_dim * hidden_dim)

---

