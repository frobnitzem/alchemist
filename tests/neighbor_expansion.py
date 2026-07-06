import torch

def neighbor_expansion(r, neighborlists):
    # r: (B, N, k) 
    # neighborlists: [(N, M1), (N, M2)] indices of neighbors for each atom
    # return: (B, N, (M1+M2)*k) 
    # Combine neighbor lists efficiently and use batched gather/indexing.
    # Ensure neighbor indices are on the same device and are long tensors.
    nbr = torch.hstack([nl.to(r.device).long() for nl in neighborlists])  # (N, M)

    B, N, k = r.shape
    M = nbr.shape[1]

    # Expand to (B, N, M) so we have per-batch, per-atom neighbor indices
    nbr = nbr.unsqueeze(0).expand(B, -1, -1)  # (B, N, M)

    # Expand r to (B, N, M, k) so we can gather along dim=1 (the atom axis)
    r_exp = r.unsqueeze(2).expand(-1, -1, M, -1)  # (B, N, M, k)

    # Index shape must match r_exp for gather: (B, N, M, k)
    idx = nbr.unsqueeze(-1).expand(-1, -1, -1, k)

    # Gather neighbor properties and flatten the last two dims to (M*k)
    out = torch.gather(r_exp, dim=1, index=idx)  # (B, N, M, k)
    return out.reshape(B, N, M * k)

a = torch.tensor([[[1,2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]) #2 batches with 3 atoms each
neighborlists = [torch.tensor([[1, 2], [0, 2], [0, 1]]), torch.tensor([[1], [2], [0]])] #each atom has 2 first neighbors and 1 second neighbor
print(neighbor_expansion(a, neighborlists)) #should return a tensor of shape (2, 2, 6) with the distances to the first neighbors for each atom in each batch