import torch

def dense_indices(sh):
    """ Return a list of length `len(sh)`.
        Each entry is an array with shape `sh`, listing
        out the indices for that entry.

        >>> dense_indices([3])
        [tensor([0, 1, 2])]
        >>> dense_indices([3,2])
        [tensor([[0, 0],
                [1, 1],
                [2, 2]]), tensor([[0, 1],
                [0, 1],
                [0, 1]])]
    """
    if len(sh) == 0:
        return []
    i0 = torch.arange(sh[0])
    if len(sh) == 1:
        return [ i0 ]
    idx = dense_indices(sh[1:])
    for s in sh[1:]: # add new axes at end
        i0 = i0.unsqueeze(-1)
    return [i0.broadcast_to(sh)] + [x.broadcast_to(sh) for x in idx]

def expand_dims(x, ind):
    """ Broadcast x by adding new indices just before the dimensions
        specified by ind. ind should contain indices in the set
        (0 [far left side], 1, ..., len(x) [far right])

        Note that these are applied by expanding x
        right-to-left.  Negative indices will be expanded last,
        so are going to be counted with respect to the already-
        expanded order.
    """
    for i in reversed(sorted(ind)):
        x = x.unsqueeze(i)
    return x

def to_coo(idxs, vals, dims, wrap=None):
    """ Take a batched list of values and starting indices
        and return a coo tensor.

        Idxs is an array of indices, whose last dimension
        holds the starting index for each sparse sub-block.
        - idxs.size(-1) == len(dims)

        Both idxs and vals have the same batch dimensions,
        but the last dimensions of vals are dense sub-blocks.
        - bshape = idxs.shape[:-1]
        - bshape + sparse_shape == vals.shape

        Dims gives the expanded shapes of the sparse dimensions.
        - bshape + dims == expanded shape of vals
        - len(sparse_shape) == len(dims)

    s = torch.sparse_coo_tensor(
            [[0, 0, 0, 1, 1, 1],
             [0, 1, 1, 0, 1, 1],
             [0, 0, 1, 2, 3, 4]] ,
            torch.Tensor([1., 2., 3., 4., 5., 6.]),
            (2, 2, 5))
    """
    if wrap is None:
        wrap = (True,)*len(dims)
    assert idxs.size(-1) == len(dims), "Need one index per sparse dim"
    assert len(wrap) == len(dims), "Need wrap = True/False for each sparse dim."
    bdim = len(idxs.shape)-1 # number of batch dims
    bshape = idxs.shape[:-1]

    assert len(vals.shape) == len(bshape) + len(dims), "Vals shape must include batch dims then sparse dims."
    assert vals.shape[:bdim] == bshape, "idxs and vals must have same batch shape"

    ndim = len(dims) # number of sparse dimensions

    sparse_idxs = dense_indices(vals.shape)
    # shift all sparse indices
    mask = True # accumulate mask here
    for i in range(ndim):
        eset = [bdim]*i + [-1]*(ndim-i)
        expanded = sparse_idxs[i+bdim].clone() \
                 + expand_dims(idxs[...,i], eset)
        if wrap[i]: # wrap here
            expanded = wrap_idx(expanded, dims[i])
        else:
            mask = mask_idx(expanded, dims[i]) & mask
        sparse_idxs[i+bdim] = expanded

    idxs = torch.stack(sparse_idxs).reshape(len(vals.shape), -1)
    vals = vals.reshape(-1)
    if mask is not True: # need to apply this mask
        mask = mask.reshape(-1)
        idxs = idxs[:,mask]
        vals = vals[mask]

    #print(torch.stack(sparse_idxs).reshape(ndim, -1))
    #print(vals.reshape(-1))
    #print(bshape + dims)

    return torch.sparse_coo_tensor(idxs, vals, bshape + dims)

def wrap_idx(idx, nmax):
    return idx - nmax*(idx//nmax)

def mask_idx(idx, nmax):
    return (idx >= 0) & (idx < nmax)

if __name__=="__main__":
    idx = torch.IntTensor([[1, 2, 0, 3],
                           [0, 2, 1, 1],
                           [3, 3, 1, 2]]).unsqueeze(-1)
    # splines produce a block size of 2 for each
    # batch, where bshape = (3,4)
    vals = torch.arange(12*2).reshape(3,4,2)
    ans = to_coo(idx, vals, (8,), (False,))
    print(ans)

    idx = torch.IntTensor([[[1, 2], [0, 3]],
                           [[0, 2], [1, 1]],
                           [[3, 3], [1, 2]]])
    # splines produce a block size of 3x2 for each
    # batch, where bshape = (3,2)
    vals = torch.arange(3*2 * 3*2).reshape(3,2,3,2)
    ans = to_coo(idx, vals, (6,5), (True, False))
    print(ans)
    print(ans.sum(1).to_dense())
