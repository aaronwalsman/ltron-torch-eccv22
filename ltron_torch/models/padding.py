import torch

def cat_padded_seqs(a, b, a_pad, b_pad, seq_dim=0, batch_dim=1, pad_value=0):
    '''
    aa____   bbb_                              aabbb____
    aaa___   b___                              aaab_____
    aaaaaa , bbb_ , (2,3,6,1), (3, 1, 3, 4) -> aaaaaabbb , (5, 4, 9, 5)
    a_____   bbbb                              abbbb____
    '''
    # compute new lengths
    ab_pad = a_pad + b_pad
    
    # add extra entries required to handle the new maximum length
    max_pad = torch.max(ab_pad)
    extra_pad = max_pad - a.shape[seq_dim]
    if extra_pad > 0:
        extra_shape = (*a.shape[:seq_dim], extra_pad, *a.shape[seq_dim+1:])
        extra_values = torch.full(
            extra_shape, pad_value, dtype=a.dtype, device=a.device)
        ab = torch.cat((a, extra_values), dim=seq_dim)
    else:
        ab = a.clone()
    
    # construct indices for reading and writing elements
    ai = get_seq_range_indices(a_pad, a_pad+b_pad)
    bi, bj = get_seq_batch_indices(b_pad)
    ab_index = get_index_tuple(ai, bj, len(ab.shape), seq_dim, batch_dim)
    b_index = get_index_tuple(bi, bj, len(b.shape), seq_dim, batch_dim)
    
    # read values from b and write into ab
    ab[ab_index] = b[b_index]
    
    return ab, ab_pad

def decat_padded_seq(ab, a_pad, b_pad, seq_dim=0, batch_dim=1, pad_value=0):
    # compute lengths
    max_a_pad = torch.max(a_pad)
    max_b_pad = torch.max(b_pad)
    
    a_shape = (*ab.shape[:seq_dim], max_a_pad, *ab.shape[seq_dim+1:])
    b_shape = (*ab.shape[:seq_dim], max_b_pad, *ab.shape[seq_dim+1:])
    
    # construct a using the first block of the tensor
    a_slices = [slice(None) for _ in ab.shape]
    a_slices[seq_dim] = slice(max_a_pad)
    a = ab[a_slices].clone()
    a_mask = make_padding_mask(a_pad, a.shape, seq_dim, batch_dim)
    a[a_mask] = pad_value
    
    # construct b using careful indexing
    ai = get_seq_range_indices(a_pad, a_pad+b_pad)
    bi, bj = get_seq_batch_indices(b_pad)
    ab_index = get_index_tuple(ai, bj, len(ab.shape), seq_dim, batch_dim)
    b_index = get_index_tuple(bi, bj, len(ab.shape), seq_dim, batch_dim)
    b = torch.full(b_shape, pad_value, dtype=ab.dtype, device=ab.device)
    b[b_index] = ab[ab_index]
    
    return a, b

def linearize_padded_seq(x, pad, seq_dim=0, batch_dim=1):
    '''
    12__
    3456
    78__ , (2,4,2,1) -> 123456789
    9___
    '''
    # return all elements from x that are inside the pad region
    xi, xj = get_seq_batch_indices(pad)
    x_index = get_index_tuple(xi, xj, len(x.shape), seq_dim, batch_dim)
    return x[x_index]

def make_padding_mask(pad, shape, seq_dim=0, batch_dim=1, mask_value=False):
    '''
                 000111
                 000011
    (3,4,1,6) -> 011111
                 000000
    '''
    # make a mask that is False inside the pad region and True everywhere else
    if mask_value:
        mask = torch.zeros(shape, dtype=torch.bool, device=pad.device)
    else:
        mask = torch.ones(shape, dtype=torch.bool, device=pad.device)
    i, j = get_seq_batch_indices(pad)
    index = get_index_tuple(i, j, len(shape), seq_dim, batch_dim)
    mask[index] = bool(mask_value)
    return mask

def get_seq_batch_indices(pad):
    # get indices for the sequence and batch dimensions for every element in
    # the pad region
    seq_indices = get_seq_indices(pad)
    batch_indices = get_batch_indices(pad)
    return seq_indices, batch_indices

def get_seq_indices(pad):
    # get indices for the sequence dimension for every element in the pad region
    seq_indices = torch.cat([torch.arange(p, device=pad.device) for p in pad])
    return seq_indices
    
def get_seq_range_indices(pad_start, pad_end):
    # get indices for the sequence dimension of a padded range
    seq_indices = torch.cat([
        torch.arange(ps, pe, device=pad_end.device)
        for ps, pe in zip(pad_start, pad_end)
    ])
    return seq_indices

def get_batch_indices(pad):
    # get indices for the batch dimension for every element in the pad region
    batch_indices = torch.cat([
        torch.full((p,), i, dtype=torch.long, device=pad.device)
        for i, p in enumerate(pad)
    ])
    return batch_indices

def get_index_tuple(seq_indices, batch_indices, dims, seq_dim, batch_dim):
    # put sequence and batch indices into an index tuple for a padded tensor
    index = [slice(None) for d in range(dims)]
    index[seq_dim] = seq_indices
    index[batch_dim] = batch_indices
    return tuple(index)
