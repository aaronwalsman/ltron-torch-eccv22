import torch

'''
def padded_causal_self_mask(t, pad):
    n, b = t.shape
    
    # make the causal mask
    causal_mask = t.view(n, 1, b) < t.view(1, n, b)
    
    # make the padding mask
    square = torch.max(torch.arange(n).view(n,1), torch.arange(n).view(1,n))
    square = square.to(pad.device)
    padding_mask = square.unsqueeze(-1) >= pad.view(1,1,b)
    
    # combine masks
    mask = causal_mask | padding_mask
    
    # make the diagonal False to avoid nan
    mask[torch.arange(n), torch.arange(n), :] = False
    
    return mask
'''

def padded_causal_mask(tq, pad_q, tk=None, pad_k=None):
    
    # fill defaults
    if tk is None:
        tk = tq
    if pad_k is None:
        pad_k = pad_q
    
    # pull out shape information
    nq, bq = tq.shape
    nk, bk = tk.shape
    assert bq == bk
    
    # make the causal mask
    causal_mask = tq.view(nq, 1, bq) < tk.view(1, nk, bk)
    
    # make the padding mask
    device = pad_q.device
    q_pad = torch.arange(nq, device=device).view(nq, 1) >= pad_q.view(1, bq)
    k_pad = torch.arange(nk, device=device).view(nk, 1) >= pad_k.view(1, bk)
    padding_mask = q_pad.view(nq, 1, bq) | k_pad.view(1, nk, bk)
    
    # combine the causal and padding masks then return
    return causal_mask | padding_mask
