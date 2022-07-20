from ltron.hierarchy import auto_pad_stack_numpy_hierarchies

def pad_stack_collate(seqs):
    seqs, pad = auto_pad_stack_numpy_hierarchies(
        *seqs, pad_axis=0, stack_axis=1)
    return seqs, pad
