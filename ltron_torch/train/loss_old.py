import torch
from torch.nn.functional import (
        binary_cross_entropy, binary_cross_entropy_with_logits)

def dense_class_label_loss(
        dense_label_logits,
        dense_label_target,
        #dense_scores,
        class_weight):
    
    class_label_loss = torch.nn.functional.cross_entropy(
            dense_label_logits,
            dense_label_target,
            weight=class_weight,
            reduction='mean')
    
    return class_label_loss

'''
def dense_seq_class_label_loss(
        dense_label_logits,
        dense_label_target,
        #dense_scores,
        class_weight,
        padding_mask):
    
    class_label_loss = torch.nn.functional.cross_entropy(
            dense_label_logits[padding_mask],
            dense_label_target[padding_mask],
            weight=class_weight,
            reduction='mean')
    
    return class_label_loss
'''

def dense_score_loss(
        dense_score_logits,
        correct,
        foreground,
        background_weight = 0.01,
        correct_weight = 0.1,
        incorrect_weight = 1.0):
    
    # get dimensions and reshape to batch_size x h x w
    # this removes any single channel
    b = dense_score_logits.shape[0]
    h, w = dense_score_logits.shape[-2:]
    dense_score_logits = dense_score_logits.view(b, h, w)
    correct = correct.view(b, h, w)
    
    # bce scores to correct
    # TODO: _with_logits
    score_loss = binary_cross_entropy_with_logits(
            dense_score_logits,
            correct.float() * foreground,
            reduction = 'none')
    
    # reweight the different regions and sum
    #foreground_score_loss = score_loss * foreground
    incorrect = 1. - correct
    summed_loss = torch.sum(score_loss * (
            foreground * correct * correct_weight +
            foreground * incorrect * incorrect_weight +
            ~foreground * background_weight))
    
    # compute the normalizer
    total_foreground_correct = torch.sum(foreground * correct)
    total_foreground_incorrect = torch.sum(foreground * incorrect)
    total_background = foreground.numel() - (
            total_foreground_correct + total_foreground_incorrect)
    normalizer = (
            total_foreground_correct * correct_weight +
            total_foreground_incorrect * incorrect_weight +
            total_background * background_weight)
    
    # normalize and return the loss
    normalized_loss = summed_loss / normalizer
    return normalized_loss

def cross_product_loss(
        logits,
        target,
        correct_weight = 1.0,
        incorrect_weight = 1.0):
    h, w = logits.shape[-2:]
    if h == 0 or w == 0:
        return 0.
    
    #import pdb
    #pdb.set_trace()
    cross_loss = binary_cross_entropy_with_logits(
            logits, target.float(), reduction='none')
    estimated_pos = logits > 0.5
    correct = estimated_pos == target
    
    #neg_weight = 1./(h*w)**0.5
    #neg_weight = 1.0
    neg_weight = (min(h,w) / (h*w)) * 2.
    pos_weight = (1. - neg_weight)
    #print('n:', neg_weight)
    #print('p:', pos_weight)
    
    correct_pos = correct * target
    correct_neg = correct * ~target
    incorrect_pos = ~correct * target
    incorrect_neg = ~correct * ~target
    summed_loss = torch.sum(cross_loss * (
            correct_pos * correct_weight * pos_weight +
            correct_neg * correct_weight * neg_weight +
            incorrect_pos * incorrect_weight * pos_weight +
            incorrect_neg * incorrect_weight * neg_weight))
    
    return summed_loss
    
    '''
    # compute the normalizer
    total_correct_pos = torch.sum(correct_pos)
    total_correct_neg = torch.sum(correct_neg)
    total_incorrect_pos = torch.sum(incorrect_pos)
    total_incorrect_neg = torch.sum(incorrect_neg)
    normalizer = (
            total_correct_pos * correct_weight +
            total_correct_neg * correct_weight * neg_weight +
            total_incorrect_pos * incorrect_weight +
            total_incorrect_neg * incorrect_weight * neg_weight)
    
    # normalize and return
    normalized_loss = summed_loss / normalizer
    return normalized_loss
    '''
