import copy

import numpy

from ltron.hierarchy import index_hierarchy, concatenate_numpy_hierarchies

from ltron_torch.dataset.episode_dataset import (
    EpisodeDatasetConfig, EpisodeDataset)

class BreakAndMakeDatasetConfig(EpisodeDatasetConfig):
    factor_cursor_distribution = False

class BreakAndMakeDataset(EpisodeDataset):
    def __getitem__(self, i):
        sequence = super().__getitem__(i)
        
        # fix cursor observations
        # why is this happening?  can we fix the next round of episodes we
        # produce to take this out?
        if isinstance(
            sequence['observations']['table_cursor']['position'], list):
            sequence['observations']['table_cursor']['position'] = (
                numpy.stack((
                    sequence['observations']['table_cursor']['position'][0],
                    sequence['observations']['table_cursor']['position'][1],
                ), axis=-1))
        
        if isinstance(
            sequence['observations']['hand_cursor']['position'], list):
            sequence['observations']['hand_cursor']['position'] = (
                numpy.stack((
                    sequence['observations']['hand_cursor']['position'][0],
                    sequence['observations']['hand_cursor']['position'][1],
                ), axis=-1))
        
        '''
        if self.config.factor_cursor_distribution:
            
            # expand spatial steps
            def expand_steps(seq, action_name, ids):
                for i in sorted(ids, reverse=True):
                    # extract original step
                    original_step = index_hierarchy(seq, [i])
                    
                    def make_activated_cursor_observations(step, activated):
                        for a in activated:
                            # for some reason the actions are one big array,
                            # but the observations are a list of two arrays,
                            # probably fix this if we make this permanent
                            step['observations'][a]['position'] = (
                                original_step['actions'][a]['position'])
                            step['observations'][a]['polarity'] = (
                                original_step['actions'][a]['polarity'])
                            # Note that the cursor has extra observations
                            # instance_id and snap_id that should be updated
                            # here, but they are not because it would require
                            # us to read from the next step, which may not
                            # exist, and also, this information isn't passed
                            # to the network anyway, so it shouldn't change
                            # anything in training.  If this becomes necessary
                            # for some reason I'm not forseeing now, we can
                            # spoof this using the snap render and cursor.
                    
                    def disable_cursor(step, region):
                        step['actions'][region]['activate'] = numpy.array(
                            [0], dtype=numpy.long)
                        step['actions'][region]['position'] = numpy.array(
                            [[0,0]], dtype=numpy.long)
                        step['actions'][region]['polarity'] = numpy.array(
                            [0], dtype=numpy.long)
                    
                    # make cursor steps
                    expanded_steps = []
                    activated = []
                    for region in 'hand_cursor', 'table_cursor':
                        if original_step['actions'][region]['activate'][0]:
                            # make a new step to move the cursor
                            step = copy.deepcopy(original_step)
                            
                            # turn the action off
                            step['actions'][action_name] = (
                                numpy.array([0], dtype=numpy.long))
                            
                            # turn other cursors off
                            for other_region in 'hand_cursor', 'table_cursor':
                                if other_region != region:
                                    disable_cursor(step, other_region)
                            
                            # make observations for perviously activated cursors
                            make_activated_cursor_observations(step, activated)
                            
                            activated.append(region)
                            expanded_steps.append(step)
                    
                    assert len(activated)
                    
                    # make action step
                    step = copy.deepcopy(original_step)
                    for region in 'hand_cursor', 'table_cursor':
                        disable_cursor(step, region)
                    
                    make_activated_cursor_observations(step, activated)
                    expanded_steps.append(step)
                    
                    # turn off all tiles except for the first
                    for step in expanded_steps[1:]:
                        step['observations']['table_tile_mask'][:] = 0
                        step['observations']['hand_tile_mask'][:] = 0
                    
                    # join everything together
                    seq = concatenate_numpy_hierarchies(
                        index_hierarchy(seq, slice(None, i)),
                        *expanded_steps,
                        index_hierarchy(seq, slice(i+1, None)),
                    )
                
                return seq
            
            disassembly_ids = numpy.where(sequence['actions']['disassembly'])[0]
            sequence = expand_steps(sequence, 'disassembly', disassembly_ids)
            
            pnp_ids = numpy.where(sequence['actions']['pick_and_place'])[0]
            sequence = expand_steps(sequence, 'pick_and_place', pnp_ids)
            
            rotate_ids = numpy.where(sequence['actions']['rotate'])[0]
            sequence = expand_steps(sequence, 'rotate', rotate_ids)
            
            sequence['observations']['step'] = numpy.arange(
                sequence['observations']['step'].shape[0])
        '''    
        return sequence

class BreakOnlyDataset(BreakAndMakeDataset):
    def __getitem__(self, i):
        data = super().__getitem__(i)
        
        i = numpy.where(data['actions']['phase'])[0]
        if len(i):
            data = index_hierarchy(data, slice(0, i[0]+1))
        
        return data
