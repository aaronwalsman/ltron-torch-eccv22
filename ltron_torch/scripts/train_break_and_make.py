from ltron_torch.train.break_and_make_lstm_bc import (
    BreakAndMakeLSTMBCConfig, train_break_and_make_lstm_bc)
from ltron_torch.train.break_and_make_transformer_bc import (
    BreakAndMakeTransformerBCConfig, train_break_and_make_transformer_bc)

class BreakAndMakeConfig(
    BreakAndMakeLSTMBCConfig,
    BreakAndMakeTransformerBCConfig,
):
    model = 'lstm'

def main():
    config = BreakAndMakeConfig.from_commandline()
    
    if config.model == 'lstm':
        train_break_and_make_lstm_bc(config, device)
    
    elif config.model == 'transformer':
        train_break_and_make_transformer_bc(config)
