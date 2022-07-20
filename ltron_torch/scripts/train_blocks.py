from ltron_torch.train.blocks_bc import (
    BlocksBCConfig, train_blocks_bc)

def main():
    config = BlocksBCConfig.from_commandline()
    train_blocks_bc(config)
