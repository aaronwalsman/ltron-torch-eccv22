import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="ltron-torch-eccv22",
    version="0.0.1",
    install_requires = [
        'ltron', 'tqdm', 'numpy', 'pyquaternion', 'tensorboard', 'conspiracy'],
    author="Aaron Walsman",
    author_email="aaronwalsman@gmail.com",
    description='LTRON Torch Experiments"',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aaronwalsman/ltron-torch",
    packages=setuptools.find_packages(),
    entry_points = {
        'console_scripts' : [
            'train_break_and_make_bc='
                'ltron_torch.train.break_and_make_bc:train_break_and_make_bc',
            'plot_break_and_make_bc='
                'ltron_torch.train.break_and_make_bc:plot_break_and_make_bc',
            'eval_break_and_make='
                'ltron_torch.train.break_and_make_eval:break_and_make_eval',
            'train_blocks_bc=ltron_torch.train.blocks_bc:train_blocks_bc',
        ]
    }
)
