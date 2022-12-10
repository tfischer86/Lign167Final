# Code for LIGN 167 final project.

### Authors

Thomas Fischer

Vikram Srinivasan

### Running the code

The code in this project was developed using pytorch 1.9.1+cu111, which was the version on UCSD Datahub.

```bash
pip install -r requirements.txt
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

First make a directory for the Amazon MASSIVE dataset.

```bash
mkdir -p assets/cache
```

To run, start main.py:
```bash
python main.py
```

By default, the sparse model is trained. You can train the dense model by passing `--dense`. You can set hyperparameters using the command line arguments. See the output from `--help` for more options.
