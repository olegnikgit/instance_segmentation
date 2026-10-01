RF-DETR based instance segmentstion package
===================

Package implements RF-DETR based instance segmentation and data management tools.


Installation
===================

```console
# Install UV following instructions here: https://github.com/astral-sh/uv
# For Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# make sure uv is added to your PATH after installation:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
# reload your shell configuration:
source ~/.bashrc


# Install Python 3.12:
uv python install 3.12

# To view available versions:
uv python list

# Create a virtual environment using it
uv venv .venv-deploy -p 3.12

# activate an environment
source .venv-deploy/bin/activate

# install pytorch:
uv pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# install instance_segmentation:
# clone instance_segmentation
cd instance_segmentation
uv pip install -e .

# install RF-DETR:
# clone RF-DETR
git pull https://github.com/roboflow/rf-detr.git
uv pip install -e .
# also install packages necessary for training if needed:
uv pip install -e ".[train,loggers]"
# the version installed and tested: rfdetr 1.8.0
# install additional packages:
uv pip install supervision

# Note, the version of torchaudio might cause some issues. Re-install it if needed:
uv pip uninstall torchaudio
uv pip install torchaudio==2.7.0+cu126 \
  --index-url https://download.pytorch.org/whl/cu126

# installing additional packages in currently active environment:
uv add --active <PACKAGE_NAME>

# Managing an environment.

# to save an environment:
uv pip freeze > all-deps.txt
# manually remove packages installed from local repos, e.g. custom-toolchain, etc.
# Save the pruned list as:
mv all-deps.txt deploy-requirements.in
# generate lockfile from your new input:
uv pip compile -o deploy-requirements.lock deploy-requirements.in

# to see the list of packages:
uv pip list

# To delete venv:
rm -rf .venv
```

