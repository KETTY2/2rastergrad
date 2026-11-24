## Environment Setup

```bash
git submodule update --init --recursive  #for simple-knn

# Create and activate environment
conda create -n H python=3.12
conda activate H

# Install PyTorch (with CUDA support, choose version for your CUDA)
# See: [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)

# Install other dependencies
pip install plyfile tqdm opencv-python joblib

# Install submodules
pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
pip install submodules/fused-ssim --no-build-isolation
# single scene
python train.py -s <dataset>/scene --eval
