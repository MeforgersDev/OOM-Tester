## Automatic Batch Size Finder for PyTorch
This script helps you automatically find the maximum batch size that can be used without running into "Out of Memory" (OOM) errors on your GPU. The script increments the batch size in small steps (0.1 by default) and tests it by running through a mini training loop. It uses mixed precision training (FP16) to optimize performance and memory usage.
## Features
- Automatically adjusts the batch size in 0.1 increments.
- Handles GPU "Out of Memory" (OOM) errors gracefully.
- Uses mixed precision training with torch.cuda.amp to optimize performance.
- Logs successful batch sizes and training times.
- Finds the most efficient batch size without manual tuning.
## Requirements
-Python 3.x
-PyTorch with GPU support
-CUDA-enabled GPU
-Python packages: torch, time
