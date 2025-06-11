import torch

def check_versions():
    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    cuda_available = torch.cuda.is_available()

    print(f"PyTorch version: {torch_version}")
    print(f"CUDA version: {cuda_version if cuda_version else 'CUDA not available'}")
    print(f"Is CUDA available: {cuda_available}")

if __name__ == "__main__":
    check_versions()
