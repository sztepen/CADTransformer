import torch
import sys

def check_versions_and_cuda():
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of CUDA Devices: {torch.cuda.device_count()}")
        print(f"NCCL Version: {torch.cuda.nccl.version()}")
        for i in range(torch.cuda.device_count()):
            print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA devices are not available. Please check your installation.")

if __name__ == "__main__":
    check_versions_and_cuda()
