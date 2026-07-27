import torch

print("Torch version :", torch.__version__)
print("CUDA disponible :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))
    print("Mémoire totale (Go) :", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))