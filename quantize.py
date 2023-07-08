import torch

X = torch.rand((5, 5))
print("Original Tensor:")
print(X)

#where X is an unquantized tensor
def zeropoint_quantize(X):
     # Calculate value range (denominator)
    x_range = torch.max(X) - torch.min(X)
    x_range = 1 if x_range == 0 else x_range

    # Calculate scale
    scale = 255 / x_range

    # Shift by zero-point
    zeropoint = (-scale * torch.min(X) - 128).round()

    # Scale and round the inputs
    X_quant = torch.clip((X * scale + zeropoint).round(), -128, 127)

    # Dequantize
    X_dequant = (X_quant - zeropoint) / scale

    return X_quant.to(torch.int8), X_dequant


X_quant, X_dequant = zeropoint_quantize(X)

print("\nQuantized Tensor:")
print(X_quant)

print("\nDequantized Tensor:")
print(X_dequant)
