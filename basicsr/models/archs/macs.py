import torch
from torchprofile import profile_macs
from akdt_arch import AKDT

model = AKDT(inp_channels=3, 
        out_channels=3, 
        dim = 34,
        num_blocks = [1,2,2,4],
        num_refinement_blocks = 3,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
        dual_pixel_task = False
    )

input_tensor = torch.randn(1, 3, 256, 256) 

macs = profile_macs(model, input_tensor)
flops = macs * 2
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

tflops = flops / (1024*1024*1024)

print(f"Model FLOPs (G): {tflops} G")
print(f"Model FLOPs (M): {tflops*1024} M")

print(f"Model MACs (G): {macs / (1024*1024*1024)} G")

print(f"Model params (M): {num_params / 1e6}")
print(f"Model params: {num_params}")
