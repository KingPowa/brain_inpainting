import time
import torch as th
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

# Function to benchmark model training

def benchmark_model(model, batch_size=1, image_size=64, num_batches=10, device='cuda'):
    model.to(device)
    model.train()
    
    # Dummy input & labels
    x = th.randn(batch_size, model.in_channels, image_size, image_size, device=device)
    timesteps = th.randint(0, 1000, (batch_size,), device=device)
    optimizer = th.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    
    print("Starting benchmarking...")
    
    # Forward pass timing
    start_time = time.time()
    for _ in range(num_batches):
        optimizer.zero_grad()
        with record_function("Forward pass"):
            output = model(x, timesteps)
        
    forward_time = (time.time() - start_time) / num_batches
    print(f"Average forward pass time: {forward_time:.4f} sec/batch")
    
    # Backward pass timing
    start_time = time.time()
    for _ in range(num_batches):
        optimizer.zero_grad()
        with record_function("Backward pass"):
            output = model(x, timesteps)
            loss = loss_fn(output, th.zeros_like(output))
            loss.backward()
        optimizer.step()
    
    backward_time = (time.time() - start_time) / num_batches
    print(f"Average backward pass time: {backward_time:.4f} sec/batch")
    
    # Profile memory usage
    max_memory = th.cuda.max_memory_allocated(device) / 1e6  # MB
    print(f"Peak memory usage: {max_memory:.2f} MB")
    
    # Layer-wise profiling
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("Model Profiling"):
            output = model(x, timesteps)
            loss = loss_fn(output, th.zeros_like(output))
            loss.backward()
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    return {
        "forward_time": forward_time,
        "backward_time": backward_time,
        "max_memory_MB": max_memory
    }

# Example usage
if __name__ == "__main__":
    from model_definition import UNetModel, model_and_diffusion_defaults  # Import your model definition
    
    defaults = model_and_diffusion_defaults()
    model = UNetModel(
        image_size=defaults['image_size'],
        in_channels=defaults['in_channels'],
        model_channels=defaults['num_channels'],
        out_channels=1,  # Assuming a single output channel for test
        num_res_blocks=defaults['num_res_blocks'],
        attention_resolutions=[int(r) for r in defaults['attention_resolutions'].split(',')],
        dropout=defaults['dropout'],
        dims=defaults['dims'],
        use_checkpoint=defaults['use_checkpoint'],
        use_fp16=defaults['use_fp16'],
        num_heads=defaults['num_heads'],
        num_head_channels=defaults['num_head_channels'],
        resblock_updown=defaults['resblock_updown'],
        use_new_attention_order=defaults['use_new_attention_order'],
        num_groups=defaults['num_groups'],
        resample_2d=defaults['resample_2d']
    )
    
    benchmark_results = benchmark_model(model, batch_size=2, image_size=128, num_batches=5, device='cuda')
    print("Benchmarking completed.")
