"""Targeted debugging of test_noise.py's exact code path."""

import torch
import torch.nn.functional as F
from model import create_model
from data import get_dataloader

# Global tracking
hook_fired_count = 0
PAD_TOKEN_ID = 50256

def create_noise_hook(eps: float):
    """EXACT copy from test_noise.py"""
    def hook(module, input, output):
        global hook_fired_count
        hook_fired_count += 1
        
        if eps <= 0:
            return output
        
        # Generate Gaussian noise
        v = torch.randn_like(output)
        
        # Compute norms
        out_norm = output.norm(dim=-1, keepdim=True)
        v_norm = v.norm(dim=-1, keepdim=True)
        
        # Relative noise: eps * v * ||output|| / ||v||
        noise = eps * v * out_norm / (v_norm + 1e-8)
        
        return output + noise
    
    return hook


def register_noise_hooks(model, targets, eps):
    """EXACT copy from test_noise.py"""
    hooks = []
    hook_fn = create_noise_hook(eps)
    
    for layer_idx, module_type in targets:
        if module_type == "attn":
            module = model.layers[layer_idx].attn
        elif module_type == "ff":
            module = model.layers[layer_idx].ff
        else:
            raise ValueError(f"Unknown module type: {module_type}")
        
        handle = module.register_forward_hook(hook_fn)
        hooks.append(handle)
    
    return hooks


def clear_hooks(hooks):
    """EXACT copy from test_noise.py"""
    for handle in hooks:
        handle.remove()


def evaluate_with_noise(model, dataloader, device, eps, targets, num_samples, batch_size):
    """EXACT copy from test_noise.py"""
    global hook_fired_count
    hook_fired_count = 0
    
    model.eval()
    
    # Register hooks
    hooks = register_noise_hooks(model, targets, eps)
    print(f"  Registered {len(hooks)} hooks for targets={targets}, eps={eps}")
    
    try:
        total_loss = 0.0
        total_tokens = 0
        samples_seen = 0
        
        for batch in dataloader:
            if samples_seen >= num_samples:
                break
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass (hooks will inject noise)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask)
            
            # Compute loss (ignore padding)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=PAD_TOKEN_ID,
                reduction="sum",
            )
            
            # Count non-pad tokens
            n_tokens = (labels != PAD_TOKEN_ID).sum().item()
            
            total_loss += loss.item()
            total_tokens += n_tokens
            samples_seen += input_ids.size(0)
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        print(f"  Hook fired {hook_fired_count} times, samples={samples_seen}, tokens={total_tokens}")
        return avg_loss
    
    finally:
        clear_hooks(hooks)


def load_model(checkpoint_path, device):
    """EXACT copy from test_noise.py"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint or use defaults
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        # Assume default config
        config = {
            "d_model": 2048,
            "n_layers": 16,
            "n_heads": 8,
            "d_ff": 4096,
        }
    
    print(f"Config from checkpoint: {config}")
    
    model = create_model(**config)
    
    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, config


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Try to load a real checkpoint
    checkpoint_path = "saved_models/weightless_step9999.pt"
    print(f"\nLoading model from {checkpoint_path}...")
    
    try:
        model, config = load_model(checkpoint_path, device)
    except FileNotFoundError:
        print("Checkpoint not found, creating a fresh model for testing")
        config = {"d_model": 256, "n_layers": 2, "n_heads": 4, "d_ff": 512}
        model = create_model(**config)
        model.to(device)
        model.eval()
    
    n_layers = len(model.layers)
    print(f"Model loaded: {n_layers} layers")
    print(f"Layer 0 type: {type(model.layers[0])}")
    print(f"Layer 0.attn type: {type(model.layers[0].attn)}")
    print(f"Layer 0.ff type: {type(model.layers[0].ff)}")
    
    # Test 1: Does baseline work?
    print("\n" + "="*60)
    print("TEST 1: Baseline evaluation (no hooks)")
    print("="*60)
    dataloader = get_dataloader(split="test", batch_size=32)
    baseline = evaluate_with_noise(
        model, dataloader, device,
        eps=0.0, targets=[],
        num_samples=500, batch_size=32
    )
    print(f"Baseline loss: {baseline:.10f}")
    
    # Test 2: Single layer, single module
    print("\n" + "="*60)
    print("TEST 2: Noise on layer 0 attn (eps=0.5)")
    print("="*60)
    dataloader = iter(get_dataloader(split="test", batch_size=32))
    loss_l0_attn = evaluate_with_noise(
        model, dataloader, device,
        eps=0.5, targets=[(0, "attn")],
        num_samples=500, batch_size=32
    )
    print(f"Loss with noise: {loss_l0_attn:.10f}")
    print(f"Difference from baseline: {loss_l0_attn - baseline:.10f}")
    
    # Test 3: Multiple targets
    print("\n" + "="*60)
    print("TEST 3: Noise on ALL layers (eps=1.0)")
    print("="*60)
    dataloader = iter(get_dataloader(split="test", batch_size=32))
    all_targets = [(i, m) for i in range(n_layers) for m in ["attn", "ff"]]
    loss_all = evaluate_with_noise(
        model, dataloader, device,
        eps=1.0, targets=all_targets,
        num_samples=500, batch_size=32
    )
    print(f"Loss with noise: {loss_all:.10f}")
    print(f"Difference from baseline: {loss_all - baseline:.10f}")
    
    # Test 4: Verify hooks are being called
    print("\n" + "="*60)
    print("TEST 4: Verifying hook execution with print statements")
    print("="*60)
    
    call_count = [0]
    def verbose_hook(module, input, output):
        call_count[0] += 1
        if call_count[0] <= 3:  # Only print first 3
            print(f"    >>> Hook called! module={type(module).__name__}, output shape={output.shape}")
        noise = torch.randn_like(output) * 0.5
        return output + noise
    
    handle = model.layers[0].attn.register_forward_hook(verbose_hook)
    
    dataloader = iter(get_dataloader(split="test", batch_size=2))
    batch = next(dataloader)
    input_ids = batch["input_ids"].to(device)
    
    print("Running forward pass...")
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = model(input_ids)
    
    handle.remove()
    print(f"Verbose hook was called {call_count[0]} times")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Baseline:         {baseline:.10f}")
    print(f"Layer 0 attn:     {loss_l0_attn:.10f}")
    print(f"All layers:       {loss_all:.10f}")
    print(f"")
    if abs(loss_l0_attn - baseline) < 0.0001 and abs(loss_all - baseline) < 0.0001:
        print("⚠️  BUG CONFIRMED: Losses are identical - hooks are NOT affecting output!")
    else:
        print("✓ Losses differ - hooks are working in this test")


if __name__ == "__main__":
    main()
