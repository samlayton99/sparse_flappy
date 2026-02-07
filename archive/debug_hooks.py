"""Debug script to test if forward hooks are working correctly."""

import torch
import torch.nn.functional as F
from model import create_model
from data import get_dataloader

# Global counter to track hook calls
hook_call_count = 0
original_outputs = []
modified_outputs = []

def debug_hook(module, input, output):
    """Hook that tracks calls and modifies output."""
    global hook_call_count, original_outputs, modified_outputs
    hook_call_count += 1
    
    # Store original output stats
    original_outputs.append({
        "call": hook_call_count,
        "shape": output.shape,
        "mean": output.mean().item(),
        "std": output.std().item(),
        "norm": output.norm().item(),
    })
    
    # Add significant noise
    eps = 1.0
    v = torch.randn_like(output)
    out_norm = output.norm(dim=-1, keepdim=True)
    v_norm = v.norm(dim=-1, keepdim=True)
    noise = eps * v * out_norm / (v_norm + 1e-8)
    modified = output + noise
    
    # Store modified output stats
    modified_outputs.append({
        "call": hook_call_count,
        "mean": modified.mean().item(),
        "std": modified.std().item(),
        "norm": modified.norm().item(),
        "diff_norm": (modified - output).norm().item(),
    })
    
    print(f"  Hook fired #{hook_call_count}: output shape {output.shape}, "
          f"norm={output.norm().item():.2f}, noise_norm={(modified-output).norm().item():.2f}")
    
    return modified


def test_hook_firing():
    """Test 1: Do hooks actually fire?"""
    global hook_call_count, original_outputs, modified_outputs
    hook_call_count = 0
    original_outputs = []
    modified_outputs = []
    
    print("=" * 60)
    print("TEST 1: Do hooks fire?")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a small model for testing
    model = create_model(
        model_class="SimpleTransformer",
        d_model=256,
        n_layers=2,
        n_heads=4,
        d_ff=512,
    )
    model.to(device)
    model.eval()
    
    print(f"\nModel has {len(model.layers)} layers")
    print(f"Layer 0 attn module: {type(model.layers[0].attn)}")
    print(f"Layer 0 ff module: {type(model.layers[0].ff)}")
    
    # Register hook on layer 0 attention
    print("\nRegistering hook on model.layers[0].attn...")
    handle = model.layers[0].attn.register_forward_hook(debug_hook)
    
    # Create dummy input
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    
    print(f"\nRunning forward pass with input shape {input_ids.shape}...")
    with torch.no_grad():
        output = model(input_ids)
    
    print(f"\nHook was called {hook_call_count} times")
    print(f"Expected: 1 call (once for layer 0 attn)")
    
    handle.remove()
    
    return hook_call_count > 0


def test_output_modification():
    """Test 2: Does the modified output actually propagate?"""
    global hook_call_count
    hook_call_count = 0
    
    print("\n" + "=" * 60)
    print("TEST 2: Does modified output propagate?")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_model(
        model_class="SimpleTransformer",
        d_model=256,
        n_layers=2,
        n_heads=4,
        d_ff=512,
    )
    model.to(device)
    model.eval()
    
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    
    # Run WITHOUT hook
    print("\nRunning forward pass WITHOUT hook...")
    with torch.no_grad():
        output_clean = model(input_ids).clone()
    
    # Register noise hook
    def noise_hook(module, input, output):
        # Add massive noise to make the difference obvious
        noise = torch.randn_like(output) * 100.0
        return output + noise
    
    handle = model.layers[0].attn.register_forward_hook(noise_hook)
    
    # Run WITH hook
    print("Running forward pass WITH hook (adding massive noise)...")
    with torch.no_grad():
        output_noisy = model(input_ids).clone()
    
    handle.remove()
    
    # Compare
    diff = (output_noisy - output_clean).abs()
    print(f"\nClean output: mean={output_clean.mean().item():.4f}, std={output_clean.std().item():.4f}")
    print(f"Noisy output: mean={output_noisy.mean().item():.4f}, std={output_noisy.std().item():.4f}")
    print(f"Difference: mean={diff.mean().item():.4f}, max={diff.max().item():.4f}")
    
    outputs_differ = diff.max().item() > 0.01
    print(f"\nOutputs differ: {outputs_differ}")
    
    return outputs_differ


def test_loss_changes():
    """Test 3: Does noise actually change the loss?"""
    print("\n" + "=" * 60)
    print("TEST 3: Does noise change the loss?")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_model(
        model_class="SimpleTransformer",
        d_model=256,
        n_layers=2,
        n_heads=4,
        d_ff=512,
    )
    model.to(device)
    model.eval()
    
    # Create dummy data
    batch_size, seq_len = 4, 128
    input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    labels = input_ids[:, 1:].contiguous()
    input_ids = input_ids[:, :-1].contiguous()
    
    def compute_loss(logits, labels):
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        ).item()
    
    # Loss without hook
    print("\nComputing loss WITHOUT noise...")
    with torch.no_grad():
        logits_clean = model(input_ids)
    loss_clean = compute_loss(logits_clean, labels)
    print(f"Clean loss: {loss_clean:.4f}")
    
    # Loss with noise hook
    def noise_hook(module, input, output):
        eps = 1.0
        v = torch.randn_like(output)
        out_norm = output.norm(dim=-1, keepdim=True)
        v_norm = v.norm(dim=-1, keepdim=True)
        noise = eps * v * out_norm / (v_norm + 1e-8)
        return output + noise
    
    handles = []
    for layer in model.layers:
        handles.append(layer.attn.register_forward_hook(noise_hook))
        handles.append(layer.ff.register_forward_hook(noise_hook))
    
    print(f"Registered {len(handles)} hooks (attn + ff for each layer)")
    print("Computing loss WITH noise (eps=1.0)...")
    
    with torch.no_grad():
        logits_noisy = model(input_ids)
    loss_noisy = compute_loss(logits_noisy, labels)
    print(f"Noisy loss: {loss_noisy:.4f}")
    
    for h in handles:
        h.remove()
    
    loss_changed = abs(loss_noisy - loss_clean) > 0.01
    print(f"\nLoss change: {loss_noisy - loss_clean:.4f}")
    print(f"Loss changed significantly: {loss_changed}")
    
    return loss_changed


def test_autocast_interaction():
    """Test 4: Does autocast affect hook behavior?"""
    print("\n" + "=" * 60)
    print("TEST 4: Does autocast affect hooks?")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Skipping autocast test - no CUDA")
        return True
    
    model = create_model(
        model_class="SimpleTransformer",
        d_model=256,
        n_layers=2,
        n_heads=4,
        d_ff=512,
    )
    model.to(device)
    model.eval()
    
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    
    def noise_hook(module, input, output):
        print(f"    Hook: output dtype={output.dtype}, shape={output.shape}")
        noise = torch.randn_like(output) * 10.0
        return output + noise
    
    handle = model.layers[0].attn.register_forward_hook(noise_hook)
    
    # Without autocast
    print("\nForward WITHOUT autocast:")
    with torch.no_grad():
        out1 = model(input_ids).clone()
    
    # With autocast
    print("\nForward WITH autocast (bfloat16):")
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out2 = model(input_ids).clone()
    
    handle.remove()
    
    print(f"\nOutput 1 dtype: {out1.dtype}")
    print(f"Output 2 dtype: {out2.dtype}")
    
    return True


def test_test_noise_hook_implementation():
    """Test 5: Test the exact hook implementation from test_noise.py"""
    print("\n" + "=" * 60)
    print("TEST 5: Test exact test_noise.py hook implementation")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_model(
        model_class="SimpleTransformer",
        d_model=256,
        n_layers=2,
        n_heads=4,
        d_ff=512,
    )
    model.to(device)
    model.eval()
    
    # This is the EXACT implementation from test_noise.py
    def create_noise_hook(eps: float):
        def hook(module, input, output):
            if eps <= 0:
                return output
            
            v = torch.randn_like(output)
            out_norm = output.norm(dim=-1, keepdim=True)
            v_norm = v.norm(dim=-1, keepdim=True)
            noise = eps * v * out_norm / (v_norm + 1e-8)
            
            return output + noise
        
        return hook
    
    batch_size, seq_len = 4, 128
    input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    labels = input_ids[:, 1:].contiguous()
    input_ids = input_ids[:, :-1].contiguous()
    
    def compute_loss(logits, labels):
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        ).item()
    
    # Test different eps values
    eps_values = [0.0, 0.1, 0.5, 1.0, 5.0]
    results = []
    
    for eps in eps_values:
        hook_fn = create_noise_hook(eps)
        handles = []
        for layer in model.layers:
            handles.append(layer.attn.register_forward_hook(hook_fn))
            handles.append(layer.ff.register_forward_hook(hook_fn))
        
        with torch.no_grad():
            logits = model(input_ids)
        loss = compute_loss(logits, labels)
        results.append((eps, loss))
        
        for h in handles:
            h.remove()
    
    print("\nResults:")
    print("-" * 30)
    for eps, loss in results:
        print(f"eps={eps:.1f}: loss={loss:.4f}")
    
    # Check if losses are different
    losses = [r[1] for r in results]
    all_same = all(abs(l - losses[0]) < 0.001 for l in losses)
    
    if all_same:
        print("\n⚠️  WARNING: All losses are identical!")
        print("This reproduces the bug!")
    else:
        print("\n✓ Losses vary with eps (hooks working)")
    
    return not all_same


if __name__ == "__main__":
    print("=" * 60)
    print("HOOK DEBUGGING TESTS")
    print("=" * 60)
    
    results = {}
    
    results["test1_hooks_fire"] = test_hook_firing()
    results["test2_output_modified"] = test_output_modification()
    results["test3_loss_changes"] = test_loss_changes()
    results["test4_autocast"] = test_autocast_interaction()
    results["test5_exact_impl"] = test_test_noise_hook_implementation()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
