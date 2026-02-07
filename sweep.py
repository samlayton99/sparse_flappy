"""Wandb hyperparameter sweep script.

Runs persistent sessions of hyperparameter sweeps using wandb.
Models are NOT saved during sweeps to conserve disk space.

Usage:
    # Create new sweep and run 20 trials
    python sweep.py --count 20 --gpus 8 --num_steps 2000

    # Resume existing sweep
    python sweep.py --sweep_id abc123 --count 10
"""

import argparse
import subprocess
import os
import wandb


# Default sweep configuration
SWEEP_CONFIG = {
    "method": "bayes",  # bayes, grid, or random
    "metric": {
        "name": "val/loss",
        "goal": "minimize",
    },
    "parameters": {
        # Model architecture
        "d_model": {
            "values": [512, 1024, 2048],
        },
        "n_layers": {
            "values": [4, 8, 12, 16],
        },
        "n_heads": {
            "values": [4, 8, 16],
        },
        "d_ff": {
            "values": [1024, 2048, 4096],
        },
        # Training params
        "max_lr": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 5e-3,
        },
        "batch_size": {
            "values": [32, 48],
        },
    },
}


def train_sweep(gpus: int, num_steps: int, eval_every: int, project: str):
    """Run a single training trial with wandb sweep config.
    
    Args:
        gpus: Number of GPUs to use
        num_steps: Training steps per trial
        eval_every: Evaluation frequency
        project: wandb project name
    """
    # Initialize wandb run (gets config from sweep)
    run = wandb.init()
    config = wandb.config
    
    # Build torchrun command
    cmd = [
        "torchrun",
        f"--nproc_per_node={gpus}",
        "train.py",
        "--no_save",  # Don't save models during sweep
        f"--d_model={config.d_model}",
        f"--n_layers={config.n_layers}",
        f"--n_heads={config.n_heads}",
        f"--d_ff={config.d_ff}",
        f"--max_lr={config.max_lr}",
        f"--batch_size={config.batch_size}",
        f"--num_steps={num_steps}",
        f"--eval_every={eval_every}",
        f"--wandb_project={project}",
    ]
    
    print(f"\n{'='*60}")
    print(f"Starting sweep trial:")
    print(f"  d_model={config.d_model}, n_layers={config.n_layers}")
    print(f"  n_heads={config.n_heads}, d_ff={config.d_ff}")
    print(f"  max_lr={config.max_lr:.2e}, batch_size={config.batch_size}")
    print(f"{'='*60}\n")
    
    # Run training
    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=False,
        )
        if result.returncode != 0:
            print(f"Warning: Training exited with code {result.returncode}")
    except KeyboardInterrupt:
        print("\nSweep trial interrupted by user")
        raise
    except Exception as e:
        print(f"Error during training: {e}")
        wandb.log({"error": str(e)})
    
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Run wandb hyperparameter sweep")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of sweep trials to run")
    parser.add_argument("--gpus", type=int, default=8,
                        help="Number of GPUs to use per trial")
    parser.add_argument("--num_steps", type=int, default=2000,
                        help="Training steps per trial (shorter for exploration)")
    parser.add_argument("--eval_every", type=int, default=50,
                        help="Evaluation frequency")
    parser.add_argument("--project", type=str, default="weightless",
                        help="wandb project name")
    parser.add_argument("--sweep_id", type=str, default=None,
                        help="Resume existing sweep by ID (e.g., 'abc123')")
    parser.add_argument("--method", type=str, default="bayes",
                        choices=["bayes", "grid", "random"],
                        help="Sweep search method")
    args = parser.parse_args()
    
    # Update sweep config with method
    SWEEP_CONFIG["method"] = args.method
    
    # Create or resume sweep
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Resuming sweep: {args.project}/{sweep_id}")
    else:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=args.project)
        print(f"Created new sweep: {args.project}/{sweep_id}")
        print(f"View at: https://wandb.ai/{os.environ.get('WANDB_ENTITY', 'your-entity')}/{args.project}/sweeps/{sweep_id}")
    
    print(f"\nRunning {args.count} trials with {args.gpus} GPUs, {args.num_steps} steps each")
    print(f"Sweep method: {args.method}")
    print(f"Models will NOT be saved during sweep\n")
    
    # Create wrapper function with fixed args
    def train_fn():
        train_sweep(
            gpus=args.gpus,
            num_steps=args.num_steps,
            eval_every=args.eval_every,
            project=args.project,
        )
    
    # Run sweep agent
    try:
        wandb.agent(sweep_id, function=train_fn, count=args.count, project=args.project)
    except KeyboardInterrupt:
        print(f"\n\nSweep interrupted. To resume later:")
        print(f"  python sweep.py --sweep_id {sweep_id} --count N")
    
    print(f"\nSweep completed. View results at:")
    print(f"  https://wandb.ai/{os.environ.get('WANDB_ENTITY', 'your-entity')}/{args.project}/sweeps/{sweep_id}")


if __name__ == "__main__":
    main()
