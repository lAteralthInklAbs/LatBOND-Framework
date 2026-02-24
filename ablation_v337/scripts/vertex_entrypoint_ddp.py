#!/usr/bin/env python3
"""
Vertex AI entrypoint with DDP support for multi-GPU training.
v3.3.4: Supports both single-GPU and multi-GPU (DDP) modes.

Usage:
    # Single GPU (standard)
    python scripts/vertex_entrypoint_ddp.py --arch streaming_cnn --budget_ms 40 --condition matched

    # Multi-GPU (via torchrun)
    torchrun --nproc_per_node=2 scripts/vertex_entrypoint_ddp.py --arch causal_transformer --budget_ms 40 --condition matched
"""
import os
import sys
import subprocess
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_distributed():
    """Initialize distributed training from environment variables."""
    # Vertex AI / torchrun sets these for multi-GPU jobs
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if world_size > 1:
        import torch.distributed as dist
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        print(f"DDP initialized: rank {rank}/{world_size}, local_rank {local_rank}")
        return True, local_rank, rank
    return False, 0, 0


def main():
    parser = argparse.ArgumentParser(description='LatBOND DDP Entrypoint')
    parser.add_argument('--arch', type=str, required=True,
                       choices=['streaming_cnn', 'causal_transformer', 'tcn'])
    parser.add_argument('--budget_ms', type=int, default=40)
    parser.add_argument('--condition', type=str, default=None,
                       choices=['matched', 'truncated', 'buffered'])
    parser.add_argument('--budgets', type=str, default=None,
                       help='Comma-separated budget list')
    parser.add_argument('--output_dir', type=str, default='/tmp/outputs')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Setup DDP if multi-GPU
    is_distributed, local_rank, global_rank = setup_distributed()

    # Set CUDA device for this process
    if is_distributed:
        import torch
        torch.cuda.set_device(local_rank)

    # Only rank 0 prints startup info
    if global_rank == 0:
        print("=" * 60)
        print("LatBOND v3.3.4 DDP Entrypoint")
        print("=" * 60)
        print(f"  Architecture:  {args.arch}")
        print(f"  Budget:        {args.budget_ms}ms")
        print(f"  Condition:     {args.condition}")
        print(f"  Distributed:   {is_distributed}")
        if is_distributed:
            import torch.distributed as dist
            print(f"  World size:    {dist.get_world_size()}")
        print("=" * 60)

    # Build command for run_full_study.py
    cmd = [
        sys.executable, "run_full_study.py",
        "--mode", "single_arch",
        "--architecture", args.arch,
        "--output_dir", args.output_dir,
        "--seed", str(args.seed),
    ]

    if args.budgets:
        cmd.extend(["--budgets", args.budgets])
    else:
        cmd.extend(["--budgets", str(args.budget_ms)])

    if args.condition:
        cmd.extend(["--condition", args.condition])

    if global_rank == 0:
        print(f"Launching: {' '.join(cmd)}")
    sys.stdout.flush()

    # Run training
    result = subprocess.run(cmd)

    # Cleanup DDP
    if is_distributed:
        import torch.distributed as dist
        dist.destroy_process_group()

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
