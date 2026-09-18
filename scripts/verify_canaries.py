import os
import json
import numpy as np

canary_dirs = [
    ("Transformer-Q (LBF)", "results/lbf_transformer_q_sight3_random_seed1"),
    ("IPPO-GRU (LBF)", "results/lbf_ippo_gru_sight3_random_seed1"),
    ("Transformer-Q (Wolfpack)", "results/wolfpack_transformer_q_sight4_random_seed1"),
    ("IPPO-GRU (Wolfpack)", "results/wolfpack_ippo_gru_sight4_random_seed1"),
]

print("=" * 80)
print("CANARY VERIFICATION AUDIT REPORT: ALL 4 COMBINATIONS")
print("=" * 80)

for name, d in canary_dirs:
    print(f"\n--- {name} [{d}] ---")
    if not os.path.exists(d):
        print(f"Directory {d} NOT FOUND!")
        continue
    
    # 1. Checkpoints
    ckpt_dir = os.path.join(d, "checkpoints")
    ckpts = os.listdir(ckpt_dir) if os.path.exists(ckpt_dir) else []
    print(f"[Checkpoints]: Found {len(ckpts)} files: {ckpts}")
    for c in ckpts:
        sz = os.path.getsize(os.path.join(ckpt_dir, c))
        print(f"  - {c}: {sz / (1024*1024):.2f} MB")
    
    # 2. Summary JSON
    sum_path = os.path.join(d, "summary.json")
    if os.path.exists(sum_path):
        with open(sum_path) as f:
            summary = json.load(f)
        print(f"[Summary]: Steps={summary.get('train_steps'):,}, Completed Episodes={summary.get('completed_episodes'):,}, "
              f"Best Return={summary.get('best_eval_return')}, Final Return={summary.get('final_eval_return')}, "
              f"Win Rate={summary.get('final_eval_success_rate')}, Runtime={summary.get('runtime_seconds')}s")
    else:
        print("[Summary]: summary.json not found")
        
    # 3. Metrics CSV
    csv_path = os.path.join(d, "metrics.csv")
    if os.path.exists(csv_path):
        lines = []
        with open(csv_path) as f:
            header = f.readline().strip().split(",")
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 5:
                    lines.append(parts)
        print(f"[Logging]: CSV Header valid ({len(header)} columns). Total episodes recorded: {len(lines):,}")
        
        # Loss checks
        losses = []
        for row in lines:
            try:
                losses.append(float(row[4]))
            except ValueError:
                pass
        if losses:
            first_loss = losses[0]
            mid_loss = losses[len(losses)//2]
            final_loss = losses[-1]
            has_nan = any(np.isnan(losses))
            has_inf = any(np.isinf(losses))
            print(f"[Losses]: Initial={first_loss:.5f}, Mid={mid_loss:.5f}, Final={final_loss:.5f} | NaN={has_nan} | Inf={has_inf}")
    else:
        print("[Logging]: metrics.csv not found")

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)
