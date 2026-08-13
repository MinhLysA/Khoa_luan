"""
main.py
=======
Entry point duy nhất cho toàn bộ pipeline RL Inventory.

Cách sử dụng
------------
  # Chạy nhanh end-to-end (dữ liệu giả lập):
  python main.py train --synthetic
  python main.py evaluate

  # Dữ liệu M5 thực tế:
  python main.py preprocess --stores CA_1 TX_1 --n_skus 30
  python main.py train --episodes 500
  python main.py evaluate --checkpoint checkpoints/best_model.pth

  # Theo dõi TensorBoard:
  tensorboard --logdir runs/

Các script con (train.py, evaluate.py, data_preprocessing.py) vẫn có thể
chạy độc lập như trước — main.py chỉ là shortcut tổng hợp.
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# Fix encoding tren Windows: bat UTF-8 mode cho stdout/stderr
# Tranh UnicodeEncodeError khi in cac ky tu tieng Viet tu scripts con
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Dam bao thu muc rl_inventory nam trong sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils import load_config, set_seed


# ---------------------------------------------------------------------------
# Subcommand: preprocess
# ---------------------------------------------------------------------------

def cmd_preprocess(args, cfg: dict) -> None:
    """Tien xu ly du lieu M5 hoac tao du lieu synthetic."""
    pre_cfg = cfg.get("preprocess", {})
    path_cfg = cfg.get("paths", {})

    stores   = args.stores     or pre_cfg.get("stores",   ["CA_1", "TX_1"])
    n_skus   = args.n_skus     or pre_cfg.get("n_skus",    30)
    raw_dir  = args.raw_dir    or str(ROOT / path_cfg.get("raw_dir",   "data/raw"))
    out_dir  = args.output_dir or str(ROOT / path_cfg.get("data_dir", "data/processed"))
    seed     = args.seed       or cfg.get("seed", 42)

    # Ghi đè args để truyền vào script
    from scripts.data_preprocessing import parse_args as dp_parse, main as dp_main
    sys.argv = ["data_preprocessing.py",
                "--stores"] + stores + [
                "--n_skus",     str(n_skus),
                "--raw_dir",    raw_dir,
                "--output_dir", out_dir,
                "--seed",       str(seed),
    ]
    if args.synthetic:
        sys.argv.append("--synthetic")

    dp_main()


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------

def cmd_train(args, cfg: dict) -> None:
    """Huấn luyện tác tử Double DQN."""
    train_cfg = cfg.get("train",   {})
    env_cfg   = cfg.get("env",     {})
    path_cfg  = cfg.get("paths",   {})
    seed      = args.seed or cfg.get("seed", 42)

    set_seed(seed)

    # Build sys.argv để train.py parse như cũ (backward-compatible)
    argv = ["train.py",
        "--episodes",      str(args.episodes      or train_cfg.get("episodes",    500)),
        "--hidden_dim",    str(args.hidden_dim     or train_cfg.get("hidden_dim",  256)),
        "--batch_size",    str(args.batch_size     or train_cfg.get("batch_size",  128)),
        "--lr",            str(args.lr             or train_cfg.get("lr",          3e-4)),
        "--gamma",         str(args.gamma          or train_cfg.get("gamma",       0.99)),
        "--buffer_cap",    str(args.buffer_cap     or train_cfg.get("buffer_cap",  100_000)),
        "--eps_start",     str(args.eps_start      or train_cfg.get("eps_start",   1.0)),
        "--eps_min",       str(args.eps_min        or train_cfg.get("eps_min",     0.05)),
        "--eps_decay",     str(args.eps_decay      or train_cfg.get("eps_decay",   50_000)),
        "--n_skus",        str(args.n_skus         or env_cfg.get("n_skus",        30)),
        "--data_dir",      str(args.data_dir       or ROOT / path_cfg.get("data_dir",      "data/processed")),
        "--log_dir",       str(args.log_dir        or ROOT / path_cfg.get("log_dir",       "runs")),
        "--checkpoint_dir",str(args.checkpoint_dir or ROOT / path_cfg.get("checkpoint_dir", "checkpoints")),
        "--save_every",    str(args.save_every     or train_cfg.get("save_every",  100)),
        "--eval_every",    str(args.eval_every     or train_cfg.get("eval_every",  50)),
        "--seed",          str(seed),
    ]
    if args.use_per or train_cfg.get("use_per", False):
        argv.append("--use_per")
    if args.synthetic:
        argv.append("--synthetic")

    sys.argv = argv

    from scripts.train import parse_args as tr_parse, train as tr_train
    tr_args = tr_parse()
    tr_train(tr_args)


# ---------------------------------------------------------------------------
# Subcommand: evaluate
# ---------------------------------------------------------------------------

def cmd_evaluate(args, cfg: dict) -> None:
    """Đánh giá và so sánh DQN với các baseline."""
    eval_cfg  = cfg.get("eval",    {})
    path_cfg  = cfg.get("paths",   {})
    seed      = args.seed or eval_cfg.get("seed", 100)

    set_seed(seed)

    checkpoint = args.checkpoint or str(
        ROOT / path_cfg.get("checkpoint_dir", "checkpoints") / "best_model.pth"
    )

    argv = ["evaluate.py",
        "--checkpoint",  checkpoint,
        "--n_episodes",  str(args.n_episodes  or eval_cfg.get("n_episodes",  20)),
        "--data_dir",    str(args.data_dir    or ROOT / path_cfg.get("data_dir",    "data/processed")),
        "--output_dir",  str(args.output_dir  or ROOT / path_cfg.get("results_dir", "results")),
        "--hidden_dim",  str(args.hidden_dim  or eval_cfg.get("hidden_dim",  256)),
        "--seed",        str(seed),
    ]
    if args.synthetic:
        argv.append("--synthetic")

    sys.argv = argv

    from scripts.evaluate import parse_args as ev_parse, evaluate as ev_evaluate
    ev_args = ev_parse()
    ev_evaluate(ev_args)


# ---------------------------------------------------------------------------
# CLI chính
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="RL Inventory - Entry point cho pipeline huan luyen va danh gia.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Vi du su dung:
  python main.py train --synthetic                  # Chay nhanh voi du lieu gia lap
  python main.py train --episodes 500               # Train day du voi M5
  python main.py evaluate                           # Danh gia checkpoint tot nhat
  python main.py preprocess --stores CA_1 TX_1      # Tien xu ly M5
  python main.py train --synthetic --use_per        # Train voi PER
""",
    )
    parser.add_argument(
        "--config", type=str, default=str(ROOT / "config.yaml"),
        help="Duong dan toi file config YAML (mac dinh: rl_inventory/config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- preprocess ----
    pre = subparsers.add_parser("preprocess", help="Tien xu ly du lieu M5 / tao synthetic")
    pre.add_argument("--stores",     nargs="+", default=None, help="Danh sach ID cua hang M5")
    pre.add_argument("--n_skus",     type=int,  default=None, help="So SKU moi cua hang")
    pre.add_argument("--raw_dir",    type=str,  default=None, help="Thu muc du lieu tho M5")
    pre.add_argument("--output_dir", type=str,  default=None, help="Thu muc dau ra")
    pre.add_argument("--synthetic",  action="store_true",     help="Tao du lieu gia lap")
    pre.add_argument("--seed",       type=int,  default=None, help="Hat giong ngau nhien")

    # ---- train ----
    tr = subparsers.add_parser("train", help="Huan luyen tac tu Double DQN")
    tr.add_argument("--episodes",       type=int,   default=None)
    tr.add_argument("--hidden_dim",     type=int,   default=None)
    tr.add_argument("--batch_size",     type=int,   default=None)
    tr.add_argument("--lr",             type=float, default=None)
    tr.add_argument("--gamma",          type=float, default=None)
    tr.add_argument("--buffer_cap",     type=int,   default=None)
    tr.add_argument("--eps_start",      type=float, default=None)
    tr.add_argument("--eps_min",        type=float, default=None)
    tr.add_argument("--eps_decay",      type=int,   default=None)
    tr.add_argument("--n_skus",         type=int,   default=None)
    tr.add_argument("--use_per",        action="store_true")
    tr.add_argument("--synthetic",      action="store_true")
    tr.add_argument("--data_dir",       type=str,   default=None)
    tr.add_argument("--log_dir",        type=str,   default=None)
    tr.add_argument("--checkpoint_dir", type=str,   default=None)
    tr.add_argument("--save_every",     type=int,   default=None)
    tr.add_argument("--eval_every",     type=int,   default=None)
    tr.add_argument("--seed",           type=int,   default=None)

    # ---- evaluate ----
    ev = subparsers.add_parser("evaluate", help="Danh gia va so sanh DQN voi baseline")
    ev.add_argument("--checkpoint",  type=str,  default=None)
    ev.add_argument("--n_episodes",  type=int,  default=None)
    ev.add_argument("--data_dir",    type=str,  default=None)
    ev.add_argument("--output_dir",  type=str,  default=None)
    ev.add_argument("--hidden_dim",  type=int,  default=None)
    ev.add_argument("--synthetic",   action="store_true")
    ev.add_argument("--seed",        type=int,  default=None)

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Load config YAML
    cfg = load_config(args.config)

    if args.command == "preprocess":
        cmd_preprocess(args, cfg)
    elif args.command == "train":
        cmd_train(args, cfg)
    elif args.command == "evaluate":
        cmd_evaluate(args, cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
