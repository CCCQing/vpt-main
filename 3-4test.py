import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


# ========= 用户要求的总输出目录（绝对路径）=========
BASE_OUTPUT_DIR = Path(r"D:\postgraduate1\project\vpt-main\vpt-main\output")

# ========= 你的 config 路径 =========
CONFIG_FILE = Path("configs/prompt/cub.yaml")

# ========= 通用开关（来自 CLI 的统一模板）=========
COMMON_OPTS = [
    "SOLVER.OVERFIT_ONE_BATCH_STEPS", "1000",
    "SOLVER.LOG_EVERY_N", "20",
    "MODEL.PROMPT.DEBUG_FLOW", "True",
    "SOLVER.DEBUG_GRAD_NORM", "True",
    "SOLVER.OVERFIT_DISABLE_PROMPT_SAMPLING", "True",
    # 建议显式保证 r_similarity 和 visual proj 打开（不依赖 yaml 默认）
    "MODEL.R_SIMILARITY.ENABLE", "True",
    "MODEL.R_SIMILARITY.VISUAL_PROJ_ENABLE", "True",
]


def run_one_experiment(
    name: str,
    extra_opts: List[str],
    base_output_dir: Path,
    config_file: Path,
    dry_run: bool = False,
) -> int:
    """
    Run: python train.py --config-file <cfg> OUTPUT_DIR <out_dir> <opts...>
    Save console to <out_dir>/console.log
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_output_dir / f"{name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 强制输出目录写到你指定的 D:\...\output 下
    opts = ["OUTPUT_DIR", str(out_dir)] + COMMON_OPTS + extra_opts

    cmd = [sys.executable, "train.py", "--config-file", str(config_file)] + opts

    # 保存命令本身，方便复现
    (out_dir / "cmd.txt").write_text(" ".join(cmd), encoding="utf-8")

    print(f"\n========== Running: {name} ==========")
    print(f"OUTPUT_DIR: {out_dir}")
    print("CMD:", " ".join(cmd))

    if dry_run:
        return 0

    log_path = out_dir / "console.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(Path.cwd()),
            env=env,
            check=False,
        )

    print(f"Exit code: {p.returncode}")
    print(f"Console log saved to: {log_path}")
    return p.returncode


def main(only: Optional[List[str]] = None, dry_run: bool = False) -> None:
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ========= 按 CLI 计划组织实验 =========
    # Step0：纯 CE + 关采样 + scale=10（校准）
    exp_step0 = [
        ("Step0_pureCE_samplingOff_scale10", [
            "SOLVER.LOSS", "softmax",
            "MODEL.AFFINITY.ENABLE", "False",
            "SOLVER.LOSS_ALPHA", "0.0",
            "MODEL.R_SIMILARITY.LOGIT_SCALE_INIT", "10.0",
        ]),
    ]

    # Step1：只改优化强度（LR/WD）
    exp_step1 = [
        ("Step1A_pureCE_LR0p005_WD0", [
            "SOLVER.LOSS", "softmax",
            "MODEL.AFFINITY.ENABLE", "False",
            "SOLVER.LOSS_ALPHA", "0.0",
            "MODEL.R_SIMILARITY.LOGIT_SCALE_INIT", "10.0",
            "SOLVER.BASE_LR", "0.005",
            "SOLVER.WEIGHT_DECAY", "0.0",
        ]),
        ("Step1B_pureCE_LR0p001_WD0", [
            "SOLVER.LOSS", "softmax",
            "MODEL.AFFINITY.ENABLE", "False",
            "SOLVER.LOSS_ALPHA", "0.0",
            "MODEL.R_SIMILARITY.LOGIT_SCALE_INIT", "10.0",
            "SOLVER.BASE_LR", "0.001",
            "SOLVER.WEIGHT_DECAY", "0.0",
        ]),
    ]

    # Step2：温度轨迹对照（可学习 vs 固定）
    # 注意：FIXED_LOGIT_SCALE 需要你已按 CLI 方案在 config.py/losses.py 加好
    exp_step2 = [
        ("Step2A_learnableScale_LR0p001_WD0", [
            "SOLVER.LOSS", "softmax",
            "MODEL.AFFINITY.ENABLE", "False",
            "SOLVER.LOSS_ALPHA", "0.0",
            "SOLVER.BASE_LR", "0.001",
            "SOLVER.WEIGHT_DECAY", "0.0",
            "MODEL.R_SIMILARITY.LOGIT_SCALE_INIT", "10.0",
            "MODEL.R_SIMILARITY.FIXED_LOGIT_SCALE", "0.0",
        ]),
        ("Step2B_fixedScale20_LR0p001_WD0", [
            "SOLVER.LOSS", "softmax",
            "MODEL.AFFINITY.ENABLE", "False",
            "SOLVER.LOSS_ALPHA", "0.0",
            "SOLVER.BASE_LR", "0.001",
            "SOLVER.WEIGHT_DECAY", "0.0",
            "MODEL.R_SIMILARITY.LOGIT_SCALE_INIT", "10.0",
            "MODEL.R_SIMILARITY.FIXED_LOGIT_SCALE", "20.0",
        ]),
        ("Step2C_fixedScale50_LR0p001_WD0", [
            "SOLVER.LOSS", "softmax",
            "MODEL.AFFINITY.ENABLE", "False",
            "SOLVER.LOSS_ALPHA", "0.0",
            "SOLVER.BASE_LR", "0.001",
            "SOLVER.WEIGHT_DECAY", "0.0",
            "MODEL.R_SIMILARITY.LOGIT_SCALE_INIT", "10.0",
            "MODEL.R_SIMILARITY.FIXED_LOGIT_SCALE", "50.0",
        ]),
    ]

    # Step3：逐级加回 align（alpha=0.001 -> 0.01）
    # 这里按 CLI 建议用 FIXED_LOGIT_SCALE=20 作为稳态基线（如果你 Step2 结论不同，可改）
    exp_step3 = [
        ("Step3A_align_alpha0p001_fixedScale20", [
            "SOLVER.LOSS", "softmax_prompt_align",
            "MODEL.AFFINITY.ENABLE", "True",
            "SOLVER.LOSS_ALPHA", "0.001",
            "SOLVER.BASE_LR", "0.001",
            "SOLVER.WEIGHT_DECAY", "0.0",
            "MODEL.R_SIMILARITY.FIXED_LOGIT_SCALE", "20.0",
        ]),
        ("Step3B_align_alpha0p01_fixedScale20", [
            "SOLVER.LOSS", "softmax_prompt_align",
            "MODEL.AFFINITY.ENABLE", "True",
            "SOLVER.LOSS_ALPHA", "0.01",
            "SOLVER.BASE_LR", "0.001",
            "SOLVER.WEIGHT_DECAY", "0.0",
            "MODEL.R_SIMILARITY.FIXED_LOGIT_SCALE", "20.0",
        ]),
    ]

    # Step4：只加回采样（单因素：OVERFIT_DISABLE_PROMPT_SAMPLING True -> False）
    # 这里以 Step3A 为基础，复制一份把 sampling 打开（False）
    exp_step4 = [
        ("Step4_samplingOn_align_alpha0p001_fixedScale20", [
            "SOLVER.LOSS", "softmax_prompt_align",
            "MODEL.AFFINITY.ENABLE", "True",
            "SOLVER.LOSS_ALPHA", "0.001",
            "SOLVER.BASE_LR", "0.001",
            "SOLVER.WEIGHT_DECAY", "0.0",
            "MODEL.R_SIMILARITY.FIXED_LOGIT_SCALE", "20.0",
            "SOLVER.OVERFIT_DISABLE_PROMPT_SAMPLING", "False",
        ]),
    ]

    experiments = exp_step0 + exp_step1 + exp_step2 + exp_step3 + exp_step4

    # 可选：只跑部分实验
    if only:
        experiments = [(n, o) for (n, o) in experiments if n in set(only)]
        if not experiments:
            raise SystemExit(f"No experiments matched --only={only}")

    # 逐个运行
    failed = []
    for name, extra_opts in experiments:
        rc = run_one_experiment(
            name=name,
            extra_opts=extra_opts,
            base_output_dir=BASE_OUTPUT_DIR,
            config_file=CONFIG_FILE,
            dry_run=dry_run,
        )
        if rc != 0:
            failed.append((name, rc))

    print("\n========== Summary ==========")
    if not failed:
        print("All experiments finished OK.")
    else:
        for name, rc in failed:
            print(f"FAILED: {name} (exit={rc})")
        raise SystemExit(1)


if __name__ == "__main__":
    # 简单参数解析（不依赖额外库）
    # 用法：
    #   python run_overfit_ablation.py
    #   python run_overfit_ablation.py --dry-run
    #   python run_overfit_ablation.py --only Step0_pureCE_samplingOff_scale10 Step1B_pureCE_LR0p001_WD0
    args = sys.argv[1:]
    dry = "--dry-run" in args
    only = None
    if "--only" in args:
        idx = args.index("--only")
        only = []
        for x in args[idx + 1:]:
            if x.startswith("--"):
                break
            only.append(x)
    main(only=only, dry_run=dry)