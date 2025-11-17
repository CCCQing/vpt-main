# tools/tsne_vis.py
"""
t-SNE 特征可视化工具
"""
import os
from collections import defaultdict
import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


@torch.no_grad()
def extract_features(model, loader, device, feat_type="cls", prompt_len=0, max_per_class=100):
    """
    feat_type: "cls" | "patch_mean" | "prompt_mean"
    prompt_len: 使用 VPT 时的 prompt 数（cfg.MODEL.PROMPT.LENGTH）
    """
    model.eval()
    feats, labels = [], []
    per_class = defaultdict(int)

    for batch in loader:
        x = batch["image"]
        y = batch["label"]
        if isinstance(x, np.ndarray): x = torch.from_numpy(x)
        if isinstance(y, np.ndarray): y = torch.from_numpy(y)
        x = x.to(device).float()
        y = y.to(device)

        if feat_type == "cls":
            # ViT 封装里自带：返回 (B, feat_dim) 的全局特征【证据：ViT.get_features】
            f = model.get_features(x)                                       # (B, D)
        else:
            # 取“整段 token 序列”的最终表示，再做切片与均值
            enc = model.enc
            tr = enc.transformer
            if hasattr(tr, "incorporate_prompt"):
                seq = tr.incorporate_prompt(x)                              # [B, 1+P+N, D]
            else:
                seq = tr.embeddings(x)                                      # [B, 1+N, D]
            tokens, _ = tr.encoder(seq)                                     # 编码后序列【Transformer.forward 的等价拆解】

            if feat_type == "patch_mean":
                start = 1 + int(prompt_len)
                f = tokens[:, start:, :].mean(dim=1)                        # 平均所有 patch token
            elif feat_type == "prompt_mean":
                if int(prompt_len) <= 0:
                    continue
                f = tokens[:, 1:1+int(prompt_len), :].mean(dim=1)           # 平均 prompt token
            else:
                raise ValueError("Unknown feat_type")

        f = f.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        for i in range(f.shape[0]):
            c = int(y_np[i])
            if per_class[c] < max_per_class:
                feats.append(f[i]); labels.append(c)
                per_class[c] += 1

    X = np.stack(feats); y = np.array(labels)
    return X, y


def run_tsne(X, y=None, pca_dim=50, perplexity=30, metric="cosine", random_state=0):
    X_in = X
    if pca_dim and X.shape[1] > pca_dim:
        X_in = PCA(n_components=pca_dim, random_state=random_state).fit_transform(X)
    tsne = TSNE(
        n_components=2, perplexity=perplexity, metric=metric,
        init="pca", learning_rate="auto", random_state=random_state
    )
    Z = tsne.fit_transform(X_in)
    return Z


def plot_tsne(Z, y, class_names=None, title="", save_path=None, markers=None):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(7, 6))
    y = np.array(y)
    classes = np.unique(y)
    for c in classes:
        idx = (y == c)
        label = str(c)
        if isinstance(class_names, list) and c < len(class_names):
            label = str(class_names[c])
        plt.scatter(Z[idx, 0], Z[idx, 1], s=8, label=label, alpha=0.7)
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

# ====== 追加：离线可视化入口（读取 .npz 或 .pth 并出图） ======
if __name__ == "__main__":
    import argparse

    def load_logits_pth(path):
        obj = torch.load(path, map_location="cpu")
        X = obj["joint_logits"].astype(np.float32)
        y = np.asarray(obj["targets"])
        return X, y, "logits"

    def load_npz(path):
        d = np.load(path, allow_pickle=True)
        X, y = d["X"], d["y"]
        tag = "feat"
        if "meta" in d:
            meta = d["meta"].item() if hasattr(d["meta"], "item") else d["meta"]
            tag = meta.get("type", tag)
        return X, y, tag

    # --- 修补：plot_tsne 保存目录为空字符串时不要 os.makedirs("") 报错 ---
    def safe_plot_tsne(Z, y, class_names=None, title="", save_path=None):
        plt.figure(figsize=(7, 6), dpi=140)
        y = np.asarray(y)
        for c in np.unique(y):
            idx = (y == c)
            label = str(c if class_names is None or c >= len(class_names) else class_names[c])
            plt.scatter(Z[idx,0], Z[idx,1], s=8, alpha=0.7, label=label)
        plt.legend(markerscale=2, fontsize=8, ncol=2, frameon=False)
        plt.title(title); plt.xticks([]); plt.yticks([]); plt.tight_layout()
        if save_path:
            dir_ = os.path.dirname(save_path)
            if dir_:
                os.makedirs(dir_, exist_ok=True)
            plt.savefig(save_path, dpi=300)
        plt.close()

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="路径：.npz (CLS/patch/prompt) 或 .pth (logits)")
    ap.add_argument("--out", required=True, help="输出图片路径，如 OUTPUT_DIR/vis/tsne.png")
    ap.add_argument("--perplexity", type=float, default=30)
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--metric", type=str, default="cosine")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_iter", type=int, default=1500)
    args = ap.parse_args()

    # 读取离线缓存
    if args.input.endswith(".pth"):
        X, y, tag = load_logits_pth(args.input)
    else:
        X, y, tag = load_npz(args.input)

    # 跑 t-SNE
    # （把 run_tsne 的迭代步数透传进去，保持可控）
    Z = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        metric=args.metric,
        init="pca",
        learning_rate=200.0,
        square_distances=True,
        random_state=args.seed,
        n_iter=args.n_iter,
        verbose=1,
    ).fit_transform(PCA(n_components=min(args.pca_dim, X.shape[1]), random_state=args.seed).fit_transform(X))

    title = f"t-SNE ({tag}) | pplex={args.perplexity}, pca={args.pca_dim}, metric={args.metric}, seed={args.seed}"
    safe_plot_tsne(Z, y, title=title, save_path=args.out)
    print(f"[OK] saved -> {args.out}")

# # 画 logits（你在 eval_classifier(save=True) 里保存的）
# python tsne_vis.py --input OUTPUT_DIR/test_CUB_logits.pth --out OUTPUT_DIR/vis/tsne_logits.png
#
# # 画 CLS（你在 eval_classifier(save=True 且 prefix=='test') 里保存的）
# 运行指令
# b16_224\lr0.25_wd0.001\run1\cache\test_CUB_cls.npz" --out "D:\postgraduate1\project\vpt-main\output\CUB-2\sup_vitb16_224\lr0.25_wd0.001\run1\vis\tsne_cls.png" --perplexity 30 --pca_dim 50 --metric cosine --seed 42 --n_iter 1500
