"""
資料視覺化 Web App
使用 Streamlit + Matplotlib + scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_iris, load_wine, fetch_california_housing
import streamlit as st

# ── 頁面設定 ──────────────────────────────────────────────────
st.set_page_config(page_title="資料視覺化工具", page_icon="📊", layout="wide")
st.title("📊 資料視覺化工具")
st.markdown("選擇左側資料集，點擊按鈕即可產生圖表。")

# ── 側邊欄選單 ────────────────────────────────────────────────
with st.sidebar:
    st.header("選擇資料集")
    dataset = st.radio(
        "",
        ["🌸 Iris 鳶尾花", "🍷 Wine 葡萄酒", "🏠 California Housing 房價"],
    )
    run = st.button("產生圖表", use_container_width=True, type="primary")

# ════════════════════════════════════════════════════════════════
#  1. Iris
# ════════════════════════════════════════════════════════════════
def plot_iris():
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = ["花萼長度", "花萼寬度", "花瓣長度", "花瓣寬度"]
    target_names  = iris.target_names
    colors        = ["#e74c3c", "#2ecc71", "#3498db"]

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Iris 鳶尾花資料集視覺化", fontsize=16, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    for i, name in enumerate(target_names):
        mask = y == i
        ax1.scatter(X[mask, 2], X[mask, 3],
                    c=colors[i], label=name, alpha=0.7, edgecolors="white", s=60)
    ax1.set_xlabel("花瓣長度 (cm)")
    ax1.set_ylabel("花瓣寬度 (cm)")
    ax1.set_title("散佈圖：花瓣長度 vs 花瓣寬度")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = fig.add_subplot(gs[0, 2])
    counts = [np.sum(y == i) for i in range(3)]
    ax2.pie(counts, labels=target_names, colors=colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax2.set_title("各品種樣本比例")

    ax3 = fig.add_subplot(gs[1, :2])
    x_pos = np.arange(len(feature_names))
    bar_w = 0.25
    for i, (name, color) in enumerate(zip(target_names, colors)):
        means = X[y == i].mean(axis=0)
        ax3.bar(x_pos + i * bar_w, means, bar_w,
                label=name, color=color, alpha=0.85, edgecolor="white")
    ax3.set_xticks(x_pos + bar_w)
    ax3.set_xticklabels(feature_names)
    ax3.set_ylabel("平均值 (cm)")
    ax3.set_title("各品種特徵平均值比較")
    ax3.legend()
    ax3.grid(axis="y", linestyle="--", alpha=0.4)

    ax4 = fig.add_subplot(gs[1, 2])
    data_by_class = [X[y == i, 0] for i in range(3)]
    bp = ax4.boxplot(data_by_class, patch_artist=True,
                     medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax4.set_xticklabels(target_names)
    ax4.set_ylabel("花萼長度 (cm)")
    ax4.set_title("花萼長度盒鬚圖")
    ax4.grid(axis="y", linestyle="--", alpha=0.4)

    return fig


# ════════════════════════════════════════════════════════════════
#  2. Wine
# ════════════════════════════════════════════════════════════════
def plot_wine():
    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names = [
        "酒精", "蘋果酸", "灰分", "灰分鹼度", "鎂",
        "酚類", "類黃酮", "非類黃酮酚", "花青素",
        "顏色強度", "色調", "OD280", "脯氨酸"
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Wine 葡萄酒資料集視覺化", fontsize=16, fontweight="bold")

    corr = np.corrcoef(X.T)
    im   = axes[0].imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_xticks(range(len(feature_names)))
    axes[0].set_yticks(range(len(feature_names)))
    axes[0].set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticklabels(feature_names, fontsize=8)
    axes[0].set_title("特徵相關係數熱力圖")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            axes[0].text(j, i, f"{corr[i, j]:.1f}",
                         ha="center", va="center", fontsize=5.5,
                         color="black" if abs(corr[i, j]) < 0.7 else "white")

    colors  = ["#e74c3c", "#2ecc71", "#3498db"]
    selected_features = feature_names[:6]
    N       = len(selected_features)
    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    X_norm  = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

    ax_r = plt.subplot(122, polar=True)
    for i, (name, color) in enumerate(zip(wine.target_names, colors)):
        means  = X_norm[y == i, :6].mean(axis=0).tolist()
        means += means[:1]
        ax_r.plot(angles, means, color=color, linewidth=2, label=name)
        ax_r.fill(angles, means, color=color, alpha=0.15)

    ax_r.set_thetagrids(np.degrees(angles[:-1]), selected_features, fontsize=9)
    ax_r.set_ylim(0, 1)
    ax_r.set_title("三種葡萄酒特徵雷達圖\n（標準化後）", pad=20)
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax_r.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════
#  3. California Housing
# ════════════════════════════════════════════════════════════════
def plot_housing():
    housing = fetch_california_housing()
    prices  = housing.target
    income  = housing.data[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("California Housing 加州房價資料集視覺化",
                 fontsize=16, fontweight="bold")

    axes[0].hist(prices, bins=50, color="#3498db", edgecolor="white", alpha=0.85)
    axes[0].axvline(np.mean(prices),   color="#e74c3c", linestyle="--",
                    linewidth=2, label=f"平均：{np.mean(prices):.2f}")
    axes[0].axvline(np.median(prices), color="#f39c12", linestyle="--",
                    linewidth=2, label=f"中位：{np.median(prices):.2f}")
    axes[0].set_xlabel("中位房價（十萬美元）")
    axes[0].set_ylabel("樣本數")
    axes[0].set_title("房價分布直方圖")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(prices), size=2000, replace=False)
    sc  = axes[1].scatter(income[idx], prices[idx],
                           c=prices[idx], cmap="plasma",
                           alpha=0.5, s=15, edgecolors="none")
    plt.colorbar(sc, ax=axes[1], label="中位房價（十萬美元）")
    axes[1].set_xlabel("平均收入（萬美元）")
    axes[1].set_ylabel("中位房價（十萬美元）")
    axes[1].set_title("平均收入 vs 中位房價散佈圖")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig


# ── 主邏輯 ────────────────────────────────────────────────────
if run:
    with st.spinner("繪製中，請稍候..."):
        if "Iris" in dataset:
            fig = plot_iris()
        elif "Wine" in dataset:
            fig = plot_wine()
        else:
            fig = plot_housing()

    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("請從左側選擇資料集，再點擊「產生圖表」。")
