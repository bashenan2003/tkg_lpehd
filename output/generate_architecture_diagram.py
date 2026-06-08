"""
生成TKG-LPEHD架构图（修正版）
修正了原图中的实线/虚线错误，并清晰标注各模块间数据流关系
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 使用中文字体
for font_name in ["SimHei", "Microsoft YaHei", "Noto Sans SC"]:
    try:
        fm.findfont(font_name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(32, 22))
ax.set_xlim(0, 32)
ax.set_ylim(0, 22)
ax.axis("off")
ax.set_facecolor("#FAFBFC")

# ============================================================
# 颜色方案
# ============================================================
C_DATA = "#E3F2FD"       # 数据模块 - 浅蓝
C_DATA_BORDER = "#1565C0"
C_PHASE = "#FFFFFF"      # 阶段模块 - 白
C_PHASE_BORDER = "#455A64"
C_PHASE5 = "#FFF8E1"     # Phase 5 - 浅黄
C_PHASE5_BORDER = "#F9A825"
C_LLM = "#F3E5F5"        # LLM模块 - 浅紫
C_LLM_BORDER = "#7B1FA2"
C_OUTPUT = "#E8F5E9"     # 输出 - 浅绿
C_OUTPUT_BORDER = "#2E7D32"
C_SOLID = "#37474F"      # 实线 - 深灰
C_DASHED = "#D84315"     # 虚线 - 橙红
C_RULE_FLOW = "#1565C0"  # 规则流 - 蓝
C_GNN_FLOW = "#6A1B9A"   # 图推理流 - 紫

# ============================================================
# 辅助函数
# ============================================================
def draw_box(ax, x, y, w, h, text, color=C_PHASE, border=C_PHASE_BORDER,
             fontsize=10, fontcolor="#212121", bold=False, text_y_offset=0,
             sub_lines=None, corner_radius=0.1):
    """绘制圆角矩形框"""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.05,rounding_size={corner_radius}",
                         facecolor=color, edgecolor=border, linewidth=1.8, zorder=2)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2 + text_y_offset, text,
            ha="center", va="center", fontsize=fontsize, color=fontcolor,
            fontweight=weight, zorder=3)
    if sub_lines:
        for i, sl in enumerate(sub_lines):
            ax.text(x + w / 2, y + h / 2 - (i + 1) * 0.85 + text_y_offset, sl,
                    ha="center", va="center", fontsize=fontsize - 1.5,
                    color="#546E7A", zorder=3)


def draw_solid_arrow(ax, x1, y1, x2, y2, color=C_SOLID, lw=1.8, zorder=1):
    """实线箭头 - 主数据流"""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                               connectionstyle="arc3,rad=0"), zorder=zorder)


def draw_dashed_arrow(ax, x1, y1, x2, y2, color=C_DASHED, lw=1.5, zorder=1, rad=0):
    """虚线箭头 - 辅助/LLM交互"""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                               linestyle="dashed",
                               connectionstyle=f"arc3,rad={rad}"), zorder=zorder)


def draw_curved_solid(ax, x1, y1, x2, y2, color=C_SOLID, lw=1.8, rad=0.3, zorder=1):
    """曲线实线箭头"""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                               connectionstyle=f"arc3,rad={rad}"), zorder=zorder)


def draw_curved_dashed(ax, x1, y1, x2, y2, color=C_DASHED, lw=1.5, rad=0.3, zorder=1):
    """曲线虚线箭头"""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                               linestyle="dashed",
                               connectionstyle=f"arc3,rad={rad}"), zorder=zorder)


# ============================================================
# 标题
# ============================================================
ax.text(16, 21.3, "TKG-LPEHD: 基于LLM动态增强的时序知识图谱推理框架",
        ha="center", va="center", fontsize=18, fontweight="bold", color="#263238")

# ============================================================
# 区域1: 数据处理与加载 (左上)
# ============================================================
# 数据源标签
for i, ds in enumerate(["ICEWS14", "ICEWS18", "ICEWS0515", "YAGO", "MHAES"]):
    x = 0.6 + i * 1.5
    draw_box(ax, x, 18.2, 1.3, 0.7, ds, color=C_DATA, border=C_DATA_BORDER,
             fontsize=7.5, bold=True, corner_radius=0.08)

# 数据加载大框
draw_box(ax, 0.3, 14.0, 7.9, 3.8,
         "Step 0: 数据处理与加载",
         color="#F5F5F5", border="#78909C", fontsize=10, bold=True, text_y_offset=1.4,
         sub_lines=["CSV/TTL/RDF 多格式解析", "四元组 (s,r,o,t) 统一格式化",
                    "实体/关系/时间戳词典构建", "时序邻接图构建"],
         corner_radius=0.12)

# 数据划分
draw_box(ax, 0.5, 12.6, 2.1, 1.0, "训练集\n80%", color="#E8EAF6", border="#5C6BC0", fontsize=8)
draw_box(ax, 3.0, 12.6, 2.1, 1.0, "验证集\n10%", color="#E8EAF6", border="#5C6BC0", fontsize=8)
draw_box(ax, 5.5, 12.6, 2.1, 1.0, "测试集\n10%", color="#E8EAF6", border="#5C6BC0", fontsize=8)

# ============================================================
# Phase 1: LLM引导三维关系路径采样
# ============================================================
draw_box(ax, 9.5, 16.5, 8.0, 5.0,
         "Phase 1: LLM引导三维关系路径采样",
         color=C_PHASE, border="#1976D2", fontsize=10, bold=True, text_y_offset=2.0,
         sub_lines=["a) 采样规则头边           d) 时间加权采样 w(t)=exp(-λ|t-T|)",
                    "b) 三维候选评分            e) LLM逻辑验证",
                    "   S=w1·S1+w2·S2+w3·S3",
                    "c) 符号时序约束              → 输出: 候选规则集 + 路径"],
         corner_radius=0.1)

# ============================================================
# Phase 2: LLM辅助时序有效性评估
# ============================================================
draw_box(ax, 9.5, 11.0, 8.0, 4.9,
         "Phase 2: LLM辅助时序有效性评估",
         color=C_PHASE, border="#1976D2", fontsize=10, bold=True, text_y_offset=1.8,
         sub_lines=["Δti = t_target - t_final",
                    "LLM确定有效期 V",
                    "构建语义样本集 T",
                    "过滤有效时序路径",
                    "→ 输出: 带有效期的规则集"],
         corner_radius=0.1)

# ============================================================
# Phase 3: 规则优先级排序
# ============================================================
draw_box(ax, 9.5, 6.0, 8.0, 4.4,
         "Phase 3: 规则优先级排序",
         color=C_PHASE, border="#1976D2", fontsize=10, bold=True, text_y_offset=1.6,
         sub_lines=["cp = Hb / B  (置信度)",
                    "C = B_valid / B_total  (覆盖度)",
                    "P = w1·cp + w2·C  (优先级)",
                    "阈值过滤: cp ≥ 0.6",
                    "→ 输出: 按P降序排列的规则集"],
         corner_radius=0.1)

# ============================================================
# Phase 4: 自适应规则演化
# ============================================================
draw_box(ax, 9.5, 1.2, 8.0, 4.2,
         "Phase 4: 自适应规则演化",
         color=C_PHASE, border="#1976D2", fontsize=10, bold=True, text_y_offset=1.5,
         sub_lines=["淘汰: P < P_th ∧ cp < C_th",
                    "用当前数据验证规则有效性",
                    "保留高价值规则，更新权重",
                    "→ 输出: 最终演化规则集 Sr"],
         corner_radius=0.1)

# ============================================================
# Phase 5: LLM动态权重融合推理 (右大框)
# ============================================================
draw_box(ax, 19.0, 1.0, 12.5, 20.5,
         "Phase 5: LLM动态权重融合推理",
         color=C_PHASE5, border=C_PHASE5_BORDER, fontsize=11, bold=True,
         text_y_offset=9.5, corner_radius=0.12)

# 5a: 规则推理
draw_box(ax, 19.6, 16.5, 5.5, 4.2,
         "5a. 规则推理",
         color="#FFF3E0", border="#E65100", fontsize=9.5, bold=True, text_y_offset=1.5,
         sub_lines=["Sr'={ρ∈Sr | score(ρ)≥θ}",
                    "(es,rq,eo,tq)←∧(es,ri,eo,ti)",
                    "Score(ρ,eo)=Σ(cρ+exp(-λ(tr-to)))"],
         corner_radius=0.08)

# 5b: 图推理 GNN
draw_box(ax, 26.0, 16.5, 5.0, 4.2,
         "5b. 图推理 GNN",
         color="#E8EAF6", border="#4527A0", fontsize=9.5, bold=True, text_y_offset=1.5,
         sub_lines=["Score_graph(eo)=⟨fg(q),eo⟩",
                    "时间感知邻居聚合",
                    "TiRGN/RE-GCN架构"],
         corner_radius=0.08)

# 5c: LLM双向验证
draw_box(ax, 21.3, 10.0, 8.0, 4.8,
         "5c. LLM双向验证",
         color="#FCE4EC", border="#C62828", fontsize=9.5, bold=True, text_y_offset=1.8,
         sub_lines=["规则→图方向验证",
                    "图→规则方向验证",
                    "交叉调整候选得分",
                    "过滤不一致预测"],
         corner_radius=0.08)

# 5d: 动态权重融合
draw_box(ax, 21.3, 3.5, 8.0, 5.0,
         "5d. 动态权重融合",
         color="#E0F2F1", border="#00695C", fontsize=9.5, bold=True, text_y_offset=1.8,
         sub_lines=["α_dynamic = α_fixed·exp(-λ·dr) /",
                    "  [α_fixed·exp(-λ·dr)+(1-α_fixed)·exp(-λ·dg)]",
                    "Score_f = α_dynamic×Score(ρ,eo)",
                    "  + (1-α_dynamic)×Score_graph(eo)"],
         corner_radius=0.08)

# ============================================================
# LLM模块
# ============================================================
draw_box(ax, 0.5, 9.0, 4.5, 3.0,
         "LLM 语言模型层",
         color=C_LLM, border=C_LLM_BORDER, fontsize=10, bold=True, text_y_offset=0.8,
         sub_lines=["GPT-4 / 模拟模式",
                    "Sentence-BERT 语义编码"],
         corner_radius=0.08)

# ============================================================
# SPARQL模块
# ============================================================
draw_box(ax, 0.5, 5.5, 4.5, 2.8,
         "SPARQL 时序模式匹配",
         color="#E0E0E0", border="#424242", fontsize=9, bold=True, text_y_offset=0.8,
         sub_lines=["6种事件匹配模式",
                    "Neo4j → rdflib → SPARQL"],
         corner_radius=0.08)

# ============================================================
# 输出
# ============================================================
draw_box(ax, 19.6, 0.2, 11.3, 1.0,
         "推理预测结果: Top-K 候选实体排序  |  评估指标: MRR, Hit@1, Hit@3, Hit@10",
         color=C_OUTPUT, border=C_OUTPUT_BORDER, fontsize=9.5, bold=False,
         corner_radius=0.08)

# ============================================================
# 连线: 实线 (主数据流)
# ============================================================

# 数据模块 → Phase 1 (数据用于路径采样)
draw_solid_arrow(ax, 8.2, 16.0, 9.5, 18.5, C_SOLID, lw=2.0)

# Phase 1 → Phase 2 (路径+规则)
draw_solid_arrow(ax, 13.5, 16.5, 13.5, 15.9, C_SOLID, lw=2.0)

# Phase 2 → Phase 3 (带有效期的规则)
draw_solid_arrow(ax, 13.5, 11.0, 13.5, 10.4, C_SOLID, lw=2.0)

# Phase 3 → Phase 4 (优先级排序的规则)
draw_solid_arrow(ax, 13.5, 6.0, 13.5, 5.4, C_SOLID, lw=2.0)

# Data → Phase 3 (dataset用于计算cp/C)
draw_curved_solid(ax, 8.2, 14.5, 9.5, 8.5, "#546E7A", lw=1.5, rad=-0.4)

# Data → Phase 4 (test data用于演化)
draw_curved_solid(ax, 8.2, 13.0, 9.5, 3.5, "#546E7A", lw=1.5, rad=-0.5)

# Phase 4 → Phase 5a (演化规则集 Sr)
draw_curved_solid(ax, 17.5, 4.0, 19.6, 18.5, C_RULE_FLOW, lw=2.2, rad=0.5)

# Data → Phase 5b (训练/验证/测试数据)
draw_curved_solid(ax, 8.2, 13.0, 26.0, 19.0, C_GNN_FLOW, lw=2.0, rad=-0.6)

# 5a → 5c (规则推理候选得分)
draw_solid_arrow(ax, 22.3, 16.5, 22.3, 14.8, C_RULE_FLOW, lw=2.0)

# 5b → 5c (图推理候选得分)
draw_solid_arrow(ax, 28.5, 16.5, 28.5, 14.8, C_GNN_FLOW, lw=2.0)

# 5c → 5d (调整后得分)
draw_solid_arrow(ax, 25.3, 10.0, 25.3, 8.5, C_SOLID, lw=2.0)

# 5d → 输出
draw_solid_arrow(ax, 25.3, 3.5, 25.3, 1.2, C_SOLID, lw=2.0)

# ============================================================
# 连线: 虚线 (LLM交互 / 辅助连接)
# ============================================================

# LLM → Phase 1 (LLM逻辑验证)
draw_dashed_arrow(ax, 5.0, 10.5, 9.5, 19.5, C_DASHED, lw=1.5, rad=0.4)

# LLM → Phase 2 (LLM确定有效期)
draw_dashed_arrow(ax, 5.0, 10.5, 9.5, 14.0, C_DASHED, lw=1.5, rad=0.3)

# LLM → Phase 5c (LLM双向验证)
draw_curved_dashed(ax, 5.0, 10.5, 21.3, 13.0, C_DASHED, lw=1.5, rad=-0.5)

# SPARQL → Output (事件匹配结果)
draw_curved_dashed(ax, 5.0, 6.5, 19.6, 0.8, "#757575", lw=1.2, rad=0.7)

# ============================================================
# 图例
# ============================================================
legend_x, legend_y = 0.5, 2.5
ax.text(legend_x + 0.5, legend_y + 1.0, "图例", fontsize=11, fontweight="bold", color="#263238")

# 实线
ax.plot([legend_x, legend_x + 1.2], [legend_y + 0.5, legend_y + 0.5],
        color=C_SOLID, lw=2.0)
ax.annotate("", xy=(legend_x + 1.2, legend_y + 0.5), xytext=(legend_x + 1.0, legend_y + 0.5),
            arrowprops=dict(arrowstyle="->", color=C_SOLID, lw=2.0))
ax.text(legend_x + 1.6, legend_y + 0.5, "实线: 主数据流 / 前向传递",
        fontsize=9, color="#37474F", va="center")

# 虚线
ax.plot([legend_x, legend_x + 1.2], [legend_y - 0.2, legend_y - 0.2],
        color=C_DASHED, lw=1.5, linestyle="dashed")
ax.annotate("", xy=(legend_x + 1.2, legend_y - 0.2), xytext=(legend_x + 1.0, legend_y - 0.2),
            arrowprops=dict(arrowstyle="->", color=C_DASHED, lw=1.5, linestyle="dashed"))
ax.text(legend_x + 1.6, legend_y - 0.2, "虚线: LLM交互 / 辅助信息流",
        fontsize=9, color=C_DASHED, va="center")

# 规则流
ax.plot([legend_x + 7.0, legend_x + 8.2], [legend_y + 0.5, legend_y + 0.5],
        color=C_RULE_FLOW, lw=2.2)
ax.annotate("", xy=(legend_x + 8.2, legend_y + 0.5), xytext=(legend_x + 8.0, legend_y + 0.5),
            arrowprops=dict(arrowstyle="->", color=C_RULE_FLOW, lw=2.2))
ax.text(legend_x + 8.6, legend_y + 0.5, "规则推理流",
        fontsize=9, color=C_RULE_FLOW, va="center")

# 图推理流
ax.plot([legend_x + 13.0, legend_x + 14.2], [legend_y + 0.5, legend_y + 0.5],
        color=C_GNN_FLOW, lw=2.0)
ax.annotate("", xy=(legend_x + 14.2, legend_y + 0.5), xytext=(legend_x + 14.0, legend_y + 0.5),
            arrowprops=dict(arrowstyle="->", color=C_GNN_FLOW, lw=2.0))
ax.text(legend_x + 14.6, legend_y + 0.5, "图推理流",
        fontsize=9, color=C_GNN_FLOW, va="center")

# ============================================================
# 标注文本: 关键流程说明
# ============================================================
# 规则流标注
ax.text(18.0, 11.0, "规则管线:\nPhase1→2→3→4→5a",
        fontsize=7.5, color=C_RULE_FLOW, ha="center", style="italic",
        bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.8, edgecolor="none"))

# GNN流标注
ax.text(30.5, 14.0, "数据管线:\nData→GNN训练",
        fontsize=7.5, color=C_GNN_FLOW, ha="center", style="italic",
        bbox=dict(boxstyle="round", facecolor="#F3E5F5", alpha=0.8, edgecolor="none"))

# 融合标注
ax.text(30.5, 7.0, "双路汇聚:\n规则+GNN→验证→融合",
        fontsize=7.5, color="#00695C", ha="center", style="italic",
        bbox=dict(boxstyle="round", facecolor="#E0F2F1", alpha=0.8, edgecolor="none"))

# ============================================================
# 保存
# ============================================================
plt.tight_layout(pad=0.5)
out_svg = "D:/MyProjects/tkg_lpehd/output/architecture_diagram.svg"
out_png = "D:/MyProjects/tkg_lpehd/output/architecture_diagram.png"
plt.savefig(out_svg, dpi=150, bbox_inches="tight", facecolor="white")
plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
print(f"图已保存: {out_svg}")
print(f"图已保存: {out_png}")
plt.close()
