"""
TKG-LPEHD 全局配置模块
======================
本模块集中管理所有超参数、路径配置和阈值设定
敏感性分析中的最优参数。
"""

import os
import torch

# ===========================
# 项目根路径
# ===========================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data_files")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

# ===========================
# 数据集配置
# ===========================
# ICEWS14, ICEWS0515, ICEWS18, YAGO
DATASET_NAME = "ICEWS14"  # 默认使用ICEWS14

DATASET_PATHS = {
    "ICEWS14": os.path.join(DATA_DIR, "icews14"),
    "ICEWS0515": os.path.join(DATA_DIR, "icews0515"),
    "ICEWS18": os.path.join(DATA_DIR, "icews18"),
    "YAGO": os.path.join(DATA_DIR, "yago"),
    "MHAES": os.path.join(DATA_DIR, "mhaes"),
}

# 各数据集对应的案例研究查询 (subject, relation, object, timestamp)
CASE_STUDY_QUERIES = {
    "ICEWS14": ("Barack_Obama", "Make_a_visit", "South_Korea", "2014-12-09"),
    "ICEWS18": ("Barack_Obama", "Make_a_visit", "China", "2018-06-15"),
    "MHAES": ("患者_0006", "确诊", "新型冠状病毒", "2022-12-20"),
}

# ===========================
# 时间衰减系数
# ===========================
# λ = 0.1 为最优值, 平衡时间敏感性与信息保留
LAMBDA_DECAY = 0.1  # 公式(6)中的λ, 公式(12)中的λ, 公式(14)中的λ

# ===========================
# 三维路径采样权重
# ===========================
# w1(语义相关性) > w2(出现频率) > w3(时间相关性)
# w1 + w2 + w3 = 1.0
W_SEMANTIC = 0.5    # w1: 语义相关性权重
W_FREQUENCY = 0.3   # w2: 出现频率权重
W_TEMPORAL = 0.2    # w3: 时间相关性权重

# ===========================
# 规则置信度与优先级
# ===========================
# 置信度阈值: cp ≥ 0.6
CONFIDENCE_THRESHOLD = 0.6  # 公式(8)后定义的阈值

# 优先级评分权重: w1*cp + w2*C
PRIORITY_W_CONFIDENCE = 0.7  # 置信度权重
PRIORITY_W_COVERAGE = 0.3    # 有效性覆盖权重

# ===========================
# 自适应规则演化
# ===========================
# 优先级阈值 P_th = 0.6 (验证的最优值)
PRIORITY_THRESHOLD = 0.6
# 置信度阈值
CONFIDENCE_EVOLUTION_THRESHOLD = 0.6

# ===========================
# 动态权重融合
# ===========================
# α_fixed: 规则推理的基础权重系数 (公式14)
ALPHA_FIXED = 0.5

# ===========================
# 图神经网络参数
# ===========================
EMBEDDING_DIM = 128      # 实体/关系嵌入维度
GNN_LAYERS = 2           # GNN层数
GNN_DROPOUT = 0.2        # Dropout率
GNN_LR = 0.001           # 学习率
GNN_EPOCHS = 20          # 训练轮数 (20轮已足够, 大数据集可用 --skip-training)
GNN_BATCH_SIZE = 256
RETRAIN_MODEL = False     # True=重新训练, False=加载缓存模型
GNN_MODEL_DIR = os.path.join(ROOT_DIR, "models")  # GNN 模型保存目录

# ===========================
# 路径采样参数
# ===========================
MAX_PATH_LENGTH = 3      # 最大路径长度 (规则长度l, 路径长度l+1)
NUM_NEGATIVE_SAMPLES = 10
TOP_K_CANDIDATES = 100

# ===========================
# LLM配置
# ===========================
# ====================
# 消融实验控制
# ====================
# None = 全模型
# "w/o G"  = 移除 LLM 引导模块 (Phase 1 路径验证)
# "w/o T"  = 移除时序有效性模块 (Phase 2 LLM 有效期)
# "w/o E"  = 移除候选实体 Prompt 模块 (Phase 5 LLM 双向验证)
ABLATION_MODE = None


def is_real_llm_enabled() -> bool:
    """运行时检查是否启用真实 LLM (环境变量 TKG_SIMULATE=1 可强制模拟)。"""
    import os
    if os.environ.get("TKG_SIMULATE", "") == "1":
        return False
    if os.environ.get("TKG_REAL_LLM", "") == "1":
        return True
    return not USE_SIMULATED_LLM


def get_ablation_mode() -> str:
    """运行时获取消融模式 (环境变量 TKG_ABLATION 优先)。"""
    import os
    env = os.environ.get("TKG_ABLATION", "")
    if env:
        return env
    return ABLATION_MODE or ""


# ====================
# Prompt 风格切换
# ====================
# "improved" = 论文 B.1/B.2 改进版 (含 TKG 上下文 + 向量相似度表)
# "original" = 论文 Appendix B 原始版 (简单 prompt)
LLM_PROMPT_STYLE = "improved"


def get_prompt_style() -> str:
    """运行时获取 prompt 风格 (环境变量 TKG_PROMPT_STYLE 优先)。"""
    import os
    env = os.environ.get("TKG_PROMPT_STYLE", "")
    if env:
        return env
    return LLM_PROMPT_STYLE

# ====================
# LLM Provider 切换
# ====================
# 可选: deepseek / gpt35_turbo / llama2_7b / prompt_paraphrasing / deterministic
# 环境变量 TKG_LLM_PROVIDER 可运行时覆盖
LLM_PROVIDER = "deepseek"

# DeepSeek API (OpenAI 兼容接口)
LLM_MODEL_NAME = "deepseek-chat"
LLM_API_KEY = "deepseek_APIkey"
LLM_API_BASE = "https://api.deepseek.com/v1"
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.1
LLM_TIMEOUT = 30
USE_SIMULATED_LLM = False
SBERT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # Sentence-BERT

# ===========================
# 评估指标
# ===========================
EVAL_METRICS = ["MRR", "Hit@1", "Hit@3", "Hit@10"]

# ===========================
# 训练/测试时间划分
# ===========================
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ===========================
# 设备选择
# ===========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# 路径与日志
# ===========================
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================
# SPARQL端点配置 (时序事件匹配)
# ===========================
SPARQL_ENDPOINT = None  # 本地RDF图查询
SPARQL_PREFIX = """
PREFIX tkg: <http://tkg.lpehd.org/ontology/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""


def print_config():
    """打印当前配置摘要"""
    print("=" * 60)
    print("TKG-LPEHD 配置参数")
    print("=" * 60)
    print(f"  数据集: {DATASET_NAME}")
    print(f"  时间衰减系数 λ: {LAMBDA_DECAY}")
    print(f"  三维采样权重: w1={W_SEMANTIC}, w2={W_FREQUENCY}, w3={W_TEMPORAL}")
    print(f"  置信度阈值: {CONFIDENCE_THRESHOLD}")
    print(f"  优先级阈值 P_th: {PRIORITY_THRESHOLD}")
    print(f"  动态融合 α_fixed: {ALPHA_FIXED}")
    print(f"  嵌入维度: {EMBEDDING_DIM}")
    print(f"  GNN层数: {GNN_LAYERS}, 学习率: {GNN_LR}")
    print(f"  设备: {DEVICE}")
    print(f"  LLM模拟模式: {USE_SIMULATED_LLM}")
    print("=" * 60)
