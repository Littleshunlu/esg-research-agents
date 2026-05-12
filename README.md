# ESG Research Agents 🔬

> 基于大语言模型的学术研究自动化辅助Agent集群 —— 面板数据实证研究全流程自动化

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 项目简介

ESG Research Agents 是一个面向面板数据实证研究的自动化Agent框架，专为《数字技术赋能上市公司ESG信息披露》类论文设计。通过三个核心Agent的协作，实现从文献解析到实证解读的端到端自动化。

### 核心痛点

实证研究存在大量重复劳动：文献梳理耗时、数据清洗繁琐、回归结果解读机械。本项目通过多Agent协作 + 长链推理，将这些环节自动化，将研究效率提升约60%。

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────┐
│                 Orchestrator                     │
│            （Agent编排调度器）                      │
│                                                  │
│  ┌──────────┐   ┌──────────────┐   ┌──────────┐ │
│  │Literature │──▶│DataCleaning  │──▶│Empirical │ │
│  │  Agent    │   │   Agent      │   │  Agent   │ │
│  │          │   │              │   │          │ │
│  │• 文献解析 │   │• 缺失值处理  │   │• 结果解读│ │
│  │• 方法提取 │   │• 滞后项生成  │   │• 模型对比│ │
│  │• 对比矩阵 │   │• 交互项构建  │   │• 机制检验│ │
│  └──────────┘   │• 缩尾处理    │   │• 学术表述│ │
│                  │• 描述统计    │   └──────────┘ │
│                  └──────────────┘                 │
└─────────────────────────────────────────────────┘
```

## 🤖 三大核心Agent

### 1. 文献智能解析Agent (LiteratureAgent)

| 功能 | 说明 |
|------|------|
| 研究框架提取 | 自动识别理论假说和分析逻辑 |
| 计量方法识别 | OLS/FE/IV/DID/GMM等方法自动标注 |
| 变量定义提取 | 被解释变量、核心解释变量、控制变量 |
| 文献对比矩阵 | 多篇文献方法差异横向对比 |

### 2. 数据清洗与变量构建Agent (DataCleaningAgent)

| 功能 | 说明 |
|------|------|
| 缺失值处理 | Listwise/Mean/Median/Interpolate |
| 滞后项生成 | 按个体分组的L1/L2等滞后项 |
| 交互项构建 | 中心化后相乘（减少共线性） |
| 对数变换 | 自动生成ln_前缀变量 |
| 缩尾处理 | 对称缩尾，自定义比例 |
| 描述性统计 | 含偏度、峰度的完整统计 |
| VIF检验 | 多重共线性诊断 |

### 3. 实证结果解读Agent (EmpiricalAgent)

| 功能 | 说明 |
|------|------|
| Stata输出解析 | 自动提取系数、标准误、显著性 |
| 回归结果解读 | 生成符合学术规范的中文表述 |
| 多模型对比 | 稳健性分析的文字总结 |
| 中介效应检验 | Sobel检验/Bootstrap结果解读 |
| 对比表导出 | 一键生成论文级回归结果表 |

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/NetizenPashunlu/esg-research-agents.git
cd esg-research-agents
pip install -r requirements.txt
```

### 配置

复制配置文件并填入API Key：

```bash
cp config/default.yaml config/local.yaml
# 编辑 config/local.yaml，填入你的API Key
```

支持的大模型：
- **DeepSeek**（推荐）：性价比高，中文理解强
- **混元 (Hunyuan)**：腾讯系模型
- **Mimo**：小米模型
- 其他OpenAI兼容API

### 使用方式

#### 方式1：Pipeline模式（全自动流水线）

```python
from core.orchestrator import Orchestrator

orchestrator = Orchestrator(config={"api_key": "your-key"})

# 执行完整流水线
result = await orchestrator.run_pipeline(
    data_path="./data/esg_panel.csv",
    entity_col="stock_code",
    time_col="year",
    dependent_var="ESG_disclosure",
    core_independent_var="digital_tech",
)
```

#### 方式2：Step模式（逐步控制）

```python
from agents.data_cleaning_agent import DataCleaningAgent, VariableSpec

agent = DataCleaningAgent()

# 加载数据
agent.load_data("./data/esg_panel.csv")
agent.set_panel_index("stock_code", "year")

# 定义变量
agent.add_variable_spec(VariableSpec(
    name="digital_tech",
    label="数字技术应用水平",
    var_type="independent",
    measurement="年报文本分析",
    source="上市公司年报",
    lag_periods=1  # 生成一阶滞后
))

# 执行
agent.generate_variables()
stats = agent.descriptive_stats()
```

#### 方式3：命令行

```bash
# 流水线模式
python main.py --mode pipeline --data ./data/sample.csv

# 单步模式
python main.py --mode step --agent data --data ./data/sample.csv
```

## 📊 示例数据

运行示例生成模拟ESG面板数据：

```python
from examples.demo import create_sample_data, demo_data_cleaning, demo_empirical_analysis

# 生成示例数据（50家上市公司 × 9年 = 450条观测）
df = create_sample_data("data/sample_esg_data.csv")

# 演示数据清洗
demo_data_cleaning()

# 演示实证解读
demo_empirical_analysis()
```

## 📁 项目结构

```
esg-research-agents/
├── agents/                     # 核心Agent模块
│   ├── literature_agent.py     # 文献解析Agent
│   ├── data_cleaning_agent.py  # 数据清洗Agent
│   └── empirical_agent.py      # 实证解读Agent
├── core/                       # 核心框架
│   ├── base.py                 # Agent基类
│   └── orchestrator.py         # 编排调度器
├── utils/                      # 工具模块
│   └── llm_client.py           # LLM统一调用接口
├── config/                     # 配置文件
│   └── default.yaml            # 默认配置
├── data/                       # 数据目录
├── examples/                   # 示例代码
│   └── demo.py                 # 完整流程演示
├── tests/                      # 单元测试
│   ├── test_literature_agent.py
│   ├── test_data_cleaning_agent.py
│   └── test_empirical_agent.py
├── main.py                     # 命令行入口
├── requirements.txt            # 依赖列表
└── README.md                   # 项目文档
```

## 🔬 研究背景

本项目应用于硕士论文《数字技术赋能上市公司ESG信息披露的实证研究》，研究框架如下：

- **研究问题**：数字技术应用是否以及如何影响上市公司ESG信息披露水平？
- **核心变量**：被解释变量（ESG披露水平）、核心解释变量（数字技术应用水平）
- **估计方法**：双向固定效应模型 + IV-2SLS工具变量法
- **稳健性检验**：替换变量测量、PSM-DID、改变样本区间
- **机制检验**：信息透明度中介效应、产权性质调节效应
- **数据来源**：CSMAR、Wind、华证ESG评级

## 🛠️ 技术栈

- **Python 3.10+**
- **pandas / numpy**：数据处理
- **statsmodels / linearmodels**：计量经济学
- **aiohttp**：异步API调用
- **DeepSeek / 混元 / Mimo**：大语言模型

## 📄 License

MIT License

## 👤 Author

**NetizenPashunlu** - 某理工大学计算机科学与技术学院

---

_如果这个项目对你有帮助，欢迎 ⭐ Star！_
