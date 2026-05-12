"""
示例：ESG实证研究完整流程
"""

import asyncio
import pandas as pd
import numpy as np

from agents.literature_agent import LiteratureAgent
from agents.data_cleaning_agent import DataCleaningAgent, VariableSpec
from agents.empirical_agent import EmpiricalAgent, RegressionResult
from core.orchestrator import Orchestrator


def create_sample_data(output_path: str = "data/sample_esg_data.csv"):
    """生成示例ESG面板数据"""
    np.random.seed(42)
    data = []

    for firm in range(1, 51):  # 50家上市公司
        for year in range(2015, 2024):  # 2015-2023年
            # 生成具有相关性的变量
            size = np.random.normal(22, 1.5)
            roa = np.random.normal(0.05, 0.08)
            lev = np.random.uniform(0.3, 0.7)
            digital_tech = np.random.uniform(0, 1)
            soe = np.random.choice([0, 1])
            board_size = np.random.randint(5, 15)
            dual = np.random.choice([0, 1])

            # ESG披露水平受数字技术正向影响
            esg = (
                30
                + 15 * digital_tech
                + 2 * size
                + 20 * roa
                - 10 * lev
                + 5 * soe
                + np.random.normal(0, 8)
            )
            esg = max(0, min(100, esg))  # 限制在0-100范围

            data.append({
                "stock_code": f"S{firm:04d}",
                "year": year,
                "ESG_disclosure": round(esg, 2),
                "digital_tech": round(digital_tech, 4),
                "Size": round(size, 4),
                "ROA": round(roa, 4),
                "Lev": round(lev, 4),
                "SOE": soe,
                "BoardSize": board_size,
                "Dual": dual,
                "Growth": round(np.random.normal(0.1, 0.3), 4),
                "CashFlow": round(np.random.normal(0.05, 0.04), 4),
                "ListAge": year - np.random.randint(2000, 2010),
                "Industry": f"Ind_{np.random.randint(1, 13):02d}",
            })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"示例数据已生成: {output_path}")
    print(f"数据形状: {df.shape}")
    print(f"\n前5行:\n{df.head()}")
    return df


def demo_data_cleaning():
    """演示数据清洗流程"""
    print("=" * 60)
    print("数据清洗Agent 演示")
    print("=" * 60)

    # 生成示例数据
    df = create_sample_data("data/sample_esg_data.csv")

    # 初始化Agent
    agent = DataCleaningAgent()

    # 加载数据
    load_res = agent.load_data("data/sample_esg_data.csv")
    print(f"\n数据加载: {load_res.message}")

    # 设置面板索引
    agent.set_panel_index("stock_code", "year")
    print("面板索引设置: stock_code + year")

    # 定义变量规格
    agent.add_variable_spec(VariableSpec(
        name="digital_tech", label="数字技术应用水平",
        var_type="independent", measurement="年报文本分析",
        source="上市公司年报", lag_periods=1
    ))
    agent.add_variable_spec(VariableSpec(
        name="Size", label="企业规模",
        var_type="control", measurement="总资产自然对数",
        source="CSMAR", log_transform=True
    ))
    agent.add_variable_spec(VariableSpec(
        name="digital_tech", label="数字技术应用水平",
        var_type="moderator", measurement="年报文本分析",
        source="上市公司年报", interaction_with="SOE"
    ))

    # 生成衍生变量
    gen_res = agent.generate_variables()
    print(f"\n变量构建: {gen_res.message}")
    print(f"生成列: {gen_res.data['generated_columns']}")

    # 描述性统计
    stats = agent.descriptive_stats()
    print(f"\n描述性统计: {stats.message}")

    # 清洗报告
    print(f"\n清洗日志:\n{agent.get_cleaning_report()}")

    # 导出
    agent.export_data("data/cleaned_esg_data.csv")
    print("\n清洗后数据已导出: data/cleaned_esg_data.csv")


def demo_empirical_analysis():
    """演示实证分析流程"""
    print("\n" + "=" * 60)
    print("实证解读Agent 演示")
    print("=" * 60)

    agent = EmpiricalAgent()

    # 模拟添加回归结果
    model1 = RegressionResult(
        model_name="FE_Baseline",
        dependent_var="ESG_disclosure",
        coefficients={
            "digital_tech": {"coef": 0.325, "se": 0.089, "t": 3.652, "p": 0.001, "sig": "***"},
            "Size": {"coef": 0.156, "se": 0.045, "t": 3.467, "p": 0.001, "sig": "***"},
            "ROA": {"coef": 0.234, "se": 0.112, "t": 2.089, "p": 0.037, "sig": "**"},
            "Lev": {"coef": -0.089, "se": 0.067, "t": -1.328, "p": 0.184, "sig": ""},
            "SOE": {"coef": 0.078, "se": 0.056, "t": 1.393, "p": 0.164, "sig": ""},
            "_cons": {"coef": 2.456, "se": 0.789, "t": 3.113, "p": 0.002, "sig": "***"},
        },
        r_squared=0.423, adj_r_squared=0.418,
        f_stat=45.67, f_pvalue=0.000,
        n_obs=15234, method="双向固定效应"
    )

    model2 = RegressionResult(
        model_name="IV_2SLS",
        dependent_var="ESG_disclosure",
        coefficients={
            "digital_tech": {"coef": 0.289, "se": 0.102, "t": 2.833, "p": 0.005, "sig": "***"},
            "Size": {"coef": 0.145, "se": 0.048, "t": 3.021, "p": 0.003, "sig": "***"},
            "ROA": {"coef": 0.198, "se": 0.118, "t": 1.678, "p": 0.093, "sig": "*"},
            "Lev": {"coef": -0.076, "se": 0.071, "t": -1.071, "p": 0.284, "sig": ""},
            "SOE": {"coef": 0.067, "se": 0.059, "t": 1.136, "p": 0.256, "sig": ""},
            "_cons": {"coef": 2.312, "se": 0.823, "t": 2.809, "p": 0.005, "sig": "***"},
        },
        r_squared=0.398, adj_r_squared=0.393,
        f_stat=38.92, f_pvalue=0.000,
        n_obs=15234, method="IV-2SLS"
    )

    agent.add_result(model1)
    agent.add_result(model2)

    # 导出对比表
    table = agent.export_results_table()
    print("\n回归结果对比表:")
    print(table.to_string(index=False))

    print(f"\n已录入 {len(agent.results)} 个回归结果")
    print("调用 interpret_result() 或 compare_models() 可获取AI解读")


if __name__ == "__main__":
    demo_data_cleaning()
    demo_empirical_analysis()
