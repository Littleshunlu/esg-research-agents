"""
ESG Research Agents - 主入口
学术研究自动化辅助Agent集群

使用方法：
    python main.py --mode pipeline --data ./data/sample.csv
    python main.py --mode step --agent literature
"""

import asyncio
import argparse
import json
import sys

from core.orchestrator import Orchestrator
from agents.literature_agent import LiteratureAgent
from agents.data_cleaning_agent import DataCleaningAgent, VariableSpec
from agents.empirical_agent import EmpiricalAgent, RegressionResult


async def run_pipeline(args):
    """运行完整流水线"""
    orchestrator = Orchestrator(config=_load_config(args.config))

    print("=" * 60)
    print("ESG Research Agents - 流水线模式")
    print("=" * 60)

    # 阶段1：文献解析
    if args.papers:
        print("\n📖 [1/3] 文献解析中...")
        lit_agent = orchestrator.literature_agent
        with open(args.papers, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        for paper in papers:
            res = await lit_agent.parse_paper(paper.get("text", ""))
            if res.success:
                print(f"  ✅ 解析成功: {paper.get('title', 'Unknown')}")
            else:
                print(f"  ❌ 解析失败: {res.message}")

    # 阶段2：数据清洗
    if args.data:
        print("\n🔧 [2/3] 数据清洗中...")
        data_agent = orchestrator.data_agent
        load_res = data_agent.load_data(args.data)
        if load_res.success:
            print(f"  ✅ 数据加载成功: {load_res.data['shape']}")
            data_agent.set_panel_index(
                args.entity_col or "stock_code",
                args.time_col or "year"
            )
            gen_res = data_agent.generate_variables()
            if gen_res.success:
                print(f"  ✅ 变量构建完成: {gen_res.data['generated_columns']}")

            # 描述性统计
            stats = data_agent.descriptive_stats()
            if stats.success:
                print("  ✅ 描述性统计生成完成")

            print(f"\n  清洗日志:\n{data_agent.get_cleaning_report()}")
        else:
            print(f"  ❌ 数据加载失败: {load_res.message}")

    # 阶段3：实证解读
    if args.empirical:
        print("\n📊 [3/3] 实证解读中...")
        print("  请通过API或代码添加回归结果后调用解读功能")

    # 生成报告
    print("\n" + orchestrator.generate_report())


async def run_step(args):
    """单步执行模式"""
    config = _load_config(args.config)
    agent_name = args.agent

    if agent_name == "literature":
        agent = LiteratureAgent(config=config)
        if args.input:
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
            res = await agent.parse_paper(text)
            print(json.dumps(res.data, ensure_ascii=False, indent=2))

    elif agent_name == "data":
        agent = DataCleaningAgent(config=config)
        if args.data:
            res = agent.load_data(args.data)
            print(json.dumps(res.data, ensure_ascii=False, indent=2))

    elif agent_name == "empirical":
        agent = EmpiricalAgent(config=config)
        if args.input:
            with open(args.input, 'r', encoding='utf-8') as f:
                stata_output = f.read()
            res = agent.parse_stata_output(stata_output)
            if res.success:
                interp = await agent.interpret_result()
                print(interp.data.get("interpretation", ""))
            else:
                print(f"解析失败: {res.message}")


def _load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="ESG Research Agents - 学术研究自动化辅助Agent集群"
    )
    parser.add_argument(
        "--mode", choices=["pipeline", "step"],
        default="pipeline",
        help="运行模式：pipeline(流水线) 或 step(单步)"
    )
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--data", type=str, help="数据文件路径")
    parser.add_argument("--papers", type=str, help="文献JSON文件路径")
    parser.add_argument("--empirical", type=str, help="实证结果文件路径")
    parser.add_argument("--input", type=str, help="单步模式输入文件路径")
    parser.add_argument(
        "--agent", choices=["literature", "data", "empirical"],
        help="单步模式指定Agent"
    )
    parser.add_argument("--entity-col", type=str, help="个体标识列名")
    parser.add_argument("--time-col", type=str, help="时间标识列名")

    args = parser.parse_args()

    if args.mode == "pipeline":
        asyncio.run(run_pipeline(args))
    else:
        asyncio.run(run_step(args))


if __name__ == "__main__":
    main()
