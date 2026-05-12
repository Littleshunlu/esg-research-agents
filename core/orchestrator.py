"""
Agent编排器 - 多Agent协作调度
实现从文献解析→数据清洗→实证解读的端到端流水线
"""

import json
from typing import Optional

from core.base import BaseAgent, AgentResponse
from agents.literature_agent import LiteratureAgent
from agents.data_cleaning_agent import DataCleaningAgent
from agents.empirical_agent import EmpiricalAgent
from utils.llm_client import LLMClient


class Orchestrator(BaseAgent):
    """
    学术研究Agent编排器

    协调三个核心Agent的工作流程：
    1. LiteratureAgent → 文献解析与对比
    2. DataCleaningAgent → 数据清洗与变量构建
    3. EmpiricalAgent → 实证结果解读

    支持两种模式：
    - Pipeline模式：串行执行完整流程
    - Step模式：单步执行，手动控制流程
    """

    ORCHESTRATION_PROMPT = """你是一位学术研究流程编排专家。
根据当前的研究进展，决定下一步应该执行哪个Agent的操作。

当前状态：
- 文献解析: {lit_status}
- 数据清洗: {data_status}
- 实证解读: {emp_status}

用户请求: {user_request}

请判断下一步操作，输出JSON:
{{"next_agent": "literature|data|empirical", "action": "具体操作", "reason": "选择理由"}}"""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(name="orchestrator", config=config)

        # 初始化三个核心Agent
        self.literature_agent = LiteratureAgent(config=config)
        self.data_agent = DataCleaningAgent(config=config)
        self.empirical_agent = EmpiricalAgent(config=config)

        # 编排状态追踪
        self.pipeline_status = {
            "literature": "pending",   # pending → running → done
            "data_cleaning": "pending",
            "empirical": "pending",
        }

        self.llm = LLMClient(config=self.config)

    async def run_pipeline(
        self,
        paper_texts: Optional[list[str]] = None,
        data_path: Optional[str] = None,
        entity_col: str = "stock_code",
        time_col: str = "year",
        dependent_var: str = "ESG_disclosure",
        core_independent_var: str = "digital_tech",
    ) -> AgentResponse:
        """
        执行完整的端到端流水线

        流程：文献解析 → 数据清洗 → 实证解读
        """
        results = {}

        # 阶段1：文献解析
        if paper_texts:
            self.pipeline_status["literature"] = "running"
            lit_results = []
            for text in paper_texts:
                res = await self.literature_agent.parse_paper(text)
                lit_results.append(res.data)

            if len(paper_texts) >= 2:
                compare_res = await self.literature_agent.compare_literatures()
                results["literature_comparison"] = compare_res.data

            self.pipeline_status["literature"] = "done"
            results["literature_parsed"] = lit_results

        # 阶段2：数据清洗
        if data_path:
            self.pipeline_status["data_cleaning"] = "running"
            load_res = self.data_agent.load_data(data_path)
            if load_res.success:
                self.data_agent.set_panel_index(entity_col, time_col)
                gen_res = self.data_agent.generate_variables()
                stats_res = self.data_agent.descriptive_stats()
                corr_res = self.data_agent.correlation_matrix()

                self.pipeline_status["data_cleaning"] = "done"
                results["data_cleaning"] = {
                    "load": load_res.data,
                    "generated": gen_res.data,
                    "descriptive_stats": stats_res.data,
                    "correlation": corr_res.data,
                }
            else:
                self.pipeline_status["data_cleaning"] = "error"
                results["data_cleaning"] = {"error": load_res.message}

        # 阶段3：实证解读
        if self.empirical_agent.results:
            self.pipeline_status["empirical"] = "running"
            interp_res = await self.empirical_agent.interpret_result()
            if len(self.empirical_agent.results) >= 2:
                comp_res = await self.empirical_agent.compare_models()
                results["model_comparison"] = comp_res.data

            self.pipeline_status["empirical"] = "done"
            results["empirical_interpretation"] = interp_res.data

        return AgentResponse(
            success=True,
            data=results,
            message="流水线执行完成"
        )

    async def smart_dispatch(self, user_request: str) -> AgentResponse:
        """
        智能调度：根据用户请求自动路由到合适的Agent
        """
        prompt = self.ORCHESTRATION_PROMPT.format(
            lit_status=self.pipeline_status["literature"],
            data_status=self.pipeline_status["data_cleaning"],
            emp_status=self.pipeline_status["empirical"],
            user_request=user_request
        )

        decision = await self.llm.chat(
            system_prompt="你是一个路由决策器，只输出JSON。",
            user_prompt=prompt
        )

        try:
            parsed = json.loads(decision)
            next_agent = parsed.get("next_agent", "")

            if next_agent == "literature":
                return AgentResponse(
                    success=True,
                    data={"routed_to": "literature_agent", "action": parsed["action"]},
                    message=parsed["reason"]
                )
            elif next_agent == "data":
                return AgentResponse(
                    success=True,
                    data={"routed_to": "data_cleaning_agent", "action": parsed["action"]},
                    message=parsed["reason"]
                )
            elif next_agent == "empirical":
                return AgentResponse(
                    success=True,
                    data={"routed_to": "empirical_agent", "action": parsed["action"]},
                    message=parsed["reason"]
                )
        except json.JSONDecodeError:
            pass

        return AgentResponse(
            success=False,
            data={},
            message="无法确定路由，请明确指定操作"
        )

    def get_pipeline_status(self) -> dict:
        """获取流水线状态"""
        return {
            "pipeline": self.pipeline_status,
            "literature": self.literature_agent.info(),
            "data_cleaning": self.data_agent.info(),
            "empirical": self.empirical_agent.info(),
        }

    def generate_report(self) -> str:
        """生成完整研究流程报告"""
        report_parts = [
            "=" * 60,
            "ESG实证研究Agent集群 - 执行报告",
            "=" * 60,
            "",
            "【流程状态】",
        ]

        for stage, status in self.pipeline_status.items():
            status_icon = {"pending": "⏳", "running": "🔄", "done": "✅", "error": "❌"}.get(status, "❓")
            report_parts.append(f"  {status_icon} {stage}: {status}")

        if self.data_agent.df is not None:
            report_parts.extend([
                "",
                "【数据清洗日志】",
                self.data_agent.get_cleaning_report(),
            ])

        if self.empirical_agent.results:
            table = self.empirical_agent.export_results_table()
            report_parts.extend([
                "",
                "【回归结果对比表】",
                table.to_string(index=False),
            ])

        report_parts.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(report_parts)
