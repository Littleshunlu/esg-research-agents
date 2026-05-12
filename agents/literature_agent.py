"""
文献智能解析Agent
自动提取学术论文中的研究框架、计量方法和核心发现
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from core.base import BaseAgent, AgentResponse
from utils.llm_client import LLMClient


@dataclass
class LiteratureInfo:
    """文献信息结构"""
    title: str = ""
    authors: list = field(default_factory=list)
    year: int = 0
    journal: str = ""
    research_framework: str = ""
    methodology: str = ""
    variables: dict = field(default_factory=dict)
    key_findings: list = field(default_factory=list)
    data_source: str = ""
    sample_period: str = ""


class LiteratureAgent(BaseAgent):
    """
    文献智能解析Agent

    功能：
    1. 从学术论文文本中提取研究框架
    2. 识别计量经济学方法（OLS, FE, IV, DID, GMM等）
    3. 提取变量定义（被解释变量、核心解释变量、控制变量）
    4. 总结核心发现与贡献
    5. 构建文献对比矩阵
    """

    SYSTEM_PROMPT = """你是一位专业的学术文献分析专家，擅长实证研究方法论。
你的任务是从给定的学术论文文本中提取关键信息，包括：

1. **研究框架**：理论假说、研究问题、分析逻辑
2. **计量方法**：所使用的估计方法、模型设定、稳健性检验
3. **变量定义**：
   - 被解释变量（Y）：名称、测量方式、数据来源
   - 核心解释变量（X）：名称、测量方式、数据来源
   - 控制变量：名称列表及选择理由
   - 中介变量/调节变量（如有）
4. **核心发现**：主要回归结果、系数方向与显著性
5. **数据信息**：样本范围、数据来源、样本量

请以结构化JSON格式输出分析结果。"""

    EXTRACTION_PROMPT = """请分析以下学术论文文本，提取研究框架和计量方法信息：

---
{paper_text}
---

请按以下JSON结构输出：
{{
    "research_framework": "研究框架描述",
    "hypotheses": ["假说1", "假说2"],
    "methodology": {{
        "estimation": ["估计方法列表"],
        "model_specification": "模型设定描述",
        "robustness_checks": ["稳健性检验方法"]
    }},
    "variables": {{
        "dependent": {{"name": "", "measurement": "", "source": ""}},
        "core_independent": {{"name": "", "measurement": "", "source": ""}},
        "controls": [{{"name": "", "reason": ""}}],
        "mediator": null,
        "moderator": null
    }},
    "key_findings": ["核心发现列表"],
    "data_info": {{
        "sample_period": "",
        "data_source": "",
        "sample_size": ""
    }}
}}"""

    COMPARISON_PROMPT = """请对比以下多篇文献的研究方法差异，生成文献对比矩阵：

{literature_summaries}

请从以下维度进行对比：
1. 研究视角差异
2. 变量测量方式差异
3. 计量方法差异
4. 核心结论异同
5. 对本研究的启示

以表格形式输出对比结果。"""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(name="literature_agent", config=config)
        self.llm = LLMClient(config=self.config)
        self.parsed_papers: list[LiteratureInfo] = []

    async def parse_paper(self, paper_text: str) -> AgentResponse:
        """解析单篇学术论文"""
        prompt = self.EXTRACTION_PROMPT.format(paper_text=paper_text)
        result = await self.llm.chat(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=prompt
        )

        try:
            # 尝试解析JSON响应
            parsed = self._extract_json(result)
            info = LiteratureInfo(
                research_framework=parsed.get("research_framework", ""),
                methodology=json.dumps(parsed.get("methodology", {}), ensure_ascii=False),
                variables=parsed.get("variables", {}),
                key_findings=parsed.get("key_findings", []),
                data_source=parsed.get("data_info", {}).get("data_source", ""),
                sample_period=parsed.get("data_info", {}).get("sample_period", ""),
            )
            self.parsed_papers.append(info)

            return AgentResponse(
                success=True,
                data=parsed,
                message="文献解析完成"
            )
        except json.JSONDecodeError:
            return AgentResponse(
                success=False,
                data={"raw_response": result},
                message="文献解析结果格式异常，请检查输入"
            )

    async def compare_literatures(self) -> AgentResponse:
        """对比已解析的文献，生成对比矩阵"""
        if len(self.parsed_papers) < 2:
            return AgentResponse(
                success=False,
                data={},
                message="至少需要解析2篇文献才能进行对比"
            )

        summaries = []
        for i, paper in enumerate(self.parsed_papers, 1):
            summaries.append(f"文献{i}: 框架={paper.research_framework}, 方法={paper.methodology}")

        prompt = self.COMPARISON_PROMPT.format(
            literature_summaries="\n".join(summaries)
        )
        result = await self.llm.chat(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=prompt
        )

        return AgentResponse(
            success=True,
            data={"comparison": result},
            message="文献对比完成"
        )

    def _extract_json(self, text: str) -> dict:
        """从LLM响应中提取JSON"""
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            return json.loads(json_match.group())
        raise json.JSONDecodeError("No JSON found", text, 0)

    def export_parsed(self) -> list[dict]:
        """导出所有已解析文献的摘要"""
        return [
            {
                "framework": p.research_framework,
                "methodology": p.methodology,
                "variables": p.variables,
                "findings": p.key_findings,
            }
            for p in self.parsed_papers
        ]
