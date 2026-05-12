"""
实证结果解读Agent
对比不同模型的回归系数差异并生成学术表述建议
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from core.base import BaseAgent, AgentResponse
from utils.llm_client import LLMClient


@dataclass
class RegressionResult:
    """回归结果结构"""
    model_name: str
    dependent_var: str
    coefficients: dict  # {var_name: {"coef": float, "se": float, "t": float, "p": float, "sig": str}}
    r_squared: float
    adj_r_squared: float
    f_stat: float
    f_pvalue: float
    n_obs: int
    method: str  # OLS, FE, RE, IV, GMM etc.


class EmpiricalAgent(BaseAgent):
    """
    实证结果解读Agent

    功能：
    1. 解析回归结果（Stata/Python输出）
    2. 多模型系数对比
    3. 显著性标注与解读
    4. 生成学术规范表述
    5. 稳健性检验对比
    6. 中介效应/调节效应分析报告
    """

    SYSTEM_PROMPT = """你是一位专业的计量经济学实证分析专家，熟悉面板数据实证研究的规范表述。
你的任务是根据回归结果生成符合学术规范的文字表述，包括：

1. **基准回归解读**：系数方向、显著性、经济学含义
2. **稳健性检验**：不同方法下结论的一致性
3. **异质性分析**：分组回归差异的经济解释
4. **机制检验**：中介效应的Sobel检验/Bootstrap结果解读

表述要求：
- 遵循中文学术论文写作规范
- 系数解读需结合经济学含义
- 显著性使用 * p<0.1, ** p<0.05, *** p<0.01
- 避免因果推断的绝对表述（面板数据只能说"影响"而非"导致"）"""

    INTERPRETATION_PROMPT = """请根据以下回归结果，生成符合学术规范的解读文字：

模型信息：
- 估计方法：{method}
- 被解释变量：{dep_var}
- 样本量：{n_obs}
- R²：{r_squared}

回归系数：
{coefficients_text}

请生成：
1. 基准回归结果解读（3-5句话）
2. 核心变量系数的经济学含义解释
3. 模型拟合度评价
4. 与理论预期的对比"""

    COMPARISON_PROMPT = """请对比以下多个模型的回归结果，分析结论的稳健性：

{models_text}

请从以下角度分析：
1. 核心变量系数在不同模型中的方向和显著性是否一致
2. 系数值的变化趋势及可能原因
3. 控制变量加入后核心变量的变化
4. 整体结论的稳健性判断"""

    MECHANISM_PROMPT = """请根据以下中介效应检验结果，生成机制分析文字：

第一步（总效应）: {step1_coef} ({step1_sig})
第二步（加入中介变量后核心变量系数）: {step2_coef} ({step2_sig})
中介变量系数: {med_coef} ({med_sig})
Sobel Z值: {sobel_z} ({sobel_sig})

请生成：
1. 中介效应是否成立的判断
2. 完全中介还是部分中介
3. 中介效应占总效应的比例
4. 经济学含义解释"""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(name="empirical_agent", config=config)
        self.llm = LLMClient(config=self.config)
        self.results: list[RegressionResult] = []

    def parse_stata_output(self, stata_text: str) -> AgentResponse:
        """解析Stata回归输出"""
        result = self._extract_stata_results(stata_text)
        if result:
            self.results.append(result)
            return AgentResponse(
                success=True,
                data=self._result_to_dict(result),
                message=f"成功解析Stata输出: {result.model_name}"
            )
        return AgentResponse(
            success=False,
            data={},
            message="无法解析Stata输出，请检查格式"
        )

    def add_result(self, result: RegressionResult) -> AgentResponse:
        """手动添加回归结果"""
        self.results.append(result)
        return AgentResponse(
            success=True,
            data=self._result_to_dict(result),
            message=f"已添加回归结果: {result.model_name}"
        )

    async def interpret_result(self, model_index: int = -1) -> AgentResponse:
        """解读单个回归结果"""
        if not self.results:
            return AgentResponse(success=False, data={}, message="暂无回归结果")

        result = self.results[model_index]
        coef_text = self._format_coefficients(result)

        prompt = self.INTERPRETATION_PROMPT.format(
            method=result.method,
            dep_var=result.dependent_var,
            n_obs=result.n_obs,
            r_squared=f"{result.r_squared:.4f}",
            coefficients_text=coef_text
        )

        interpretation = await self.llm.chat(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=prompt
        )

        return AgentResponse(
            success=True,
            data={"interpretation": interpretation, "model": result.model_name},
            message="回归结果解读完成"
        )

    async def compare_models(self) -> AgentResponse:
        """对比多个回归模型"""
        if len(self.results) < 2:
            return AgentResponse(
                success=False,
                data={},
                message="至少需要2个模型才能进行对比"
            )

        models_text = ""
        for i, r in enumerate(self.results, 1):
            models_text += f"\n模型{i}（{r.model_name}）:\n"
            models_text += self._format_coefficients(r)
            models_text += f"\nR²={r.r_squared:.4f}, N={r.n_obs}\n"

        prompt = self.COMPARISON_PROMPT.format(models_text=models_text)
        comparison = await self.llm.chat(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=prompt
        )

        return AgentResponse(
            success=True,
            data={"comparison": comparison},
            message="模型对比完成"
        )

    async def analyze_mechanism(
        self,
        total_effect_coef: float, total_effect_sig: str,
        direct_effect_coef: float, direct_effect_sig: str,
        mediator_coef: float, mediator_sig: str,
        sobel_z: float, sobel_sig: str
    ) -> AgentResponse:
        """中介效应分析"""
        prompt = self.MECHANISM_PROMPT.format(
            step1_coef=total_effect_coef, step1_sig=total_effect_sig,
            step2_coef=direct_effect_coef, step2_sig=direct_effect_sig,
            med_coef=mediator_coef, med_sig=mediator_sig,
            sobel_z=sobel_z, sobel_sig=sobel_sig
        )

        analysis = await self.llm.chat(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=prompt
        )

        return AgentResponse(
            success=True,
            data={"mechanism_analysis": analysis},
            message="中介效应分析完成"
        )

    def _format_coefficients(self, result: RegressionResult) -> str:
        """格式化回归系数为文本"""
        lines = []
        for var, info in result.coefficients.items():
            sig = info.get('sig', '')
            lines.append(
                f"  {var}: 系数={info['coef']:.4f}{sig}, "
                f"标准误={info['se']:.4f}, t值={info['t']:.3f}"
            )
        return "\n".join(lines)

    def _extract_stata_results(self, text: str) -> Optional[RegressionResult]:
        """从Stata输出文本中提取回归结果"""
        try:
            # 提取系数表
            coef_pattern = r'(\w+)\s*\|\s*([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)'
            coefficients = {}
            for match in re.finditer(coef_pattern, text):
                var = match.group(1)
                coef = float(match.group(2))
                se = float(match.group(3))
                t = float(match.group(4))
                p = float(match.group(5))
                sig = ""
                if p < 0.01: sig = "***"
                elif p < 0.05: sig = "**"
                elif p < 0.1: sig = "*"
                coefficients[var] = {"coef": coef, "se": se, "t": t, "p": p, "sig": sig}

            # 提取R²
            r2_match = re.search(r'R-squared\s*=\s*([-]?\d+\.\d+)', text)
            r2 = float(r2_match.group(1)) if r2_match else 0.0

            # 提取样本量
            n_match = re.search(r'Number of obs\s*=\s*(\d+)', text)
            n_obs = int(n_match.group(1)) if n_match else 0

            return RegressionResult(
                model_name="Stata_Model",
                dependent_var="",
                coefficients=coefficients,
                r_squared=r2,
                adj_r_squared=0.0,
                f_stat=0.0,
                f_pvalue=0.0,
                n_obs=n_obs,
                method="Unknown"
            )
        except Exception:
            return None

    def _result_to_dict(self, result: RegressionResult) -> dict:
        return {
            "model_name": result.model_name,
            "dependent_var": result.dependent_var,
            "coefficients": result.coefficients,
            "r_squared": result.r_squared,
            "n_obs": result.n_obs,
            "method": result.method,
        }

    def export_results_table(self) -> pd.DataFrame:
        """导出回归结果对比表"""
        rows = []
        all_vars = set()
        for r in self.results:
            all_vars.update(r.coefficients.keys())

        for var in sorted(all_vars):
            row = {"变量": var}
            for r in self.results:
                if var in r.coefficients:
                    info = r.coefficients[var]
                    row[r.model_name] = f"{info['coef']:.4f}{info['sig']}"
                else:
                    row[r.model_name] = ""
            rows.append(row)

        # 添加模型统计量
        for stat_name, stat_key in [("N", "n_obs"), ("R²", "r_squared")]:
            row = {"变量": stat_name}
            for r in self.results:
                val = getattr(r, stat_key)
                row[r.model_name] = f"{val:.4f}" if isinstance(val, float) else str(val)
            rows.append(row)

        return pd.DataFrame(rows)
