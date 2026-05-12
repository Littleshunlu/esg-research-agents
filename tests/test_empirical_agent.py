"""
单元测试 - 实证解读Agent
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from agents.empirical_agent import EmpiricalAgent, RegressionResult
from core.base import AgentResponse


@pytest.fixture
def agent():
    return EmpiricalAgent(config={"api_key": "test-key"})


@pytest.fixture
def sample_result():
    return RegressionResult(
        model_name="Model_1_FE",
        dependent_var="ESG_disclosure",
        coefficients={
            "digital_tech": {"coef": 0.325, "se": 0.089, "t": 3.652, "p": 0.001, "sig": "***"},
            "Size": {"coef": 0.156, "se": 0.045, "t": 3.467, "p": 0.001, "sig": "***"},
            "ROA": {"coef": 0.234, "se": 0.112, "t": 2.089, "p": 0.037, "sig": "**"},
            "Lev": {"coef": -0.089, "se": 0.067, "t": -1.328, "p": 0.184, "sig": ""},
            "_cons": {"coef": 2.456, "se": 0.789, "t": 3.113, "p": 0.002, "sig": "***"},
        },
        r_squared=0.423,
        adj_r_squared=0.418,
        f_stat=45.67,
        f_pvalue=0.000,
        n_obs=15234,
        method="双向固定效应"
    )


class TestEmpiricalAgent:
    def test_init(self, agent):
        assert agent.name == "empirical_agent"
        assert agent.results == []

    def test_add_result(self, agent, sample_result):
        result = agent.add_result(sample_result)
        assert result.success is True
        assert len(agent.results) == 1

    @pytest.mark.asyncio
    async def test_interpret_result(self, agent, sample_result):
        agent.add_result(sample_result)
        agent.llm = MagicMock()
        agent.llm.chat = AsyncMock(return_value="数字技术对ESG披露具有显著正向影响...")

        result = await agent.interpret_result()
        assert result.success is True
        assert "interpretation" in result.data

    @pytest.mark.asyncio
    async def test_compare_models(self, agent, sample_result):
        agent.add_result(sample_result)
        result2 = RegressionResult(
            model_name="Model_2_IV",
            dependent_var="ESG_disclosure",
            coefficients={
                "digital_tech": {"coef": 0.289, "se": 0.102, "t": 2.833, "p": 0.005, "sig": "***"},
                "Size": {"coef": 0.145, "se": 0.048, "t": 3.021, "p": 0.003, "sig": "***"},
                "_cons": {"coef": 2.312, "se": 0.823, "t": 2.809, "p": 0.005, "sig": "***"},
            },
            r_squared=0.398,
            adj_r_squared=0.393,
            f_stat=38.92,
            f_pvalue=0.000,
            n_obs=15234,
            method="IV-2SLS"
        )
        agent.add_result(result2)

        agent.llm = MagicMock()
        agent.llm.chat = AsyncMock(return_value="核心变量系数在不同模型中保持一致...")

        result = await agent.compare_models()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_compare_insufficient_models(self, agent):
        result = await agent.compare_models()
        assert result.success is False

    def test_export_results_table(self, agent, sample_result):
        agent.add_result(sample_result)
        table = agent.export_results_table()
        assert "变量" in table.columns
        assert "Model_1_FE" in table.columns

    def test_format_coefficients(self, agent, sample_result):
        text = agent._format_coefficients(sample_result)
        assert "digital_tech" in text
        assert "0.3250***" in text


class TestRegressionResult:
    def test_creation(self, sample_result):
        assert sample_result.model_name == "Model_1_FE"
        assert sample_result.method == "双向固定效应"
        assert sample_result.n_obs == 15234
        assert "digital_tech" in sample_result.coefficients
