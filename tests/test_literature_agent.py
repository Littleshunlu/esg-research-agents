"""
单元测试 - 文献解析Agent
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from agents.literature_agent import LiteratureAgent, LiteratureInfo
from core.base import AgentResponse


@pytest.fixture
def agent():
    return LiteratureAgent(config={"api_key": "test-key"})


class TestLiteratureAgent:
    def test_init(self, agent):
        assert agent.name == "literature_agent"
        assert agent.parsed_papers == []

    def test_info(self, agent):
        info = agent.info()
        assert info["name"] == "literature_agent"

    def test_export_parsed_empty(self, agent):
        result = agent.export_parsed()
        assert result == []

    @pytest.mark.asyncio
    async def test_parse_paper(self, agent):
        mock_response = '''
        {
            "research_framework": "数字技术通过降低信息不对称促进ESG披露",
            "hypotheses": ["数字技术显著提升ESG信息披露水平"],
            "methodology": {
                "estimation": ["双向固定效应", "IV-2SLS"],
                "model_specification": "ESG = α + β*Digital + γ*Controls + ε",
                "robustness_checks": ["替换变量", "PSM-DID"]
            },
            "variables": {
                "dependent": {"name": "ESG_disclosure", "measurement": "华证ESG评级", "source": "Wind"},
                "core_independent": {"name": "digital_tech", "measurement": "文本分析", "source": "年报"},
                "controls": [{"name": "Size", "reason": "企业规模"}],
                "mediator": null,
                "moderator": null
            },
            "key_findings": ["数字技术显著提升ESG披露水平"],
            "data_info": {
                "sample_period": "2015-2023",
                "data_source": "CSMAR/Wind",
                "sample_size": "15000+"
            }
        }
        '''

        agent.llm = MagicMock()
        agent.llm.chat = AsyncMock(return_value=mock_response)

        result = await agent.parse_paper("示例论文文本")
        assert result.success is True
        assert len(agent.parsed_papers) == 1

    def test_extract_json(self, agent):
        text = 'Some text before {"key": "value"} some text after'
        result = agent._extract_json(text)
        assert result == {"key": "value"}

    def test_extract_json_no_json(self, agent):
        with pytest.raises(Exception):
            agent._extract_json("No JSON here")


class TestLiteratureInfo:
    def test_default_values(self):
        info = LiteratureInfo()
        assert info.title == ""
        assert info.authors == []
        assert info.year == 0
        assert info.key_findings == []
