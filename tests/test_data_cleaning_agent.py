"""
单元测试 - 数据清洗Agent
"""

import pytest
import pandas as pd
import numpy as np

from agents.data_cleaning_agent import DataCleaningAgent, VariableSpec
from core.base import AgentResponse


@pytest.fixture
def agent():
    return DataCleaningAgent(config={})


@pytest.fixture
def sample_df():
    """创建示例面板数据"""
    np.random.seed(42)
    data = []
    for firm in range(1, 6):
        for year in range(2015, 2024):
            data.append({
                "stock_code": f"F{firm:03d}",
                "year": year,
                "ESG_disclosure": np.random.uniform(0, 100),
                "digital_tech": np.random.uniform(0, 1),
                "Size": np.random.uniform(18, 24),
                "ROA": np.random.uniform(-0.1, 0.2),
                "Lev": np.random.uniform(0.2, 0.8),
            })
    return pd.DataFrame(data)


class TestDataCleaningAgent:
    def test_init(self, agent):
        assert agent.name == "data_cleaning_agent"
        assert agent.df is None

    def test_load_csv(self, agent, sample_df, tmp_path):
        csv_path = tmp_path / "test_data.csv"
        sample_df.to_csv(csv_path, index=False)

        result = agent.load_data(str(csv_path))
        assert result.success is True
        assert result.data["shape"][0] == 45  # 5 firms * 9 years

    def test_load_unsupported_format(self, agent):
        result = agent.load_data("test.parquet")
        assert result.success is False

    def test_set_panel_index(self, agent, sample_df):
        agent.df = sample_df.copy()
        result = agent.set_panel_index("stock_code", "year")
        assert result.success is True
        assert agent.df.index.names == ["stock_code", "year"]

    def test_generate_lag_variables(self, agent, sample_df):
        agent.df = sample_df.copy()
        agent.set_panel_index("stock_code", "year")
        agent.add_variable_spec(VariableSpec(
            name="digital_tech",
            label="数字技术",
            var_type="independent",
            measurement="文本分析",
            source="年报",
            lag_periods=1
        ))
        result = agent.generate_variables()
        assert result.success is True
        assert "digital_tech_L1" in result.data["generated_columns"]

    def test_log_transform(self, agent, sample_df):
        agent.df = sample_df.copy()
        agent.set_panel_index("stock_code", "year")
        agent.add_variable_spec(VariableSpec(
            name="Size",
            label="企业规模",
            var_type="control",
            measurement="总资产对数",
            source="CSMAR",
            log_transform=True
        ))
        result = agent.generate_variables()
        assert result.success is True
        assert "ln_Size" in result.data["generated_columns"]

    def test_interaction_term(self, agent, sample_df):
        agent.df = sample_df.copy()
        agent.set_panel_index("stock_code", "year")
        agent.add_variable_spec(VariableSpec(
            name="digital_tech",
            label="数字技术",
            var_type="independent",
            measurement="文本分析",
            source="年报",
            interaction_with="Size"
        ))
        result = agent.generate_variables()
        assert result.success is True
        assert "digital_tech_x_Size" in result.data["generated_columns"]

    def test_winsorize(self, agent, sample_df):
        agent.df = sample_df.copy()
        agent.set_panel_index("stock_code", "year")
        agent.add_variable_spec(VariableSpec(
            name="ROA",
            label="资产收益率",
            var_type="control",
            measurement="净利润/总资产",
            source="CSMAR",
            winsorize=0.01
        ))
        result = agent.generate_variables()
        assert result.success is True

    def test_descriptive_stats(self, agent, sample_df):
        agent.df = sample_df.copy()
        result = agent.descriptive_stats(["ESG_disclosure", "digital_tech"])
        assert result.success is True
        assert "ESG_disclosure" in result.data

    def test_correlation_matrix(self, agent, sample_df):
        agent.df = sample_df.copy()
        result = agent.correlation_matrix(["ESG_disclosure", "digital_tech"])
        assert result.success is True

    def test_handle_missing_listwise(self, agent, sample_df):
        sample_df.loc[0, "ESG_disclosure"] = np.nan
        agent.df = sample_df.copy()
        result = agent.handle_missing("listwise")
        assert result.success is True
        assert result.data["dropped"] >= 1

    def test_get_cleaning_report(self, agent, sample_df):
        agent.df = sample_df.copy()
        agent.set_panel_index("stock_code", "year")
        report = agent.get_cleaning_report()
        assert len(report) > 0

    def test_no_data_error(self, agent):
        result = agent.set_panel_index("a", "b")
        assert result.success is False
