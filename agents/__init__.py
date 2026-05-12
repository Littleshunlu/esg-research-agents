"""
Agent模块初始化
"""

from agents.literature_agent import LiteratureAgent
from agents.data_cleaning_agent import DataCleaningAgent, VariableSpec
from agents.empirical_agent import EmpiricalAgent, RegressionResult

__all__ = [
    "LiteratureAgent",
    "DataCleaningAgent",
    "VariableSpec",
    "EmpiricalAgent",
    "RegressionResult",
]
