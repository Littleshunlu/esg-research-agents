"""
数据清洗与变量构建Agent
处理上市公司ESG披露数据并生成回归所需变量
"""

import json
import warnings
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

from core.base import BaseAgent, AgentResponse
from utils.llm_client import LLMClient


@dataclass
class VariableSpec:
    """变量规格定义"""
    name: str
    label: str
    var_type: str  # dependent, independent, control, mediator, moderator
    measurement: str
    source: str
    lag_periods: int = 0
    interaction_with: Optional[str] = None
    log_transform: bool = False
    winsorize: Optional[float] = None  # 缩尾比例，如0.01


class DataCleaningAgent(BaseAgent):
    """
    数据清洗与变量构建Agent

    功能：
    1. 自动识别和处理缺失值
    2. 生成滞后项（L1, L2等）
    3. 构建交互项（调节效应）
    4. 对数变换
    5. 缩尾处理
    6. 描述性统计生成
    7. 相关性矩阵计算
    8. VIF检验（多重共线性）
    """

    SYSTEM_PROMPT = """你是一位专业的计量经济学数据处理专家。
你的任务是根据用户的变量定义，生成对应的数据处理代码和统计报告。

注意以下规范：
1. 面板数据需要设置个体和时间双重索引
2. 滞后项需按个体分组计算
3. 交互项需要先中心化再相乘（减少多重共线性）
4. 缩尾处理使用对称缩尾
5. 所有变量需生成描述性统计和相关性矩阵"""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(name="data_cleaning_agent", config=config)
        self.llm = LLMClient(config=self.config)
        self.df: Optional[pd.DataFrame] = None
        self.variable_specs: list[VariableSpec] = []
        self.cleaning_log: list[str] = []

    def load_data(self, file_path: str, **kwargs) -> AgentResponse:
        """加载数据文件"""
        try:
            if file_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(file_path, **kwargs)
            elif file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path, **kwargs)
            elif file_path.endswith('.dta'):
                self.df = pd.read_stata(file_path, **kwargs)
            else:
                return AgentResponse(
                    success=False,
                    data={},
                    message=f"不支持的文件格式: {file_path}"
                )

            self.cleaning_log.append(f"加载数据: {file_path}, 形状: {self.df.shape}")
            return AgentResponse(
                success=True,
                data={"shape": self.df.shape, "columns": list(self.df.columns)},
                message="数据加载成功"
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                data={},
                message=f"数据加载失败: {str(e)}"
            )

    def set_panel_index(self, entity_col: str, time_col: str) -> AgentResponse:
        """设置面板数据索引"""
        if self.df is None:
            return AgentResponse(success=False, data={}, message="请先加载数据")

        self.df = self.df.set_index([entity_col, time_col])
        self.cleaning_log.append(f"设置面板索引: 个体={entity_col}, 时间={time_col}")
        return AgentResponse(
            success=True,
            data={"index_names": self.df.index.names},
            message="面板索引设置成功"
        )

    def add_variable_spec(self, spec: VariableSpec) -> None:
        """添加变量规格"""
        self.variable_specs.append(spec)

    def generate_variables(self) -> AgentResponse:
        """根据变量规格生成所有衍生变量"""
        if self.df is None:
            return AgentResponse(success=False, data={}, message="请先加载数据")

        generated = []
        for spec in self.variable_specs:
            col_name = spec.name

            # 滞后项
            if spec.lag_periods > 0:
                lag_col = f"{col_name}_L{spec.lag_periods}"
                self.df[lag_col] = self.df.groupby(level=0)[col_name].shift(spec.lag_periods)
                generated.append(lag_col)
                self.cleaning_log.append(f"生成滞后项: {lag_col}")

            # 对数变换
            if spec.log_transform:
                log_col = f"ln_{col_name}"
                self.df[log_col] = np.log(self.df[col_name].replace(0, np.nan))
                generated.append(log_col)
                self.cleaning_log.append(f"对数变换: {log_col}")

            # 交互项
            if spec.interaction_with:
                inter_col = f"{col_name}_x_{spec.interaction_with}"
                # 先中心化
                x_centered = self.df[col_name] - self.df[col_name].mean()
                m_centered = self.df[spec.interaction_with] - self.df[spec.interaction_with].mean()
                self.df[inter_col] = x_centered * m_centered
                generated.append(inter_col)
                self.cleaning_log.append(f"生成交互项: {inter_col}")

            # 缩尾处理
            if spec.winsorize:
                lower = self.df[col_name].quantile(spec.winsorize)
                upper = self.df[col_name].quantile(1 - spec.winsorize)
                self.df[col_name] = self.df[col_name].clip(lower, upper)
                self.cleaning_log.append(f"缩尾处理: {col_name}, 比例={spec.winsorize}")

        return AgentResponse(
            success=True,
            data={"generated_columns": generated, "shape": self.df.shape},
            message=f"成功生成 {len(generated)} 个衍生变量"
        )

    def handle_missing(self, method: str = "listwise") -> AgentResponse:
        """处理缺失值"""
        if self.df is None:
            return AgentResponse(success=False, data={}, message="请先加载数据")

        before_count = len(self.df)
        if method == "listwise":
            self.df = self.df.dropna()
        elif method == "fill_mean":
            self.df = self.df.fillna(self.df.mean())
        elif method == "fill_median":
            self.df = self.df.fillna(self.df.median())
        elif method == "interpolate":
            self.df = self.df.groupby(level=0).apply(
                lambda g: g.interpolate(method='linear')
            )

        after_count = len(self.df)
        dropped = before_count - after_count
        self.cleaning_log.append(f"缺失值处理: 方法={method}, 删除{dropped}行")

        return AgentResponse(
            success=True,
            data={"before": before_count, "after": after_count, "dropped": dropped},
            message=f"缺失值处理完成，删除 {dropped} 行"
        )

    def descriptive_stats(self, columns: Optional[list] = None) -> AgentResponse:
        """生成描述性统计"""
        if self.df is None:
            return AgentResponse(success=False, data={}, message="请先加载数据")

        cols = columns or self.df.columns.tolist()
        desc = self.df[cols].describe().T
        desc['skewness'] = self.df[cols].skew()
        desc['kurtosis'] = self.df[cols].kurtosis()

        return AgentResponse(
            success=True,
            data=desc.to_dict(),
            message="描述性统计生成完成"
        )

    def correlation_matrix(self, columns: Optional[list] = None) -> AgentResponse:
        """生成相关性矩阵"""
        if self.df is None:
            return AgentResponse(success=False, data={}, message="请先加载数据")

        cols = columns or self.df.columns.tolist()
        corr = self.df[cols].corr()

        return AgentResponse(
            success=True,
            data=corr.to_dict(),
            message="相关性矩阵生成完成"
        )

    def vif_check(self, columns: list) -> AgentResponse:
        """VIF检验（多重共线性）"""
        if self.df is None:
            return AgentResponse(success=False, data={}, message="请先加载数据")

        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            X = self.df[columns].dropna()
            vif_data = pd.DataFrame()
            vif_data['variable'] = columns
            vif_data['VIF'] = [
                variance_inflation_factor(X.values, i)
                for i in range(X.shape[1])
            ]

            return AgentResponse(
                success=True,
                data=vif_data.to_dict(),
                message="VIF检验完成"
            )
        except ImportError:
            return AgentResponse(
                success=False,
                data={},
                message="需要安装statsmodels: pip install statsmodels"
            )

    def export_data(self, file_path: str) -> AgentResponse:
        """导出清洗后的数据"""
        if self.df is None:
            return AgentResponse(success=False, data={}, message="请先加载数据")

        try:
            if file_path.endswith('.csv'):
                self.df.to_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.df.to_excel(file_path)
            elif file_path.endswith('.dta'):
                self.df.to_stata(file_path)

            return AgentResponse(
                success=True,
                data={"path": file_path},
                message="数据导出成功"
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                data={},
                message=f"导出失败: {str(e)}"
            )

    def get_cleaning_report(self) -> str:
        """获取数据清洗报告"""
        return "\n".join(f"[{i+1}] {log}" for i, log in enumerate(self.cleaning_log))
