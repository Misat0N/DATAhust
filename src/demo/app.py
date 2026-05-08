"""Streamlit demo for the end-to-end credit risk pipeline."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline import CreditRiskPipeline  # noqa: E402


st.set_page_config(
    page_title="信贷风控智能演示系统",
    page_icon="💳",
    layout="wide",
)


@st.cache_resource(show_spinner=True)
def load_pipeline() -> CreditRiskPipeline:
    """Load and cache the end-to-end pipeline."""

    return CreditRiskPipeline()


def apply_custom_style() -> None:
    """Apply simple CSS to improve page presentation."""

    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #16324f;
            margin-bottom: 0.4rem;
        }
        .sub-title {
            font-size: 1.1rem;
            color: #4f5d75;
            margin-bottom: 1rem;
        }
        .metric-card {
            padding: 1rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #eef4ff 0%, #f9fbff 100%);
            border: 1px solid #d8e4ff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_home_page() -> None:
    """Render the landing page."""

    st.markdown('<div class="main-title">信贷风控智能演示系统</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">融合风控建模、额度测算与 LLM 策略生成的一体化演示平台</div>',
        unsafe_allow_html=True,
    )

    intro_col, arch_col = st.columns([1.1, 1.2], gap="large")
    with intro_col:
        st.markdown("### 项目介绍")
        st.write(
            "该系统面向信贷风控场景，覆盖从用户特征预处理、风险分级、额度测算，"
            "到贷后策略生成的完整链路，适用于模型演示、方案验证与业务沟通。"
        )
        st.markdown("### 核心功能说明")
        st.markdown(
            "- 风险分级：输出 4 类风险等级及概率分布\n"
            "- 额度测算：结合监管硬约束输出合规授信额度\n"
            "- 策略生成：按贷后场景生成话术与运营策略\n"
            "- 批量处理：支持 CSV/Excel 导入与 Excel 结果导出\n"
            "- 可视化看板：集中展示分布、指标与特征重要性"
        )

    with arch_col:
        st.markdown("### 系统架构图")
        st.graphviz_chart(
            """
            digraph G {
                rankdir=LR;
                node [shape=box, style="rounded,filled", color="#5b8def", fillcolor="#eef4ff"];
                input [label="用户输入 / 批量数据"];
                preprocess [label="数据预处理\\nCreditDataPreprocessor"];
                selector [label="特征筛选\\nCreditFeatureSelector"];
                risk [label="风险分级模型\\nCreditRiskClassifier"];
                score [label="额度测算模型\\nCreditScorer"];
                llm [label="LLM 策略生成\\nCreditStrategyGenerator"];
                output [label="结果展示与导出"];
                input -> preprocess -> selector -> risk -> score -> llm -> output;
            }
            """
        )


def render_single_user_page(pipeline: CreditRiskPipeline) -> None:
    """Render the single-user assessment page."""

    st.markdown("## 单用户查询")
    st.write("输入用户核心特征后，点击按钮即可完成端到端评估。")

    with st.form("single_user_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            annual_inc = st.number_input("年收入", min_value=0.0, value=180000.0, step=1000.0)
            dti = st.number_input("负债比", min_value=0.0, max_value=100.0, value=22.5, step=0.5)
            loan_amnt = st.number_input("申请额度", min_value=0.0, value=20000.0, step=500.0)
        with col2:
            emp_length = st.selectbox(
                "工作年限",
                options=[
                    "1年以内",
                    "1年",
                    "2年",
                    "3年",
                    "5年",
                    "10年以上",
                ],
                index=4,
            )
            home_ownership = st.selectbox(
                "住房情况",
                options=["租房", "按揭", "自有住房", "其他"],
                index=1,
            )
            verification_status = st.selectbox(
                "收入核验",
                options=["已核验", "来源核验", "未核验"],
                index=0,
            )
        with col3:
            purpose = st.selectbox(
                "贷款用途",
                options=[
                    "债务整合",
                    "信用卡周转",
                    "房屋改善",
                    "大额消费",
                    "小微经营",
                ],
                index=0,
            )
            grade = st.selectbox("内部评级", options=["A", "B", "C", "D", "E", "F", "G"], index=1)
            term = st.selectbox("贷款期限", options=["36个月", "60个月"], index=0)

        submitted = st.form_submit_button("开始评估", use_container_width=True)

    if not submitted:
        return

    user_payload = {
        "annual_inc": annual_inc,
        "dti": dti,
        "loan_amnt": loan_amnt,
        "emp_length": {
            "1年以内": "< 1 year",
            "1年": "1 year",
            "2年": "2 years",
            "3年": "3 years",
            "5年": "5 years",
            "10年以上": "10+ years",
        }[emp_length],
        "home_ownership": {
            "租房": "RENT",
            "按揭": "MORTGAGE",
            "自有住房": "OWN",
            "其他": "OTHER",
        }[home_ownership],
        "verification_status": {
            "已核验": "Verified",
            "来源核验": "Source Verified",
            "未核验": "Not Verified",
        }[verification_status],
        "purpose": {
            "债务整合": "debt_consolidation",
            "信用卡周转": "credit_card",
            "房屋改善": "home_improvement",
            "大额消费": "major_purchase",
            "小微经营": "small_business",
        }[purpose],
        "grade": grade,
        "sub_grade": f"{grade}3",
        "term": {"36个月": " 36 months", "60个月": " 60 months"}[term],
        "issue_d": "Jan-2018",
        "loan_status": "Current",
    }

    with st.spinner("系统正在完成风控评估..."):
        result = pipeline.run_single(user_payload)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("风险等级", result["risk_label_name"])
    metric_col2.metric("授信额度", f"{result['approved_credit_limit']:,.0f} 元")
    metric_col3.metric("策略场景", result["strategy_scenario"])

    chart_col1, chart_col2 = st.columns([1.1, 1.1], gap="large")
    with chart_col1:
        st.markdown("### 风险概率分布")
        probability_df = pd.DataFrame(
            {
                "risk_level": list(result["risk_probabilities"].keys()),
                "probability": list(result["risk_probabilities"].values()),
            }
        )
        fig = px.bar(
            probability_df,
            x="risk_level",
            y="probability",
            color="probability",
            text="probability",
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=380, xaxis_title="", yaxis_title="概率")
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        st.markdown("### 额度结果")
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=result["approved_credit_limit"],
                number={"suffix": " 元"},
                title={"text": "合规授信额度"},
                gauge={
                    "axis": {"range": [0, max(result["approved_credit_limit"] * 1.5, 10000)]},
                    "bar": {"color": "#5b8def"},
                    "steps": [
                        {"range": [0, result["approved_credit_limit"]], "color": "#dce7ff"},
                    ],
                },
            )
        )
        gauge_fig.update_layout(height=380)
        st.plotly_chart(gauge_fig, use_container_width=True)

    st.markdown("### 贷后策略与话术")
    source_name = {
        "deepseek": "Deepseek 在线生成",
        "local_fallback": "本地模板兜底",
    }.get(result["strategy_source"], result["strategy_source"])
    st.info(f"策略来源：`{source_name}`")
    st.markdown(result["strategy_content"].replace("\n", "  \n"))


def render_batch_page(pipeline: CreditRiskPipeline) -> None:
    """Render the batch processing page."""

    st.markdown("## 批量处理")
    st.write("上传 Excel 或 CSV 文件，系统将自动完成整批用户的风险评估与策略生成。")

    uploaded_file = st.file_uploader(
        "上传批量用户数据",
        type=["csv", "xlsx"],
        help="建议包含 annual_inc、dti、loan_amnt、emp_length、home_ownership、purpose 等字段。",
    )

    if uploaded_file is None:
        st.caption("未上传文件时，可先下载模板并准备数据。")
        template_df = pd.DataFrame(
            [
                {
                    "annual_inc": 180000,
                    "dti": 21.5,
                    "loan_amnt": 20000,
                    "emp_length": "5 years",
                    "home_ownership": "MORTGAGE",
                    "verification_status": "Verified",
                    "purpose": "debt_consolidation",
                    "grade": "B",
                    "term": " 36 months",
                    "issue_d": "Jan-2018",
                    "loan_status": "Current",
                }
            ]
        )
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            template_df.to_excel(writer, index=False, sheet_name="template")
        st.download_button(
            "下载模板 Excel",
            data=buffer.getvalue(),
            file_name="批量处理模板.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        return

    if uploaded_file.name.lower().endswith(".csv"):
        batch_df = pd.read_csv(uploaded_file)
    else:
        batch_df = pd.read_excel(uploaded_file)

    st.markdown("### 原始数据预览")
    st.dataframe(batch_df.head(10), use_container_width=True)

    if not st.button("批量处理", use_container_width=True):
        return

    progress_bar = st.progress(0)

    def update_progress(progress_value: float) -> None:
        progress_bar.progress(min(max(int(progress_value * 100), 0), 100))

    pipeline.progress_callback = update_progress
    with st.spinner("系统正在处理批量数据，请稍候..."):
        result_df = pipeline.run_batch(batch_df)
    pipeline.progress_callback = None
    progress_bar.progress(100)

    st.markdown("### 处理结果")
    st.dataframe(result_df, use_container_width=True)

    output_buffer = BytesIO()
    with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="results")

    st.download_button(
        "下载 Excel 结果报告",
        data=output_buffer.getvalue(),
        file_name="批量评估结果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def render_dashboard_page(pipeline: CreditRiskPipeline) -> None:
    """Render the dashboard page."""

    st.markdown("## 可视化看板")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("加权 F1 分数", f"{pipeline.dashboard_metrics['weighted_f1']:.4f}")
    metric_col2.metric("KS 值", f"{pipeline.dashboard_metrics['ks_value']:.4f}")
    metric_col3.metric("R²", f"{pipeline.dashboard_metrics['r2']:.4f}")

    chart_col1, chart_col2 = st.columns([1.0, 1.2], gap="large")
    with chart_col1:
        st.markdown("### 用户风险等级分布")
        test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
        if test_path.exists():
            test_df = pd.read_csv(test_path, low_memory=False)
            risk_distribution = (
                test_df["preloan_risk_label_name"]
                .fillna("未知")
                .value_counts()
                .reset_index()
            )
            risk_distribution.columns = ["risk_level", "count"]
            pie_fig = px.pie(
                risk_distribution,
                names="risk_level",
                values="count",
                hole=0.45,
                color_discrete_sequence=px.colors.sequential.Blues_r,
            )
            pie_fig.update_layout(height=420)
            st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.warning("未找到测试集分布数据。")

    with chart_col2:
        st.markdown("### 特征重要性排序")
        importance_df = pipeline.feature_importance_df.head(15).copy()
        importance_fig = px.bar(
            importance_df.sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Tealgrn",
        )
        importance_fig.update_layout(height=420, yaxis_title="", xaxis_title="重要性")
        st.plotly_chart(importance_fig, use_container_width=True)


def main() -> None:
    """Run the Streamlit application."""

    apply_custom_style()
    pipeline = load_pipeline()

    st.sidebar.title("导航")
    page = st.sidebar.radio(
        "选择页面",
        options=["首页", "单用户查询", "批量处理", "可视化看板"],
    )

    if page == "首页":
        render_home_page()
    elif page == "单用户查询":
        render_single_user_page(pipeline)
    elif page == "批量处理":
        render_batch_page(pipeline)
    else:
        render_dashboard_page(pipeline)


if __name__ == "__main__":
    main()
