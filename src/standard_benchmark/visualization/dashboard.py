"""Streamlit dashboard for interactive analysis."""

import json
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from standard_benchmark.data.models import EvaluationResult
from standard_benchmark.storage.json_store import JSONStore


# Page config
st.set_page_config(
    page_title="Standard Benchmark Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def recalculate_scores(
    evaluations: dict,
    faithfulness_weight: float,
    completeness_weight: float,
    source_quality_weight: float,
    hf_classifier_weight: float,
    hf_grader_weight: float,
    hf_quality_weight: float,
    bertscore_relevance_weight: float,
    bertscore_faithfulness_weight: float,
) -> dict:
    """Recalculate three independent scores with adjusted weights.

    Args:
        evaluations: Original evaluations dict
        faithfulness_weight: Weight for faithfulness within Ragas
        completeness_weight: Weight for completeness within Ragas
        source_quality_weight: Weight for source quality within Ragas
        hf_classifier_weight: Weight for HF correctness classifier
        hf_grader_weight: Weight for HF correctness grader
        hf_quality_weight: Weight for HF response quality
        bertscore_relevance_weight: Weight for BERTScore relevance within automated
        bertscore_faithfulness_weight: Weight for BERTScore
            faithfulness within automated

    Returns:
        Updated evaluations dict with recalculated three independent scores
    """
    updated_evaluations = {}

    for query_id, systems in evaluations.items():
        updated_systems = {}

        for system, eval_result in systems.items():
            # Create a copy of scores
            scores = dict(eval_result.scores)

            # Calculate Ragas score (weighted average of 3 metrics)
            if "faithfulness" in scores or "completeness" in scores:
                ragas_metrics = {
                    "faithfulness": scores.get("faithfulness", 5.0),
                    "completeness": scores.get("completeness", 5.0),
                    "source_quality": scores.get("source_quality", 5.0),
                }

                scores["ragas_score"] = (
                    ragas_metrics["faithfulness"] * faithfulness_weight
                    + ragas_metrics["completeness"] * completeness_weight
                    + ragas_metrics["source_quality"] * source_quality_weight
                )

            # Calculate HuggingFace score
            has_hf_scores = any(k.startswith("hf_") for k in scores.keys())

            if has_hf_scores:
                hf_metrics = {
                    "classifier": scores.get("hf_correctness_classifier", 5.5),
                    "grader": scores.get("hf_correctness_grader", 5.5),
                    "quality": scores.get("hf_response_quality", 5.5),
                }

                scores["huggingface_score"] = (
                    hf_metrics["classifier"] * hf_classifier_weight
                    + hf_metrics["grader"] * hf_grader_weight
                    + hf_metrics["quality"] * hf_quality_weight
                )

            # Calculate Automated score (BERTScore only)
            if "automated_bertscore_relevance" in scores:
                automated_metrics = {
                    "bertscore_relevance": scores.get(
                        "automated_bertscore_relevance", 0.5
                    ),
                    "bertscore_faithfulness": scores.get(
                        "automated_bertscore_faithfulness", 0.5
                    ),
                }

                scores["automated_score"] = (
                    automated_metrics["bertscore_relevance"] * 9 + 1
                ) * bertscore_relevance_weight + (
                    automated_metrics["bertscore_faithfulness"] * 9 + 1
                ) * bertscore_faithfulness_weight

            # Recompute final_score from available scores
            available = [
                v
                for k, v in scores.items()
                if k in ("ragas_score", "huggingface_score", "automated_score")
            ]
            if available:
                scores["final_score"] = sum(available) / len(available)

            # Create updated EvaluationResult
            updated_result = EvaluationResult(
                query_id=eval_result.query_id,
                system=eval_result.system,
                scores=scores,
                metadata=eval_result.metadata,
                timestamp=eval_result.timestamp,
            )

            updated_systems[system] = updated_result

        updated_evaluations[query_id] = updated_systems

    return updated_evaluations


@st.cache_data
def load_data(
    queries_path: str, evaluations_path: str, responses_path: Optional[str] = None
):
    """Load all data files."""
    store = JSONStore()

    # Load queries
    queries = store.load_queries(queries_path, use_jsonl=True)

    # Load evaluations
    with open(evaluations_path, "r") as f:
        eval_data = json.load(f)

    # Convert to EvaluationResult objects
    evaluations = {}
    for query_id, systems in eval_data.items():
        evaluations[query_id] = {
            system: EvaluationResult(**eval_dict)
            for system, eval_dict in systems.items()
        }

    # Load responses if provided
    responses = None
    if responses_path and Path(responses_path).exists():
        with open(responses_path, "r") as f:
            responses = json.load(f)

    return queries, evaluations, responses


def main():
    st.title("🔍 Search API Benchmark Dashboard")
    st.markdown("---")

    # Sidebar - Data Loading
    st.sidebar.header("📁 Data Files")

    # Add clear cache button at the top
    if st.sidebar.button("🔄 Clear Cache & Reload Data"):
        st.cache_data.clear()
        st.rerun()

    queries_file = st.sidebar.text_input(
        "Queries File", value="data/queries.jsonl"
    )
    evaluations_file = st.sidebar.text_input(
        "Evaluations File", value="data/evaluations.json"
    )
    responses_file = st.sidebar.text_input(
        "Responses File (optional)", value="data/responses.json"
    )

    # Load data
    try:
        queries, evaluations, responses = load_data(
            queries_file,
            evaluations_file,
            responses_file if Path(responses_file).exists() else None,
        )
        st.sidebar.success(
            f"✅ Loaded {len(queries)} queries, {len(evaluations)} evaluations"
        )

        # Check if HuggingFace scores are available in the data
        hf_scores_available = False
        if evaluations:
            first_query_id = list(evaluations.keys())[0]
            first_system = list(evaluations[first_query_id].keys())[0]
            first_eval = evaluations[first_query_id][first_system]
            hf_scores_available = any(
                k.startswith("hf_") for k in first_eval.scores.keys()
            )

            if hf_scores_available:
                st.sidebar.info("✨ HuggingFace judges scores detected")
            else:
                st.sidebar.warning(
                    "⚠️ No HuggingFace scores found."
                    " Run evaluation with"
                    " --use-huggingface to enable."
                )

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Score weightage adjustment
    st.sidebar.header("⚖️ Score Weights")

    with st.sidebar.expander("Adjust Score Calculation", expanded=False):
        # Warn if HuggingFace mode selected but no scores available
        if not hf_scores_available:
            st.warning(
                "⚠️ HuggingFace scores not available."
                " Run evaluation with"
                " --use-huggingface first."
            )

        st.info(
            "💡 Three independent scores are computed:"
            " Ragas, HuggingFace, and Automated."
            " Adjust weights within each"
            " scoring system below."
        )

        st.markdown("---")
        st.markdown("**Ragas Metrics Distribution**")
        equal_ragas = st.checkbox("Equal weights for all Ragas metrics", value=True)

        if equal_ragas:
            faithfulness_weight = 0.45
            completeness_weight = 0.45
            source_quality_weight = 0.1
        else:
            faithfulness_weight = st.slider("Faithfulness", 0.0, 1.0, 0.45, 0.05)
            completeness_weight = st.slider("Completeness", 0.0, 1.0, 0.45, 0.05)
            source_quality_weight = st.slider("Source Quality", 0.0, 1.0, 0.1, 0.05)

            # Normalize
            total_ragas = (
                faithfulness_weight + completeness_weight + source_quality_weight
            )
            if total_ragas > 0:
                faithfulness_weight /= total_ragas
                completeness_weight /= total_ragas
                source_quality_weight /= total_ragas

        st.markdown("---")
        st.markdown("**HuggingFace Judges Distribution**")
        equal_hf = st.checkbox("Equal weights for all HF judges", value=True)

        if equal_hf:
            hf_classifier_weight = 0.45
            hf_grader_weight = 0.45
            hf_quality_weight = 0.1
        else:
            hf_classifier_weight = st.slider(
                "HF Correctness Classifier", 0.0, 1.0, 0.45, 0.05
            )
            hf_grader_weight = st.slider("HF Correctness Grader", 0.0, 1.0, 0.45, 0.05)
            hf_quality_weight = st.slider("HF Response Quality", 0.0, 1.0, 0.1, 0.05)

            # Normalize
            total_hf = hf_classifier_weight + hf_grader_weight + hf_quality_weight
            if total_hf > 0:
                hf_classifier_weight /= total_hf
                hf_grader_weight /= total_hf
                hf_quality_weight /= total_hf

        st.markdown("---")
        st.markdown("**Automated Metrics Distribution**")
        equal_auto = st.checkbox("Equal weights for BERTScore metrics", value=False)

        if equal_auto:
            bertscore_relevance_weight = 0.5
            bertscore_faithfulness_weight = 0.5
        else:
            bertscore_relevance_weight = st.slider(
                "BERTScore Relevance (Q-R)", 0.0, 1.0, 0.55, 0.05
            )
            bertscore_faithfulness_weight = st.slider(
                "BERTScore Faithfulness (R-S)", 0.0, 1.0, 0.45, 0.05
            )

            # Normalize
            total_auto = bertscore_relevance_weight + bertscore_faithfulness_weight
            if total_auto > 0:
                bertscore_relevance_weight /= total_auto
                bertscore_faithfulness_weight /= total_auto

        st.caption(
            "📝 Note: entity_coverage and"
            " unique_domains are descriptive"
            " statistics only (not scored)"
        )

    # Recalculate scores with adjusted weights if they differ from defaults
    weights_changed = (
        not equal_ragas
        or not equal_hf
        or not equal_auto
        or bertscore_relevance_weight != 0.55
        or bertscore_faithfulness_weight != 0.45
    )

    if weights_changed:
        evaluations = recalculate_scores(
            evaluations,
            faithfulness_weight,
            completeness_weight,
            source_quality_weight,
            hf_classifier_weight,
            hf_grader_weight,
            hf_quality_weight,
            bertscore_relevance_weight,
            bertscore_faithfulness_weight,
        )

    # Sidebar - Score selection for comparison
    st.sidebar.header("📊 Comparison Score")
    score_for_comparison = st.sidebar.selectbox(
        "Score metric for comparisons",
        [k for k in ["final_score", "ragas_score", "huggingface_score", "automated_score"]],
        index=0,
        help="Select which scoring system to use in comparison views",
    )

    # Main tabs
    tab0, tab1, tab2, tab3, tab4 = st.tabs(
        ["Key Metrics", "Overview", "Query Explorer", "Analytics", "Export"]
    )

    # Shared: determine all systems
    all_systems_set = set()
    for sys_evals in evaluations.values():
        all_systems_set.update(sys_evals.keys())
    _preferred_order = ["linkup", "perplexity", "exa", "tavily"]
    all_systems_global = sorted(
        all_systems_set,
        key=lambda s: _preferred_order.index(s) if s in _preferred_order else len(_preferred_order),
    )

    # TAB 0: Key Metrics
    with tab0:
        st.markdown(
            "These metrics capture three dimensions of search reliability "
            "that matter most for enterprise decision-making: breadth of sources, "
            "faithfulness to cited material, and completeness of answers."
        )

        # --- 1. Source Diversity ---
        st.markdown("---")
        st.subheader("Source Diversity")
        st.caption(
            "Average number of unique domains retrieved per query. "
            "Higher domain diversity reduces dependence on any single source "
            "and increases the likelihood of capturing heterogeneous viewpoints."
        )

        domain_data = []
        for query_id, sys_evals in evaluations.items():
            for system, eval_result in sys_evals.items():
                val = eval_result.scores.get("automated_unique_domains")
                if val is not None:
                    domain_data.append({"System": system, "Unique Domains": val})

        if domain_data:
            df_domains = pd.DataFrame(domain_data)
            avg_domains = (
                df_domains.groupby("System")["Unique Domains"]
                .mean()
                .reset_index()
                .sort_values("Unique Domains", ascending=True)
            )

            col_chart, col_table = st.columns([2, 1])
            with col_chart:
                fig = px.bar(
                    avg_domains,
                    y="System",
                    x="Unique Domains",
                    orientation="h",
                    color_discrete_sequence=["#94a3b8"],
                )
                fig.update_layout(
                    showlegend=False,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=20, t=10, b=0),
                    height=max(200, len(avg_domains) * 50),
                    xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title=None),
                    yaxis=dict(showgrid=False, title=None),
                )
                fig.update_traces(
                    texttemplate="%{x:.1f}", textposition="outside",
                    marker_line_width=0,
                )
                st.plotly_chart(fig, use_container_width=True)
            with col_table:
                st.dataframe(
                    avg_domains.sort_values("Unique Domains", ascending=False)
                    .set_index("System")
                    .round(1),
                    use_container_width=True,
                )
        else:
            st.info("No unique-domain data available in evaluations.")

        # --- 2. Hallucination Risk ---
        st.markdown("---")
        st.subheader("Hallucination Risk")
        st.caption(
            "Derived from faithfulness scores (RAGAS claim-level analysis). "
            "Each response is decomposed into atomic claims and checked against "
            "cited sources. Lower hallucination risk means stronger grounding "
            "and fewer unsupported claims."
        )

        faith_data = []
        for query_id, sys_evals in evaluations.items():
            for system, eval_result in sys_evals.items():
                val = eval_result.scores.get("faithfulness")
                if val is not None:
                    faith_data.append({"System": system, "Faithfulness": val})

        if faith_data:
            df_faith = pd.DataFrame(faith_data)
            avg_faith = (
                df_faith.groupby("System")["Faithfulness"]
                .mean()
                .reset_index()
            )
            # Hallucination risk = inverse of faithfulness (on a 1-10 scale)
            avg_faith["Hallucination Risk"] = 10 - avg_faith["Faithfulness"]
            avg_faith = avg_faith.sort_values("Hallucination Risk", ascending=True)

            col_chart, col_table = st.columns([2, 1])
            with col_chart:
                fig = px.bar(
                    avg_faith,
                    y="System",
                    x="Hallucination Risk",
                    orientation="h",
                    color_discrete_sequence=["#94a3b8"],
                )
                fig.update_layout(
                    showlegend=False,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=20, t=10, b=0),
                    height=max(200, len(avg_faith) * 50),
                    xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title=None),
                    yaxis=dict(showgrid=False, title=None, autorange="reversed"),
                )
                fig.update_traces(
                    texttemplate="%{x:.2f}", textposition="outside",
                    marker_line_width=0,
                )
                st.plotly_chart(fig, use_container_width=True)
            with col_table:
                display_df = avg_faith[["System", "Faithfulness", "Hallucination Risk"]].copy()
                display_df = display_df.sort_values("Hallucination Risk").set_index("System").round(2)
                st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No faithfulness data available in evaluations.")

        # --- 3. Answer Completeness ---
        st.markdown("---")
        st.subheader("Answer Completeness")
        st.caption(
            "Measured via entity coverage: the proportion of key entities and "
            "sub-queries addressed in each response. Higher coverage means fewer "
            "missing entities and less need for follow-up queries."
        )

        entity_data = []
        for query_id, sys_evals in evaluations.items():
            for system, eval_result in sys_evals.items():
                val = eval_result.scores.get("automated_entity_coverage")
                if val is not None:
                    entity_data.append(
                        {"System": system, "Entity Coverage (%)": val * 100}
                    )

        if entity_data:
            df_entity = pd.DataFrame(entity_data)
            avg_entity = (
                df_entity.groupby("System")["Entity Coverage (%)"]
                .mean()
                .reset_index()
            )
            # Information gap = missing entities (inverse of completeness)
            avg_entity["Missing Entities (%)"] = 100 - avg_entity["Entity Coverage (%)"]
            avg_entity = avg_entity.sort_values("Missing Entities (%)", ascending=True)

            col_chart, col_table = st.columns([2, 1])
            with col_chart:
                fig = px.bar(
                    avg_entity,
                    y="System",
                    x="Missing Entities (%)",
                    orientation="h",
                    color_discrete_sequence=["#94a3b8"],
                )
                fig.update_layout(
                    showlegend=False,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=20, t=10, b=0),
                    height=max(200, len(avg_entity) * 50),
                    xaxis=dict(
                        showgrid=True, gridcolor="#f1f5f9", title=None,
                        range=[0, 100],
                    ),
                    yaxis=dict(showgrid=False, title=None, autorange="reversed"),
                )
                fig.update_traces(
                    texttemplate="%{x:.1f}%", textposition="outside",
                    marker_line_width=0,
                )
                st.plotly_chart(fig, use_container_width=True)
            with col_table:
                display_df = (
                    avg_entity[["System", "Entity Coverage (%)", "Missing Entities (%)"]]
                    .sort_values("Entity Coverage (%)", ascending=False)
                    .set_index("System")
                    .round(1)
                )
                st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No entity coverage data available in evaluations.")

        st.markdown("---")
        st.caption(
            "These metrics do not capture usability factors such as latency, "
            "UI experience, or writing coherence. See the Analytics tab for "
            "performance and additional metric breakdowns."
        )

    # TAB 1: Overview
    with tab1:
        all_systems = all_systems_global

        # Compute per-system averages
        system_avg_scores = {}
        for system in all_systems:
            scores = []
            for query_id, sys_evals in evaluations.items():
                if system in sys_evals:
                    s = sys_evals[system].scores.get(score_for_comparison, 0)
                    scores.append(s)
            if scores:
                system_avg_scores[system] = sum(scores) / len(scores)

        # Top-level metrics
        cols = st.columns(len(all_systems) + 1)
        with cols[0]:
            st.metric("Total Queries", len(queries))
        for i, system in enumerate(all_systems):
            with cols[i + 1]:
                avg = system_avg_scores.get(system, 0)
                st.metric(f"{system.title()} Avg", f"{avg:.2f}")

        st.markdown("---")

        # Performance metrics
        if responses:
            st.subheader("⚡ Performance Metrics")

            # Calculate latency and throughput
            perf_data = []
            for query_id, sys_responses in responses.items():
                for system, response_data in sys_responses.items():
                    latency = response_data.get("latency_ms")
                    response_text = response_data.get("response_text", "")

                    if latency and latency > 0:
                        words = len(response_text.split())
                        tokens = int(words * 1.3)
                        throughput = (tokens / latency) * 1000 if latency > 0 else 0

                        perf_data.append(
                            {
                                "System": system,
                                "Latency (ms)": latency,
                                "Output Tokens": tokens,
                                "Throughput (tokens/s)": throughput,
                            }
                        )

            if perf_data:
                col1, col2 = st.columns(2)

                with col1:
                    df_perf = pd.DataFrame(perf_data)
                    avg_throughput = (
                        df_perf.groupby("System")["Throughput (tokens/s)"]
                        .mean()
                        .reset_index()
                    )
                    fig = px.bar(
                        avg_throughput,
                        x="System",
                        y="Throughput (tokens/s)",
                        color="System",
                        title="Average Throughput by System",
                        category_orders={"System": all_systems},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    summary = (
                        df_perf.groupby("System")
                        .agg(
                            {
                                "Throughput (tokens/s)": ["mean", "min", "max"],
                            }
                        )
                        .round(2)
                        .reindex(all_systems)
                    )
                    st.markdown("**Performance Summary**")
                    st.dataframe(summary, use_container_width=True)

        # Descriptive metrics (entity coverage & unique domains)
        if evaluations:
            st.subheader("📊 Descriptive Metrics (not in score)")

            desc_data = []
            for query_id, sys_evals in evaluations.items():
                for system, eval_result in sys_evals.items():
                    entity_cov = eval_result.scores.get(
                        "automated_entity_coverage", None
                    )
                    unique_dom = eval_result.scores.get(
                        "automated_unique_domains", None
                    )

                    if entity_cov is not None or unique_dom is not None:
                        desc_data.append(
                            {
                                "System": system,
                                "Entity Coverage (%)": (
                                    entity_cov * 100 if entity_cov else 0
                                ),
                                "Unique Domains": int(unique_dom) if unique_dom else 0,
                            }
                        )

            if desc_data:
                col1, col2, col3 = st.columns(3)

                with col1:
                    df_desc = pd.DataFrame(desc_data)
                    avg_entity = (
                        df_desc.groupby("System")["Entity Coverage (%)"]
                        .mean()
                        .reset_index()
                    )
                    fig = px.bar(
                        avg_entity,
                        x="System",
                        y="Entity Coverage (%)",
                        color="System",
                        title="Average Entity Coverage by System",
                        category_orders={"System": all_systems},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    avg_domains = (
                        df_desc.groupby("System")["Unique Domains"].mean().reset_index()
                    )
                    fig = px.bar(
                        avg_domains,
                        x="System",
                        y="Unique Domains",
                        color="System",
                        title="Average Unique Domains by System",
                        category_orders={"System": all_systems},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col3:
                    summary_desc = (
                        df_desc.groupby("System")
                        .agg(
                            {
                                "Entity Coverage (%)": ["mean", "min", "max"],
                                "Unique Domains": ["mean", "min", "max"],
                            }
                        )
                        .round(2)
                        .reindex(all_systems)
                    )
                    st.markdown("**Descriptive Summary**")
                    st.dataframe(summary_desc, use_container_width=True)

        st.markdown("---")

        # Score comparison
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Score Distribution")

            # Collect available scores
            score_data = []
            for query_id, sys_evals in evaluations.items():
                for system, eval_result in sys_evals.items():
                    for score_type in [
                        "ragas_score",
                        "huggingface_score",
                        "automated_score",
                    ]:
                        score = eval_result.scores.get(score_type)
                        if score is not None:
                            score_data.append(
                                {
                                    "System": system,
                                    "Score": score,
                                    "Score Type": score_type.replace("_", " ").title(),
                                }
                            )

            df_scores = pd.DataFrame(score_data)
            if not df_scores.empty:
                fig = px.box(
                    df_scores,
                    x="System",
                    y="Score",
                    color="Score Type",
                    title="Score Distribution by System",
                    facet_col="Score Type",
                    category_orders={"System": all_systems},
                )
                fig.for_each_annotation(lambda a: a.update(text=""))
                fig.for_each_xaxis(lambda axis: axis.update(title=""))
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Average Scores by System")

            avg_data = []
            for system in all_systems:
                for score_type in ["ragas_score", "huggingface_score", "automated_score"]:
                    scores = []
                    for query_id, sys_evals in evaluations.items():
                        if system in sys_evals:
                            s = sys_evals[system].scores.get(score_type)
                            if s is not None:
                                scores.append(s)
                    if scores:
                        avg_data.append(
                            {
                                "System": system,
                                "Score Type": score_type.replace("_", " ").title(),
                                "Average Score": sum(scores) / len(scores),
                            }
                        )

            if avg_data:
                df_avg = pd.DataFrame(avg_data)
                fig = px.bar(
                    df_avg,
                    x="System",
                    y="Average Score",
                    color="Score Type",
                    barmode="group",
                    title="Average Scores by System and Type",
                    category_orders={"System": all_systems},
                )
                st.plotly_chart(fig, use_container_width=True)

    # TAB 2: Query Explorer
    with tab2:
        st.subheader("🔍 Detailed Query Explorer")

        # Check if we have queries
        if not queries:
            st.warning("No queries found. Please load evaluation data first.")
        else:
            # Query selector
            query_options = {
                f"{i + 1}. {q.text[:80]}...": q for i, q in enumerate(queries)
            }
            selected_query_str = st.selectbox(
                "Select Query", options=list(query_options.keys())
            )

            if selected_query_str is None:
                st.info("Please select a query from the dropdown above.")
            else:
                selected_query = query_options[selected_query_str]

                if selected_query.id in evaluations:
                    st.markdown("---")

                    # Query details
                    st.markdown(f"### Query: {selected_query.text}")

                    # System comparison
                    sys_evals = evaluations[selected_query.id]

                    # Create tabs for each system
                    system_tabs = st.tabs(list(sys_evals.keys()))

                    for tab, (system, eval_result) in zip(
                        system_tabs, sys_evals.items()
                    ):
                        with tab:
                            col1, col2 = st.columns([2, 1])

                            with col1:
                                st.markdown("#### Response")
                                if responses and selected_query.id in responses:
                                    resp_data = responses[selected_query.id].get(
                                        system, {}
                                    )
                                    response_text = resp_data.get(
                                        "response_text", "No response"
                                    )
                                    st.markdown(response_text)

                                    st.markdown("#### Sources")
                                    sources = resp_data.get("sources", [])
                                    for i, source in enumerate(sources[:5], 1):
                                        url = source.get("url", "No URL")
                                        title = source.get("title", "Untitled")
                                        st.markdown(f"{i}. [{title}]({url})")
                                else:
                                    st.info("Load responses file to see response text")

                            with col2:
                                st.markdown("#### Scores")
                                score_cols = st.columns(3)
                                ragas_val = eval_result.scores.get("ragas_score")
                                hf_val = eval_result.scores.get("huggingface_score")
                                auto_val = eval_result.scores.get("automated_score")
                                with score_cols[0]:
                                    if ragas_val is not None:
                                        st.metric("Ragas", f"{ragas_val:.2f}")
                                with score_cols[1]:
                                    if hf_val is not None:
                                        st.metric("HF", f"{hf_val:.2f}")
                                with score_cols[2]:
                                    if auto_val is not None:
                                        st.metric("Auto", f"{auto_val:.2f}")

                                # Performance metrics
                                if responses and selected_query.id in responses:
                                    resp_data = responses[selected_query.id].get(
                                        system, {}
                                    )
                                    latency = resp_data.get("latency_ms")
                                    response_text = resp_data.get("response_text", "")

                                    if latency:
                                        st.markdown("#### Performance")
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            st.metric("Latency", f"{latency:.0f} ms")

                                        if response_text and latency > 0:
                                            words = len(response_text.split())
                                            tokens = int(words * 1.3)
                                            throughput = (tokens / latency) * 1000
                                            with col_b:
                                                st.metric(
                                                    "Throughput",
                                                    f"{throughput:.1f} tok/s",
                                                )

                                            st.caption(
                                                f"Output: {tokens} tokens"
                                                f" (~{words} words)"
                                            )

                                # Ragas scores
                                with st.expander("Ragas Metrics"):
                                    ragas_metrics = [
                                        "faithfulness",
                                        "completeness",
                                        "source_quality",
                                        "overall_quality",
                                    ]
                                    for key in ragas_metrics:
                                        if key in eval_result.scores:
                                            value = eval_result.scores[key]
                                            metric_name = key.replace("_", " ").title()
                                            if isinstance(value, (int, float)):
                                                st.markdown(
                                                    f"**{metric_name}:** {value:.1f}"
                                                )
                                            else:
                                                st.markdown(
                                                    f"**{metric_name}:** {value}"
                                                )

                                # Automated descriptive metrics
                                with st.expander(
                                    "📊 Descriptive Metrics (not in score)"
                                ):
                                    desc_metrics = [
                                        "automated_entity_coverage",
                                        "automated_unique_domains",
                                    ]
                                    for key in desc_metrics:
                                        if key in eval_result.scores:
                                            value = eval_result.scores[key]
                                            if key == "automated_entity_coverage":
                                                st.markdown(
                                                    "**Entity Coverage:**"
                                                    f" {value*100:.1f}%"
                                                )
                                            elif key == "automated_unique_domains":
                                                st.markdown(
                                                    "**Unique Domains:**"
                                                    f" {int(value)}"
                                                    " domains"
                                                )

                                # HuggingFace judges metrics
                                with st.expander("HuggingFace Judges Metrics"):
                                    hf_metrics = [
                                        "hf_correctness_classifier",
                                        "hf_correctness_grader",
                                        "hf_response_quality",
                                        "hf_overall_quality",
                                    ]
                                    has_hf = any(
                                        k in eval_result.scores for k in hf_metrics
                                    )

                                    if has_hf:
                                        for key in hf_metrics:
                                            if key in eval_result.scores:
                                                value = eval_result.scores[key]
                                                metric_name = (
                                                    key.replace("hf_", "HF ")
                                                    .replace("_", " ")
                                                    .title()
                                                )
                                                if isinstance(value, (int, float)):
                                                    st.markdown(
                                                        f"**{metric_name}:**"
                                                        f" {value:.1f}"
                                                    )
                                                else:
                                                    st.markdown(
                                                        f"**{metric_name}:** {value}"
                                                    )
                                    else:
                                        st.info(
                                            "No HuggingFace scores"
                                            " available. Run evaluation"
                                            " with --use-huggingface."
                                        )

                                # Automated metrics
                                with st.expander("Automated Metrics"):
                                    for key, value in eval_result.scores.items():
                                        if key.startswith("automated_"):
                                            metric_name = (
                                                key.replace("automated_", "")
                                                .replace("_", " ")
                                                .title()
                                            )
                                            if isinstance(value, (int, float)):
                                                st.markdown(
                                                    f"**{metric_name}:** {value:.2f}"
                                                )
                                            else:
                                                st.markdown(
                                                    f"**{metric_name}:** {value}"
                                                )
                else:
                    st.warning("No evaluation found for this query")

    # TAB 3: Analytics
    with tab3:
        st.subheader("📈 Advanced Analytics")

        # Metric comparison
        st.markdown("### Metric Comparison")

        metrics_to_compare = st.multiselect(
            "Select Metrics",
            options=[
                "faithfulness",
                "completeness",
                "source_quality",
                "overall_quality",
                "automated_bertscore_relevance",
                "automated_bertscore_faithfulness",
                "automated_entity_coverage",
                "automated_unique_domains",
            ],
            default=["faithfulness", "automated_bertscore_relevance"],
        )

        if metrics_to_compare:
            metric_data = []
            for query_id, sys_evals in evaluations.items():
                for system, eval_result in sys_evals.items():
                    for metric in metrics_to_compare:
                        if metric in eval_result.scores:
                            score_value = eval_result.scores[metric]
                            if isinstance(score_value, (int, float)):
                                metric_data.append(
                                    {
                                        "System": system,
                                        "Metric": metric.replace("automated_", "")
                                        .replace("_", " ")
                                        .title(),
                                        "Score": score_value,
                                    }
                                )

            if metric_data:
                df_metrics = pd.DataFrame(metric_data)
                fig = px.box(
                    df_metrics,
                    x="Metric",
                    y="Score",
                    color="System",
                    title="Metric Comparison Across Systems",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric data available for selected metrics.")

        # Latency vs Throughput Analysis
        if responses:
            st.markdown("### ⚡ Latency vs Throughput Analysis")

            perf_analysis = []
            for query_id, sys_responses in responses.items():
                for system, response_data in sys_responses.items():
                    latency = response_data.get("latency_ms")
                    response_text = response_data.get("response_text", "")

                    if latency and latency > 0:
                        words = len(response_text.split())
                        tokens = int(words * 1.3)
                        throughput = (tokens / latency) * 1000

                        quality_score = 0
                        if query_id in evaluations and system in evaluations[query_id]:
                            quality_score = evaluations[query_id][system].scores.get(
                                score_for_comparison, 0
                            )

                        perf_analysis.append(
                            {
                                "System": system,
                                "Query ID": query_id,
                                "Latency (ms)": latency,
                                "Throughput (tokens/s)": throughput,
                                "Output Tokens": tokens,
                                "Quality Score": quality_score,
                            }
                        )

            if perf_analysis:
                df_perf_analysis = pd.DataFrame(perf_analysis)

                col1, col2 = st.columns(2)

                with col1:
                    fig = px.scatter(
                        df_perf_analysis,
                        x="Latency (ms)",
                        y="Throughput (tokens/s)",
                        color="System",
                        size="Output Tokens",
                        hover_data=["Query ID", "Quality Score"],
                        title="Latency vs Throughput",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = px.scatter(
                        df_perf_analysis,
                        x="Latency (ms)",
                        y="Quality Score",
                        color="System",
                        hover_data=["Query ID", "Throughput (tokens/s)"],
                        title="Latency vs Quality Score",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Efficiency table
                st.markdown("**Efficiency Analysis (Quality per Latency Unit)**")
                df_perf_analysis["Efficiency"] = df_perf_analysis["Quality Score"] / (
                    df_perf_analysis["Latency (ms)"] / 1000
                )
                efficiency_summary = (
                    df_perf_analysis.groupby("System")
                    .agg(
                        {
                            "Latency (ms)": "mean",
                            "Throughput (tokens/s)": "mean",
                            "Quality Score": "mean",
                            "Efficiency": "mean",
                        }
                    )
                    .round(2)
                    .sort_values("Efficiency", ascending=False)
                )
                st.dataframe(efficiency_summary, use_container_width=True)

        st.markdown("---")

        # Score comparison across systems
        st.markdown(
            f"### Score Comparison Across Systems ({score_for_comparison})"
        )

        # Per-query score comparison
        comparison_data = []
        for query_id, sys_evals in evaluations.items():
            row = {"Query": query_id}
            for system, eval_result in sys_evals.items():
                row[system] = eval_result.scores.get(score_for_comparison, 0)
            comparison_data.append(row)

        if comparison_data:
            df_comp = pd.DataFrame(comparison_data)
            system_cols = [c for c in df_comp.columns if c != "Query"]

            fig = px.box(
                df_comp.melt(id_vars=["Query"], value_vars=system_cols,
                             var_name="System", value_name="Score"),
                x="System",
                y="Score",
                color="System",
                title=f"Score Distribution ({score_for_comparison})",
            )
            st.plotly_chart(fig, use_container_width=True)

    # TAB 4: Export
    with tab4:
        st.subheader("💾 Export Data")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Export Full Evaluation")

            if st.button("Generate Full Report"):
                full_export = []
                for query in queries:
                    if query.id in evaluations:
                        sys_evals = evaluations[query.id]
                        row = {
                            "query_id": query.id,
                            "query_text": query.text,
                        }

                        for system, eval_result in sys_evals.items():
                            final = eval_result.scores.get("final_score")
                            ragas = eval_result.scores.get("ragas_score")
                            hf = eval_result.scores.get("huggingface_score")
                            auto = eval_result.scores.get("automated_score")
                            if final is not None:
                                row[f"{system}_final"] = round(final, 2)
                            if ragas is not None:
                                row[f"{system}_ragas"] = round(ragas, 2)
                            if hf is not None:
                                row[f"{system}_hf"] = round(hf, 2)
                            if auto is not None:
                                row[f"{system}_auto"] = round(auto, 2)

                        full_export.append(row)

                df_full = pd.DataFrame(full_export)

                csv = df_full.to_csv(index=False)
                st.download_button(
                    label="Download Full CSV",
                    data=csv,
                    file_name="benchmark_full_evaluation.csv",
                    mime="text/csv",
                )

                st.success(
                    f"✅ Ready to download full evaluation"
                    f" for {len(full_export)} queries"
                )

        with col2:
            st.markdown("### Export Raw JSON")

            if st.button("Download Evaluations JSON"):
                with open(evaluations_file, "r") as f:
                    json_data = f.read()

                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="evaluations.json",
                    mime="application/json",
                )


if __name__ == "__main__":
    main()
