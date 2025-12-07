"""
🏆 Judge Agent v3 - Streamlit Dashboard

Interactive UI for:
- Running agent evaluations
- Viewing evaluation history
- Comparing agents
- Analyzing dimension scores
"""

import os
import sys
import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge_agent_supervisor import (
    JudgeAgentSupervisor,
    AgentConfig,
    MockAgent,
    EvaluationDimension
)
from utils import create_audit_logger, AuditLogger


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Judge Agent v3",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stMetric label {
        color: rgba(255,255,255,0.8) !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
    }
    .pass-badge {
        background: #22c55e;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
    }
    .fail-badge {
        background: #ef4444;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
    }
    .dimension-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_async(coro):
    """Run async coroutine in Streamlit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def get_audit_logger() -> AuditLogger:
    """Get or create audit logger."""
    return create_audit_logger(local_only=True)


def score_color(score: float) -> str:
    """Get color based on score."""
    if score >= 0.9:
        return "#22c55e"  # Green
    elif score >= 0.7:
        return "#eab308"  # Yellow
    else:
        return "#ef4444"  # Red


def format_score(score: float) -> str:
    """Format score as percentage."""
    return f"{score:.1%}"


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🏆 Judge Agent v3")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🔬 Run Evaluation", "📊 Dashboard", "📈 History", "⚖️ Compare Agents"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Settings")
    
    aws_region = st.text_input("AWS Region", value="us-east-1")
    demo_mode = st.checkbox("Demo Mode (Mock Agent)", value=True)
    
    st.markdown("---")
    st.markdown("""
    **Judge Agent v3**  
    Enterprise AI Agent Evaluation  
    
    [📖 Documentation](https://github.com/rgslaughterjr/JudgeAgent_v3)
    """)


# ============================================================================
# RUN EVALUATION PAGE
# ============================================================================

if page == "🔬 Run Evaluation":
    st.header("🔬 Run Agent Evaluation")
    st.markdown("Evaluate an AI agent for production readiness across 8 dimensions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Agent Configuration")
        
        agent_id = st.text_input("Agent ID", value="agent-001")
        agent_name = st.text_input("Agent Name", value="Customer Service Bot")
        framework = st.selectbox(
            "Framework",
            ["langchain", "crewai", "autogen", "aws_bedrock", "langgraph", "llamaindex"]
        )
        risk_level = st.selectbox(
            "Risk Level",
            ["low", "medium", "high", "critical"],
            index=1
        )
    
    with col2:
        st.subheader("Additional Details")
        
        description = st.text_area(
            "Description",
            value="Handles customer inquiries and support requests"
        )
        data_access = st.text_area(
            "Data Access (one per line)",
            value="customer_database\nsupport_tickets\nknowledge_base"
        )
        evaluator_user = st.text_input("Evaluator Email", value="evaluator@company.com")
    
    st.markdown("---")
    
    # Dimension selection
    st.subheader("Evaluation Dimensions")
    
    dim_cols = st.columns(4)
    dimensions = list(EvaluationDimension)
    selected_dims = []
    
    for i, dim in enumerate(dimensions):
        with dim_cols[i % 4]:
            if st.checkbox(dim.value.replace("_", " ").title(), value=True, key=f"dim_{dim.value}"):
                selected_dims.append(dim)
    
    st.markdown("---")
    
    # Run evaluation button
    if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
        with st.spinner("Running evaluation... This may take a few minutes."):
            try:
                config = AgentConfig(
                    agent_id=agent_id,
                    name=agent_name,
                    framework=framework,
                    risk_level=risk_level,
                    description=description,
                    data_access=data_access.strip().split("\n") if data_access.strip() else []
                )
                
                # Use mock agent in demo mode
                connector = MockAgent()
                judge = JudgeAgentSupervisor(connector)
                
                # Run evaluation
                result = run_async(judge.evaluate_parallel(config, evaluator_user))
                
                # Store in session state
                st.session_state["last_evaluation"] = result
                st.session_state["last_config"] = config
                
                # Display results
                st.success("✅ Evaluation complete!")
                
                # Summary metrics
                st.markdown("### Results Summary")
                
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Overall Score", format_score(result["overall_score"]))
                with metric_cols[1]:
                    status = "✅ PASSED" if result["passes_gate"] else "❌ FAILED"
                    st.metric("Production Gate", status)
                with metric_cols[2]:
                    st.metric("Dimensions Tested", len(result["dimension_results"]))
                with metric_cols[3]:
                    total_tests = sum(d.get("tests_run", 0) for d in result["dimension_results"])
                    st.metric("Tests Run", total_tests)
                
                # Dimension scores chart
                st.markdown("### Dimension Scores")
                
                dim_data = []
                for dim_result in result["dimension_results"]:
                    dim_data.append({
                        "Dimension": dim_result["dimension"].replace("_", " ").title(),
                        "Score": dim_result["score"],
                        "Passed": "✅" if dim_result["passed"] else "❌",
                        "Tests": dim_result.get("tests_run", 0)
                    })
                
                df = pd.DataFrame(dim_data)
                
                fig = px.bar(
                    df,
                    x="Dimension",
                    y="Score",
                    color="Score",
                    color_continuous_scale=["red", "yellow", "green"],
                    range_color=[0, 1]
                )
                fig.add_hline(y=0.7, line_dash="dash", line_color="orange", annotation_text="Threshold")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results table
                st.markdown("### Detailed Results")
                st.dataframe(df, use_container_width=True)
                
                # Error tracking
                if result.get("error_tracking"):
                    st.markdown("### ⚠️ Errors Encountered")
                    st.json(result["error_tracking"])
                
            except Exception as e:
                st.error(f"❌ Evaluation failed: {str(e)}")
                st.exception(e)


# ============================================================================
# DASHBOARD PAGE
# ============================================================================

elif page == "📊 Dashboard":
    st.header("📊 Evaluation Dashboard")
    
    # Check for last evaluation
    if "last_evaluation" in st.session_state:
        result = st.session_state["last_evaluation"]
        config = st.session_state.get("last_config")
        
        # Header with agent info
        st.markdown(f"### Agent: {config.name if config else 'Unknown'}")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            score = result["overall_score"]
            st.metric(
                "Overall Score",
                format_score(score),
                delta=f"{'Above' if score >= 0.7 else 'Below'} threshold"
            )
        
        with col2:
            st.metric(
                "Production Ready",
                "Yes ✅" if result["passes_gate"] else "No ❌"
            )
        
        with col3:
            dims_passed = sum(1 for d in result["dimension_results"] if d["passed"])
            st.metric(
                "Dimensions Passed",
                f"{dims_passed}/{len(result['dimension_results'])}"
            )
        
        with col4:
            if config:
                st.metric("Risk Level", config.risk_level.upper())
        
        st.markdown("---")
        
        # Radar chart for dimensions
        st.subheader("Dimension Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create radar chart
            dim_names = [d["dimension"].replace("_", " ").title()[:15] for d in result["dimension_results"]]
            dim_scores = [d["score"] for d in result["dimension_results"]]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=dim_scores + [dim_scores[0]],  # Close the loop
                theta=dim_names + [dim_names[0]],
                fill='toself',
                name='Agent Score',
                line_color='#667eea'
            ))
            
            # Add threshold line
            fig.add_trace(go.Scatterpolar(
                r=[0.7] * (len(dim_names) + 1),
                theta=dim_names + [dim_names[0]],
                mode='lines',
                name='Threshold',
                line=dict(color='orange', dash='dash')
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=450
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Score Breakdown")
            for dim_result in result["dimension_results"]:
                dim_name = dim_result["dimension"].replace("_", " ").title()
                score = dim_result["score"]
                passed = dim_result["passed"]
                
                status = "✅" if passed else "❌"
                color = score_color(score)
                
                st.markdown(f"""
                <div style="margin-bottom: 0.5rem;">
                    <strong>{status} {dim_name}</strong><br>
                    <span style="color: {color}; font-size: 1.2rem; font-weight: bold;">
                        {format_score(score)}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        
    else:
        st.info("👆 Run an evaluation first to see the dashboard.")
        
        # Show sample dashboard
        st.markdown("### Sample Dashboard Preview")
        sample_data = {
            "Dimension": ["Security", "Privacy", "Accuracy", "Performance", "UX", "Bias", "Harm", "Guardrails"],
            "Score": [0.85, 0.78, 0.92, 0.88, 0.75, 0.90, 0.82, 0.79]
        }
        fig = px.bar(pd.DataFrame(sample_data), x="Dimension", y="Score", color="Score",
                     color_continuous_scale=["red", "yellow", "green"], range_color=[0, 1])
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# HISTORY PAGE
# ============================================================================

elif page == "📈 History":
    st.header("📈 Evaluation History")
    
    # Query parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        agent_filter = st.text_input("Filter by Agent ID", value="")
    with col2:
        days_filter = st.slider("Days to show", 1, 90, 30)
    with col3:
        passed_only = st.checkbox("Show passed only")
    
    # Load history
    try:
        audit_logger = get_audit_logger()
        
        entries = run_async(audit_logger.query_history(
            agent_id=agent_filter if agent_filter else None,
            passed_only=passed_only
        ))
        
        if entries:
            # Convert to DataFrame
            data = []
            for entry in entries:
                data.append({
                    "Timestamp": entry.timestamp,
                    "Agent ID": entry.agent_id,
                    "Agent Name": entry.agent_name,
                    "Score": entry.overall_score,
                    "Passed": "✅" if entry.passed else "❌",
                    "Framework": entry.framework,
                    "Risk Level": entry.risk_level,
                    "Evaluator": entry.evaluator_user
                })
            
            df = pd.DataFrame(data)
            
            # Summary stats
            st.markdown("### Summary")
            stat_cols = st.columns(4)
            
            with stat_cols[0]:
                st.metric("Total Evaluations", len(df))
            with stat_cols[1]:
                pass_rate = df["Passed"].str.contains("✅").sum() / len(df) if len(df) > 0 else 0
                st.metric("Pass Rate", f"{pass_rate:.1%}")
            with stat_cols[2]:
                avg_score = df["Score"].mean() if len(df) > 0 else 0
                st.metric("Avg Score", f"{avg_score:.1%}")
            with stat_cols[3]:
                unique_agents = df["Agent ID"].nunique()
                st.metric("Unique Agents", unique_agents)
            
            st.markdown("---")
            
            # Score trend chart
            st.subheader("Score Trend")
            
            # Convert timestamp to datetime for charting
            df["Date"] = pd.to_datetime(df["Timestamp"])
            
            fig = px.line(
                df.sort_values("Date"),
                x="Date",
                y="Score",
                color="Agent ID" if agent_filter == "" else None,
                markers=True
            )
            fig.add_hline(y=0.7, line_dash="dash", line_color="orange")
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.subheader("Evaluation Records")
            st.dataframe(df, use_container_width=True)
            
        else:
            st.info("No evaluation history found. Run some evaluations to see them here.")
            
    except Exception as e:
        st.warning(f"Could not load history: {e}")
        st.info("Run an evaluation first to generate history data.")


# ============================================================================
# COMPARE AGENTS PAGE
# ============================================================================

elif page == "⚖️ Compare Agents":
    st.header("⚖️ Compare Agents")
    st.markdown("Compare evaluation results between different agents.")
    
    # Load available agents from history
    try:
        audit_logger = get_audit_logger()
        entries = run_async(audit_logger.query_history())
        
        if entries:
            # Get unique agents
            agents = {}
            for entry in entries:
                if entry.agent_id not in agents:
                    agents[entry.agent_id] = entry
            
            if len(agents) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    agent1_id = st.selectbox(
                        "Agent A",
                        list(agents.keys()),
                        format_func=lambda x: f"{agents[x].agent_name} ({x})"
                    )
                
                with col2:
                    agent2_id = st.selectbox(
                        "Agent B",
                        [k for k in agents.keys() if k != agent1_id],
                        format_func=lambda x: f"{agents[x].agent_name} ({x})"
                    )
                
                if st.button("Compare", type="primary"):
                    agent1 = agents[agent1_id]
                    agent2 = agents[agent2_id]
                    
                    st.markdown("---")
                    
                    # Compare metrics side by side
                    st.subheader("Comparison")
                    
                    comp_cols = st.columns(3)
                    
                    with comp_cols[0]:
                        st.markdown(f"### {agent1.agent_name}")
                        st.metric("Score", f"{agent1.overall_score:.1%}")
                        st.metric("Status", "✅ Passed" if agent1.passed else "❌ Failed")
                        st.metric("Framework", agent1.framework)
                    
                    with comp_cols[1]:
                        st.markdown("### Comparison")
                        diff = agent1.overall_score - agent2.overall_score
                        st.metric("Score Difference", f"{abs(diff):.1%}", 
                                  delta=f"{'A better' if diff > 0 else 'B better'}")
                    
                    with comp_cols[2]:
                        st.markdown(f"### {agent2.agent_name}")
                        st.metric("Score", f"{agent2.overall_score:.1%}")
                        st.metric("Status", "✅ Passed" if agent2.passed else "❌ Failed")
                        st.metric("Framework", agent2.framework)
                    
                    # Dimension comparison chart
                    st.markdown("---")
                    st.subheader("Dimension Comparison")
                    
                    if agent1.dimension_scores and agent2.dimension_scores:
                        dim_data = []
                        for dim in agent1.dimension_scores:
                            dim_data.append({
                                "Dimension": dim.replace("_", " ").title(),
                                "Agent A": agent1.dimension_scores[dim].get("score", 0),
                                "Agent B": agent2.dimension_scores.get(dim, {}).get("score", 0)
                            })
                        
                        df = pd.DataFrame(dim_data)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(name=agent1.agent_name, x=df["Dimension"], y=df["Agent A"]))
                        fig.add_trace(go.Bar(name=agent2.agent_name, x=df["Dimension"], y=df["Agent B"]))
                        fig.update_layout(barmode='group', height=400)
                        fig.add_hline(y=0.7, line_dash="dash", line_color="orange")
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
            else:
                st.info("Need at least 2 agents to compare. Run more evaluations first.")
                
        else:
            st.info("No agents found. Run some evaluations first.")
            
    except Exception as e:
        st.warning(f"Could not load agents: {e}")
        st.info("Run evaluations first to compare agents.")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">
    Judge Agent v3 | Enterprise AI Agent Evaluation | 
    <a href="https://github.com/rgslaughterjr/JudgeAgent_v3" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)
