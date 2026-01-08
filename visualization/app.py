# visualization/app.py
"""
Streamlit Dashboard for Multi-Agent Learning System
Interactive visualization of profiles, paths, recommendations, and XAI
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.data_loader import DashboardDataLoader
from visualization.charts import (
    create_profile_radar,
    create_path_network,
    create_shap_waterfall,
    create_recommendation_bars
)

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Learning Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.big-metric {
    font-size: 24px;
    font-weight: bold;
    color: #1f77b4;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding-left: 20px;
    padding-right: 20px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'loader' not in st.session_state:
    st.session_state.loader = DashboardDataLoader()
    st.session_state.students_loaded = False

# Header
st.title("🎓 Multi-Agent Explainable Learning System")
st.markdown("**Real-time visualization of AI-powered personalized learning**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Load students button
    if not st.session_state.students_loaded:
        if st.button("🔄 Load Sample Students from OULAD", type="primary"):
            with st.spinner("Loading students and running pipeline..."):
                try:
                    st.session_state.loader.initialize_system()
                    student_ids = st.session_state.loader.load_sample_students(num_students=3)
                    st.session_state.students_loaded = True
                    st.session_state.student_ids = student_ids
                    st.success(f"✅ Loaded {len(student_ids)} students")
                except Exception as e:
                    st.error(f"❌ Error loading students: {e}")
    
    # Student selector
    if st.session_state.students_loaded:
        st.markdown("---")
        st.subheader("👤 Select Student")
        student_ids = st.session_state.loader.get_all_students()
        
        if student_ids:
            selected_student = st.selectbox(
                "Student ID",
                student_ids,
                format_func=lambda x: f"Student {x.split('_')[1]}" if '_' in x else x
            )
        else:
            st.warning("No students available")
            selected_student = None
    else:
        selected_student = None
        st.info("👆 Click the button above to load students")

# Main content
if selected_student:
    # Load student data
    profile = st.session_state.loader.get_student_profile(selected_student)
    learning_path = st.session_state.loader.get_learning_path(selected_student)
    recommendations = st.session_state.loader.get_recommendations(selected_student)
    explanations = st.session_state.loader.get_explanations(selected_student)
    
    # Check if data exists
    if not profile:
        st.error("❌ No profile data available for this student")
        st.stop()
    
    # Student header
    st.header(f"📊 Dashboard for {selected_student}")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📚 Level", profile.get('level', 'N/A').title())
    
    with col2:
        st.metric("🎨 Learning Style", profile.get('learning_style', 'N/A').title())
    
    with col3:
        path_length = len(learning_path.get('path', [])) if learning_path else 0
        st.metric("🗺️ Path Steps", path_length)
    
    with col4:
        rec_count = len(recommendations.get('recommendations', [])) if recommendations else 0
        st.metric("🎯 Recommendations", rec_count)
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["👤 Profile", "🗺️ Learning Path", "🎯 Recommendations", "🔍 XAI Explanations"])
    
    # Tab 1: Profile
    with tab1:
        st.subheader("Learner Profile")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📋 Profile Details")
            st.write(f"**User ID**: {selected_student}")
            st.write(f"**Proficiency Level**: {profile.get('level', 'N/A').title()}")
            st.write(f"**Learning Style**: {profile.get('learning_style', 'N/A').title()}")
            st.write(f"**Interests**: {', '.join(profile.get('interests', ['N/A']))}")
            st.write(f"**Strengths**: {', '.join(profile.get('strengths', ['N/A']))}")
            st.write(f"**Weaknesses**: {', '.join(profile.get('weaknesses', ['N/A']))}")
            
            # LLM Summary
            st.markdown("### 💬 AI-Generated Summary")
            st.info(profile.get('summary', 'No summary available'))
        
        with col2:
            st.markdown("### 📊 Knowledge Radar")
            fig_radar = create_profile_radar(profile)
            st.plotly_chart(fig_radar, width='stretch')
    
    # Tab 2: Learning Path
    with tab2:
        st.subheader("Personalized Learning Path")
        
        if learning_path and learning_path.get('path'):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 🗺️ Path Visualization")
                fig_path = create_path_network(learning_path)
                st.plotly_chart(fig_path, width='stretch')
            
            with col2:
                st.markdown("### 📝 Path Details")
                st.write(f"**Current Level**: {learning_path.get('current_level', 'N/A')}")
                st.write(f"**Target Level**: {learning_path.get('target_level', 'N/A')}")
                st.write(f"**Total Steps**: {learning_path.get('total_steps', 0)}")
                st.write(f"**Estimated Duration**: {learning_path.get('estimated_duration_minutes', 0)} min")
                
                st.markdown("---")
                st.markdown("**💡 AI Explanation**")
                st.caption(learning_path.get('explanation', 'No explanation available'))
            
            # Step-by-step list
            st.markdown("### 📚 Detailed Steps")
            for step in learning_path['path'][:10]:  # Show first 10
                with st.expander(f"Step {step['step']}: {step['title']}"):
                    st.write(f"**Type**: {step['type']}")
                    st.write(f"**Level**: {step['level']}")
                    st.write(f"**Duration**: {step['duration']} minutes")
                    st.write(f"**Topic**: {step.get('topic', 'N/A')}")
                    if step.get('prerequisites'):
                        st.write(f"**Prerequisites**: {', '.join(step['prerequisites'])}")
        else:
            st.info("No learning path generated for this student")
    
    # Tab 3: Recommendations
    with tab3:
        st.subheader("Top Recommendations")
        
        if recommendations and recommendations.get('recommendations'):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 📊 Priority Scores")
                fig_recs = create_recommendation_bars(recommendations)
                st.plotly_chart(fig_recs, width='stretch')
            
            with col2:
                st.markdown("### 📈 Statistics")
                st.write(f"**Total Candidates**: {recommendations.get('total_candidates', 0)}")
                st.write(f"**Sources**:")
                for source, count in recommendations.get('sources', {}).items():
                    st.write(f"  - {source}: {count}")
            
            # Detailed recommendations
            st.markdown("### 🎯 Detailed Recommendations")
            for i, rec in enumerate(recommendations['recommendations'], 1):
                with st.expander(f"#{i} - {rec['title']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Type**: {rec['type']}")
                        st.write(f"**Level**: {rec['level']}")
                        st.write(f"**Duration**: {rec.get('duration', 'N/A')} min")
                    with col2:
                        st.write(f"**Priority Score**: {rec.get('priority_score', 0):.2f}")
                        st.write(f"**Source**: {rec.get('source', 'N/A')}")
                    
                    st.markdown(f"**💡 Reason**: {rec.get('reason', 'N/A')}")
                    
                    if st.button(f"✅ Accept Recommendation #{i}", key=f"accept_{i}"):
                        st.success("Recommendation accepted!")
        else:
            st.info("No recommendations generated for this student")
    
    # Tab 4: XAI
    with tab4:
        st.subheader("Explainable AI Insights")
        
        if explanations:
            # SHAP Feature Importance
            st.markdown("### 📊 Feature Importance (SHAP)")
            fig_shap = create_shap_waterfall(explanations)
            st.plotly_chart(fig_shap, width='stretch')
            
            # Explanations sections
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🧠 Profile Explanation")
                profile_exp = explanations.get('profile_explanation', {})
                st.write(profile_exp.get('explanation', 'No explanation available'))
                
                st.markdown("### 🗺️ Path Explanation")
                path_exp = explanations.get('path_explanation', {})
                st.write(path_exp.get('explanation', 'No explanation available'))
            
            with col2:
                st.markdown("### 🎯 Recommendation Explanation")
                rec_exp = explanations.get('recommendation_explanation', {})
                st.write(rec_exp.get('explanation', 'No explanation available'))
            
            # Counterfactuals
            st.markdown("### 🔄 Counterfactual Scenarios")
            counterfactuals = explanations.get('counterfactuals', [])
            
            if counterfactuals:
                for i, cf in enumerate(counterfactuals, 1):
                    # Handle both string and dict formats
                    if isinstance(cf, dict):
                        with st.expander(f"💭 Scenario {i}: {cf.get('scenario', 'What-if')}"):
                            st.write(f"**Change**: {cf.get('change', 'N/A')}")
                            st.write(f"**Impact**: {cf.get('impact', 'N/A')}")
                    else:
                        # If it's a string
                        with st.expander(f"💭 Scenario {i}"):
                            st.write(cf)
            else:
                st.info("No counterfactual scenarios available")
            
            # Global summary
            st.markdown("### 📝 Global Summary")
            st.info(explanations.get('global_summary', 'No summary available'))
        else:
            st.info("No XAI explanations available for this student")

else:
    # Welcome screen
    st.info("👈 Load sample students from OULAD using the sidebar to get started")
    
    st.markdown("""
    ### 🎯 About This Dashboard
    
    This interactive dashboard visualizes the **Explainable Multi-Agent Generative Recommendation System** in action.
    
    **Features:**
    - 👤 **Learner Profiling**: AI-powered analysis of learning styles and proficiency levels
    - 🗺️ **Learning Path Planning**: Graph search + reinforcement learning for optimal paths
    - 🎯 **Smart Recommendations**: Hybrid filtering with LLM ranking
    - 🔍 **XAI Explanations**: SHAP, LIME, and counterfactual reasoning
    
    **Technologies:**
    - LLMs (Ollama), Embeddings, K-Means Clustering
    - A* Graph Search, Q-Learning
    - RAG (Retrieval-Augmented Generation)
    - SHAP, LIME
    
    **Dataset:** OULAD (Open University Learning Analytics Dataset)
    
    ---
    
    📊 **73 automated tests** validate the entire system.
    """)

# Footer
st.markdown("---")
st.caption("🎓 Multi-Agent Explainable Learning System | Powered by Streamlit & Ollama")
