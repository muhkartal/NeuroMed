import streamlit as st
import pandas as pd
import numpy as np
import base64
from PIL import Image
import io

# Configure page
st.set_page_config(
    page_title="MedExplain AI Pro - Personal Health Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more enterprise look and feel
def add_custom_css():
    st.markdown("""
    <style>
        /* Overall theme improvements */
        :root {
            --primary-color: #4361EE;
            --primary-light: rgba(67, 97, 238, 0.1);
            --primary-dark: #3A56D4;
            --secondary-color: #3A0CA3;
            --accent-color: #F72585;
            --success-color: #4CC9B0;
            --warning-color: #F9C74F;
            --danger-color: #F94144;
            --text-color: #F8F9FA;
            --text-secondary: #CED4DA;
            --bg-dark: #121212;
            --bg-card: #1E1E1E;
            --border-color: #333;
        }

        /* Override default Streamlit theme */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        /* Dark mode styling */
        body {
            background-color: var(--bg-dark);
            color: var(--text-color);
        }

        /* Header styling */
        h1, h2, h3 {
            color: white;
            font-weight: 600;
        }

        h1 {
            font-size: 3rem !important;
            letter-spacing: -0.05em;
            margin-bottom: 1rem !important;
            padding-bottom: 1rem;
            background: linear-gradient(90deg, #4361EE, #7209B7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-block;
        }

        h2 {
            font-size: 2rem !important;
            margin-top: 2rem !important;
            margin-bottom: 1rem !important;
        }

        h3 {
            font-size: 1.5rem !important;
            color: var(--text-color);
            margin-top: 1.5rem !important;
            margin-bottom: 0.75rem !important;
        }

        /* Paragraph styling */
        p {
            color: var(--text-secondary);
            font-size: 1.1rem;
            line-height: 1.6;
        }

        /* Card styling */
        .feature-card {
            background-color: var(--bg-card);
            border-radius: 10px;
            padding: 1.5rem;
            border: 1px solid var(--border-color);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            height: 100%;
        }

        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
            border-color: var(--primary-color);
        }

        .feature-icon {
            background-color: var(--primary-light);
            color: var(--primary-color);
            width: 50px;
            height: 50px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1rem;
        }

        /* Action button styling */
        .action-button {
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
            text-align: center;
            margin-bottom: 1rem;
            display: block;
            text-decoration: none;
        }

        .action-button:hover {
            box-shadow: 0 5px 15px rgba(67, 97, 238, 0.3);
            filter: brightness(110%);
        }

        .action-button-secondary {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid var(--border-color);
        }

        .action-button-secondary:hover {
            background: rgba(255, 255, 255, 0.15);
            box-shadow: none;
        }

        /* Hero section */
        .hero-section {
            position: relative;
            padding: 2rem 0;
            overflow: hidden;
        }

        .hero-glow {
            position: absolute;
            top: -100px;
            right: -100px;
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, rgba(67, 97, 238, 0.2) 0%, rgba(67, 97, 238, 0) 70%);
            border-radius: 50%;
            z-index: -1;
        }

        /* Stats counter */
        .stats-box {
            background-color: var(--bg-card);
            border-radius: 10px;
            padding: 1.25rem;
            border: 1px solid var(--border-color);
            text-align: center;
            height: 100%;
        }

        .stats-number {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: 0.5rem;
        }

        .stats-label {
            color: var(--text-secondary);
            font-size: 0.875rem;
        }

        /* Section divider */
        .divider {
            border-top: 1px solid var(--border-color);
            margin: 3rem 0;
        }

        /* Badge styling */
        .badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            background-color: var(--primary-light);
            color: var(--primary-color);
            border-radius: 30px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        /* Footer styling */
        .footer {
            border-top: 1px solid var(--border-color);
            padding-top: 2rem;
            margin-top: 3rem;
            color: var(--text-secondary);
            font-size: 0.875rem;
        }

        /* Quick action buttons */
        .quick-action {
            background: linear-gradient(135deg, var(--bg-card), #2a2a2a);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1.25rem;
            transition: all 0.3s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .quick-action:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
            border-color: var(--primary-color);
        }

        .quick-action-icon {
            background-color: var(--primary-color);
            width: 40px;
            height: 40px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1rem;
        }

        .quick-action-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: white;
            margin-bottom: 0.5rem;
        }

        .quick-action-desc {
            color: var(--text-secondary);
            font-size: 0.875rem;
            margin-bottom: 1rem;
        }

        /* Dashboard preview section */
        .dashboard-preview {
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        .dashboard-header {
            background-color: var(--bg-card);
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
        }

        .window-controls {
            display: flex;
            gap: 6px;
            margin-right: 1rem;
        }

        .window-control {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }

        .window-control-red {
            background-color: #ff5f56;
        }

        .window-control-yellow {
            background-color: #ffbd2e;
        }

        .window-control-green {
            background-color: #27c93f;
        }

        /* Testimonial section */
        .testimonial {
            background-color: var(--bg-card);
            border-radius: 10px;
            padding: 1.5rem;
            border: 1px solid var(--border-color);
            height: 100%;
        }

        .testimonial-content {
            font-style: italic;
            color: var(--text-secondary);
            margin-bottom: 1rem;
        }

        .testimonial-author {
            font-weight: 600;
            color: white;
        }

        .testimonial-position {
            color: var(--text-secondary);
            font-size: 0.875rem;
        }

        /* Fix Streamlit UI elements */
        .stButton > button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.5rem 1.5rem !important;
            transition: all 0.3s ease !important;
        }

        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1) !important;
        }

        /* Make containers look nicer */
        [data-testid="stVerticalBlock"] {
            gap: 1rem !important;
        }

        /* Remove padding from expanders */
        .streamlit-expanderContent {
            padding: 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Function to create base64 encoded image placeholder
def get_base64_placeholder(width, height, color="#4361EE"):
    """Create a colored placeholder image of specified size."""
    img = Image.new('RGB', (width, height), color=color)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Generate a dashboard preview
def dashboard_preview():
    return f"""
    <div class="dashboard-preview">
        <div class="dashboard-header">
            <div class="window-controls">
                <div class="window-control window-control-red"></div>
                <div class="window-control window-control-yellow"></div>
                <div class="window-control window-control-green"></div>
            </div>
            <div style="color: var(--text-secondary); font-size: 14px;">MedExplain AI Pro Dashboard</div>
        </div>
        <div style="padding: 1rem; background-color: var(--bg-card);">
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                <div style="background-color: rgba(67, 97, 238, 0.1); border-radius: 8px; padding: 1rem; text-align: center;">
                    <div style="color: white; font-size: 14px;">Risk Score</div>
                    <div style="color: var(--primary-color); font-size: 24px; font-weight: 700;">Low</div>
                </div>
                <div style="background-color: rgba(67, 97, 238, 0.1); border-radius: 8px; padding: 1rem; text-align: center;">
                    <div style="color: white; font-size: 14px;">Health Index</div>
                    <div style="color: var(--success-color); font-size: 24px; font-weight: 700;">86%</div>
                </div>
                <div style="background-color: rgba(67, 97, 238, 0.1); border-radius: 8px; padding: 1rem; text-align: center;">
                    <div style="color: white; font-size: 14px;">Last Check</div>
                    <div style="color: var(--text-color); font-size: 18px; font-weight: 600;">2 days ago</div>
                </div>
            </div>
            <div style="background-color: #252525; height: 200px; border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                <img src="{get_base64_placeholder(400, 180, "#4361EE33")}" style="max-width: 100%; max-height: 100%; object-fit: contain;" />
            </div>
            <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 1rem;">
                <div style="background-color: #252525; border-radius: 8px; padding: 1rem;">
                    <h4 style="color: white; font-size: 16px; margin-bottom: 1rem;">Recent Symptoms</h4>
                    <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: var(--text-secondary);">Headache</span>
                            <span style="color: var(--warning-color);">Moderate</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: var(--text-secondary);">Fatigue</span>
                            <span style="color: var(--warning-color);">Moderate</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: var(--text-secondary);">Dizziness</span>
                            <span style="color: var(--success-color);">Mild</span>
                        </div>
                    </div>
                </div>
                <div style="background-color: #252525; border-radius: 8px; padding: 1rem;">
                    <h4 style="color: white; font-size: 16px; margin-bottom: 1rem;">Health Recommendations</h4>
                    <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <div style="width: 8px; height: 8px; border-radius: 50%; background-color: var(--primary-color);"></div>
                            <span style="color: var(--text-secondary);">Increase water intake to reduce headache frequency</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <div style="width: 8px; height: 8px; border-radius: 50%; background-color: var(--primary-color);"></div>
                            <span style="color: var(--text-secondary);">Consider sleep assessment to address fatigue</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <div style="width: 8px; height: 8px; border-radius: 50%; background-color: var(--primary-color);"></div>
                            <span style="color: var(--text-secondary);">Schedule follow-up health check in 2 weeks</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """

# Function to create a feature card
def feature_card(icon, title, description):
    return f"""
    <div class="feature-card">
        <div class="feature-icon">
            <span style="font-size: 1.5rem;">{icon}</span>
        </div>
        <h3>{title}</h3>
        <p>{description}</p>
    </div>
    """

# Function to create a quick action button
def quick_action(icon, title, description):
    return f"""
    <div class="quick-action">
        <div>
            <div class="quick-action-icon">
                <span style="font-size: 1.25rem; color: white;">{icon}</span>
            </div>
            <div class="quick-action-title">{title}</div>
            <div class="quick-action-desc">{description}</div>
        </div>
        <div>
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--primary-color);">
                <line x1="5" y1="12" x2="19" y2="12"></line>
                <polyline points="12 5 19 12 12 19"></polyline>
            </svg>
        </div>
    </div>
    """

# Apply custom CSS
add_custom_css()

# Main layout
def main():
    # Hero section
    st.markdown('<div class="hero-section">', unsafe_allow_html=True)
    st.markdown('<div class="hero-glow"></div>', unsafe_allow_html=True)

    # Title and subtitle
    st.markdown('<h1>Welcome to MedExplain AI Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.5rem; max-width: 800px;">Your advanced personal health assistant powered by artificial intelligence</p>', unsafe_allow_html=True)

    # Hero content
    hero_col1, hero_col2 = st.columns([3, 2])

    with hero_col1:
        st.markdown("""
        <div style="padding: 1.5rem 0;">
            <h2>Enterprise-Grade Healthcare Analytics at Your Fingertips</h2>
            <p style="margin-bottom: 2rem;">
                MedExplain AI Pro combines advanced medical knowledge with
                state-of-the-art artificial intelligence to help you understand
                and manage your health with unprecedented clarity and precision.
            </p>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 2rem;">
                <a href="#" class="action-button">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 8px;">
                        <circle cx="11" cy="11" r="8"></circle>
                        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                    </svg>
                    Analyze Symptoms
                </a>
                <a href="#" class="action-button action-button-secondary">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 8px;">
                        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                    </svg>
                    View Dashboard
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Key features
        st.markdown("""
        <div>
            <p style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--success-color); margin-right: 10px;">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                <span style="color: var(--text-secondary);">Analyze your symptoms with ensemble machine learning models</span>
            </p>
            <p style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--success-color); margin-right: 10px;">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                <span style="color: var(--text-secondary);">Track your health over time with interactive visualizations</span>
            </p>
            <p style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--success-color); margin-right: 10px;">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                <span style="color: var(--text-secondary);">Uncover patterns in your symptoms and health data</span>
            </p>
            <p style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--success-color); margin-right: 10px;">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                <span style="color: var(--text-secondary);">Identify potential risks based on your comprehensive health profile</span>
            </p>
            <p style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--success-color); margin-right: 10px;">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                <span style="color: var(--text-secondary);">Chat naturally about medical topics with our AI assistant</span>
            </p>
            <p style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--success-color); margin-right: 10px;">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                <span style="color: var(--text-secondary);">Access medical literature summarized in plain language</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with hero_col2:
        st.markdown(dashboard_preview(), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Stats section
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

    with stats_col1:
        st.markdown("""
        <div class="stats-box">
            <div class="stats-number">93%</div>
            <div class="stats-label">Symptom Detection Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with stats_col2:
        st.markdown("""
        <div class="stats-box">
            <div class="stats-number">5,000+</div>
            <div class="stats-label">Users Trust MedExplain</div>
        </div>
        """, unsafe_allow_html=True)

    with stats_col3:
        st.markdown("""
        <div class="stats-box">
            <div class="stats-number">200+</div>
            <div class="stats-label">Medical Conditions Analyzed</div>
        </div>
        """, unsafe_allow_html=True)

    with stats_col4:
        st.markdown("""
        <div class="stats-box">
            <div class="stats-number">24/7</div>
            <div class="stats-label">Health Insights Available</div>
        </div>
        """, unsafe_allow_html=True)

    # Key features section
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="badge">ENTERPRISE FEATURES</div>', unsafe_allow_html=True)
    st.markdown('<h2>Key Advanced Features</h2>', unsafe_allow_html=True)
    st.markdown('<p style="max-width: 800px; margin-bottom: 2rem;">MedExplain AI Pro delivers enterprise-grade healthcare analytics through innovative features powered by advanced machine learning algorithms.</p>', unsafe_allow_html=True)

    feature_col1, feature_col2, feature_col3 = st.columns(3)

    with feature_col1:
        st.markdown(feature_card(
            "🧠",
            "AI-Powered Analysis",
            "Ensemble ML models analyze your health data using multiple algorithms for more accurate insights."
        ), unsafe_allow_html=True)

        st.markdown(feature_card(
            "🔍",
            "Pattern Recognition",
            "Advanced algorithms identify correlations in your symptoms to recognize potential health patterns."
        ), unsafe_allow_html=True)

    with feature_col2:
        st.markdown(feature_card(
            "📊",
            "Interactive Health Dashboard",
            "Comprehensive visualizations of your health patterns and trends with drill-down capabilities."
        ), unsafe_allow_html=True)

        st.markdown(feature_card(
            "📈",
            "Predictive Insights",
            "Risk assessment and early warning indicators based on your health trends and medical research."
        ), unsafe_allow_html=True)

    with feature_col3:
        st.markdown(feature_card(
            "💬",
            "Medical NLP Interface",
            "Discuss your health in plain language with our AI that understands medical context."
        ), unsafe_allow_html=True)

        st.markdown(feature_card(
            "🔒",
            "Enterprise Security",
            "HIPAA-compliant data encryption and privacy protection for your sensitive health information."
        ), unsafe_allow_html=True)

    # Quick actions section
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2>Quick Actions</h2>', unsafe_allow_html=True)

    action_col1, action_col2, action_col3, action_col4 = st.columns(4)

    with action_col1:
        st.markdown(quick_action(
            "🔍",
            "Analyze Symptoms",
            "Start a symptom analysis to assess your health"
        ), unsafe_allow_html=True)

    with action_col2:
        st.markdown(quick_action(
            "💬",
            "Health Chat",
            "Chat with the AI about your health concerns"
        ), unsafe_allow_html=True)

    with action_col3:
        st.markdown(quick_action(
            "📊",
            "View Dashboard",
            "See your health trends and analytics dashboard"
        ), unsafe_allow_html=True)

    with action_col4:
        st.markdown(quick_action(
            "📈",
            "Advanced Analytics",
            "Explore detailed analytics about your health data"
        ), unsafe_allow_html=True)

    # Call to action section
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: linear-gradient(90deg, rgba(67, 97, 238, 0.1), rgba(58, 12, 163, 0.1)); border: 1px solid rgba(67, 97, 238, 0.2); border-radius: 12px; padding: 2rem; text-align: center;">
        <h2 style="margin-top: 0;">Ready to Transform Your Healthcare Experience?</h2>
        <p style="max-width: 700px; margin: 0 auto 1.5rem auto;">
            Take control of your health with powerful AI-driven insights and personalized recommendations.
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem;">
            <a href="#" class="action-button" style="max-width: 200px; margin: 0 auto;">
                Get Started Now
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <div style="background-color: var(--primary-color); width: 24px; height: 24px; border-radius: 4px; display: flex; align-items: center; justify-content: center; margin-right: 8px;">
                        💊
                    </div>
                    <strong style="color: white;">MedExplain AI Pro</strong>
                </div>
                <p style="margin: 0; font-size: 0.75rem;">© 2025 MedExplain AI. All rights reserved.</p>
            </div>

            <div>
                <div style="display: flex; gap: 1.5rem;">
                    <a href="#" style="color: var(--text-secondary); text-decoration: none;">Terms</a>
                    <a href="#" style="color: var(--text-secondary); text-decoration: none;">Privacy</a>
                    <a href="#" style="color: var(--text-secondary); text-decoration: none;">Contact</a>
                    <a href="#" style="color: var(--text-secondary); text-decoration: none;">Help</a>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Medical disclaimer
    st.markdown("""
    <div style="margin-top: 1rem; padding: 0.75rem; background-color: rgba(255, 255, 255, 0.05); border-radius: 6px; font-size: 0.75rem; color: var(--text-secondary);">
        <strong>MEDICAL DISCLAIMER:</strong> MedExplain AI Pro is for educational purposes only. Always consult healthcare professionals for medical advice, diagnosis, or treatment.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
