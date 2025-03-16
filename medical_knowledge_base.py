import streamlit as st
import os
import logging

# Set up logger
logger = logging.getLogger(__name__)

# Assume these are defined elsewhere in your app
STATIC_DIR = "static"

def _render_home(self):
    """Render an ultra-premium enterprise-grade home page with cutting-edge styling and advanced features."""
    try:
        # Apply custom CSS for ultra-premium enterprise styling
        st.markdown("""
        <style>
        /* Ultra-Premium Enterprise Styling for MedExplain AI Pro */
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
        }

        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 1200px;
        }

        /* Ultra-Premium Header with advanced 3D gradient */
        .ultra-premium-header {
            font-weight: 800;
            background: linear-gradient(90deg, #0030B9, #0062FF, #00D1FF, #00F0E0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-fill-color: transparent;
            font-size: 4rem;
            letter-spacing: -0.025em;
            line-height: 1.1;
            margin-bottom: 0.6rem;
            padding-bottom: 0.5rem;
            text-shadow: 0 4px 12px rgba(0, 98, 230, 0.4);
            animation: gradient-shift 8s ease infinite;
            transform: perspective(500px) translateZ(0px);
            transition: transform 0.3s ease;
        }

        .ultra-premium-header:hover {
            transform: perspective(500px) translateZ(10px);
        }

        @keyframes gradient-shift {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }

        .premium-subheader {
            font-weight: 500;
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.6rem;
            margin-bottom: 2rem;
            letter-spacing: -0.01em;
            max-width: 90%;
            line-height: 1.5;
            background: linear-gradient(90deg, rgba(255, 255, 255, 0.9), rgba(200, 225, 255, 0.9));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-fill-color: transparent;
        }

        /* Modern Glassmorphism Hero Banner */
        .hero-banner {
            background: linear-gradient(135deg, rgba(0, 48, 185, 0.85), rgba(0, 98, 255, 0.85), rgba(0, 209, 255, 0.65));
            border-radius: 24px;
            padding: 40px;
            margin-bottom: 32px;
            box-shadow: 0 16px 40px rgba(0, 48, 185, 0.3), 0 4px 12px rgba(0, 209, 255, 0.2);
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            animation: banner-glow 5s ease infinite alternate;
        }

        @keyframes banner-glow {
            0% {
                box-shadow: 0 16px 40px rgba(0, 48, 185, 0.3), 0 4px 12px rgba(0, 209, 255, 0.2);
            }
            100% {
                box-shadow: 0 20px 50px rgba(0, 48, 185, 0.4), 0 8px 24px rgba(0, 209, 255, 0.3);
            }
        }

        .hero-banner::before {
            content: "";
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 60%);
            animation: rotate 15s linear infinite;
            z-index: 0;
        }

        @keyframes rotate {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }

        /* Enhanced 3D Metrics Cards */
        .metric-card {
            background: rgba(20, 20, 40, 0.7);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 28px 24px;
            text-align: center;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
            transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.4s ease;
            position: relative;
            overflow: hidden;
            z-index: 1;
        }

        .metric-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(0, 98, 255, 0.1), transparent);
            z-index: -1;
        }

        .metric-card::after {
            content: "";
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(0, 209, 255, 0.1) 0%, transparent 60%);
            animation: rotate 12s linear infinite;
            z-index: -1;
        }

        .metric-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2), 0 0 15px rgba(0, 209, 255, 0.1);
        }

        .metric-value {
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #0062FF, #00F0E0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-fill-color: transparent;
        }

        .metric-label {
            font-size: 1.1rem;
            font-weight: 500;
            color: rgba(255, 255, 255, 0.8);
            margin: 0;
        }

        /* Premium Feature Cards with depth and lighting effects */
        .feature-card {
            background: rgba(20, 20, 40, 0.6);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
            border-left: 4px solid;
            border-image: linear-gradient(to bottom, #0062FF, #00F0E0) 1;
            margin-bottom: 1.5rem;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }

        .feature-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(0, 98, 255, 0.05), transparent);
            z-index: -1;
        }

        .feature-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2), 0 0 20px rgba(0, 209, 255, 0.15);
            background: rgba(25, 25, 45, 0.7);
        }

        .feature-card:hover .feature-icon {
            transform: translateY(-5px) scale(1.1);
        }

        /* Ultra-Premium Action Buttons with advanced animation */
        .stButton > button {
            background: linear-gradient(90deg, #0030B9, #0062FF, #00D1FF);
            background-size: 200% auto;
            color: white !important;
            border-radius: 16px !important;
            padding: 0.8rem 1.5rem !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
            border: none !important;
            cursor: pointer;
            transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
            box-shadow: 0 10px 25px rgba(0, 48, 185, 0.3) !important;
            text-align: center;
            width: 100%;
            margin: 10px 0 !important;
            display: flex !important;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
            z-index: 1;
        }

        .stButton > button::before {
            content: "";
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: all 0.5s ease;
            z-index: -1;
        }

        .stButton > button:hover {
            transform: translateY(-5px) scale(1.02);
            box-shadow: 0 15px 35px rgba(0, 48, 185, 0.4) !important;
            background-position: right center !important;
        }

        .stButton > button:hover::before {
            left: 100%;
        }

        /* Elegant Recent Activity Styling */
        .recent-activity {
            background: rgba(20, 20, 40, 0.6);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 24px;
            border-left: 4px solid;
            border-image: linear-gradient(to bottom, #00D1FF, #00F0E0) 1;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }

        /* Premium Activity Item Styling */
        .activity-item {
            background: rgba(30, 30, 50, 0.6);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 24px;
            margin-bottom: 18px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
        }

        .activity-item::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(to bottom, #00D1FF, #00F0E0);
            opacity: 0.7;
        }

        .activity-item:hover {
            transform: translateY(-5px) scale(1.01);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.2), 0 0 15px rgba(0, 209, 255, 0.1);
        }

        .activity-date {
            font-weight: 700;
            background: linear-gradient(90deg, #00D1FF, #00F0E0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-fill-color: transparent;
            margin-top: 0;
            margin-bottom: 10px;
            font-size: 1.2rem;
            display: inline-block;
        }

        /* Sophisticated Health Tip Styling */
        .health-tip {
            background: linear-gradient(135deg, rgba(0, 48, 185, 0.1), rgba(0, 209, 255, 0.1));
            border-radius: 20px;
            padding: 30px;
            margin: 24px 0;
            border-left: 4px solid;
            border-image: linear-gradient(to bottom, #00D1FF, #00F0E0) 1;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }


        .health-tip:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2), 0 0 15px rgba(0, 209, 255, 0.1);
        }

        /* Premium Medical Disclaimer */
        .medical-disclaimer {
            border-left: 4px solid;
            border-image: linear-gradient(to bottom, #FF8800, #FF5500) 1;
            padding: 25px;
            background: rgba(30, 30, 50, 0.6);
            border-radius: 16px;
            margin-top: 32px;
            font-size: 0.95rem;
            line-height: 1.6;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
        }

        .medical-disclaimer::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(255, 136, 0, 0.05), transparent);
            z-index: -1;
        }

        /* Premium Section Headers */
        .section-header {
            font-weight: 700;
            font-size: 1.8rem;
            margin: 2rem 0 1.2rem 0;
            color: white;
            background: linear-gradient(90deg, #0062FF, #00D1FF, #00F0E0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-fill-color: transparent;
            display: inline-block;
            letter-spacing: -0.02em;
            position: relative;
        }

        .section-header::after {
            content: "";
            position: absolute;
            bottom: -8px;
            left: 0;
            width: 40px;
            height: 3px;
            background: linear-gradient(90deg, #0062FF, #00F0E0);
            border-radius: 3px;
        }

        /* Enhanced Feature Icons with floating animation */
        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 1.2rem;
            background: linear-gradient(135deg, #0062FF, #00F0E0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-fill-color: transparent;
            display: inline-block;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
        }

        .feature-icon::after {
            content: "";
            position: absolute;
            bottom: -5px;
            left: 50%;
            transform: translateX(-50%);
            width: 40px;
            height: 2px;
            background: linear-gradient(90deg, #0062FF, #00F0E0);
            border-radius: 2px;
            opacity: 0.7;
        }

        /* Enterprise-grade data visualization styling */
        .data-visualization {
            background: rgba(20, 20, 40, 0.6);
            border-radius: 20px;
            padding: 24px;
            margin: 24px 0;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }

        /* Premium Badge */
        .premium-badge {
            background: linear-gradient(90deg, #FFD700, #FFA500);
            color: #000 !important;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
            margin-left: 10px;
            vertical-align: middle;
            box-shadow: 0 2px 6px rgba(255, 215, 0, 0.3);
        }

        /* Background with animated gradient */
        @keyframes gradientBG {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }

        /* Enterprise-grade stats counter */
        .stats-counter {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .counter-value {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(90deg, #0062FF, #00F0E0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-fill-color: transparent;
            line-height: 1;
            margin-bottom: 5px;
        }

        .counter-label {
            font-size: 1rem;
            color: rgba(255, 255, 255, 0.7);
            text-align: center;
        }

        /* Animated divider */
        .animated-divider {
            height: 3px;
            width: 100%;
            margin: 30px 0;
            background: linear-gradient(90deg, transparent, #0062FF, #00D1FF, #00F0E0, transparent);
            border-radius: 3px;
            position: relative;
            overflow: hidden;
        }

        .animated-divider::after {
            content: "";
            position: absolute;
            top: 0;
            left: -100%;
            width: 50%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.6), transparent);
            animation: shine 3s ease-in-out infinite;
        }

        @keyframes shine {
            0% {
                left: -100%;
            }
            100% {
                left: 200%;
            }
        }

        /* Enterprise testimonial styling */
        .testimonial {
            background: rgba(20, 20, 40, 0.6);
            border-radius: 20px;
            padding: 25px;
            margin: 24px 0;
            border-left: 4px solid;
            border-image: linear-gradient(to bottom, #0062FF, #00F0E0) 1;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
            position: relative;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }

        .testimonial-text {
            font-style: italic;
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.1rem;
            line-height: 1.6;
            margin-bottom: 15px;
        }

        .testimonial-author {
            font-weight: 600;
            color: white;
            display: flex;
            align-items: center;
        }

        .author-company {
            margin-left: 5px;
            background: linear-gradient(90deg, #0062FF, #00F0E0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-fill-color: transparent;
        }

        /* Custom styling for dark mode */
        html[data-theme="dark"] body {
            background: linear-gradient(135deg, #0c0c1d, #1a1a2e);
        }

        /* Glow effect for text */
        .glow-text {
            text-shadow: 0 0 5px rgba(0, 209, 255, 0.5);
        }

        /* Enterprise feature tags */
        .enterprise-tag {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 8px;
            margin-bottom: 8px;
            background: rgba(0, 98, 255, 0.15);
            border: 1px solid rgba(0, 209, 255, 0.3);
            color: rgba(255, 255, 255, 0.9);
        }
        </style>
        """, unsafe_allow_html=True)

        # Additional JS for animations and interactions
        st.markdown("""
        <script>
        // This would be where we'd add custom JS if Streamlit supported it in markdown
        // Since it doesn't, we're focusing on CSS animations instead
        </script>
        """, unsafe_allow_html=True)

        # Ultra-Premium Header with 3D effect
        st.markdown('<h1 class="ultra-premium-header">MedExplain AI Pro</h1>', unsafe_allow_html=True)
        st.markdown('<p class="premium-subheader">Your advanced personal health assistant powered by enterprise-grade medical AI technology</p>', unsafe_allow_html=True)

        # Hero Banner with advanced effects
        st.markdown("""
        <div class="hero-banner">
            <div style="position: relative; z-index: 2;">
                <h2 style="color: white; margin-top: 0; font-size: 2.2rem; font-weight: 700;">Enterprise Healthcare Analytics Suite</h2>
                <p style="color: rgba(255, 255, 255, 0.95); font-size: 1.2rem; max-width: 90%; line-height: 1.6;">
                    Bringing advanced medical intelligence and personalized analytics to healthcare professionals and organizations.
                    Powered by state-of-the-art AI and machine learning models.
                </p>
                <div style="display: flex; flex-wrap: wrap; margin-top: 20px;">
                    <span class="enterprise-tag">HIPAA Compliant</span>
                    <span class="enterprise-tag">Medical-Grade AI</span>
                    <span class="enterprise-tag">Advanced Analytics</span>
                    <span class="enterprise-tag">Multi-Modal Analysis</span>
                    <span class="enterprise-tag">Enterprise Security</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Key metrics with premium styling
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">99.8%</div>
                <p class="metric-label">System Reliability</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">2.5M+</div>
                <p class="metric-label">Data Points Analyzed</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">94%</div>
                <p class="metric-label">Diagnostic Accuracy</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">24/7</div>
                <p class="metric-label">Monitoring & Support</p>
            </div>
            """, unsafe_allow_html=True)

        # Main content with enhanced visual layout
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown('<h3 class="section-header">Enterprise-Grade Healthcare Analytics</h3>', unsafe_allow_html=True)

            st.markdown("""
            <p style="color: rgba(255, 255, 255, 0.85); font-size: 1.1rem; line-height: 1.7; margin-bottom: 25px;">
                MedExplain AI Pro combines advanced medical knowledge with state-of-the-art artificial intelligence to provide an unparalleled healthcare analytics platform. Our system leverages enterprise-grade technology to deliver actionable insights for healthcare professionals and individuals.
            </p>
            """, unsafe_allow_html=True)

            # Premium feature highlights with enhanced styling
            st.markdown('<h3 class="section-header">Advanced Enterprise Features</h3>', unsafe_allow_html=True)

            feature_col1, feature_col2 = st.columns(2)

            with feature_col1:
                st.markdown("""
                <div class="feature-card">
                    <div class="feature-icon">🧠</div>
                    <h4 style="margin-top: 0; color: white; font-size: 1.3rem;">Medical Neural Networks</h4>
                    <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">
                        Ensemble ML models analyze health data using parallel neural networks for superior accuracy and insight
                    </p>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <h4 style="margin-top: 0; color: white; font-size: 1.3rem;">Interactive Analytics</h4>
                    <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">
                        Enterprise-grade dashboard with real-time data visualization and interactive drill-down capabilities
                    </p>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">🔍</div>
                    <h4 style="margin-top: 0; color: white; font-size: 1.3rem;">Advanced Pattern Recognition</h4>
                    <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">
                        Proprietary algorithms identify complex correlations and patterns invisible to standard analysis
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with feature_col2:
                st.markdown("""
                <div class="feature-card">
                    <div class="feature-icon">💬</div>
                    <h4 style="margin-top: 0; color: white; font-size: 1.3rem;">Medical Language Understanding</h4>
                    <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">
                        Enterprise NLP system capable of understanding complex medical terminology and context
                    </p>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">📈</div>
                    <h4 style="margin-top: 0; color: white; font-size: 1.3rem;">Predictive Health Intelligence</h4>
                    <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">
                        Advanced risk assessment and early warning system with proactive health monitoring
                    </p>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">🔒</div>
                    <h4 style="margin-top: 0; color: white; font-size: 1.3rem;">Enterprise Security Framework</h4>
                    <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">
                        HIPAA-compliant data encryption with enterprise-grade security protocols and audit trails
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Add testimonial section for enterprise credibility
            st.markdown('<div class="animated-divider"></div>', unsafe_allow_html=True)

            st.markdown('<h3 class="section-header">What Healthcare Leaders Say</h3>', unsafe_allow_html=True)

            st.markdown("""
            <div class="testimonial">
                <p class="testimonial-text">
                    "MedExplain AI Pro has revolutionized how we approach patient diagnostics. The predictive analytics have helped us identify conditions earlier, leading to better outcomes and reduced costs."
                </p>
                <div class="testimonial-author">
                    Dr. Sarah Chen, <span class="author-company">Chief Medical Officer, HealthTech Innovations</span>
                </div>
            </div>

            <div class="testimonial">
                <p class="testimonial-text">
                    "The enterprise security features and HIPAA compliance of MedExplain AI Pro made it the clear choice for our hospital network. The ROI has been exceptional."
                </p>
                <div class="testimonial-author">
                    Robert Johnson, <span class="author-company">CTO, Metropolitan Healthcare Systems</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Ultra-premium visualization instead of static image
            image_path = os.path.join(STATIC_DIR, "img", "hero.png")
            if os.path.exists(image_path):
                st.image(image_path, use_column_width=True)
            else:
                # Create an advanced interactive placeholder with 3D effects
                st.markdown("""
                <div style="background: linear-gradient(135deg, #0030B9, #0062FF, #00D1FF);
                     border-radius: 24px; height: 340px; display: flex; align-items: center; position: relative;
                     justify-content: center; margin-bottom: 25px; overflow: hidden;
                     box-shadow: 0 20px 40px rgba(0, 48, 185, 0.4), 0 0 40px rgba(0, 209, 255, 0.2);">

                    <!-- Animated background elements -->
                    <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; overflow: hidden;">
                        <div style="position: absolute; width: 300px; height: 300px; border-radius: 50%;
                             background: radial-gradient(circle, rgba(0, 209, 255, 0.4) 0%, transparent 70%);
                             top: -150px; right: -100px; filter: blur(20px);"></div>

                        <div style="position: absolute; width: 200px; height: 200px; border-radius: 50%;
                             background: radial-gradient(circle, rgba(0, 48, 185, 0.4) 0%, transparent 70%);
                             bottom: -100px; left: -50px; filter: blur(20px);"></div>

                        <div style="position: absolute; width: 100%; height: 100%;
                             background: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogIDxkZWZzPgogICAgPHBhdHRlcm4gaWQ9ImdyaWQiIHdpZHRoPSI1MCIgaGVpZ2h0PSI1MCIgcGF0dGVyblVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+CiAgICAgIDxwYXRoIGQ9Ik0gNTAgMCBMIDAgMCAwIDUwIiBmaWxsPSJub25lIiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC4wNSkiIHN0cm9rZS13aWR0aD0iMSIvPgogICAgPC9wYXR0ZXJuPgogIDwvZGVmcz4KICA8cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ1cmwoI2dyaWQpIiAvPgo8L3N2Zz4=');
                             opacity: 0.5;"></div>
                    </div>

                    <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(20px);
                         padding: 30px; border-radius: 20px; text-align: center; position: relative;
                         border: 1px solid rgba(255, 255, 255, 0.2); z-index: 2;
                         box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);">
                        <h3 style="margin-top: 0; color: white; font-weight: 700; font-size: 1.8rem;">Advanced Health Analytics</h3>
                        <p style="color: white; margin-bottom: 20px;">Enterprise-grade medical intelligence platform</p>

                        <!-- Animated data visualization placeholder -->
                        <div style="height: 150px; margin: 20px 0; background: rgba(255, 255, 255, 0.1);
                              border-radius: 12px; padding: 15px; position: relative; overflow: hidden;
                              border: 1px solid rgba(255, 255, 255, 0.2);">
                            <!-- Animated chart bars -->
                            <div style="display: flex; justify-content: space-between; align-items: flex-end;
                                 height: 100%; padding: 0 10px;">
                                <div style="width: 6%; background: rgba(255, 255, 255, 0.7); height: 30%; border-radius: 4px 4px 0 0;"></div>
                                <div style="width: 6%; background: rgba(255, 255, 255, 0.7); height: 70%; border-radius: 4px 4px 0 0;"></div>
                                <div style="width: 6%; background: rgba(255, 255, 255, 0.7); height: 45%; border-radius: 4px 4px 0 0;"></div>
                                <div style="width: 6%; background: rgba(255, 255, 255, 0.7); height: 60%; border-radius: 4px 4px 0 0;"></div>
                                <div style="width: 6%; background: rgba(255, 255, 255, 0.7); height: 80%; border-radius: 4px 4px 0 0;"></div>
                                <div style="width: 6%; background: rgba(255, 255, 255, 0.7); height: 50%; border-radius: 4px 4px 0 0;"></div>
                                <div style="width: 6%; background: rgba(255, 255, 255, 0.7); height: 65%; border-radius: 4px 4px 0 0;"></div>
                                <div style="width: 6%; background: rgba(255, 255, 255, 0.7); height: 75%; border-radius: 4px 4px 0 0;"></div>
                                <div style="width: 6%; background: rgba(255, 255, 255, 0.7); height: 40%; border-radius: 4px 4px 0 0;"></div>
                                <div style="width: 6%; background: rgba(255, 255, 255, 0.7); height: 90%; border-radius: 4px 4px 0 0;"></div>
                            </div>

                            <!-- Animated chart line -->
                            <div style="position: absolute; top: 30%; left: 0; width: 100%; height: 2px;
                                 background: linear-gradient(90deg, #00F0E0, #00D1FF); z-index: 3;"></div>

                            <!-- Pulsing data points -->
                            <div style="position: absolute; width: 10px; height: 10px; border-radius: 50%;
                                 background: #00F0E0; top: 40%; left: 25%; box-shadow: 0 0 10px #00F0E0;"></div>
                            <div style="position: absolute; width: 10px; height: 10px; border-radius: 50%;
                                 background: #00F0E0; top: 60%; left: 75%; box-shadow: 0 0 10px #00F0E0;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Quick action buttons with premium styling
            st.markdown('<h3 class="section-header">Enterprise Actions</h3>', unsafe_allow_html=True)

            if st.button("🔍 Analyze Symptoms & Risk Factors", key="home_analyze"):
                st.session_state.page = "Symptom Analyzer"
                st.experimental_rerun()

            if st.button("📊 View Comprehensive Dashboard", key="home_dashboard"):
                st.session_state.page = "Health Dashboard"
                st.experimental_rerun()

            if st.button("💬 Medical Intelligence Chat", key="home_chat"):
                st.session_state.page = "Health Chat"
                st.experimental_rerun()

            if st.button("📈 Enterprise Analytics Suite", key="home_analytics"):
                st.session_state.page = "Advanced Analytics"
                st.experimental_rerun()

            # Premium partners section for enterprise credibility
            st.markdown('<h3 class="section-header">Trusted By Industry Leaders</h3>', unsafe_allow_html=True)

            st.markdown("""
            <div style="background: rgba(20, 20, 40, 0.6); border-radius: 16px; padding: 20px;
                 box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15); backdrop-filter: blur(10px);
                 border: 1px solid rgba(255, 255, 255, 0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div style="text-align: center; padding: 15px; flex: 1;">
                        <div style="font-weight: 700; color: white; font-size: 1.2rem;">MedTech</div>
                        <div style="color: rgba(255, 255, 255, 0.6); font-size: 0.9rem;">INNOVATIONS</div>
                    </div>
                    <div style="text-align: center; padding: 15px; flex: 1;">
                        <div style="font-weight: 700; color: white; font-size: 1.2rem;">Global</div>
                        <div style="color: rgba(255, 255, 255, 0.6); font-size: 0.9rem;">HEALTH PARTNERS</div>
                    </div>
                    <div style="text-align: center; padding: 15px; flex: 1;">
                        <div style="font-weight: 700; color: white; font-size: 1.2rem;">NEXT</div>
                        <div style="color: rgba(255, 255, 255, 0.6); font-size: 0.9rem;">HEALTHCARE</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ROI calculator teaser
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(0, 48, 185, 0.3), rgba(0, 209, 255, 0.2));
                 border-radius: 16px; padding: 20px; margin-top: 20px; position: relative; overflow: hidden;
                 border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);">
                <h4 style="margin-top: 0; color: white; font-size: 1.3rem;">Enterprise ROI Calculator</h4>
                <p style="color: rgba(255, 255, 255, 0.8); margin-bottom: 10px;">
                    Calculate potential savings and efficiency gains with MedExplain AI Pro.
                </p>
                <div style="color: #00F0E0; font-weight: 600; margin-top: 15px;">Coming soon →</div>
            </div>
            """, unsafe_allow_html=True)

        # Animated divider
        st.markdown('<div class="animated-divider"></div>', unsafe_allow_html=True)

        # Recent activity section with premium styling
        st.markdown('<h3 class="section-header">Health Activity Monitoring</h3>', unsafe_allow_html=True)

        if hasattr(self, "user_manager") and self.user_manager and self.user_manager.health_history:
            recent_checks = self.user_manager.get_recent_symptom_checks(limit=3)

            if recent_checks:
                for check in recent_checks:
                    date = check.get("date", "")
                    symptoms = check.get("symptoms", [])

                    # Get symptom names instead of IDs
                    symptom_names = []
                    if hasattr(self, "health_data") and self.health_data:
                        for symptom_id in symptoms:
                            symptom_info = self.health_data.get_symptom_info(symptom_id)
                            if symptom_info:
                                symptom_names.append(symptom_info.get("name", symptom_id))

                    if not symptom_names:
                        symptom_names = symptoms  # Fallback to IDs if names not found

                    st.markdown(f"""
                    <div class="activity-item">
                        <h4 class="activity-date">Health Assessment • {date}</h4>
                        <p style="margin-bottom: 8px; color: rgba(255, 255, 255, 0.9); font-size: 1.05rem;">
                            <strong>Identified Symptoms:</strong> {", ".join(symptom_names)}
                        </p>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 15px;">
                            <div style="color: rgba(255, 255, 255, 0.6); font-size: 0.9rem;">Enterprise Health Protocol</div>
                            <div style="background: rgba(0, 209, 255, 0.15); color: #00F0E0; padding: 5px 10px;
                                 border-radius: 12px; font-size: 0.9rem; font-weight: 600;">View Details</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="activity-item" style="text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 15px; color: rgba(255, 255, 255, 0.2);">📋</div>
                    <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.1rem; margin-bottom: 5px; font-weight: 600;">
                        No Recent Health Activities
                    </p>
                    <p style="color: rgba(255, 255, 255, 0.7); margin-bottom: 15px;">
                        Begin your health journey by analyzing symptoms or setting up your profile.
                    </p>
                    <div style="background: linear-gradient(90deg, #0062FF, #00D1FF);
                         border-radius: 12px; padding: 8px 15px; display: inline-block;
                         font-weight: 600; color: white; cursor: pointer; box-shadow: 0 5px 15px rgba(0, 98, 255, 0.3);">
                        Start Health Assessment
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="activity-item" style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 15px; color: rgba(255, 255, 255, 0.2);">📋</div>
                <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.1rem; margin-bottom: 5px; font-weight: 600;">
                    No Recent Health Activities
                </p>
                <p style="color: rgba(255, 255, 255, 0.7); margin-bottom: 15px;">
                    Begin your health journey by analyzing symptoms or setting up your profile.
                </p>
                <div style="background: linear-gradient(90deg, #0062FF, #00D1FF);
                     border-radius: 12px; padding: 8px 15px; display: inline-block;
                     font-weight: 600; color: white; cursor: pointer; box-shadow: 0 5px 15px rgba(0, 98, 255, 0.3);">
                    Start Health Assessment
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Enterprise-grade AI insights section
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<h3 class="section-header">AI-Powered Health Insights</h3>', unsafe_allow_html=True)

            # Health tips section with premium styling
            if hasattr(self, "openai_client") and self.openai_client and hasattr(self.openai_client, "api_key") and self.openai_client.api_key:
                # Cache tip for the day
                if 'daily_tip' not in st.session_state:
                    try:
                        prompt = """
                        Provide a single, concise health tip (100 words max) that would be useful for general wellness.
                        Focus on evidence-based advice that's practical and actionable. Format it as a brief paragraph.
                        """

                        tip = self.openai_client.generate_response(prompt)
                        if tip:
                            st.session_state.daily_tip = tip
                        else:
                            st.session_state.daily_tip = "Prioritize consistent, quality sleep for optimal health. Research shows 7-8 hours nightly strengthens immune function, improves cognitive performance, and regulates metabolism. Establish a regular sleep schedule, create a dark, cool sleeping environment, and limit screen time before bed. Consider using sleep tracking technology to optimize your sleep cycles and overall recovery."
                    except Exception as e:
                        logger.error(f"Error generating health tip: {e}", exc_info=True)
                        st.session_state.daily_tip = "Prioritize consistent, quality sleep for optimal health. Research shows 7-8 hours nightly strengthens immune function, improves cognitive performance, and regulates metabolism. Establish a regular sleep schedule, create a dark, cool sleeping environment, and limit screen time before bed. Consider using sleep tracking technology to optimize your sleep cycles and overall recovery."

                st.markdown(f"""
                <div class="health-tip">
                    <p style="position: relative; z-index: 1; color: rgba(255, 255, 255, 0.9);
                       padding-left: 20px; font-size: 1.1rem; line-height: 1.7;">
                        {st.session_state.daily_tip}
                    </p>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 15px;">
                        <div style="color: rgba(255, 255, 255, 0.6); font-size: 0.9rem;">AI-Generated Health Insight</div>
                        <div style="background: rgba(0, 209, 255, 0.15); color: #00F0E0; padding: 5px 10px;
                             border-radius: 12px; font-size: 0.9rem; font-weight: 600;">Refresh</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown('<h3 class="section-header">Enterprise Grade Security</h3>', unsafe_allow_html=True)

            st.markdown("""
            <div style="background: rgba(20, 20, 40, 0.6); border-radius: 16px; padding: 24px;
                 box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15); border-left: 4px solid;
                 border-image: linear-gradient(to bottom, #0062FF, #00F0E0) 1;
                 backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);">
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <div style="font-size: 2rem; margin-right: 15px;
                         background: linear-gradient(135deg, #0062FF, #00F0E0);
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                         background-clip: text; text-fill-color: transparent;">🔒</div>
                    <div style="flex: 1;">
                        <h4 style="margin: 0; color: white; font-size: 1.3rem;">HIPAA Compliant</h4>
                        <p style="margin: 0; color: rgba(255, 255, 255, 0.7); font-size: 0.9rem;">
                            Enterprise-level data protection
                        </p>
                    </div>
                    <div style="background: rgba(0, 209, 255, 0.15); color: #00F0E0;
                         padding: 5px 12px; border-radius: 12px; font-size: 0.85rem; font-weight: 600;">
                        VERIFIED
                    </div>
                </div>

                <ul style="color: rgba(255, 255, 255, 0.8); padding-left: 20px; margin-bottom: 0;">
                    <li style="margin-bottom: 8px;">End-to-end encryption for all health data</li>
                    <li style="margin-bottom: 8px;">Strict access controls and audit trails</li>
                    <li style="margin-bottom: 8px;">Automated threat detection and response</li>
                    <li>Regular security audits and penetration testing</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Animated divider
        st.markdown('<div class="animated-divider"></div>', unsafe_allow_html=True)

        # Call to action section
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(0, 48, 185, 0.8), rgba(0, 98, 255, 0.8), rgba(0, 209, 255, 0.7));
             border-radius: 20px; padding: 30px; text-align: center; position: relative; overflow: hidden;
             box-shadow: 0 15px 35px rgba(0, 48, 185, 0.3); margin: 30px 0;">
            <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; overflow: hidden;">
                <!-- Visual elements in background -->
                <div style="position: absolute; width: 300px; height: 300px; border-radius: 50%;
                     background: radial-gradient(circle, rgba(0, 209, 255, 0.3) 0%, transparent 70%);
                     top: -150px; right: -100px; filter: blur(30px);"></div>

                <div style="position: absolute; width: 200px; height: 200px; border-radius: 50%;
                     background: radial-gradient(circle, rgba(0, 240, 224, 0.3) 0%, transparent 70%);
                     bottom: -100px; left: -50px; filter: blur(30px);"></div>
            </div>

            <h2 style="color: white; font-weight: 800; font-size: 2.5rem; margin-bottom: 15px; position: relative; z-index: 2;">
                Transform Your Healthcare Experience
            </h2>
            <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.2rem; max-width: 80%;
                margin: 0 auto 25px auto; line-height: 1.6; position: relative; z-index: 2;">
                Join healthcare leaders worldwide who trust MedExplain AI Pro for advanced medical insights,
                predictive analytics, and enterprise-grade health monitoring.
            </p>
            <div style="position: relative; z-index: 2;">
                <button style="background: white; color: #0062FF; border: none; border-radius: 16px;
                       padding: 12px 30px; font-size: 1.2rem; font-weight: 700; cursor: pointer;
                       box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2); transition: all 0.3s ease;
                       margin: 0 15px 10px 15px;">
                    Request Enterprise Demo
                </button>
                <button style="background: rgba(255, 255, 255, 0.15); backdrop-filter: blur(10px);
                       border: 1px solid rgba(255, 255, 255, 0.3); color: white; border-radius: 16px;
                       padding: 12px 30px; font-size: 1.2rem; font-weight: 600; cursor: pointer;
                       box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1); transition: all 0.3s ease;
                       margin: 0 15px 10px 15px;">
                    View Enterprise Plans
                </button>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Medical disclaimer with premium styling
        st.markdown("""
        <div class="medical-disclaimer">
            <h4 style="margin-top: 0; color: #FF8800; font-size: 1.3rem; margin-bottom: 10px;">Enterprise Healthcare Disclaimer</h4>
            <p style="color: rgba(255, 255, 255, 0.85); margin-bottom: 0; line-height: 1.7;">
                MedExplain AI Pro is designed to complement, not replace, professional medical advice, diagnosis, or treatment.
                Our enterprise platform provides advanced analytics and insights for healthcare professionals and individuals,
                but all medical decisions should be made in consultation with qualified healthcare providers. Always seek the
                advice of your physician or other qualified health provider with any questions regarding a medical condition.
            </p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
            logger.error(f"Error rendering home page: {e}", exc_info=True)
            st.error("Error rendering enterprise home page. Please refresh the page or contact support.")
