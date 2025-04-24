"""
Chat Interface UI for MedExplain AI Pro.

This module provides the user interface for the natural language
chat interface that allows users to interact with the AI assistant.
"""

import streamlit as st
import logging
from typing import Dict, List, Any, Optional, Union

# Configure logger
logger = logging.getLogger(__name__)

class ChatInterface:
    """
    Streamlit interface for the natural language chat component.

    This class is responsible for:
    - Rendering the chat interface
    - Managing chat history
    - Handling user interactions with the chat
    """

    def __init__(self, nlp_interface):
        """
        Initialize the chat interface.

        Args:
            nlp_interface: The natural language processing interface for generating responses
        """
        self.nlp_interface = nlp_interface
        logger.info("ChatInterface initialized")

    def render(self, user_profile: Optional[Dict[str, Any]] = None,
              health_data: Optional[Any] = None) -> None:
        """
        Render the chat interface in Streamlit.

        Args:
            user_profile: Optional user profile data
            health_data: Optional health data manager
        """
        st.title("Medical Information Chat")

        # Add disclaimer
        st.markdown("""
        <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
        <strong>Medical Disclaimer:</strong> This chat interface provides educational information only and is not a substitute
        for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or
        other qualified health provider with any questions you may have regarding a medical condition.
        </div>
        """, unsafe_allow_html=True)

        # Initialize chat history in session state if it doesn't exist
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a health question..."):
            self._process_chat_input(prompt, user_profile, health_data)

        # Add option to clear chat history
        self._render_chat_controls()

    def _process_chat_input(self, prompt: str, user_profile: Optional[Dict[str, Any]] = None,
                           health_data: Optional[Any] = None) -> None:
        """
        Process a chat input from the user.

        Args:
            prompt: The user's input text
            user_profile: Optional user profile data
            health_data: Optional health data manager
        """
        # Check if NLP interface is available
        if not self.nlp_interface.is_available():
            st.error("Please add your OpenAI API key in Settings to use the chat interface.")
            return

        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = self.nlp_interface.generate_response(prompt, user_profile, health_data)
                    st.markdown(response)

                    # Add assistant response to chat history
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    logger.error(error_msg)
                    st.error(error_msg)

    def _render_chat_controls(self) -> None:
        """Render control elements for the chat interface."""
        # Only show controls if there are messages
        if st.session_state.chat_messages:
            st.markdown("---")

            col1, col2 = st.columns([1, 4])

            with col1:
                if st.button("Clear Chat", key="clear_chat"):
                    st.session_state.chat_messages = []
                    self.nlp_interface.clear_conversation_history()
                    st.experimental_rerun()

            with col2:
                st.markdown("""
                <div style="font-size: 0.8em; color: #666;">
                You can ask about health topics, symptoms, conditions, or general medical information.
                </div>
                """, unsafe_allow_html=True)


def render_chat_interface(nlp_interface, user_profile: Optional[Dict[str, Any]] = None,
                        health_data: Optional[Any] = None) -> None:
    """
    Render the chat interface.

    Args:
        nlp_interface: The natural language processing interface
        user_profile: Optional user profile data
        health_data: Optional health data manager
    """
    # Use data from session state if not provided
    if user_profile is None and "user_manager" in st.session_state:
        user_profile = st.session_state.user_manager.profile

    if health_data is None and "health_data_manager" in st.session_state:
        health_data = st.session_state.health_data_manager

    # Create and render chat interface
    chat_interface = ChatInterface(nlp_interface)
    chat_interface.render(user_profile, health_data)
