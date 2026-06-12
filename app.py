import streamlit as st
import os
from agents.lie_detect_agent import LieDetectAgent

# --- Page Configuration ---
st.set_page_config(
    page_title="Multi-Modal Lie Detector",
    page_icon="🕵️",
    layout="wide"
)

# --- App Title and Description ---
st.title("🕵️ Multi-Modal Lie Detection Agent")
st.write("Upload a video and provide the corresponding text statement to analyze for deception.")

# --- UI Components for Input ---
st.header("1. Provide Input")
uploaded_video = st.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi'])
text_statement = st.text_area("Enter the text statement made in the video", height=100)
analyze_button = st.button("Analyze Video", type="primary")

# --- Analysis Section ---
st.header("2. Analysis Report")

# A placeholder to show the report
report_placeholder = st.empty()
report_placeholder.info("Please upload a video and enter a statement to see the analysis report.")

if analyze_button:
    # Check if both video and text are provided
    if uploaded_video is not None and text_statement:
        # Create a temporary directory to store the uploaded file
        temp_dir = "temp_uploads"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Save the uploaded file to the temporary directory
        video_path = os.path.join(temp_dir, uploaded_video.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        # Show a spinner while the agent is working
        with st.spinner("Agent is analyzing... This might take a moment."):
            try:
                # --- Initialize and run the agent ---
                agent = LieDetectAgent()
                decision, score, reasoning = agent.analyze_from_video(
                    video_path=video_path,
                    text_input=text_statement
                )
                
                # --- Display the Report ---
                with report_placeholder.container():
                    st.subheader(f"Video File: `{uploaded_video.name}`")
                    
                    # Display the final decision with color
                    if "LIE" in decision.upper():
                        st.error(f"Final Decision: {decision.upper()}")
                    else:
                        st.success(f"Final Decision: {decision.upper()}")
                    
                    # Display the confidence score as a metric
                    st.metric(label="Lie Confidence Score", value=f"{score:.2f}")

                    # Display the reasoning trace
                    with st.expander("Show Reasoning Trace"):
                        for step in reasoning:
                            st.write(step)
            
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

    else:
        # Show a warning if inputs are missing
        report_placeholder.warning("⚠️ Please make sure you've uploaded a video and entered a statement.")