from functions import YoutubeProcessor
from dotenv import load_dotenv
import streamlit as st
import time  # To measure runtime

# Load environment variables
load_dotenv()

# Streamlit UI
st.title("YouTube Video Summarizer")

# Input field for YouTube video URL
video_url = st.text_input("Enter YouTube Video URL:")

# Check if a URL is provided
if video_url:
    try:
        # Create a Processor object with the input YouTube URL
        processor = YoutubeProcessor(video_url)

        # Display the YouTube video preview
        st.video(video_url)

        # Generate and display the summary with runtime measurement
        with st.spinner("Generating summary..."):
            start_time = time.time()
            transcript = (
                processor.extract_transcript()
            )  # Transcript is still needed to generate summary
            if transcript:
                summary = processor.generate_summary("", transcript)
                summary_runtime = time.time() - start_time
            else:
                summary = "No transcript available for this video."
                summary_runtime = 0

        # Display the summary and runtime
        st.subheader("Summary:")
        if summary:
            st.write(summary)
            if summary_runtime > 0:
                st.caption(
                    f"Summary generation completed in {summary_runtime:.2f} seconds."
                )
        else:
            st.error("Failed to generate summary.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("Please enter a valid YouTube URL.")
