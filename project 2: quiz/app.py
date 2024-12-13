from functions import YoutubeProcessor, PineconeDB
from dotenv import load_dotenv
import streamlit as st
from sentence_transformers import SentenceTransformer
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize text embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Pinecone
quiz_test = PineconeDB("quiz-test", 384)
index = quiz_test.create_index()

# Streamlit UI
st.title("YouTube Video Summarizer")

# User inputs
video_url = st.text_input("Enter YouTube Video URL:")
date_added = st.date_input("Enter the date added (YYYY-MM-DD):")

if video_url:
    # Display the YouTube video preview
    st.video(video_url)

if st.button("Generate Summary"):
    if video_url and date_added:
        try:
            # Create Youtube Processor
            processor = YoutubeProcessor(video_url)

            # Generate and display the summary with runtime measurement
            with st.spinner("Generating summary..."):
                summary, summary_runtime, title = processor.generate_summary()

            # Display the summary and runtime
            st.subheader("Summary:")
            if summary:
                st.write(summary)
                if summary_runtime > 0:
                    st.caption(
                        f"Summary generation completed in {summary_runtime:.2f} seconds."
                    )

                # Generate embeddings for the summary
                with st.spinner("Generating embeddings..."):
                    embedding = embedding_model.encode(summary).tolist()

                    # Use manually entered date
                    date_added = date_added.strftime("%Y-%m-%d")
                    metadata = {
                        "url": video_url,
                        "title": title,
                        "date_added": date_added,
                    }

                    # Store the embeddings in Pinecone with metadata
                    index.upsert(
                        [{"id": video_url, "values": embedding, "metadata": metadata}]
                    )
                    st.success(f"Embedding successfully stored with date: {date_added}")

            else:
                st.error("Failed to generate summary.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid YouTube URL/date.")
