import streamlit as st
import lancedb
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from typing import List

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from openai import OpenAI
from tokenizer import OpenAITokenizerWrapper
import os

import lancedb

# --------------------------------------------------------------
# Connect to the database
# --------------------------------------------------------------

uri = "data/lancedb"
db = lancedb.connect(uri)


# --------------------------------------------------------------
# Load the table
# --------------------------------------------------------------

table = db.open_table("docling")


# --------------------------------------------------------------
# Search the table
# --------------------------------------------------------------

result = table.search(query="pdf").limit(5)
result.to_pandas()

# Load environment variables
load_dotenv()

# For testing only - remove in production
os.environ["OPENAI_API_KEY"] = "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"

# Initialize OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
    st.stop()

client = OpenAI(
    api_key=api_key
)

def get_context(query: str, table, num_results: int = 3) -> str:
    """Search the database for relevant context.

    Args:
        query: User's question
        table: LanceDB table object
        num_results: Number of results to return

    Returns:
        str: Concatenated context from relevant chunks with source information
    """
    results = table.search(query).limit(num_results).to_pandas()
    contexts = []

    for _, row in results.iterrows():
        # Extract metadata
        filename = row["metadata"]["filename"]
        page_numbers = row["metadata"]["page_numbers"]
        title = row["metadata"]["title"]

        # Build source citation
        source_parts = []
        if filename:
            source_parts.append(filename)
        if isinstance(page_numbers, (list, np.ndarray)) and len(page_numbers) > 0:
            source_parts.append(f"p. {', '.join(str(p) for p in page_numbers)}")

        source = f"\nSource: {' - '.join(source_parts)}"
        if title:
            source += f"\nTitle: {title}"

        contexts.append(f"{row['text']}{source}")

    return "\n\n".join(contexts)


def get_chat_response(messages, context: str) -> str:
    """Get streaming response from OpenAI API.

    Args:
        messages: Chat history
        context: Retrieved context from database

    Returns:
        str: Model's response
    """
    system_prompt = f"""You are an intelligent and specialized assistant that provides comprehensive and detailed answers. When responding to questions:
1. Provide a complete and thorough answer that covers all aspects of the question.
2. Explain concepts in depth, offering examples when needed.
3. Organize your answer in a well-structured manner using bullet points, paragraphs, tables, and clear text formatting.
4. Ensure that the answer is visually appealing, professional, and logically organized.
5. Present information in sections with clear headings, bullet points, and tables to enhance readability.
6. When displaying a Table of Contents or similarly structured information:
   - Use formatting that ensures clarity and ease of reading.
   - Organize the content in a neat and professional manner.
   - Do not use excessive filler characters such as long dotted lines (e.g., "............................................................"). Instead, use structured tables or columns to separate titles and page numbers.
   - If numbering is used, format it consistently and clearly.
7. Connect related information to offer a cohesive and comprehensive response.
8. Use the provided context, and if any details are missing, clearly indicate that.

Context:
{context}

    """

    messages_with_context = [{"role": "system", "content": system_prompt}, *messages]

    # Create the streaming response
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_with_context,
        temperature=0.7,
        stream=True,
    )

    # Use Streamlit's built-in streaming capability
    response = st.write_stream(stream)
    return response


# Initialize Streamlit app
st.title("📚 Document Q&A")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize database connection
table = init_db()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the document"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get relevant context
    with st.status("Searching document...", expanded=False) as status:
        context = get_context(prompt, table)
        st.markdown(
            """
            <style>
            .search-result {
                margin: 10px 0;
                padding: 10px;
                border-radius: 4px;
                background-color: #f0f2f6;
            }
            .search-result summary {
                cursor: pointer;
                color: #0f52ba;
                font-weight: 500;
            }
            .search-result summary:hover {
                color: #1e90ff;
            }
            .metadata {
                font-size: 0.9em;
                color: #666;
                font-style: italic;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

        st.write("Found relevant sections:")
        for chunk in context.split("\n\n"):
            # Split into text and metadata parts
            parts = chunk.split("\n")
            text = parts[0]
            metadata = {
                line.split(": ")[0]: line.split(": ")[1]
                for line in parts[1:]
                if ": " in line
            }

            source = metadata.get("Source", "Unknown source")
            title = metadata.get("Title", "Untitled section")

            st.markdown(
                f"""
                <div class="search-result">
                    <details>
                        <summary>{source}</summary>
                        <div class="metadata">Section: {title}</div>
                        <div style="margin-top: 8px;">{text}</div>
                    </details>
                </div>
            """,
                unsafe_allow_html=True,
            )

    # Display assistant response first
    with st.chat_message("assistant"):
        # Get model response with streaming
        response = get_chat_response(st.session_state.messages, context)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
