import streamlit as st
from typing import List
from pydantic import BaseModel, Field
import os
import logging
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent  # type: ignore[attr-defined]

load_dotenv()  # Loads variables from .env into environment

st.title("Hello Streamlit-er 👋")

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize LangChain LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# --------------------------------------------------------------
# Step 1: Define search tool using LangChain
# --------------------------------------------------------------


@tool
def search_internet(query: str, max_results: int = 5) -> str:
    """Search the internet for recent and relevant information about a topic. Returns URLs, titles, publish dates, and content snippets."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return "Error: TAVILY_API_KEY not found in environment variables"

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answer": True,
        "include_raw_content": False,
        "include_images": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        # Format results
        results = []
        for idx, result in enumerate(data.get("results", []), 1):
            results.append(f"""
Result {idx}:
Title: {result.get('title', 'N/A')}
URL: {result.get('url', 'N/A')}
Published: {result.get('published_date', 'Date not available')}
Content: {result.get('content', 'N/A')}
Score: {result.get('score', 'N/A')}
---""")

        return "\n".join(results)
    except Exception as e:
        logger.error(f"Error searching: {str(e)}")
        return f"Error searching: {str(e)}"


# List of tools for the agent
tools = [search_internet]

# --------------------------------------------------------------
# Step 2: Define the data models
# --------------------------------------------------------------


class SearchResult(BaseModel):
    """Search result with summary"""

    title: str = Field(description="Title of the article/page")
    url: str = Field(description="The url of the link")
    publish_date: str = Field(description="The publish date of the link")
    summary: str = Field(description="A comprehensive paragraph summary of the content (3-5 sentences)")
    relevance_score: float = Field(description="Relevance score (0-1)")


class SearchResults(BaseModel):
    """Collection of search results"""

    results: List[SearchResult] = Field(description="List of relevant search results sorted by date and relevance")


# --------------------------------------------------------------
# Step 3: Implement finder using LangChain agent
# --------------------------------------------------------------


class ContentFinder:
    def __init__(self):
        # Create a ReAct agent using LangGraph - it handles tool execution automatically!
        self.agent = create_react_agent(llm, tools)
        self.structured_llm = llm.with_structured_output(SearchResults)

    def get_results(self, topic: str, target_length: int, style: str) -> SearchResults:
        """Get search results using LangGraph agent with automatic tool execution"""

        logger.info(f"Starting search for: {topic}")

        # Create the system message with instructions
        system_message = f"""You are a research assistant that finds the most recent and relevant information about a topic.

Use the search_internet tool to find information about: {topic}
Target: {target_length} results
Focus: {style}"""

        # Run the agent - it will automatically call tools as needed
        agent_response = self.agent.invoke({
            "messages": [
                SystemMessage(content=system_message),
                HumanMessage(content=f"Find the most recent and relevant information about: {topic}")
            ]
        })

        # Get the final response from the agent
        final_message = agent_response["messages"][-1]
        search_output = final_message.content

        logger.info(f"Agent completed search")

        # Now use structured output to format the results
        structured_response = self.structured_llm.invoke([
            SystemMessage(content=f"""You are analyzing search results about: {topic}

Create a structured list with a comprehensive paragraph summary for each result.
Each summary should be 3-5 sentences capturing the key points and insights.
Sort results by relevance and recency.
Target: {target_length} results
Focus: {style}"""),
            HumanMessage(content=f"Here are the search results:\n\n{search_output}\n\nProvide a structured list with comprehensive summaries for each result.")
        ])

        return structured_response

    def find_content(
        self, topic: str, target_length: int = 5, style: str = "technical"
    ) -> SearchResults:
        """Process the entire research task"""
        logger.info(f"Starting research process for: {topic}")

        # Get search results
        results = self.get_results(topic, target_length, style)
        logger.info(f"Found {len(results.results)} relevant results")
        logger.info(f"\nResults:\n{results.model_dump_json(indent=2)}")

        return results


# --------------------------------------------------------------
# Step 4: Streamlit interface
# --------------------------------------------------------------

def find_content_for_topic(topic: str, style: str = "technical") -> str:
    """Find and format content for a given topic"""
    finder = ContentFinder()

    # Search for recent articles
    search_topic = f"recent articles about {topic}"
    result = finder.find_content(
        topic=search_topic,
        target_length=5,
        style=style
    )

    output_lines = []

    for idx, search_result in enumerate(result.results, 1):
        output_lines.extend([
            f"\n### {idx}. {search_result.title}",
            f"**Published:** {search_result.publish_date} | **Relevance:** {search_result.relevance_score:.2f}",
            f"\n{search_result.summary}",
            f"\n[Read more]({search_result.url})",
            ""
        ])

    return "\n".join(output_lines)

st.markdown(
"""
This is a simple Streamlit app that uses LangChain to search the internet for recent and relevant information about a topic.
"""
)

user_input = st.text_input(label='Enter your topic here:')

style_options = [
    "Travel/Nature",
    "Health/Wellness",
    "Food/Recipes",
    "Fashion/Beauty",
    "Sports/Fitness",
    "Entertainment/Movies",
    "History/Culture",
    "Religion/Spirituality",
    "Philosophy/Mindfulness",
]

selected_style = st.selectbox(
    label='Select content style:',
    options=style_options,
    index=0  # Default to "Technical"
)

if user_input:
    result = find_content_for_topic(user_input, style=selected_style)
    st.markdown(result)
