import streamlit as st
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
import os
import logging
import requests
import json
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env into environment

st.title("Hello Streamlit-er 👋")

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o-mini"

# --------------------------------------------------------------
# Step 1: Define search function
# --------------------------------------------------------------


def search_internet(query: str, max_results: int = 5) -> str:
    """
    Search the internet using Tavily API
    Returns formatted search results with URLs, titles, and content
    """
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


# --------------------------------------------------------------
# Step 2: Define tools for OpenAI function calling
# --------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_internet",
            "description": "Search the internet for recent and relevant information about a topic. Returns URLs, titles, publish dates, and content snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant information"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# --------------------------------------------------------------
# Step 3: Define the data models
# --------------------------------------------------------------


class SubTask(BaseModel):
    """URL on the topic"""

    url: str = Field(description="The url of the link")
    title: str = Field(description="Title of the article/page")
    target_audience: str = Field(description="Intended audience for the content")
    publish_date: str = Field(description="The publish date of the link")
    description: str = Field(description="What this content covers")
    relevance_score: float = Field(description="Relevance score (0-1)")


class OrchestratorPlan(BaseModel):
    """Orchestrator's structured search results"""

    urls: List[SubTask] = Field(description="List of relevant URLs sorted by date and relevance")


# --------------------------------------------------------------
# Step 4: Implement orchestrator with search capability
# --------------------------------------------------------------


class UrlFinder:
    def __init__(self):
        self.sections_content = {}

    def get_plan(self, topic: str, target_length: int, style: str) -> OrchestratorPlan:
        """Get URLs using function calling and structured output"""

        messages = [
            {
                "role": "system",
                "content": f"""You are a research assistant that finds the most recent and relevant links about a topic.

Use the search_internet function to find information about: {topic}
Then analyze the results and return them in a structured format, sorted by relevance and recency.
Target: {target_length} links
Focus: {style}"""
            },
            {
                "role": "user",
                "content": f"Find the most recent and relevant links about: {topic}"
            }
        ]

        # First call: Let the model decide to use the search tool
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        # Process tool calls
        while response.choices[0].message.tool_calls:
            messages.append(response.choices[0].message)

            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                logger.info(f"Calling {function_name} with args: {function_args}")

                if function_name == "search_internet":
                    function_response = search_internet(**function_args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": function_response
                    })

            # Get next response
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

        # Now get structured output
        messages.append({
            "role": "user",
            "content": "Based on the search results, provide a structured list of the most relevant URLs with all the details."
        })

        completion = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=OrchestratorPlan,
        )

        return completion.choices[0].message.parsed

    def find_urls(
        self, topic: str, target_length: int = 5, style: str = "technical"
    ) -> OrchestratorPlan:
        """Process the entire research task"""
        logger.info(f"Starting research process for: {topic}")

        # Get search results
        plan = self.get_plan(topic, target_length, style)
        logger.info(f"Found {len(plan.urls)} relevant links")
        logger.info(f"\nResults:\n{plan.model_dump_json(indent=2)}")

        return plan


# --------------------------------------------------------------
# Step 5: Example usage
# --------------------------------------------------------------

def find_urls(topic: str) -> str:
    urlFinder = UrlFinder()

    # Example: Search for recent articles
    topic = f"recent articles about {topic}"
    result = urlFinder.find_urls(
        topic=topic,
        target_length=5,
        style="technical"
    )

    output_lines = [
        "\n" + "="*60,
        "SEARCH RESULTS",
        "="*60
    ]

    for idx, url_info in enumerate(result.urls, 1):
        output_lines.extend([
            f"\n{idx}. {url_info.title}",
            f"   URL: {url_info.url}",
            f"   Published: {url_info.publish_date}",
            f"   Relevance: {url_info.relevance_score}",
            f"   Audience: {url_info.target_audience}",
            f"   Description: {url_info.description}"
        ])

    return "\n".join(output_lines)

st.markdown(
"""
This is a simple Streamlit app that uses the OpenAI API to search the internet for recent and relevant information about a topic.
"""
)

user_input = st.text_input(label='Enter your topic here:')

if user_input:
    result = find_urls(user_input)
    st.markdown(result)
