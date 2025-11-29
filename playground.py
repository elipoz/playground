import streamlit as st
from typing import List, Optional
from pydantic import BaseModel, Field
import os
import logging
import requests
import pandas as pd
from datetime import datetime, date
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent  # type: ignore[attr-defined]

load_dotenv()  # Loads variables from .env into environment

st.set_page_config(page_title="Travel Itinerary Planner", page_icon="✈️", layout="wide")
st.title("✈️ Travel Itinerary Planner")

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
def search_travel_activities(query: str, max_results: int = 20) -> str:
    """Search for travel activities, attractions, and itineraries. Returns titles, descriptions, URLs, and images."""
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
        "include_images": True
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        # Format results with images
        results = []
        images = data.get("images", [])

        for idx, result in enumerate(data.get("results", []), 1):
            # Try to associate an image with each result
            image_url = images[idx - 1] if idx - 1 < len(images) else "No image available"

            results.append(f"""
Activity {idx}:
Title: {result.get('title', 'N/A')}
URL: {result.get('url', 'N/A')}
Content: {result.get('content', 'N/A')}
Image: {image_url}
Score: {result.get('score', 'N/A')}
---""")

        return "\n".join(results)
    except Exception as e:
        logger.error(f"Error searching: {str(e)}")
        return f"Error searching: {str(e)}"


# List of tools for the agent
tools = [search_travel_activities]

# --------------------------------------------------------------
# Step 2: Define the data models
# --------------------------------------------------------------


class Activity(BaseModel):
    """A travel activity or attraction"""

    title: str = Field(description="Name of the activity or attraction")
    description: str = Field(description="One paragraph description of the activity (3-5 sentences)")
    location: str = Field(description="Specific location within the destination")
    duration: str = Field(description="Estimated duration (e.g., '2 hours', 'Half day', 'Full day')")
    image_url: Optional[str] = Field(description="URL of a representative image")
    source_url: Optional[str] = Field(description="Source URL for more information")
    latitude: Optional[float] = Field(description="Latitude coordinate of the activity location")
    longitude: Optional[float] = Field(description="Longitude coordinate of the activity location")


class Hub(BaseModel):
    """A geographic hub with nearby activities"""

    hub_name: str = Field(description="Name of the main hub city or area")
    hub_description: str = Field(description="2-3 sentence description of the hub area, why it's a good base, amenities, transportation")
    latitude: float = Field(description="Latitude coordinate of the hub")
    longitude: float = Field(description="Longitude coordinate of the hub")
    activities: List[Activity] = Field(description="Activities within 50-100 miles of this hub")
    radius_miles: int = Field(description="Approximate radius in miles covering all activities from this hub")


class ActivityList(BaseModel):
    """Collection of travel activities grouped by geographic hubs"""

    hubs: List[Hub] = Field(description="List of geographic hubs, each with their nearby activities (up to 20 activities total across all hubs)")


class DayItinerary(BaseModel):
    """Itinerary for a single day"""

    date: str = Field(description="Date in YYYY-MM-DD format")
    day_number: int = Field(description="Day number of the trip")
    activities: List[str] = Field(description="List of activity titles for this day")
    notes: str = Field(description="Additional notes, travel tips, or recommendations for the day")


class TravelItinerary(BaseModel):
    """Complete travel itinerary"""

    destination: str = Field(description="Travel destination")
    start_date: str = Field(description="Start date of trip")
    end_date: str = Field(description="End date of trip")
    daily_itinerary: List[DayItinerary] = Field(description="Day-by-day itinerary")
    summary: str = Field(description="Overall trip summary and recommendations")


# --------------------------------------------------------------
# Step 3: Implement activity finder and itinerary generator
# --------------------------------------------------------------


class TravelPlanner:
    def __init__(self):
        # Create a ReAct agent using LangGraph - it handles tool execution automatically!
        self.agent = create_react_agent(llm, tools)
        self.activity_llm = llm.with_structured_output(ActivityList)
        self.itinerary_llm = llm.with_structured_output(TravelItinerary)

    def find_activities(self, destination: str, preferences: Optional[str] = None) -> ActivityList:
        """Find activities and attractions for a destination, grouped by geographic hubs"""

        logger.info(f"Searching for activities in: {destination}")
        if preferences:
            logger.info(f"With preferences: {preferences}")

        # Build preference guidance
        preference_text = ""
        if preferences:
            preference_text = f"\n\nIMPORTANT PREFERENCES: Prioritize and filter activities based on these user preferences: {preferences}"

        # Create the system message with instructions
        system_message = f"""You are a travel expert who finds the best activities and attractions for travelers.

Use the search_travel_activities tool to find popular activities, attractions, and experiences in: {destination}
Look for a diverse mix of activities including cultural sites, outdoor activities, food experiences, entertainment, and unique local experiences.
Focus on identifying major geographic hubs (cities/areas) where travelers can stay.
Target: Up to 20 activities total across all hubs{preference_text}"""

        # Build human message with preferences
        human_message = f"Find the top activities and attractions in {destination}. Include information about major cities and regions."
        if preferences:
            human_message += f"\n\nUser preferences to consider: {preferences}"

        # Run the agent - it will automatically call tools as needed
        agent_response = self.agent.invoke({
            "messages": [
                SystemMessage(content=system_message),
                HumanMessage(content=human_message)
            ]
        })

        # Get the final response from the agent
        final_message = agent_response["messages"][-1]
        search_output = final_message.content

        logger.info(f"Agent completed search")

        # Build structured output prompt with preferences
        structured_system_content = f"""You are analyzing search results about activities in {destination}.

IMPORTANT: Group activities by geographic hubs. Each hub should be a main city or area where travelers can stay.
All activities in a hub group should be within 50-100 miles radius of that hub.{preference_text}

For EACH HUB provide:
- Hub name (the main city/area to stay in)
- Hub description (2-3 sentences: why it's a good base, what amenities it has, transportation options)
- Hub coordinates (latitude/longitude - use actual coordinates for the hub city)
- List of activities near this hub

For EACH ACTIVITY provide:
- A clear, engaging title
- A one-paragraph description (3-5 sentences) highlighting what makes it special
- Specific location within the destination
- Estimated duration
- Coordinates (latitude/longitude - estimate based on location)
- Image URL if available in the search results
- Source URL for more information

Create 2-4 geographic hubs with up to 20 activities total across all hubs.
Focus on popular, highly-rated activities that would appeal to various types of travelers."""

        # Now use structured output to format the activities grouped by hubs
        structured_response = self.activity_llm.invoke([
            SystemMessage(content=structured_system_content),
            HumanMessage(content=f"Here are the search results:\n\n{search_output}\n\nProvide activities grouped by geographic hubs.")
        ])

        return structured_response

    def create_itinerary(
        self, destination: str, start_date: date, end_date: date, selected_activities: List[Activity]
    ) -> TravelItinerary:
        """Create a day-by-day itinerary from selected activities"""

        logger.info(f"Creating itinerary for {destination}: {start_date} to {end_date}")

        # Format selected activities
        activities_text = "\n\n".join([
            f"- {act.title}\n  Location: {act.location}\n  Duration: {act.duration}\n  Description: {act.description}"
            for act in selected_activities
        ])

        # Calculate number of days
        num_days = (end_date - start_date).days + 1

        # Generate itinerary using structured output
        itinerary = self.itinerary_llm.invoke([
            SystemMessage(content=f"""You are an expert travel planner creating a {num_days}-day itinerary for {destination}.

Create a logical, well-paced day-by-day itinerary that:
- Distributes the selected activities across the trip dates
- Groups activities that are in similar locations on the same day
- Considers activity durations and travel time
- Balances busy days with more relaxed days
- Includes practical tips and recommendations

Trip details:
- Destination: {destination}
- Start date: {start_date}
- End date: {end_date}
- Number of days: {num_days}"""),
            HumanMessage(content=f"""Create a complete itinerary using these selected activities:

{activities_text}

Organize them into a day-by-day plan with dates, and include helpful notes for each day.""")
        ])

        return itinerary


# --------------------------------------------------------------
# Step 4: Streamlit interface with session state
# --------------------------------------------------------------

# Initialize session state
if 'activities' not in st.session_state:
    st.session_state.activities = None
if 'selected_activities' not in st.session_state:
    st.session_state.selected_activities = []
if 'final_itinerary' not in st.session_state:
    st.session_state.final_itinerary = None

st.markdown("""
### Plan your perfect trip! 🌍
Enter your destination and travel dates, then select from curated activities to create a personalized itinerary.
""")

# Step 1: Input destination and dates
st.subheader("📍 Step 1: Where and when?")

col1, col2, col3 = st.columns(3)

with col1:
    destination = st.text_input("Destination Country:", placeholder="e.g., Japan, Italy, Thailand")

with col2:
    start_date = st.date_input("Start Date:", min_value=date.today())

with col3:
    end_date = st.date_input("End Date:", min_value=start_date if start_date else date.today())

# Date validation
date_valid = True
if start_date and end_date:
    if end_date <= start_date:
        st.error("❌ End date must be after start date")
        date_valid = False
    else:
        # Show trip duration
        trip_days = (end_date - start_date).days + 1
        st.info(f"📅 Trip duration: {trip_days} day{'s' if trip_days != 1 else ''}")

# Preferences input
preferences = st.text_input(
    "Preferences (optional):",
    placeholder="e.g., family-friendly, budget-conscious, adventure activities, cultural experiences, vegetarian food options",
    help="Add specific preferences to guide the activity search"
)

if st.button("🔍 Search for Activities", type="primary", disabled=not destination or not date_valid):
    with st.spinner(f"Searching for amazing activities in {destination}..."):
        planner = TravelPlanner()
        st.session_state.activities = planner.find_activities(destination, preferences=preferences)
        st.session_state.selected_activities = []
        st.session_state.final_itinerary = None
        st.rerun()

# Step 2: Display and select activities grouped by hubs
if st.session_state.activities:
    st.divider()
    st.subheader("🎯 Step 2: Select your activities")

    # Count total activities across all hubs
    total_activities = sum(len(hub.activities) for hub in st.session_state.activities.hubs)
    st.markdown(f"Found **{total_activities}** activities across **{len(st.session_state.activities.hubs)}** geographic hubs. Select the ones you'd like to include:")

    # Global activity index for unique keys
    global_activity_idx = 0

    # Display each hub
    for hub_idx, hub in enumerate(st.session_state.activities.hubs):
        st.markdown("---")
        st.markdown(f"## 🏛️ {hub.hub_name}")

        # Hub description
        st.info(f"**Base Location:** {hub.hub_description}")

        # Display map for the hub
        if hub.latitude and hub.longitude:
            st.markdown(f"📍 **Hub Coordinates:** {hub.latitude:.4f}, {hub.longitude:.4f} | **Radius:** ~{hub.radius_miles} miles")

            # Create map data - hub as the center point
            map_data = pd.DataFrame({
                'lat': [hub.latitude],
                'lon': [hub.longitude]
            })

            # Display map centered on hub
            st.map(map_data, zoom=8)

        st.markdown(f"### Activities near {hub.hub_name} ({len(hub.activities)} activities)")

        # Display activities for this hub
        for activity in hub.activities:
            # Create two columns for layout
            col_text, col_image = st.columns([2, 1])

            with col_text:
                # Checkbox for selection
                st.checkbox(
                    f"**{activity.title}**",
                    key=f"activity_{global_activity_idx}",
                    value=False
                )

                # Activity details
                st.markdown(f"📍 {activity.location} | ⏱️ {activity.duration}")
                st.markdown(activity.description)

                if activity.source_url:
                    st.markdown(f"[Learn more]({activity.source_url})")

            with col_image:
                # Display image if available
                if activity.image_url and activity.image_url != "No image available":
                    try:
                        st.image(activity.image_url, use_container_width=True)
                    except:
                        st.info("🖼️ Image unavailable")
                else:
                    st.info("🖼️ No image")

            st.markdown("")  # Spacing
            global_activity_idx += 1

    # Show selection summary and generate button
    st.divider()

    # Count selected activities
    all_activities = []
    global_idx = 0
    for hub in st.session_state.activities.hubs:
        for activity in hub.activities:
            if st.session_state.get(f"activity_{global_idx}"):
                all_activities.append(activity)
            global_idx += 1

    selected_count = len(all_activities)

    if selected_count > 0:
        st.success(f"✅ {selected_count} activities selected")

        if st.button("🗓️ Generate My Itinerary", type="primary"):
            with st.spinner("Creating your personalized itinerary..."):
                planner = TravelPlanner()
                st.session_state.final_itinerary = planner.create_itinerary(
                    destination=destination,
                    start_date=start_date,
                    end_date=end_date,
                    selected_activities=all_activities
                )
                st.rerun()
    else:
        st.warning("⚠️ Please select at least one activity to generate an itinerary.")

# Step 3: Display final itinerary
if st.session_state.final_itinerary:
    st.divider()
    st.subheader("🎉 Your Personalized Itinerary")

    itinerary = st.session_state.final_itinerary

    # Summary
    st.markdown(f"### {itinerary.destination}")
    st.markdown(f"**{itinerary.start_date}** to **{itinerary.end_date}**")
    st.info(itinerary.summary)

    st.divider()

    # Day-by-day breakdown
    for day in itinerary.daily_itinerary:
        with st.expander(f"📅 Day {day.day_number} - {day.date}", expanded=True):
            st.markdown("**Activities:**")
            for activity_title in day.activities:
                st.markdown(f"- {activity_title}")

            if day.notes:
                st.markdown(f"\n**Notes:** {day.notes}")

    # Download option
    st.divider()
    if st.button("📥 Download Itinerary (JSON)"):
        st.download_button(
            label="Download",
            data=itinerary.model_dump_json(indent=2),
            file_name=f"{destination}_itinerary.json",
            mime="application/json"
        )
