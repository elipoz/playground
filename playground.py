import streamlit as st
from typing import List, Optional
from pydantic import BaseModel, Field
import os
import logging
import requests
import pandas as pd
import pydeck as pdk
import math
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
            # Try to associate an image with each result - only include valid URLs
            image_url = ""
            if idx - 1 < len(images):
                img = images[idx - 1]
                if img and isinstance(img, str) and img.startswith(("http://", "https://")):
                    image_url = img

            image_line = f"Image: {image_url}" if image_url else "Image: (none available)"

            results.append(f"""
Activity {idx}:
Title: {result.get('title', 'N/A')}
URL: {result.get('url', 'N/A')}
Content: {result.get('content', 'N/A')}
{image_line}
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

    def find_activities(self, destination: str, preferences: Optional[str] = None, num_days: int = 5, num_activities: int = 10, num_hubs: int = 2) -> ActivityList:
        """Find activities and attractions for a destination, grouped by geographic hubs"""

        logger.info(f"Searching for activities in: {destination}")
        logger.info(f"Trip duration: {num_days} days, Target activities: {num_activities}, Target hubs: {num_hubs}")
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
Focus on identifying exactly {num_hubs} DISTINCT major geographic hub{'s' if num_hubs != 1 else ''} (specific cities/areas) where travelers can stay.
Pay attention to the actual locations mentioned in search results (city names, specific areas).
Target: {num_activities} activities total across all {num_hubs} hub{'s' if num_hubs != 1 else ''} (based on {num_days}-day trip){preference_text}"""

        # Build human message with preferences
        human_message = f"""Find the top activities and attractions in {destination}.

When searching, pay attention to:
- Specific city or area names mentioned in the results
- Geographic locations of activities
- Major tourist hubs/cities in {destination}

Include information about major cities and specific regions."""
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

CRITICAL GEOGRAPHIC REQUIREMENTS:
1. Each hub MUST be a distinct major city or region where travelers can stay
2. STRICTLY verify that ALL activities assigned to a hub are within 50-100 miles (80-160 km) of that hub
3. DO NOT group activities from different cities/regions together unless they are truly within the distance limit
4. If an activity's location is mentioned (like "in Alta", "in Tromsø", "in Narvik"), it MUST be assigned to a hub for that specific city
5. Use accurate real-world coordinates for hubs and activities - do not guess randomly
6. Calculate approximate distances to verify grouping is correct

GEOGRAPHIC VALIDATION CHECKLIST:
- Is each activity's stated location actually near the hub city?
- Are activities from different named cities kept in separate hubs?
- Would a traveler realistically stay at this hub to do all these activities?
- Are the coordinates realistic for the actual locations mentioned?{preference_text}

For EACH HUB provide:
- Hub name (the EXACT main city/area name, e.g., "Tromsø", "Alta", "Narvik" - not generic names)
- Hub description (2-3 sentences: why it's a good base, what amenities it has, transportation options)
- Hub coordinates (latitude/longitude - use ACCURATE real-world coordinates for this specific city/location)
- Radius in miles (approximate radius to cover all activities in this hub, must be 50-100 miles)
- List of activities ONLY if they are actually near this hub

For EACH ACTIVITY provide:
- A clear, engaging title
- A one-paragraph description (3-5 sentences) highlighting what makes it special
- Specific location (be precise - include city/area name)
- Estimated duration
- Coordinates (latitude/longitude - use accurate coordinates based on the actual location mentioned)
- Image URL: ONLY include if a valid HTTPS image URL is found in search results. If no image is available, set to null/empty. Do NOT include placeholder text like "No image available"
- Source URL for more information

BEFORE FINALIZING: Double-check that activities in "City A" are not placed under a hub for "City B".

Create exactly {num_hubs} distinct geographic hub{'s' if num_hubs != 1 else ''} with {num_activities} activities total across all hubs.
Each hub must have at least 2 activities (never less than 2).
Distribute activities as evenly as possible across the {num_hubs} hub{'s' if num_hubs != 1 else ''}.
Focus on popular, highly-rated activities that would appeal to various types of travelers."""

        # Now use structured output to format the activities grouped by hubs
        structured_response = self.activity_llm.invoke([
            SystemMessage(content=structured_system_content),
            HumanMessage(content=f"""Here are the search results:\n\n{search_output}\n\n
Provide activities grouped by geographic hubs.

REMEMBER:
- Create exactly {num_hubs} hub{'s' if num_hubs != 1 else ''}
- Target: {num_activities} total activities distributed across {num_hubs} hub{'s' if num_hubs != 1 else ''}
- Each hub MUST have at least 2 activities (minimum 2, no exceptions)
- Create separate hubs for different cities (e.g., Tromsø, Alta, Narvik should be separate hubs)
- Only group activities that are actually within 50-100 miles of each hub
- Match activity locations to the correct hub city
- Use accurate real-world coordinates for each location""")
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
if 'selected_hub_for_zoom' not in st.session_state:
    st.session_state.selected_hub_for_zoom = None

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
trip_days = 1
if start_date and end_date:
    if end_date <= start_date:
        st.error("❌ End date must be after start date")
        date_valid = False
    else:
        # Show trip duration
        trip_days = (end_date - start_date).days + 1
        st.info(f"📅 Trip duration: {trip_days} day{'s' if trip_days != 1 else ''}")

# Hub count slider
max_hubs = max(1, trip_days // 2) if date_valid else 4
num_hubs = st.slider(
    "Number of different locations to visit:",
    min_value=1,
    max_value=max_hubs,
    value=min(2, max_hubs),  # Default to 2 or max_hubs if less
    disabled=not date_valid,
    help=f"Choose how many different cities/areas you want to base yourself in. More hubs = more travel between locations. Maximum {max_hubs} for a {trip_days}-day trip."
)

# Preferences input
preferences = st.text_input(
    "Preferences (optional):",
    placeholder="e.g., family-friendly, budget-conscious, adventure activities, cultural experiences, vegetarian food options",
    help="Add specific preferences to guide the activity search"
)

if st.button("🔍 Search for Activities", type="primary", disabled=not destination or not date_valid):
    # Calculate number of activities based on trip duration (2x number of days)
    trip_days = (end_date - start_date).days + 1
    num_activities = trip_days * 2

    with st.spinner(f"Searching for activities in {num_hubs} location{'s' if num_hubs != 1 else ''} in {destination}..."):
        planner = TravelPlanner()
        st.session_state.activities = planner.find_activities(
            destination=destination,
            preferences=preferences,
            num_days=trip_days,
            num_activities=num_activities,
            num_hubs=num_hubs
        )
        st.session_state.selected_activities = []
        st.session_state.final_itinerary = None
        st.rerun()

# Step 2: Display and select activities grouped by hubs
if st.session_state.activities:
    st.divider()
    st.subheader("🎯 Step 2: Select your activities")

    # Filter out hubs with fewer than 2 activities
    valid_hubs = [hub for hub in st.session_state.activities.hubs if len(hub.activities) >= 2]

    # Count total activities across all hubs
    total_activities = sum(len(hub.activities) for hub in valid_hubs)
    st.markdown(f"Found **{total_activities}** activities across **{len(valid_hubs)}** geographic hubs.")

    # Display overview map with all hubs
    hub_coordinates = []
    for hub in valid_hubs:
        if hub.latitude and hub.longitude:
            hub_coordinates.append({
                'lat': hub.latitude,
                'lon': hub.longitude,
                'hub_name': hub.hub_name
            })

    if hub_coordinates:
        st.markdown("**📍 Hub Locations Overview:**")

        # Create two columns: map on left (half width), distance matrix on right
        col_map, col_distances = st.columns([1, 1])

        with col_map:
            # Determine map center and zoom based on selected hub or overall view
            if st.session_state.selected_hub_for_zoom is not None:
                # Zoom to specific hub
                selected_hub_data = hub_coordinates[st.session_state.selected_hub_for_zoom]
                view_state = pdk.ViewState(
                    latitude=selected_hub_data['lat'],
                    longitude=selected_hub_data['lon'],
                    zoom=8,
                    pitch=0
                )
            else:
                # Show all hubs
                avg_lat = sum(h['lat'] for h in hub_coordinates) / len(hub_coordinates)
                avg_lon = sum(h['lon'] for h in hub_coordinates) / len(hub_coordinates)
                view_state = pdk.ViewState(
                    latitude=avg_lat,
                    longitude=avg_lon,
                    zoom=5,
                    pitch=0
                )

            # Create pydeck map with labels
            hub_map_df = pd.DataFrame(hub_coordinates)

            # Scatterplot layer for hub points
            scatterplot_layer = pdk.Layer(
                'ScatterplotLayer',
                data=hub_map_df,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=15000,
                pickable=True
            )

            # Text layer for hub names
            text_layer = pdk.Layer(
                'TextLayer',
                data=hub_map_df,
                get_position='[lon, lat]',
                get_text='hub_name',
                get_size=16,
                get_color='[0, 0, 0]',
                get_angle=0,
                get_text_anchor='"middle"',
                get_alignment_baseline='"bottom"'
            )

            # Render map (using open street map style, no API key needed)
            st.pydeck_chart(pdk.Deck(
                map_style='https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json',
                initial_view_state=view_state,
                layers=[scatterplot_layer, text_layer],
                tooltip={"text": "{hub_name}"}
            ))

        with col_distances:
            # Calculate and display distance matrix
            def haversine_distance(lat1, lon1, lat2, lon2):
                """Calculate the great circle distance in miles between two points"""
                R = 3959  # Radius of Earth in miles

                lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1

                a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                c = 2 * math.asin(math.sqrt(a))

                return R * c

            st.markdown("**Distances Between Hubs (miles):**")

            # Create distance matrix
            n_hubs = len(hub_coordinates)
            if n_hubs > 1:
                distance_data = []
                for i in range(n_hubs):
                    row = [hub_coordinates[i]['hub_name']]
                    for j in range(n_hubs):
                        if i == j:
                            row.append("-")
                        else:
                            dist = haversine_distance(
                                hub_coordinates[i]['lat'], hub_coordinates[i]['lon'],
                                hub_coordinates[j]['lat'], hub_coordinates[j]['lon']
                            )
                            row.append(f"{dist:.0f}")
                    distance_data.append(row)

                # Create DataFrame
                columns = ["From \\ To"] + [h['hub_name'] for h in hub_coordinates]
                distance_df = pd.DataFrame(distance_data, columns=columns)
                st.dataframe(distance_df, hide_index=True, use_container_width=True)
            else:
                st.info("Add more hubs to see distances")

        # Clickable hub names below map
        st.markdown("**Click to zoom to a hub:**")
        cols = st.columns(len(hub_coordinates) + 1)

        # Reset button
        with cols[0]:
            if st.button("🌍 All Hubs", key="zoom_all", use_container_width=True):
                st.session_state.selected_hub_for_zoom = None
                st.rerun()

        # Individual hub buttons
        for idx, hub_data in enumerate(hub_coordinates):
            with cols[idx + 1]:
                if st.button(f"📍 {hub_data['hub_name']}", key=f"zoom_{idx}", use_container_width=True):
                    st.session_state.selected_hub_for_zoom = idx
                    st.rerun()

    st.markdown(f"**Select the activities you'd like to include:**")

    # Global activity index for unique keys
    global_activity_idx = 0

    # Display each hub
    for hub_idx, hub in enumerate(valid_hubs):
        st.markdown("---")

        # Hub name with map icon
        col_hub_title, col_map_icon = st.columns([0.95, 0.05])

        with col_hub_title:
            st.markdown(f"## 🏛️ {hub.hub_name}")

        with col_map_icon:
            # Map icon with popover
            if hub.latitude and hub.longitude:
                with st.popover("🗺️"):
                    st.markdown(f"**{hub.hub_name}**")
                    st.markdown(f"📍 Coordinates: {hub.latitude:.4f}, {hub.longitude:.4f}")
                    st.markdown(f"**Radius:** ~{hub.radius_miles} miles")

                    # Create map data - hub as the center point
                    map_data = pd.DataFrame({
                        'lat': [hub.latitude],
                        'lon': [hub.longitude]
                    })

                    # Display map
                    st.map(map_data, zoom=8)

        # Hub description
        st.info(f"**Base Location:** {hub.hub_description}")

        st.markdown(f"### Activities near {hub.hub_name} ({len(hub.activities)} activities)")

        # Display activities for this hub
        for activity in hub.activities:
            # Create layout with checkbox and popover for image
            col_check, col_image_icon = st.columns([0.95, 0.05])

            with col_check:
                # Checkbox for selection
                st.checkbox(
                    f"**{activity.title}**",
                    key=f"activity_{global_activity_idx}",
                    value=False
                )

            with col_image_icon:
                # Image preview icon with popover - only show if valid URL exists
                has_valid_image = (
                    activity.image_url
                    and activity.image_url not in ["No image available", ""]
                    and isinstance(activity.image_url, str)
                    and activity.image_url.startswith(("http://", "https://"))
                )

                if has_valid_image:
                    with st.popover("🖼️"):
                        try:
                            st.image(activity.image_url, use_container_width=True, caption=activity.title)
                        except Exception as e:
                            st.warning("🖼️ Image temporarily unavailable")
                            st.caption(activity.title)
                else:
                    # Show empty space to maintain alignment, but no icon
                    st.write("")

            # Activity details
            st.markdown(f"📍 {activity.location} | ⏱️ {activity.duration}")
            st.markdown(activity.description)

            if activity.source_url:
                st.markdown(f"[Learn more]({activity.source_url})")

            st.markdown("")  # Spacing
            global_activity_idx += 1

    # Show selection summary and generate button
    st.divider()

    # Count selected activities (using same filtered hubs)
    all_activities = []
    global_idx = 0
    for hub in valid_hubs:
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
