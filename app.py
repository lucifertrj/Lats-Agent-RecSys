import streamlit as st
import nest_asyncio
from llama_index.core import Settings
from llama_index.llms.sambanovasystems import SambaNovaCloud
from duckduckgo_search import DDGS
from llama_index.core.tools import FunctionTool
from llama_index.agent.lats import LATSAgentWorker
from llama_index.core.agent import AgentRunner
import os

# Apply nest_asyncio
nest_asyncio.apply()

# Product category configurations
PRODUCT_CATEGORIES = {
    "Cameras": {
        "features": [
            "Low Light Performance",
            "4K Video",
            "Image Stabilization",
            "Weather Sealing",
            "Compact Size",
            "WiFi Connectivity",
            "Touch Screen"
        ],
        "types": [
            "Mirrorless",
            "DSLR",
            "Point and Shoot",
            "Medium Format"
        ],
        "use_cases": [
            "Professional Photography",
            "Vlogging",
            "Travel Photography",
            "Sports Photography",
            "Wildlife Photography"
        ]
    },
    "Laptops": {
        "features": [
            "Long Battery Life",
            "Dedicated Graphics",
            "Touch Screen",
            "Backlit Keyboard",
            "Fingerprint Reader",
            "Thunderbolt Ports",
            "5G Connectivity"
        ],
        "types": [
            "Ultrabook",
            "Gaming Laptop",
            "Business Laptop",
            "2-in-1 Convertible",
            "Budget Laptop"
        ],
        "use_cases": [
            "Gaming",
            "Content Creation",
            "Business",
            "Student",
            "Programming"
        ]
    },
    "Smartphones": {
        "features": [
            "5G Support",
            "Wireless Charging",
            "Water Resistance",
            "Face Recognition",
            "Multiple Cameras",
            "Fast Charging",
            "NFC"
        ],
        "types": [
            "Flagship",
            "Mid-range",
            "Budget",
            "Gaming Phone",
            "Compact"
        ],
        "use_cases": [
            "Photography",
            "Gaming",
            "Business",
            "Basic Use",
            "Content Creation"
        ]
    },
    "Smart Home Devices": {
        "features": [
            "Voice Control",
            "Mobile App Control",
            "Energy Monitoring",
            "Motion Detection",
            "Smart Scheduling",
            "Multi-user Support",
            "Integration Capabilities"
        ],
        "types": [
            "Smart Speakers",
            "Security Cameras",
            "Smart Lights",
            "Smart Thermostats",
            "Smart Displays"
        ],
        "use_cases": [
            "Home Security",
            "Energy Management",
            "Entertainment",
            "Home Automation",
            "Family Organization"
        ]
    }
}

def initialize_llm():
    """Initialize the SambaNova LLM with specific parameters"""
    return SambaNovaCloud(
        model="Meta-Llama-3.1-70B-Instruct",
        context_window=10000,
        max_tokens=2048,
        temperature=0.1,
        top_k=1,
        top_p=0.95,
        additional_kwargs={
            "return_raw": True,  # Get raw response
            "format_response": False  # Disable automatic formatting
        }
    )


def search(query: str) -> str:
    """
    Perform DuckDuckGo search
    Args:
        query: user prompt
    return:
        context (str): search results to the user query
    """
    try:
        req = DDGS()
        response = req.text(query, max_results=4)
        context = ""
        for result in response:
            context += result['body']
        return context
    except Exception as e:
        return f"Search failed: {str(e)}"

def setup_agent():
    """Setup the LATS agent with search tool"""
    try:
        llm = initialize_llm()
        Settings.llm = llm
        
        search_tool = FunctionTool.from_defaults(
            fn=search,
            name="search",
            description="Search for product information and reviews"
        )
        
        agent_worker = LATSAgentWorker(
            tools=[search_tool],
            num_expansions=2,
            max_rollouts=2,
            verbose=True,
            llm=llm
        )
        
        return AgentRunner(agent_worker)
    except Exception as e:
        st.error(f"Agent setup failed: {str(e)}")
        return None

def process_recommendation(query: str, agent: AgentRunner):
    """Process the recommendation query with error handling"""
    try:
        response = agent.chat(query).response
        if "I am still thinking." in response:
            return agent.list_tasks()[-1].extra_state["root_node"].children[0].children[0].current_reasoning[-1].observation
        else:
            return response
    except Exception as e:
        return f"An error occurred while processing your request: {str(e)}"

def main():
    st.set_page_config(page_title="Smart Product Recommendation System", layout="wide")
    
    # Title and description
    st.title("🎯 Smart Product Recommendation System")
    st.write("""
    Get personalized product recommendations based on your requirements. 
    Our AI-powered system analyzes current market offerings to find the best match for your needs.
    """)
    
    # Initialize session state for agent
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    
    # Sidebar for API key and preferences
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter SambaNova API Key:", type="password")
        if api_key:
            os.environ["SAMBANOVA_API_KEY"] = api_key
            if st.session_state.agent is None:
                st.session_state.agent = setup_agent()
    
    st.header("What are you looking for?")
    
    # Product category selection
    category = st.selectbox("Select Product Category", list(PRODUCT_CATEGORIES.keys()))
    
    # Create columns for input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        budget = st.number_input("Budget (USD)", min_value=0, max_value=10000, value=1000)
    
    with col2:
        features = st.multiselect(
            "Important Features",
            PRODUCT_CATEGORIES[category]["features"]
        )
    
    custom_requirements = st.text_area("Any additional requirements or preferences?", height=100)
    
    # Generate recommendations
    if st.button("Get Recommendations", type="primary"):
        if not api_key:
            st.error("Please enter your SambaNova API Key in the sidebar first.")
            return
            
        if st.session_state.agent is None:
            st.error("Failed to initialize the recommendation agent. Please check your API key and try again.")
            return
            
        try:
            # Construct the query
            query = f"Looking for a {category.lower()} under ${budget}"
            if features:
                query += f" with {', '.join(features)}"
            if custom_requirements:
                query += f". Additional requirements: {custom_requirements}"
            
            with st.spinner("Analyzing current market offerings..."):
                recommendation = process_recommendation(query, st.session_state.agent)
                
            # Display recommendations
            st.header(f"🎯 Recommended {category}")
            st.write(recommendation)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
    # Help section
    with st.expander("Need Help?"):
        st.write("""
        How to use this recommendation system:
        1. Enter your SambaNova API key in the sidebar
        2. Select a product category
        3. Set your budget and preferences
        4. Specify your requirements
        5. Click 'Get Recommendations' to receive personalized suggestions
        """)

if __name__ == "__main__":
    main()