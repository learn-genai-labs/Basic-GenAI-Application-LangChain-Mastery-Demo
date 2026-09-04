import os
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. Page Configuration
st.set_page_config(
    page_title="Rasoi & Global · Culinary Advisor",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Clean Crisp White Design System with High-Contrast Typography
st.markdown("""
<style>
    /* Global Reset & Pure White Canvas */
    html, body, [class*="css"], .stApp {
        background-color: #ffffff !important;
        color: #111827 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1350px !important;
    }

    /* Header styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #111827 !important;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #6b7280 !important;
        margin-bottom: 1rem;
    }

    /* Top Banner */
    .top-banner {
        background-color: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        padding: 0.7rem 1rem !important;
        font-size: 0.88rem !important;
        color: #4b5563 !important;
        margin-bottom: 1.25rem !important;
    }

    /* Section Labels */
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280 !important;
        margin-bottom: 0.85rem;
    }

    /* Left Panel Card */
    .card-box {
        background-color: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        padding: 1.15rem !important;
        margin-top: 1rem !important;
    }
    .card-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280 !important;
        margin-bottom: 0.35rem;
    }
    .card-val {
        font-size: 0.92rem;
        font-weight: 600;
        color: #111827 !important;
        margin-bottom: 0.75rem;
    }

    /* Target Chat Cards and Enforce Full High-Contrast Dark Text */
    [data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        padding: 1.25rem !important;
        margin-bottom: 0.85rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    }

    [data-testid="stChatMessage"] * {
        color: #111827 !important;
    }

    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3,
    [data-testid="stChatMessage"] h4 {
        color: #111827 !important;
        font-weight: 700 !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        color: #1f2937 !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
    }

    [data-testid="stChatMessage"] ul,
    [data-testid="stChatMessage"] ol {
        color: #1f2937 !important;
        margin-left: 1.25rem !important;
        padding-left: 0.5rem !important;
    }

    [data-testid="stChatMessage"] li {
        color: #1f2937 !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
        margin-bottom: 0.35rem !important;
    }

    [data-testid="stChatMessage"] strong {
        color: #111827 !important;
        font-weight: 700 !important;
    }

    /* Blue Action Button */
    .stButton > button {
        background-color: #0071e3 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.15s ease-in-out !important;
    }
    .stButton > button:hover {
        background-color: #005bb5 !important;
    }

    /* Chat Input Styling */
    [data-testid="stChatInput"] {
        padding-bottom: 1.25rem !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 1.5px solid #111827 !important;
        border-radius: 8px !important;
        font-size: 0.92rem !important;
        padding: 0.75rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #0071e3 !important;
        box-shadow: 0 0 0 1px #0071e3 !important;
    }

    /* Clean Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 3. Environment & LangChain Setup
load_dotenv()

# Pre-check validator to detect explicit dish mismatches
@st.cache_resource
def get_validation_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict Culinary Origin Detector.\n"
         "Analyze if the user's input specifies a distinct PREPARED DISH and identify its authentic culture/cuisine of origin.\n"
         "Rules:\n"
         "1. Single ingredients, produce, or conversational modification instructions (e.g., 'tomato', 'paneer', 'make it vegan', 'add chicken', 'more spicy') are NOT dish mismatches. Answer: 'INGREDIENT'.\n"
         "2. If it is a prepared dish, name ONLY the single broad cuisine origin from this list: "
         "['Pan-Indian', 'South Indian', 'North Indian', 'Chettinad', 'Italian', 'Mediterranean', 'Mexican', 'East Asian', 'Middle Eastern', 'French'].\n"
         "Examples:\n"
         "- 'pizza' -> 'Italian'\n"
         "- 'pasta' -> 'Italian'\n"
         "- 'sambar' -> 'South Indian'\n"
         "- 'tacos' -> 'Mexican'\n"
         "- 'make this as chicken pizza' -> 'Italian'\n"
         "- 'tomato' -> 'INGREDIENT'\n"
         "Reply with ONLY the single word."
        ),
        ("human", "{user_text}")
    ])
    return prompt | llm | StrOutputParser()

# Recipe generator chain
@st.cache_resource
def get_recipe_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, streaming=True)

    system_instructions = (
        "You are Rasoi, an elite Culinary Consultant with deep expertise in regional Indian traditions "
        "and authentic international cuisines.\n\n"
        "### CONVERSATIONAL MODIFICATIONS & ADAPTABILITY\n"
        "- Follow all conversational requests, modifications, and ingredient swaps (e.g., 'make this as chicken pizza', 'no onion/garlic', 'make it spicy').\n"
        "- When modifying a previous dish, preserve the base preparation while seamlessly updating ingredients, cook steps, and technique.\n\n"
        "### OUTPUT STRUCTURE\n"
        "Format every recipe cleanly with these exact markdown headers:\n"
        "### Dish Overview\n"
        "Name, regional origin, and flavor profile.\n\n"
        "### Key Ingredients\n"
        "Itemized list with measurements and approved substitutes.\n\n"
        "### Step-by-Step Method\n"
        "Clear, numbered steps detailing heat control, cooking sequence, and seasoning.\n\n"
        "### Chef's Pro-Tip\n"
        "Specific advice on spice balance, technique, or texture preservation.\n\n"
        "### Custom Dietary & Modification Notes\n"
        "Explicitly detail any ingredient additions, swaps, or exclusions requested by the user in this turn."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instructions),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Selected Cuisine: {region} | Base Diet: {diet}\nUser Request: {user_text}")
    ])

    return prompt | llm | StrOutputParser()

validator_chain = get_validation_chain()
base_chain = get_recipe_chain()

# 4. Session State & Memory Setup
if "session_store" not in st.session_state:
    st.session_state.session_store = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "mismatch_notice" not in st.session_state:
    st.session_state.mismatch_notice = None

def get_history(session_id: str):
    if session_id not in st.session_state.session_store:
        st.session_state.session_store[session_id] = ChatMessageHistory()
    return st.session_state.session_store[session_id]

chain_with_history = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="user_text",
    history_messages_key="history"
)

# 5. Page Headers
st.markdown('<div class="main-title">Global & Indian Culinary Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-assisted authentic regional & international recipes with conversational customization.</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="top-banner">Enter an ingredient, a dish name, or ask for modifications. New requests and recipes appear right at the top for easy reading.</div>',
    unsafe_allow_html=True
)

# 6. Two-Column Layout (Left: Settings & Controls, Right: Conversational Feed)
col_left, col_right = st.columns([0.33, 0.67], gap="large")

with col_left:
    st.markdown('<div class="section-label">Culinary Preferences</div>', unsafe_allow_html=True)

    region = st.selectbox(
        "Cuisine Style",
        [
            "Pan-Indian",
            "South Indian (Tamil / Kerala / Andhra)",
            "North Indian (Punjabi / Awadhi)",
            "Chettinad",
            "Italian",
            "Mediterranean & Greek",
            "Mexican",
            "East Asian (Japanese / Thai / Chinese)",
            "Middle Eastern / Levantine",
            "Continental & French"
        ],
        index=0
    )

    diet = st.selectbox(
        "Dietary Preference",
        [
            "Standard",
            "Vegetarian",
            "Vegan",
            "Jain (No Root Vegetables)",
            "Gluten-Free",
            "High Protein"
        ],
        index=0
    )

    if st.button("Reset Conversation", use_container_width=True):
        st.session_state.session_store.clear()
        st.session_state.chat_history = []
        st.session_state.mismatch_notice = None
        st.rerun()

    # Engine Specs Card
    st.markdown("""
    <div class="card-box">
        <div class="card-title">AI Engine</div>
        <div class="card-val">gpt-4o-mini via LangChain</div>
        <div class="card-title">Top-Feed Layout</div>
        <div style="font-size: 0.85rem; color: #4b5563; line-height: 1.4; margin-bottom: 0.75rem;">
            Newest recipe renders directly at the top. Older conversation flows underneath.
        </div>
        <div class="card-title">Cuisine Validation</div>
        <div style="font-size: 0.85rem; color: #4b5563; line-height: 1.4;">
            Flags explicit dish mismatches against the selected category.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-label">Conversation Workspace</div>', unsafe_allow_html=True)

    # Display Warning Banner if a mismatch was caught
    if st.session_state.mismatch_notice:
        st.warning(st.session_state.mismatch_notice, icon="⚠️")
        if st.button("Dismiss Alert", key="dismiss_alert"):
            st.session_state.mismatch_notice = None
            st.rerun()

    # 1. Chat Input positioned at the top of the workspace
    prompt_text = st.chat_input("Ask for a recipe, ingredient swap, or dietary modification...")

    # Placeholder reserved for live streaming the newest turn directly below input
    active_stream_box = st.empty()

    if prompt_text:
        st.session_state.mismatch_notice = None

        # Pre-Validation Check
        detected_origin = validator_chain.invoke({"user_text": prompt_text}).strip()

        is_mismatch = (
            detected_origin != "INGREDIENT" and
            detected_origin != "VALID" and
            detected_origin.lower() not in region.lower()
        )

        if is_mismatch:
            st.session_state.mismatch_notice = (
                f"**Cuisine Mismatch:** '{prompt_text}' is traditionally an **{detected_origin}** dish, "
                f"but your active selection is **{region}**.\n\n"
                f"Please select **{detected_origin}** in the sidebar to view the authentic version, "
                f"or ask for an Indian fusion version (e.g., 'Desi Masala {prompt_text}')."
            )
            st.rerun()
        else:
            # Stream directly into the top container
            with active_stream_box.container():
                with st.chat_message("user"):
                    st.markdown(prompt_text)

                with st.chat_message("assistant"):
                    def generate_recipe():
                        stream = chain_with_history.stream(
                            {
                                "region": region,
                                "diet": diet,
                                "user_text": prompt_text
                            },
                            config={"configurable": {"session_id": "culinary_session"}}
                        )
                        for chunk in stream:
                            yield chunk

                    response_output = st.write_stream(generate_recipe)

            # Store in chronological order for LangChain history
            st.session_state.chat_history.append({"role": "user", "content": prompt_text})
            st.session_state.chat_history.append({"role": "assistant", "content": response_output})
            st.rerun()

    # 2. Render Feed in Reverse Order (Newest turn on top, older entries below)
    if st.session_state.chat_history:
        # Group messages into user/assistant turns
        history = st.session_state.chat_history
        turns = [history[i:i + 2] for i in range(0, len(history), 2)]

        # Reverse so newest turn is at the top
        for turn in reversed(turns):
            for msg in turn:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
    elif not st.session_state.mismatch_notice:
        st.markdown(
            """
            <div style="border: 1px dashed #d1d5db; border-radius: 8px; padding: 2.5rem; text-align: center; color: #6b7280;">
                <p style="font-size: 1.1rem; font-weight: 600; color: #374151; margin-bottom: 0.4rem;">What would you like to cook?</p>
                <p style="font-size: 0.9rem; margin-bottom: 0;">Type any dish or ingredient in the input box above. Your recipe will appear right here at the top.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# 7. JavaScript: Automatically keep the cursor focused in the top input box
components.html(
    """
    <script>
        const focusTopInput = () => {
            const inputArea = window.parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
            if (inputArea) {
                inputArea.focus();
            }
        };
        setTimeout(focusTopInput, 150);
        setTimeout(focusTopInput, 400);
    </script>
    """,
    height=0,
    width=0
)