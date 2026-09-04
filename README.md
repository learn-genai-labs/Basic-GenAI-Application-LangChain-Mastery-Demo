# Architecture & Development Workflow: Global & Indian Culinary Advisor

An interactive, production-ready Generative AI culinary assistant built using **LangChain Expression Language (LCEL)**, **OpenAI**, and **Streamlit**.

---

Pattern 1: Direct Prompting & Memory (What you just built)

How it works: Relies purely on the model's pre-trained parametric knowledge, system prompt rules, and conversational context windows.

Best for: Creative tasks, general reasoning, summarization, brainstorming, and format transformation.

Limitations: Prone to hallucinating obscure facts; lacks access to private internal company data; limited by context window limits and token costs.

---

## 🔄 Architectural Deep Dive: Specialized Modules

### 1. Two-Tier Classification & Cuisine Mismatch Guardrail

#### Problem Statement
LLMs are naturally permissive. If a user sets the cuisine category to **Pan-Indian** and requests **Pasta** or **Pizza**, naive models generate an Italian recipe without acknowledging the cross-cultural conflict. Conversely, rigid keyword matching falsely flags generic universal ingredients (e.g., classifying raw "tomato" as an invalid dish).

#### Solution: Two-Tier Gatekeeper Pattern
Before invoking the primary generative model, user queries pass through an upstream classification chain (`validator_chain`):

---

#### Key Implementation Details:
* **Zero Temperature:** `temperature=0.0` ensures deterministic, repeatable entity classification.
* **Separation of Concerns:** The validator strictly identifies the cultural origin of the query without generating recipe prose, keeping token usage under ~20 tokens.
* **Non-Blocking Feedback:** If a mismatch is detected, execution halts before calling the heavier generation chain, preventing wasted tokens and displaying a dismissible warning banner.

---

### 2. Top-Feed Inverted Feed & Auto-Focus Mechanism

#### Problem Statement
In chat interfaces where inputs append to the bottom of the screen, users must repeatedly scroll past lengthy past outputs to see new generations. Furthermore, Streamlit's full-page reruns cause the browser to reset scroll positions to the top and lose focus on text inputs.

#### Solution: Inverted DOM Flow with Target Streaming

1. **Top-Pinned Layout Architecture:**
   * `st.chat_input()` is positioned at the top of the workspace.
   * Directly beneath the input, an `st.empty()` container is reserved exclusively for the active token stream.
   * Historical turns are rendered below the stream using Python's `reversed()` function:

   ```python
   # Group linear message history into discrete user/assistant turns
   turns = [history[i:i + 2] for i in range(0, len(history), 2)]

   # Render in reverse order: newest turn at the top, older turns below
   for turn in reversed(turns):
       for msg in turn:
           with st.chat_message(msg["role"]):
               st.markdown(msg["content"])
   

DOM Cursor Auto-Focus:

Streamlit widgets unmount and remount during script reruns.

A zero-height JavaScript iframe hooks into the parent document DOM to restore cursor focus to the textarea without manual user clicks:

JavaScript
const focusInput = () => {
    const textarea = window.parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
    if (textarea) textarea.focus();
};
setTimeout(focusInput, 150);
User Experience Impact:
Zero Scrolling: The latest generated output always begins immediately below the input field.

Continuous Typing: Cursor focus is restored automatically, allowing rapid multi-turn follow-ups without manual navigation.

---

## 🛠️ Step-by-Step Implementation

### Step 1: Environment Setup & API Key Isolation
* **Objective:** Establish an isolated runtime and prevent credential leaks.
* **LangChain Integration:** Used `python-dotenv` to decouple secrets from source code.
* **Mechanism:**
  * Created a virtual environment (`venv`) to avoid dependency conflicts.
  * Extracted the `OPENAI_API_KEY` into a `.env` file, loaded dynamically at runtime via `load_dotenv()`.

---

### Step 2: The Core LangChain Expression Language (LCEL) Pipeline
* **Objective:** Construct a deterministic, modular pipeline connecting prompts, models, and parsers.
* **LangChain Components:**
  * **`ChatPromptTemplate`**: Formats raw system instructions and dynamic user parameters into a structured message payload (`SystemMessage` + `HumanMessage`).
  * **`ChatOpenAI`**: Connects to the foundation model (`gpt-4o-mini`) configured with `streaming=True`.
  * **`StrOutputParser`**: Extracts the raw text string from the model's `AIMessage` payload, stripping metadata, token counts, and HTTP headers.
  * **LCEL Pipe Operator (`|`)**: Chains components declaratively:
    ```text
    prompt | llm | output_parser
    ```

---

### Step 3: Conversation Memory Integration
* **Objective:** Enable multi-turn dialogue so the model remembers earlier recipes and modifications (e.g., modifying "pizza" to "chicken pizza").
* **LangChain Components:**
  * **`MessagesPlaceholder(variable_name="history")`**: Reserves a dynamic insertion point in the prompt template for past conversation history.
  * **`ChatMessageHistory`**: An in-memory data store holding chronological lists of `HumanMessage` and `AIMessage` objects.
  * **`RunnableWithMessageHistory`**: Wraps the base chain to automatically load context for a specific `session_id`, pass it to the model, and persist the new turn.

---

### Step 4: Multi-Chain Routing & Cuisine Validation
* **Objective:** Prevent hallucinations and cross-cuisine inconsistencies (e.g., asking for Italian "Pasta" under a "Pan-Indian" setting).
* **LangChain Architecture:**
  * **Classification Chain (`validator_chain`)**: A deterministic zero-temperature LLM call evaluates whether user input is a generic base ingredient (`tomato`) or an explicit prepared dish (`pizza`) originating from another cuisine.
  * **Generation Chain (`base_chain`)**: Only invoked if validation passes. If an authentic dish mismatch occurs, the application stops generation and raises a non-blocking warning banner.

---

### Step 5: Real-Time Token Streaming
* **Objective:** Eliminate perceived API latency with instant typewriter-style output.
* **Mechanism:**
  * Swapped static `.invoke()` for `chain.stream()`.
  * Fed LangChain's chunk generator directly into Streamlit's native `st.write_stream()`, rendering tokens to the DOM the millisecond they are returned by OpenAI.

---

### Step 6: Frontend & UX Optimization (Streamlit)
* **Objective:** Deliver a clean, responsive enterprise interface.
* **UI Features:**
  * **Pure White Canvas**: Overrode Streamlit's dark-mode defaults using scoped CSS to enforce `#ffffff` backgrounds and high-contrast dark typography (`#111827`).
  * **Two-Column Layout**: Left rail houses parameters (Cuisine, Diet, Engine metadata); the right column serves as the conversation workspace.
  * **Inverted Feed (Top-Down Flow)**: Positioned `st.chat_input` and the active streaming container at the top of the workspace, rendering past messages underneath in reverse order to eliminate manual scrolling.
  * **DOM Auto-Focus**: Injected a lightweight JavaScript hook via `components.html` to keep the text cursor focused inside the chat input box across all state reruns.

---

## 🧩 LangChain Component Reference

| Component | Class / Method | Purpose in Project |
| :--- | :--- | :--- |
| **Prompt Template** | `ChatPromptTemplate` | Injects system personas, user parameters, and dietary constraints |
| **Chat Model** | `ChatOpenAI` | Handles inference with `gpt-4o-mini` and token streaming |
| **Output Parsing** | `StrOutputParser` | Converts raw `AIMessage` payloads into clean markdown strings |
| **Stateful Memory** | `RunnableWithMessageHistory` | Manages multi-turn conversation logs linked to session IDs |
| **Execution Engine** | `chain.stream()` | Yields text tokens incrementally for real-time rendering |
