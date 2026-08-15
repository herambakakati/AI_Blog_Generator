import os
import streamlit as st
from textwrap import dedent
from typing import TypedDict
from urllib.parse import quote_plus

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


# ==========================================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Blog Generator",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================================
# OPENAI API CONFIGURATION
# ==========================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Streamlit Cloud / secrets.toml support
if not OPENAI_API_KEY:
    try:
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        OPENAI_API_KEY = None


# ==========================================================
# API KEY / BILLING CHECK
# ==========================================================

if not OPENAI_API_KEY:

    st.html("""
    <div style="
        max-width:720px;
        margin:70px auto;
        padding:40px;
        background:rgba(255,255,255,0.98);
        border-radius:22px;
        border:1px solid #e2e8f0;
        box-shadow:0 20px 50px rgba(15,23,42,0.12);
        text-align:center;
        font-family:Arial,sans-serif;
    ">

        <div style="
            font-size:48px;
            margin-bottom:12px;
        ">
            🔐
        </div>

        <div style="
            font-size:28px;
            font-weight:800;
            color:#0f172a;
            margin-bottom:12px;
        ">
            AI Access Requires an API Key
        </div>

        <div style="
            font-size:15px;
            line-height:1.7;
            color:#475569;
            margin-bottom:22px;
        ">
            Your OpenAI API key was not found.<br>
            To continue using the AI Blog Generator,
            please add a valid OpenAI API key with
            available billing/credits.
        </div>

        <div style="
            padding:16px 20px;
            border-radius:14px;
            background:#eff6ff;
            border:1px solid #bfdbfe;
            color:#1e40af;
            font-size:14px;
            font-weight:600;
            line-height:1.6;
        ">
            💳 Please add billing/credits to your OpenAI
            account if your API usage requires payment,
            then configure <strong>OPENAI_API_KEY</strong>
            in your environment variables or Streamlit secrets.
        </div>

        <div style="
            margin-top:22px;
            font-size:13px;
            color:#64748b;
        ">
            Required configuration:
            <strong>OPENAI_API_KEY</strong>
        </div>

    </div>
    """)

    st.stop()


# ==========================================================
# OPENAI LLM
# ==========================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=OPENAI_API_KEY
)


# ==========================================================
# SAFE LLM INVOKE
# ==========================================================

def safe_llm_invoke(prompt):

    try:
        return llm.invoke(prompt)

    except Exception as e:

        error_text = str(e).lower()

        # API quota / billing / rate limit
        if any(word in error_text for word in [
            "insufficient_quota",
            "quota",
            "billing",
            "payment_required",
            "rate_limit",
            "429"
        ]):

            st.error(
                "💳 OpenAI API usage limit reached. "
                "Please add billing/credits to your OpenAI account "
                "and try again."
            )
            st.stop()

        # Invalid API key
        if any(word in error_text for word in [
            "authentication",
            "invalid_api_key",
            "incorrect api key",
            "401"
        ]):

            st.error(
                "🔐 Invalid OpenAI API key. "
                "Please configure a valid OPENAI_API_KEY "
                "and try again."
            )
            st.stop()

        # Other API errors
        st.error(
            "⚠️ The AI service could not process your request. "
            "Please check your API configuration and try again."
        )
        st.stop()



# ==========================================================
# GRAPH STATE
# ==========================================================

class GraphState(TypedDict):
    topic: str
    language: str
    blog: str
    decision: str

# IDEA AGENT
def idea_agent(state: GraphState):

    prompt = f"""
Create a clear and structured blog outline for the following topic:

Topic:
{state["topic"]}

Include:
1. Introduction
2. Main points
3. Supporting ideas
4. Conclusion
"""

    outline = safe_llm_invoke(prompt).content

    return {
        "blog": outline
    }

# ==========================================================
# WRITER AGENT
# ==========================================================

def writer_agent(state: GraphState):

    topic = state["topic"]
    language = state["language"]
    outline = state["blog"]

    prompt = f"""
Write a complete, professional and engaging blog.

TOPIC:
{topic}

SELECTED LANGUAGE:
{language}

OUTLINE:
{outline}


LANGUAGE REQUIREMENTS:
- Write the ENTIRE blog in {language}.
- Write the title in {language}.
- Write all headings and subheadings in {language}.
- Write the introduction in {language}.
- Write all body content in {language}.
- Write the conclusion in {language}.
- Do not unnecessarily switch to another language.
- Use natural and grammatically correct {language}.
- If the selected language is Assamese, write naturally in Assamese
  using Assamese script.
- Do not simply translate the title while keeping the body in English.


CONTENT REQUIREMENTS:
- Create a clear and engaging title.
- Use headings and subheadings.
- Write naturally and professionally.
- Make the content informative and useful.
- Include a strong introduction.
- Develop the main points clearly.
- Include a meaningful conclusion.
- Keep the writing easy to read.
- Return only the completed blog.


TOPIC-FRIENDLY EMOJI RULES:
- Use emojis selectively and naturally.
- Choose emojis according to the ACTUAL BLOG TOPIC.
- Do NOT use the same fixed emojis for every blog.
- Emojis should visually support the topic or section.
- Use emojis in the title or selected headings when appropriate.
- Do not put an emoji in every paragraph.
- Do not use unrelated or excessive emojis.
- Keep the blog professional.
- Normally use approximately 3–8 relevant emojis.


TOPIC EXAMPLES:

AI / Technology:
🤖 🧠 💻 ⚙️ 🔬

Finance / Investment:
💰 📈 📊 💳 🏦

Business / Entrepreneurship:
💼 🚀 📊 🤝 🎯

Education / Learning:
🎓 📚 📝 🧠 💡

Health / Fitness:
❤️ 🏥 🩺 🏃 🥗

Travel:
✈️ 🌍 🗺️ 🧳 🏖️

Environment / Sustainability:
🌱 🌍 ♻️ 🌿 🌳

Food / Cooking:
🍳 🍲 👨‍🍳 🥗 🍴

Sports:
🏆 ⚽ 🏏 🏀 🏃

Marketing / Social Media:
📣 📱 🎯 📢 💻

Politics / Government:
🏛️ 🗳️ 📜 🌐 🤝

Science:
🔬 🧪 🧬 🌌 ⚛️

Culture / Festivals:
🎉 🎊 🪔 🎭 🌸

If the topic does not match these examples,
intelligently select suitable topic-specific emojis.

Never force emojis where they do not improve readability.

Return ONLY the complete blog.
"""

    blog = safe_llm_invoke(prompt).content

    return {
        "blog": blog
    }

# ==========================================================
# REWRITE AGENT
# ==========================================================

def rewrite_agent(state: GraphState):

    topic = state["topic"]
    language = state["language"]
    blog = state["blog"]

    prompt = f"""
Improve the following blog while preserving its original topic,
meaning and important information.

TOPIC:
{topic}

SELECTED LANGUAGE:
{language}

CURRENT BLOG:
{blog}


LANGUAGE REQUIREMENTS:
- The improved blog MUST remain completely in {language}.
- Keep the title in {language}.
- Keep all headings in {language}.
- Keep all body content in {language}.
- Keep the conclusion in {language}.
- Do not unnecessarily switch to English.
- If the selected language is Assamese, use natural Assamese
  language and Assamese script.
- Maintain correct grammar and natural writing.


IMPROVE:
- clarity
- grammar
- structure
- readability
- professional tone
- logical flow
- heading quality
- overall presentation


TOPIC-FRIENDLY EMOJI RULES:
- Keep relevant existing emojis.
- Remove emojis that are unrelated to the topic.
- Add emojis when they improve readability or presentation.
- Choose emojis according to the ACTUAL BLOG TOPIC.
- Do NOT use a fixed emoji set.
- Do NOT add an emoji to every paragraph.
- Avoid excessive, childish or unrelated emojis.
- Keep the final blog professional and natural.


Return ONLY the improved blog.
"""

    improved_blog = safe_llm_invoke(prompt).content

    return {
        "blog": improved_blog,
        "decision": ""
    }


#PUBLISH AGENT
def publish_agent(state: GraphState):
    return state

# Add nodes
workflow = StateGraph(GraphState)

# Add generation nodes
workflow.add_node("idea", idea_agent)
workflow.add_node("writer", writer_agent)

# Generation workflow
workflow.add_edge(START, "idea")
workflow.add_edge("idea", "writer")
workflow.add_edge("writer", END)

# Compile generation graph
app_graph = workflow.compile()


# ==========================================================
# 10. CUSTOM CSS
# ==========================================================

st.markdown(
    dedent("""
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        min-height: 100vh;

        background:
            linear-gradient(
                rgba(239, 246, 255, 0.68),
                rgba(248, 250, 252, 0.74)
            ),
            url("https://portalprompts.com.br/blog-image/ia-substitui-redator-entenda-o-que-muda-1779893677.png");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }

    .main .block-container,
    [data-testid="stMainBlockContainer"] {
        max-width: 1700px !important;
        padding-top: 0 !important;
        padding-bottom: 0.5rem !important;
        margin-top: -1px !important;
    }

    [data-testid="stAppViewContainer"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    [data-testid="stAppViewContainer"] > .main {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

   
    [data-testid="stHeader"] {
        height: 0 !important;
        min-height: 0 !important;
        background: transparent !important;
    }

   
    /* ---------- HERO ---------- */

    .hero-banner {
        position: relative;
        overflow: hidden;
        padding: 24px 30px;
        border-radius: 22px;

        background:
            linear-gradient(
                120deg,
                #2563eb,
                #06b6d4,
                #7c3aed,
                #2563eb
            );

        background-size: 300% 300%;
        animation: gradientMove 10s ease infinite;

        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.22);

        box-shadow:
            0 18px 45px rgba(37, 99, 235, 0.22),
            0 4px 12px rgba(15, 23, 42, 0.08);

        margin-top: -4px !important;
        margin-bottom: 20px;
    }

    
    @keyframes gradientMove {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }

    .hero-title {
        position: relative;
        z-index: 2;
        font-size: 36px;
        font-weight: 850;
        margin: 0;
        letter-spacing: -1.0px;
        line-height: 1.1;
        text-shadow:
            0 2px 12px rgba(0, 0, 0, 0.12);
    }

    .hero-subtitle {
        position: relative;
        z-index: 2;
        font-size: 16px;
        margin-top: 1px;
        opacity: 0.96;
        max-width: 850px;
        line-height: 1.55;
    }

    .live-badge {
        position: relative;
        z-index: 2;

        display: inline-flex;
        align-items: center;
        gap: 8px;

        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.4px;
        margin-bottom: 12px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }

    .live-dot {
        width: 9px;
        height: 9px;
        background: #22c55e;
        border-radius: 50%;
        animation: pulse 1.6s infinite;
        box-shadow: 0 0 8px rgba(34, 197, 94, 0.75);
    }

    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.65);
        }

        70% {
            box-shadow: 0 0 0 8px rgba(34, 197, 94, 0);
        }

        100% {
            box-shadow: 0 0 0 0 rgba(34, 197, 94, 0);
        }
    }

    /* ---------- WORKFLOW CARDS ---------- */

    .status-card {
        position: relative;
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(255, 255, 255, 0.96);
        border-radius: 18px;
        padding: 14px 16px;
        text-align: left;
        box-shadow:
            0 10px 28px rgba(15, 23, 42, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.90);

        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        min-height: 104px;
        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;

        display: grid;
        grid-template-columns: 52px 1fr;
        grid-template-rows: auto auto auto;
        column-gap: 12px;
        align-items: center;
        margin-bottom: 17px;
    }

    .status-card:hover {
        transform: translateY(-3px);

        box-shadow:
            0 16px 35px rgba(15, 23, 42, 0.12);
    }

    .status-icon {
        grid-row: 1 / 4;
        font-size: 30px;
        line-height: 1;

        width: 50px;
        height: 50px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 50%;

        background: linear-gradient(
            135deg,
            #eff6ff,
            #eef2ff
        );

        margin-bottom: 0;
    }

    .status-title {
        font-size: 13px;
        color: #64748b;
        font-weight: 600;
    }

    .status-value {
        font-size: 16px;
        font-weight: 800;
        color: #0f172a;
        margin-top: 3px;
    }

    .status-description {
        font-size: 12px;
        color: #64748b;
        margin: -3px 0 0 0;
        line-height: 1.1;
    }

    /* ==========================================================
   ULTRA PREMIUM HUMAN REVIEW
   ========================================================== */

    .review-section {
        margin-top: 12px;
        margin-bottom: 6px;
        padding: 0;
    }

    .review-heading {
        display: flex;
        align-items: center;
        gap: 10px;

        margin: 0 0 5px 0;

        color: #0f172a !important;

        font-size: 25px;
        font-weight: 900;

        letter-spacing: -0.6px;
        line-height: 1.2;

        text-shadow:
            0 1px 2px rgba(255,255,255,0.85);
    }

    .review-heading-icon {
        width: 34px;
        height: 34px;

        display: inline-flex;
        align-items: center;
        justify-content: center;

        border-radius: 11px;

        background:
            linear-gradient(
                135deg,
                #ede9fe,
                #e0e7ff
            );

        font-size: 19px;

        box-shadow:
            0 5px 14px rgba(79,70,229,0.14),
            inset 0 1px 0 rgba(255,255,255,0.95);
    }

    .review-description {
        margin: 0 0 12px 0;

        color: #1e293b !important;

        font-size: 13px;
        font-weight: 600;

        line-height: 1.45;

        text-shadow:
            0 1px 2px rgba(255,255,255,0.9);
    }


    /* ==========================================================
    HUMAN REVIEW BUTTONS
    ========================================================== */


    /* Secondary buttons — Improve & Rewrite */

    div.stButton > button[kind="secondary"] {
        min-height: 48px !important;

        border-radius: 14px !important;

        color: #ffffff !important;

        background:
            linear-gradient(
                135deg,
                #2563eb 0%,
                #0891b2 52%,
                #06b6d4 100%
            ) !important;

        border: 1px solid rgba(255,255,255,0.28) !important;

        font-size: 14px !important;
        font-weight: 800 !important;

        box-shadow:
            0 10px 25px rgba(6,182,212,0.22),
            0 3px 8px rgba(15,23,42,0.10),
            inset 0 1px 0 rgba(255,255,255,0.24);

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }

    div.stButton > button[kind="secondary"]:hover {
        color: #ffffff !important;

        transform: translateY(-3px);

        box-shadow:
            0 16px 34px rgba(6,182,212,0.30),
            0 5px 12px rgba(37,99,235,0.14),
            inset 0 1px 0 rgba(255,255,255,0.28);
    }


        /* Reduce vertical spacing between Streamlit elements */

    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }

    
    /* ==========================================================
    BLOG DRAFT BADGE
    ========================================================== */

    .blog-label {
        display: inline-flex;

        align-items: center;

        gap: 6px;

        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.96),
                rgba(239,246,255,0.94)
            );

        color: #2563eb !important;

        padding: 7px 14px;

        border-radius: 999px;

        font-size: 10px;

        font-weight: 850;

        letter-spacing: 0.55px;

        border: 1px solid rgba(255,255,255,0.95);

        box-shadow:
            0 7px 18px rgba(15,23,42,0.08),
            inset 0 1px 0 rgba(255,255,255,1);

        backdrop-filter: blur(12px);

        -webkit-backdrop-filter: blur(12px);
    }
    /* ---------- BLOG ---------- */

    .blog-card {
        background: transparent !important;
        border: none !important;
        border-radius: 0;
        padding: 0;
        box-shadow: none !important;

        margin-top: 2px;
        margin-bottom: 4px;
    }


    /* ---------- GENERATED BLOG TEXT ---------- */

    /* Main generated blog text */
    div[data-testid="stMarkdownContainer"] {
        color: #0f172a !important;
    }

    /* Paragraphs */
    div[data-testid="stMarkdownContainer"] p {
        color: #0f172a !important;
    }

    /* Headings */
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4,
    div[data-testid="stMarkdownContainer"] h5,
    div[data-testid="stMarkdownContainer"] h6 {
        color: #0f172a !important;
    }

    /* Lists */
    div[data-testid="stMarkdownContainer"] ul,
    div[data-testid="stMarkdownContainer"] ol,
    div[data-testid="stMarkdownContainer"] li {
        color: #0f172a !important;
    }

    /* Bold text */
    div[data-testid="stMarkdownContainer"] strong {
        color: #0f172a !important;
    }

    /* Italic text */
    div[data-testid="stMarkdownContainer"] em {
        color: #0f172a !important;
    }

        /* ---------- BUTTONS ---------- */

    div.stButton > button {
        width: 100%;
        min-height: 46px;
        border-radius: 13px;
        font-weight: 750;
        font-size: 14px;
        color: #0f172a;
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid rgba(203, 213, 225, 0.95);
        box-shadow:
            0 6px 18px rgba(15, 23, 42, 0.06);

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease,
            border-color 0.2s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        border-color: #6366f1;
        color: #4f46e5;
        box-shadow:
            0 12px 25px rgba(79, 70, 229, 0.14);
    }

    /* ==========================================================
    SOCIAL SHARE CARDS
    ========================================================== */

    .social-card-link {
        display: block;
        width: 100%;
        text-decoration: none !important;
    }

    .social-card {
        height: 72px;
        padding: 8px 12px;

        border-radius: 15px;

        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;

        text-align: center;

        color: #ffffff !important;

        border: 1px solid rgba(255,255,255,0.28);

        box-shadow:
            0 10px 25px rgba(15,23,42,0.16),
            inset 0 1px 0 rgba(255,255,255,0.28);

        cursor: pointer;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease,
            filter 0.2s ease;
    }

    .social-card:hover {
        transform: translateY(-3px);

        filter: brightness(1.06);

        box-shadow:
            0 16px 32px rgba(15,23,42,0.22),
            inset 0 1px 0 rgba(255,255,255,0.32);
    }


    /* Facebook */

    .social-facebook {
        background:
            linear-gradient(
                135deg,
                #1877F2,
                #0866FF
            );
    }


    /* LinkedIn */

    .social-linkedin {
        background:
            linear-gradient(
                135deg,
                #0A66C2,
                #005596
            );
    }


    /* X */

    .social-x {
        background:
            linear-gradient(
                135deg,
                #000000,
                #202020
            );
    }


    /* WhatsApp */

    .social-whatsapp {
        background:
            linear-gradient(
                135deg,
                #25D366,
                #128C7E
            );
    }


    /* Small logos */

    .social-logo {
        width: 25px;
        height: 25px;

        margin-bottom: 4px;

        object-fit: contain;
        display: block;

        filter:
            drop-shadow(
                0 2px 4px rgba(0,0,0,0.18)
            );
    }

        /* ---------- LinkedIn Logo ---------- */

    .linkedin-logo {
        width: 25px;
        height: 25px;

        margin-bottom: 4px;

        display: flex;
        align-items: center;
        justify-content: center;

        background: #ffffff;
        color: #0A66C2;

        border-radius: 3px;

        font-family: Arial, Helvetica, sans-serif;
        font-size: 17px;
        font-weight: 900;

        line-height: 1;

        letter-spacing: -1px;

        box-shadow:
            0 2px 5px rgba(0,0,0,0.18);
    }


    /* Platform name */

    .social-name {
        color: #ffffff !important;

        font-size: 11.5px;
        font-weight: 850;

        line-height: 1.1;
    }


    /* Description */

    .social-description {
        color: rgba(255,255,255,0.88) !important;

        font-size: 9px;
        font-weight: 500;

        line-height: 1.1;

        margin-top: 2px;
    }

    


    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        padding: 10px 10px 6px;
        color: #475569;
        font-size: 12px;
        letter-spacing: 0.1px;
        opacity: 0.92;
    }

    .developer {
        color: #2563eb;
        font-weight: 800;
    }


    /* ---------- BENEFIT STRIP ---------- */

    .benefit-strip {
        margin-top: 12px;
        padding: 12px 0 8px;

        background: transparent !important;
        border: none !important;
        box-shadow: none !important;

        backdrop-filter: none;
        -webkit-backdrop-filter: none;
    }

    .benefit-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0;
        max-width: 1000px;
        margin: 0 auto;
    }

    .benefit-item {
        flex: 1;
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0 22px;
    }

    .benefit-icon {
        width: 48px;
        height: 48px;

        display: flex;
        align-items: center;
        justify-content: center;

        flex-shrink: 0;

        border-radius: 50%;

        background: linear-gradient(
            135deg,
            #eff6ff,
            #eef2ff
        );

        font-size: 25px;

        box-shadow:
            0 5px 14px rgba(37, 99, 235, 0.08);
    }

    .benefit-content {
        line-height: 1.15;
    }

    .benefit-title {
        font-size: 15px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 4px;
    }

    .benefit-description {
        font-size: 12px;
        color: #64748b;
    }

    .benefit-divider {
        width: 1px;
        height: 42px;
        background: #e2e8f0;
    }

    .made-with {
        text-align: center;
        margin-top: 14px;
        padding-top: 12px;

        border-top: 1px solid rgba(226, 232, 240, 0.75);

        color: #64748b;
        font-size: 16px;
        font-weight: 600;
    }

        .made-with-heart {
            color: #ef4444;
            font-size: 20px;
            margin: 0 3px;
        }


    /* ==========================================================
       PREMIUM CREATE BLOG CONTROLS
       ========================================================== */

    .section-title {
        color: #0f172a !important;
        font-size: 25px !important;
        font-weight: 850 !important;
        letter-spacing: -0.5px;
        margin-top: 5px !important;
        margin-bottom: 10px !important;
    }


    /* ==========================================================
       GENERATE BUTTON
       ========================================================== */

    div.stButton {
        margin-top: 2px !important;
    }

    div.stButton > button[kind="primary"] {
        min-height: 48px !important;
        padding: 0 22px !important;
        border-radius: 14px !important;
        color: #ffffff !important;
        background:
            linear-gradient(
                135deg,
                #7c3aed 0%,
                #6366f1 45%,
                #2563eb 100%
            ) !important;

        border: 1px solid rgba(255,255,255,0.22) !important;

        font-size: 14px !important;
        font-weight: 800 !important;

        box-shadow:
            0 10px 25px rgba(79,70,229,0.26),
            0 3px 8px rgba(15,23,42,0.08),
            inset 0 1px 0 rgba(255,255,255,0.22) !important;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        color: #ffffff !important;

        transform: translateY(-2px);

        box-shadow:
            0 16px 35px rgba(79,70,229,0.32),
            0 5px 12px rgba(37,99,235,0.12) !important;
    }

    /* ==========================================================
    HUMAN REVIEW — EXACTLY EQUAL BUTTON WIDTH
    ========================================================== */

    div[data-testid="stHorizontalBlock"] div[data-testid="column"] .stButton > button {
        width: 100% !important;
        min-width: 100% !important;
        box-sizing: border-box !important;
    }


    /* ==========================================================
    FINAL CREATE BLOG INPUTS
    ONE SOURCE OF TRUTH
    ========================================================== */

    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"] {
        color-scheme: light !important;
    }


    /* ==========================================================
    LANGUAGE LABEL
    ========================================================== */

    .language-label {
        display: flex !important;
        align-items: center !important;
        gap: 7px !important;

        color: #0f172a !important;
        font-size: 13px !important;
        font-weight: 850 !important;

        margin: 0 0 6px 0 !important;
        padding: 5px !important;
    }


    /* ==========================================================
    LANGUAGE SELECTBOX
    ========================================================== */

    div[data-testid="stSelectbox"] {
        width: 100% !important;

        margin: 0 !important;
        padding: 0 !important;

        min-height: 0 !important;

        background: transparent !important;
        color-scheme: light !important;
    }

    div[data-testid="stSelectbox"] > div {
        margin: 0 !important;
        padding: 0 !important;
    }

    div[data-testid="stSelectbox"] label {
        display: none !important;
    }

    div[data-testid="stSelectbox"]
    div[data-baseweb="select"] {
        width: 100% !important;

        margin: 0 !important;
        padding: 0 !important;

        background: transparent !important;
        color-scheme: light !important;
    }

    div[data-testid="stSelectbox"]
    div[data-baseweb="select"] > div {
        width: 100% !important;

        height: 58px !important;
        min-height: 58px !important;

        box-sizing: border-box !important;

        background: #ffffff !important;
        background-color: #ffffff !important;

        border: 1px solid #dbe3ef !important;
        border-radius: 16px !important;

        box-shadow:
            0 8px 24px rgba(15, 23, 42, 0.08),
            inset 0 1px 0 rgba(255,255,255,1) !important;

        color: #0f172a !important;
        color-scheme: light !important;
    }

    div[data-testid="stSelectbox"]
    div[data-baseweb="select"] span {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;

        font-size: 15px !important;
        font-weight: 700 !important;
    }

    div[data-testid="stSelectbox"]
    div[data-baseweb="select"] svg {
        color: #4f46e5 !important;
        fill: #4f46e5 !important;
    }

    div[data-testid="stSelectbox"]
    div[data-baseweb="select"] > div:hover {
        background: #ffffff !important;
        background-color: #ffffff !important;

        border-color: rgba(99,102,241,0.45) !important;
    }


    /* ==========================================================
    BLOG TOPIC INPUT
    LARGE WHITE BOX
    ========================================================== */

    div[data-testid="stTextInput"] {
        width: 100% !important;

        margin: 10px 0 !important;
        padding: 0 !important;

        background: transparent !important;
        color-scheme: light !important;
    }

    div[data-testid="stTextInput"] label {
        display: none !important;
    }

    div[data-testid="stTextInput"]
    div[data-baseweb="input"] {
        width: 100% !important;

        height: 72px !important;
        min-height: 72px !important;

        box-sizing: border-box !important;

        background: #ffffff !important;
        background-color: #ffffff !important;

        border: 1px solid #dbe3ef !important;
        border-radius: 18px !important;

        box-shadow:
            0 10px 28px rgba(15,23,42,0.08),
            inset 0 1px 0 rgba(255,255,255,1) !important;

        color-scheme: light !important;
    }

    div[data-testid="stTextInput"]
    div[data-baseweb="input"] > div {
        height: 70px !important;
        min-height: 70px !important;

        background: #ffffff !important;
        background-color: #ffffff !important;

        border-radius: 18px !important;
    }

    div[data-testid="stTextInput"]
    div[data-baseweb="input"] input {
        width: 100% !important;

        height: 70px !important;
        min-height: 70px !important;

        box-sizing: border-box !important;

        padding: 0 22px !important;

        background: #ffffff !important;
        background-color: #ffffff !important;

        border: none !important;
        outline: none !important;

        border-radius: 18px !important;

        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;

        font-size: 17px !important;
        font-weight: 600 !important;

        color-scheme: light !important;
    }

    div[data-testid="stTextInput"]
    div[data-baseweb="input"] input::placeholder {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;

        opacity: 1 !important;

        font-size: 16px !important;
        font-weight: 500 !important;
    }

    div[data-testid="stTextInput"]
    div[data-baseweb="input"]:focus-within {
        background: #ffffff !important;
        background-color: #ffffff !important;

        border-color: #6366f1 !important;

        box-shadow:
            0 0 0 4px rgba(99,102,241,0.10),
            0 15px 35px rgba(15,23,42,0.10) !important;
    }


    /* ==========================================================
    DROPDOWN — FORCE WHITE
    ========================================================== */

    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] > div,
    ul[role="listbox"] {
        background: #ffffff !important;
        background-color: #ffffff !important;

        color: #0f172a !important;
        color-scheme: light !important;
    }

    li[role="option"],
    li[role="option"] span {
        background: #ffffff !important;
        background-color: #ffffff !important;

        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }

    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background: #eff6ff !important;
        background-color: #eff6ff !important;

        color: #1d4ed8 !important;
    }


    /* ==========================================================
    BASEWEB DARK-THEME RESET
    ========================================================== */

    div[data-testid="stTextInput"]
    div[data-baseweb="input"],
    div[data-testid="stTextInput"]
    div[data-baseweb="input"] > div,
    div[data-testid="stTextInput"]
    div[data-baseweb="input"] input,
    div[data-testid="stSelectbox"]
    div[data-baseweb="select"],
    div[data-testid="stSelectbox"]
    div[data-baseweb="select"] > div {
        color-scheme: light !important;
    }


    </style>
        """),
        unsafe_allow_html=True
    )

# ==========================================================
# 11. HERO BANNER
# ==========================================================

hero_html = (
    '<div class="hero-banner">'
    '<div class="live-badge"><span class="live-dot"></span>AI ENGINE LIVE</div>'
    '<div class="hero-title">✍️ AI Writing Studio</div>'
    '<div class="hero-subtitle">Turn your ideas into polished, engaging content — crafted, refined, and ready to share.</div>'
    '</div>'
)

st.markdown(hero_html, unsafe_allow_html=True)


# ==========================================================
# 12. WORKFLOW STATUS
# ==========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        dedent("""
        <div class="status-card">
            <div class="status-icon">💡</div>
            <div class="status-value">Idea Generation</div>
            <div class="status-title">Start with an idea</div>
            <div class="status-description">Turn your idea into a clear outline</div>
        </div>
        """),
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        dedent("""
        <div class="status-card">
            <div class="status-icon">📝</div>
            <div class="status-value">Blog Writer</div>
            <div class="status-title">Write your story</div>
            <div class="status-description">Create a complete first draft</div>
        </div>
        """),
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        dedent("""
        <div class="status-card">
            <div class="status-icon">👤</div>            
            <div class="status-value">Review</div>
            <div class="status-title">Your choice</div>
            <div class="status-description">Review and approve before sharing</div>
        </div>
        """),
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        dedent("""
        <div class="status-card">
            <div class="status-icon">🚀</div>
            <div class="status-value">Publish</div>
            <div class="status-title">Share with others</div>            
            <div class="status-description">Get your finished blog ready to share</div>
        </div>
        """),
        unsafe_allow_html=True
    )


# ==========================================================
# 13. CREATE BLOG
# ==========================================================

st.markdown(
    """
    <div class="section-title">
        🎯 Create Your Blog
    </div>
    """,
    unsafe_allow_html=True
)



# ----------------------------------------------------------
# LANGUAGE SELECTION
# ----------------------------------------------------------

st.markdown(
    """
    <div class="language-label">
        <span style="font-size:15px;">🌐</span>
        <span>Select Blog Language</span>
    </div>
    """,
    unsafe_allow_html=True
)

language = st.selectbox(
    "Choose the language for your blog",
    [
        "English",
        "অসমীয়া (Assamese)"
    ],
    index=0,
    label_visibility="collapsed",
    key="blog_language"
)


# ----------------------------------------------------------
# BLOG TOPIC
# ----------------------------------------------------------

topic = st.text_input(
    "Blog Topic",
    placeholder="Write a blog about...",
    label_visibility="collapsed"
)

# ==========================================================
# GENERATE BLOG
# ==========================================================

if st.button(
    "✨ Generate Professional Blog",
    type="primary",
    key="generate_blog"
):

    if not topic.strip():
        st.warning("Please enter a blog topic.")
        st.stop()

    with st.spinner("Creating your blog..."):

        initial_state: GraphState = {
            "topic": topic.strip(),
            "language": language,
            "blog": "",
            "decision": ""
        }

        result = app_graph.invoke(initial_state)

        st.session_state.topic = topic.strip()
        st.session_state.language = language
        st.session_state.blog = result["blog"]
        st.session_state.decision = ""
        st.session_state.status = "review"

    st.rerun()


# ==========================================================
# 14. BLOG PREVIEW
# ==========================================================

if "blog" in st.session_state:
    
    st.markdown(
        dedent("""
        <div class="blog-card">
            <div class="blog-label">
                YOUR BLOG DRAFT
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )

    selected_language = st.session_state.get(
        "language",
        "English"
    )

    st.markdown(
        f"""
        <div style="
            display:inline-flex;
            align-items:center;
            padding:8px 12px;
            margin:12px 0 10px 0;
            border-radius:999px;
            background:rgba(37,99,235,0.10);
            border:1px solid rgba(37,99,235,0.18);
            color:#1d4ed8;
            font-size:11px;
            font-weight:800;
        ">
            🌐 {selected_language}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(st.session_state.blog)
 

    # ==========================================================
    # 15. HUMAN REVIEW
    # ==========================================================

    if st.session_state.get("status") == "review":

        # ------------------------------------------------------
        # Human Review heading
        # Use st.html() so HTML is rendered as HTML,
        # not displayed as a code block.
        # ------------------------------------------------------

        review_html = dedent("""
        <div class="review-section">

            <div class="review-heading">
                <span class="review-heading-icon">👤</span>
                <span>Human Review</span>
            </div>

            <div class="review-description">
                Review the AI-generated content before publishing.
            </div>

        </div>
        """)

        st.html(review_html)

        # ------------------------------------------------------
        # Review actions
        # ------------------------------------------------------

        review_col1, review_col2 = st.columns(2)

        # ------------------------------------------------------
        # APPROVE & PUBLISH
        # ------------------------------------------------------

        with review_col1:

            if st.button(
                "🚀 Approve & Publish",
                type="primary",
                key="approve_publish",
                use_container_width=True
            ):

                publish_state: GraphState = {
                    "topic": st.session_state.topic,
                    "language": st.session_state.language,
                    "blog": st.session_state.blog,
                    "decision": "approve"
                }

                result = publish_agent(publish_state)

                st.session_state.blog = result["blog"]
                st.session_state.decision = "approve"
                st.session_state.status = "published"

                st.rerun()

        # ------------------------------------------------------
        # IMPROVE & REWRITE
        # ------------------------------------------------------

        with review_col2:

            if st.button(
                "🔄 Improve & Rewrite",
                type="secondary",
                key="rewrite_blog",
                use_container_width=True
            ):

                rewrite_state: GraphState = {
                    "topic": st.session_state.topic,
                    "language": st.session_state.language,
                    "blog": st.session_state.blog,
                    "decision": "rewrite"
                }

                with st.spinner("AI is improving your blog..."):

                    result = rewrite_agent(rewrite_state)

                st.session_state.blog = result["blog"]
                st.session_state.decision = ""
                st.session_state.status = "review"

                st.rerun()

            
    # ==========================================================
    # 16. PUBLISHED STATE
    # ==========================================================

    if st.session_state.get("status") == "published":

        # ------------------------------------------------------
        # Prepare sharing content
        # ------------------------------------------------------

        blog_text = st.session_state.get("blog", "")
        topic_text = st.session_state.get("topic", "AI Blog")

        share_text = (
            f"{topic_text}\n\n"
            f"Created with AI Blog Generator.\n\n"
            f"{blog_text[:500]}"
        )

        encoded_text = quote_plus(share_text)


        # ------------------------------------------------------
        # Social sharing URLs
        # ------------------------------------------------------

        # Facebook
        facebook_url = (
            "https://www.facebook.com/sharer/sharer.php"
            f"?quote={encoded_text}"
        )


        # LinkedIn
        # LinkedIn Share
        linkedin_url = (
            "https://www.linkedin.com/feed/?shareActive=true"
            f"&text={encoded_text}"
        )


        # X
        x_url = (
            "https://x.com/intent/post"
            f"?text={encoded_text}"
        )


        # WhatsApp
        whatsapp_url = (
            "https://wa.me/"
            f"?text={encoded_text}"
        )

        # ------------------------------------------------------
        # APPROVED MESSAGE
        # ------------------------------------------------------

        st.html("""
        <div style="
            margin-top:18px;
            padding:18px 22px;
            border-radius:18px;
            
        ">

            <div style="
                font-size:22px;
                font-weight:850;
                color:#166534;
                margin-bottom:5px;
            ">
                ✓ Blog Approved & Ready to Share
            </div>

            <div style="
                font-size:14px;
                font-weight:600;
                color:#475569;
            ">
                Your blog has passed human review.
                Choose a platform below to share it with your audience.
            </div>

        </div>
        """)

        # ------------------------------------------------------
        # SHARE HEADING
        # ------------------------------------------------------

        st.markdown(
            '<div class="section-title">📣 Share Your Blog</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="
                color: #000000 !important;
                font-size: 13px;
                font-weight: 600;
                margin-top: -4px;
                margin-bottom: 12px;
            ">
                Share your approved blog directly through your preferred platform.
            </div>
            """,
            unsafe_allow_html=True
        )

        # ------------------------------------------------------
        # SOCIAL MEDIA LOGO CARDS
        # One card = one direct share action
        # ------------------------------------------------------

        social1, social2, social3, social4 = st.columns(4)


        # ======================================================
        # FACEBOOK
        # ======================================================

        with social1:

            st.html(f"""
            <a
                class="social-card-link"
                href="{facebook_url}"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="Share on Facebook"
            >

                <div class="social-card social-facebook">

                    <img
                        class="social-logo"
                        src="https://cdn.simpleicons.org/facebook/FFFFFF"
                        alt="Facebook"
                    >

                    <div class="social-name">
                        Facebook
                    </div>

                    <div class="social-description">
                        Share with your audience
                    </div>

                </div>

            </a>
            """)


        # ======================================================
        # LINKEDIN
        # ======================================================

        with social2:

            st.html(f"""
            <a
                class="social-card-link"
                href="{linkedin_url}"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="Share on LinkedIn"
            >

                <div class="social-card social-linkedin">

                    <div class="linkedin-logo" aria-label="LinkedIn">
                        in
                    </div>

                    <div class="social-name">
                        LinkedIn
                    </div>

                    <div class="social-description">
                        Share professionally
                    </div>

                </div>

            </a>
            """)
        # ======================================================
        # X
        # ======================================================

        with social3:

            st.html(f"""
            <a
                class="social-card-link"
                href="{x_url}"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="Share on X"
            >

                <div class="social-card social-x">

                    <img
                        class="social-logo"
                        src="https://cdn.simpleicons.org/x/FFFFFF"
                        alt="X"
                    >

                    <div class="social-name">
                        X
                    </div>

                    <div class="social-description">
                        Share your blog
                    </div>

                </div>

            </a>
            """)


        # ======================================================
        # WHATSAPP
        # ======================================================

        with social4:

            st.html(f"""
            <a
                class="social-card-link"
                href="{whatsapp_url}"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="Share on WhatsApp"
            >

                <div class="social-card social-whatsapp">

                    <img
                        class="social-logo"
                        src="https://cdn.simpleicons.org/whatsapp/FFFFFF"
                        alt="WhatsApp"
                    >

                    <div class="social-name">
                        WhatsApp
                    </div>

                    <div class="social-description">
                        Share with contacts
                    </div>

                </div>

            </a>
            """)
            
# ==========================================================
# 17. BENEFITS
# ==========================================================

benefits_html = dedent("""
<div class="benefit-strip">

    <div class="benefit-container">

        <div class="benefit-item">

            <div class="benefit-icon">⚡</div>

            <div class="benefit-content">
                <div class="benefit-title">
                    Fast & Simple
                </div>

                <div class="benefit-description">
                    Generate a professional first draft in minutes
                </div>
            </div>

        </div>

        <div class="benefit-divider"></div>

        <div class="benefit-item">

            <div class="benefit-icon">🛡️</div>

            <div class="benefit-content">
                <div class="benefit-title">
                    Human-in-the-Loop
                </div>

                <div class="benefit-description">
                    Review, refine, and approve before publishing
                </div>
            </div>

        </div>

        <div class="benefit-divider"></div>

        <div class="benefit-item">

            <div class="benefit-icon">🌐</div>

            <div class="benefit-content">
                <div class="benefit-title">
                    Ready to Share
                </div>

                <div class="benefit-description">
                    Prepare your content for your preferred platform
                </div>
            </div>

        </div>

    </div>

    <div class="made-with">
        Made with
        <span class="made-with-heart">♥</span>
        to help you create better content
    </div>

</div>
""")

st.html(benefits_html)


# ==========================================================
# 18. FOOTER
# ==========================================================

footer_html = (
    '<div class="footer">'
    '<div>Developed by <span class="developer">Heramba Kakati</span></div>'
    '<div style="margin-top:0px;">AI Blog Generator • Create, review, refine, and share</div>'
    '</div>'
)

st.markdown(footer_html, unsafe_allow_html=True)
