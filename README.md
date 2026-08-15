# ✍️ AI Writing Studio

### AI-Powered Blog Generation, Human Review & Social Sharing

**AI Writing Studio** is a Streamlit-based Generative AI application that turns a simple topic into a polished, professional blog through a structured agentic workflow.

The application uses **LangGraph** to orchestrate idea generation and blog writing, **OpenAI GPT-4o-mini** for content generation and refinement, and **Streamlit** for the interactive user experience.

> **Workflow:** 💡 Idea Generation → 📝 Blog Writing → 👤 Human Review → 🔄 Improve & Rewrite → 🚀 Approve & Publish → 📣 Share

---

## 📌 Project Overview

Creating high-quality blog content often requires several stages: developing an outline, writing the first draft, reviewing the content, refining the writing, and preparing the final version for publication.

AI Writing Studio brings these steps together in one application.

The user enters a topic and selects a language. The application generates a structured outline, writes the complete blog, presents the result for human review, and allows the user to either approve the content or request an AI-powered rewrite.

The approved content can then be prepared for sharing through supported social platforms.

---

## ✨ Key Features

- 🤖 **AI-powered blog generation**
- 💡 **Automatic outline and idea generation**
- 📝 **Professional blog writing**
- 🌐 **Multi-language content generation**
- 🇮🇳 **English and Assamese support**
- 🔄 **AI-powered content improvement**
- 👤 **Human-in-the-Loop review**
- 🚀 **Approve & Publish workflow**
- 📣 **Social sharing through Facebook, LinkedIn, X and WhatsApp**
- 🎨 **Premium Streamlit user interface**
- 🔐 **OpenAI API key and billing/credit validation**
- ⚡ **LangGraph-based workflow orchestration**
- 🎯 **Topic-aware emoji selection**

The application is designed so that the generated content remains in the selected language, including the title, headings, body, and conclusion. Assamese content is specifically instructed to use Assamese script.

---

## 🧠 Agentic AI Workflow

The application separates content creation into dedicated AI stages.

```text
                    ┌───────────────────┐
                    │    User Topic     │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   Idea Agent      │
                    │ Create Blog       │
                    │ Outline           │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  Writer Agent     │
                    │ Generate Complete │
                    │ Blog              │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   Human Review    │
                    └─────────┬─────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
       ┌──────────────────┐      ┌──────────────────┐
       │ Improve & Rewrite │      │ Approve & Publish│
       └─────────┬────────┘      └─────────┬────────┘
                 │                         │
                 └────────────┐   ┌────────┘
                              ▼   ▼
                         📣 Share
```

### LangGraph Flow

The generation graph is implemented with:

```python
START → idea → writer → END
```

The application then uses the generated blog as the basis for human review, rewriting, approval, and sharing.

---

## 🔄 Application Workflow

### 1. Enter a Topic

Example:

```text
The Future of Artificial Intelligence in Education
```

### 2. Select a Language

Available options:

- English
- অসমীয়া (Assamese)

### 3. Generate the Blog

Click:

**✨ Generate Professional Blog**

The Idea Agent first creates a structured outline containing:

1. Introduction
2. Main points
3. Supporting ideas
4. Conclusion

The Writer Agent then converts that outline into a complete blog.

### 4. Review the Draft

The generated blog is displayed as a draft.

The user can choose:

**🚀 Approve & Publish**

or

**🔄 Improve & Rewrite**

### 5. Improve & Rewrite

The Rewrite Agent improves:

- Clarity
- Grammar
- Structure
- Readability
- Professional tone
- Logical flow
- Heading quality
- Overall presentation

The original topic, meaning, and important information are preserved.

### 6. Approve & Publish

After human review, the user can approve the blog.

The application then displays:

> **✓ Blog Approved & Ready to Share**

### 7. Share

The approved content can be prepared for:

- Facebook
- LinkedIn
- X
- WhatsApp

---

## 🌐 Language Support

### 🇬🇧 English

Generate complete professional blog content in English.

### অসমীয়া Assamese

Generate complete blog content in Assamese using Assamese script.

The application explicitly instructs the AI not to unnecessarily switch languages and to keep the title, headings, body, and conclusion in the selected language.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core application development |
| **Streamlit** | Web application and user interface |
| **LangGraph** | Agent workflow orchestration |
| **LangChain OpenAI** | OpenAI LLM integration |
| **OpenAI GPT-4o-mini** | Blog generation and rewriting |
| **python-dotenv** | Environment variable management |
| **HTML/CSS** | Premium UI customization |

---

## 🏗️ Project Structure

```text
AI_Blog_Generator/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

### `app.py`

Main application containing:

- OpenAI configuration
- API key validation
- LangGraph workflow
- Idea Agent
- Writer Agent
- Rewrite Agent
- Publish workflow
- Human review
- Social sharing
- Streamlit UI
- Custom CSS

### `requirements.txt`

Contains the Python packages required to run the application.

### `.env`

Stores the OpenAI API key for local development.

### `.gitignore`

Prevents sensitive and unnecessary files from being committed to GitHub.

### `README.md`

Project documentation and setup instructions.

---

## ⚙️ Installation & Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/AI_Blog_Generator.git
```

### Step 2 — Open the Project

```bash
cd AI_Blog_Generator
```

### Step 3 — Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4 — Activate the Environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
source venv/bin/activate
```

### Step 5 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6 — Configure the OpenAI API Key

Create a file named:

```text
.env
```

Add:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Step 7 — Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🔐 API Key & Billing Configuration

The application checks whether `OPENAI_API_KEY` is available before initializing the AI model.

If the key is missing, the application displays an access message requesting a valid OpenAI API key with available billing/credits.

### Local Development

Use:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Streamlit Deployment

For Streamlit deployment, configure the key through **Streamlit Secrets**.

Do **not** place your real API key directly inside `app.py`.

Do **not** commit `.env` to GitHub.

---

## 📄 Recommended `.gitignore`

Create `.gitignore` with:

```gitignore
.env
venv/
.venv/
__pycache__/
*.pyc
.streamlit/secrets.toml
.DS_Store
```

---

## 🧪 Example

### Input

```text
Topic:
How Artificial Intelligence is Transforming Education

Language:
English
```

### Processing

```text
User Topic
    ↓
Idea Agent
    ↓
Structured Outline
    ↓
Writer Agent
    ↓
Professional Blog
    ↓
Human Review
    ↓
Improve & Rewrite
    ↓
Approve & Publish
    ↓
Social Sharing
```

### Result

A complete blog with:

- Professional title
- Structured headings
- Introduction
- Main content
- Supporting ideas
- Conclusion
- Topic-relevant emojis where appropriate

---

## 👤 Human-in-the-Loop

Human review is a core part of the application.

AI-generated content is presented to the user before publication.

The user maintains control over the final content by choosing whether to:

- Approve and publish
- Improve and rewrite
- Review the revised version
- Approve the final version

This approach combines **AI automation with human judgment** rather than treating the first AI output as the final result.

---

## 📣 Social Sharing

After approval, the application prepares sharing actions for:

**Facebook · LinkedIn · X · WhatsApp**

The sharing workflow uses the approved topic and a portion of the generated blog content to create platform-specific sharing links.

---

## 🎨 User Interface

The application provides a modern, premium-style interface featuring:

- ✍️ AI Writing Studio hero banner
- 🟢 AI Engine Live indicator
- 💡 Idea Generation workflow card
- 📝 Blog Writer workflow card
- 👤 Human Review workflow card
- 🚀 Publish workflow card
- 🌐 Language selector
- 🎯 Blog topic input
- ✨ Professional Blog generation button
- 📄 Blog draft preview
- 👤 Human Review controls
- 🚀 Approve & Publish button
- 🔄 Improve & Rewrite button
- 📣 Social sharing cards
- ⚡ Fast & Simple benefits
- 🛡️ Human-in-the-Loop benefits
- 🌐 Ready-to-Share benefits

---

## 🎯 Use Cases

AI Writing Studio can support:

- 📝 Blog creation
- 🤖 AI and technology articles
- 💼 Business content
- 📚 Educational content
- 📣 Marketing content
- 🌍 Multilingual content creation
- 📱 Social content preparation
- ✍️ First-draft generation
- 👤 Human-reviewed publishing workflows

---

## 🔮 Future Enhancements

Potential future improvements include:

- Additional language support
- Adjustable blog length
- Tone selection
- SEO keyword generation
- SEO optimization
- Content history
- User authentication
- Blog export to PDF/DOCX
- Image generation
- Multiple LLM provider support
- Direct publishing integrations
- Content analytics
- Saved drafts and projects

---

## ⚠️ Important Notes

- An OpenAI API key is required for AI functionality.
- OpenAI API usage may incur charges according to the account and model usage.
- Keep API keys private.
- Never commit `.env` or Streamlit secrets to a public repository.
- Generated AI content should be reviewed before publication.

---

## 👨‍💻 Developer

**Heramba Kakati**

### Project Focus

**Generative AI · Agentic AI · LangGraph · LangChain · OpenAI · Streamlit**

This project demonstrates how an LLM-based application can be structured as a practical agentic workflow with separate generation, refinement, human-review, publishing, and sharing stages.

---

## 📌 Project Summary

| Item | Details |
|---|---|
| **Project Name** | AI Blog Generator |
| **Product/UI Name** | AI Writing Studio |
| **Application Type** | Generative AI / Agentic AI |
| **Framework** | Streamlit |
| **Workflow** | LangGraph |
| **LLM** | OpenAI GPT-4o-mini |
| **Languages** | English, Assamese |
| **Review Model** | Human-in-the-Loop |
| **Sharing** | Facebook, LinkedIn, X, WhatsApp |
| **Developer** | Heramba Kakati |

---

---

### ✍️ AI Writing Studio

**Turn your ideas into polished, engaging content — crafted, refined, and ready to share.**
