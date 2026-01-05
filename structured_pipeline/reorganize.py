from langchain_core.prompts import ChatPromptTemplate
from .common import str_parser_chain

reorganize_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Context-Preserving Data Archivist. Your task is to **reorganize** the email into functional sections while strictly **preserving the context** of every piece of information.\n\n"
     
     "## Golden Rule: The 'Standalone' Test\n"
     "Each section below must make sense on its own. If you extract a deadline, you MUST also include the text that explains **WHAT** that deadline is for (e.g., The name of event).\n"
     "**It is better to repeat information (redundancy) than to lose context.**\n\n"
     
     "## Target Sections:\n\n"
     
     "### 1. [SCOPE & IDENTITY]\n"
     "- Goal: Define who and what. Identify the Host event.\n"
     "- Action: Copy sentences naming the events. If it's a joint event, describe the relationship verbatim.\n\n"
     
     "### 2. [TIMELINE & DATES] (CRITICAL: EXPLICIT ASSOCIATION)\n"
     "- Goal: All temporal information linked to their specific events.\n"
     "- **Rule:** Every date entry MUST explicitly state which **Event** or **Track** it belongs to.\n"
     "- **Action:** Look for the 'owner' of the deadline (e.g., 'Research Track', 'Workshop A', 'Main Conference') and write it before the date.\n"
     "- **Format:** `[Event] + [Track Name] Original Date Sentence`\n"
     "- **Example:** \n"
     "   - (Bad): 'Submission Deadline: June 1st'\n"
     "   - (Good): '[MLOps Workshop] Camera-ready due: July 15th'\n\n"
     "   - (Good): '[MLOps Workshop] [Research Track] Camera-ready due: July 15th'\n\n"
     
     "### 3. [ACTIONABLE REQUIREMENTS] (CRITICAL: EXPLICIT SCOPE)\n"
     "- Goal: What is the user asked to do? (Paper submission, Proposal, Registration) linked to specific tracks.\n"
     "- **Rule:** Every requirement (topics, page limits, submission types) MUST explicitly state which **Event** or **Track** it applies to.\n"
     "- **Format:** `[Event/Track Name] Requirement Description`\n"
     "- **Example:**\n"
     "   - (Bad): 'Up to 10 pages for full papers.'\n"
     "   - (Good): '[Research Track] Full papers: up to 10 pages.'\n"
     "   - (Good): '[Doctoral Symposium] Submit a 2-page research proposal.'\n"
     "   - (Good): '[Main Conference] Topics of interest include AI, SE, and Data Science.'\n\n"
     
     "### 4. [LINKS & CONTACTS] (CRITICAL: EXPLICIT SCOPE)\n"
     "- Goal: URLs and emails linked to their specific purpose/owner.\n"
     "- **Rule:** Every link MUST explicitly state which **Event** or **Specific Resource** it belongs to.\n"
     "- **Action:** Find the context (e.g., 'Submission for Workshop A', 'Main Conference Website') and write it before the URL.\n"
     "- **Format:** `[Event/Context] Description: URL`\n"
     "- **Example:**\n"
     "   - (Bad): 'https://easychair.org/my-conf'\n"
     "   - (Good): '[Main Conference] Submission Site: https://easychair.org/my-conf'\n"
     "   - (Good): '[MLOps Workshop] Homepage: https://mlops.org'\n"
     "   - (Good): '[Springer] LNCS Formatting Guidelines: https://springer.com/lncs'\n\n"
     
     "## Output Constraints\n"
     "- **Verbatim Policy:** Use the original text as much as possible.\n"
     "- **Hierarchy Policy:** Use Markdown bullets (`-`) or indentation to show which info belongs to which event/track.\n"
     "- **No Summary:** Do not shorten sentences. Keep them full and descriptive."),
    ("human",
     "### Raw Email:\n{mail_text}")
])

reorganize_chain = str_parser_chain(reorganize_prompt)

def reorganize_node(state) -> dict:

    mail_text = state.get("mail_text", "").strip()
    result_text = reorganize_chain.invoke({"mail_text": mail_text})

    return {
        "summary": result_text
    }