# mailflow/summary.py
from langchain_core.prompts import ChatPromptTemplate
from .common import Summary, parser_chain

summ_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful academic-email summarizer. Output strict JSON matching the schema. "
     "Return: purpose (1 to 10 sentences, depending on the contents of the mail), and evidence (short sentences copied verbatim from the email) "
     "that directly justify the purpose. "
     "Constraints for evidence:\n"
     "- Copy text verbatim from the email\n"
    ),
    ("human",
     "Summarize the following email in 1 to 10 sentences (purpose), and extract evidence sentences.\n"
     "=== EMAIL START ===\n{mail_text}\n=== EMAIL END ===\n"
     "Return JSON with keys: purpose, evidence")
])
summ_chain = parser_chain(summ_prompt, Summary)

def summarize(state) -> dict:
    
    mail_text = state.get("mail_text", "").strip()
    out: Summary = summ_chain.invoke({"mail_text": mail_text})
    ev = out.evidence or []
    return {
        "purpose": out.purpose,
        "evidence_sentences": ev,
        "len_mail_text": len(mail_text),
        "len_purpose": len(out.purpose),
    }