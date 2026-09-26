from langchain.agents import create_agent
from langchain_core.tools import tool
from demo import create_llm
from rag import (load_vector_store,rag_chain)
import gradio as gr

from tool_calls import (resume_to_jd_match,interview_slot_lookup,book_interview_slot)


# Load LLM
llm = create_llm()

# Load existing FAISS vector store
vector_store = load_vector_store()

# POLICY FAQ TOOL
@tool
def candidate_policy_faq(query: str) -> str:
    """
    Answer candidate questions using only company policy
    and benefits information from the company handbook.
    """

    response = rag_chain(
        query,
        vector_store,
        "policy"
    )

    return response["result"]


# AGENT TOOLS


tools = [
    candidate_policy_faq,
    resume_to_jd_match,
    interview_slot_lookup,
    book_interview_slot
]


# SYSTEM PROMPT


system_prompt = """
You are a Hiring and Onboarding Agent.

You have THREE completely separate workflows.

The resume and job description are already stored in the system. Do NOT ask the user to provide or upload them.
The company handbook and interview schedule are also already stored in the system.

1. CANDIDATE FAQ

For questions about:
- company policies
- employee benefits
- leave
- holidays
- working hours
- insurance
- onboarding
- company rules

Use the candidate_policy_faq tool. Use ONLY the company handbook through the policy RAG system.
Never use resume information to answer policy questions.

2. RESUME SCREENING

The resume and job description are already stored in the system. Do NOT ask the user to provide or upload them.

When the user asks anything related to:
- compare resume with JD
- screen candidate
- match resume to JD
- matched skills
- missing skills
- candidate qualifications
- candidate experience

You MUST call the resume_to_jd_match tool.

The screening result must be based only on:
- candidate resume
- job description

Do not use company policy information.

3. INTERVIEW SCHEDULING

For questions about:
- available interview slots
- interview dates
- interview times
- booking an interview

Use the interview_slot_lookup or book_interview_slot tools.
Use the mock interview calendar.
Do not retrieve resume or company policy information for scheduling.


IMPORTANT DATA SEPARATION

Policy questions:
    → company handbook ONLY

Screening:
    → resume + job description ONLY

Scheduling:
    → interview_slots.json ONLY

Never mix information between these workflows.
"""

# CREATE AGENT

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)
def deploy_agent():

    def run_agent(query):
        if not query.strip():
            return "Please enter a question."

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            }
        )

        return result["messages"][-1].content

    with gr.Blocks(
        title="Hiring & Onboarding Agent"
    ) as demo:

        gr.Markdown(
            """
            # 🤖 Hiring & Onboarding Agent

            AI-powered recruitment assistant for FAQ, resume screening,
            and interview scheduling.
            """
        )

        query = gr.Textbox(
            label="Enter your question",
            placeholder="Ask about company policy, candidate screening, or interview scheduling...",
            lines=4
        )

        submit = gr.Button(
            "Submit",
            variant="primary"
        )

        output = gr.Markdown(
            label="Agent Response"
        )

        submit.click(
            fn=run_agent,
            inputs=query,
            outputs=output
        )

        gr.Markdown(
            """
            ---
            ### Workflows

            📘 **Candidate FAQ** → Company Handbook

            📄 **Resume Screening** → Resume + Job Description

            📅 **Interview Scheduling** → Interview Calendar
            """
        )

    demo.launch()


if __name__ == "__main__":

    # print("Hiring & Onboarding Agent")
    # print("--------------------------------")

    # query = input("Enter your question: ")

    # result = agent.invoke(
    #     {
    #         "messages": [
    #             {
    #                 "role": "user",
    #                 "content": query
    #             }
    #         ]
    #     }
    # )

    # print("\nAgent Response:")
    # print(result["messages"][-1].content)
    deploy_agent()