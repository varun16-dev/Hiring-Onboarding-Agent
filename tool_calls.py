from langchain_core.tools import tool 
from langchain_community.tools import DuckDuckGoSearchRun
import os
import requests 
from dotenv import load_dotenv
from rag import (ingest_data,load_vector_store,get_resume_retriever,get_jd_retriever)
import json
from demo import create_llm
load_dotenv() 

print("All libraries imported")
# TOOL 1: RESUME → JOB DESCRIPTION MATCH

@tool
def resume_to_jd_match(candidate_id: str = "candidate_01") -> dict:
    """
    Compare the candidate resume with the job description.
    MUST be used for:
    - comparing a candidate resume with a job description
    - resume screening
    - resume-to-JD matching
    - matched skills
    - missing skills
    - candidate qualifications
    - candidate experience

    The candidate resume and job description are already stored in the system.
    Do NOT ask the user to provide or upload them.
    Returns structured comparison containing matched_skills, missing_skills,
    experience, education, and supporting evidence.
    """

    vector_store = load_vector_store()

    # Retrieve resume chunks only
    resume_retriever = get_resume_retriever(vector_store)

    # Retrieve job description chunks only
    jd_retriever = get_jd_retriever(vector_store)

    resume_docs = resume_retriever.invoke(
        "skills experience education qualifications"
    )

    jd_docs = jd_retriever.invoke(
        "required skills experience education qualifications"
    )

    resume_text = "\n\n".join(
        doc.page_content for doc in resume_docs
    )

    jd_text = "\n\n".join(
        doc.page_content for doc in jd_docs
    )

    llm = create_llm()

    prompt = f"""
Compare the candidate resume with the job description.

Candidate ID:
{candidate_id}

Resume:
{resume_text}

Job Description:
{jd_text}

Return ONLY JSON in this format:

{{
    "candidate": "{candidate_id}",
    "matched_skills": [],
    "missing_skills": [],
    "experience": {{
        "required": "",
        "candidate": ""
    }},
    "education": {{
        "required": "",
        "candidate": ""
    }},
    "evidence": []
}}

Rules:
- matched_skills are skills found in both resume and JD.
- missing_skills are required JD skills not found in resume.
- compare required vs candidate experience.
- compare required vs candidate education.
- evidence must come only from resume and JD.
- do not use company handbook information.
- do not give a vague overall opinion.
- return valid JSON only.
"""

    response = llm.invoke(prompt)

    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content.strip())

    except (json.JSONDecodeError, Exception):
        return {
            "candidate": candidate_id,
            "matched_skills": [],
            "missing_skills": [],
            "experience": {
                "required": "",
                "candidate": ""
            },
            "education": {
                "required": "",
                "candidate": ""
            },
            "evidence": [response.content]
        }



# TOOL 2: INTERVIEW SLOT LOOKUP

@tool
def interview_slot_lookup(date: str) -> dict:
    """
    Look up available interview slots for a specified date
    from the mock interview calendar.
    """

    with open("./interview_slots.json", "r") as file:
        calendar = json.load(file)

    if date not in calendar:
        return {
            "date": date,
            "available_slots": [],
            "message": "No interview slots found for this date."
        }

    available_slots = []

    for slot in calendar[date]:

        if isinstance(slot, dict) and slot.get("available") is True:
            available_slots.append(slot.get("time"))

    return {
        "date": date,
        "available_slots": available_slots
    }


# TOOL 3: BOOK INTERVIEW SLOT

@tool
def book_interview_slot(
    date: str,
    time: str,
    candidate_name: str
) -> dict:
    """
    Book an available interview slot for a candidate
    in the mock interview calendar.
    """

    with open("./interview_slots.json", "r") as file:
        calendar = json.load(file)

    if date not in calendar:
        return {
            "status": "failed",
            "message": "Date not found."
        }

    for slot in calendar[date]:

        if (
            isinstance(slot, dict)
            and slot.get("time") == time
            and slot.get("available") is True
        ):
            slot["available"] = False

            with open("./interview_slots.json", "w") as file:
                json.dump(calendar, file, indent=4)

            return {
                "status": "booked",
                "candidate": candidate_name,
                "date": date,
                "time": time
            }

    return {
        "status": "failed",
        "message": "The requested interview slot is not available."
    }