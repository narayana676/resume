import os
import re
import json
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langserve import add_routes


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash"
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY environment variable is missing."
    )


# ============================================================
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)


# ============================================================
# INPUT
# ============================================================

class ResumeRequest(BaseModel):
    resume_text: str = Field(
        ...,
        description="Resume text"
    )

    target_role: str = Field(
        ...,
        description="Target job role"
    )

    github_username: Optional[str] = None


# ============================================================
# RESUME PROFILE
# ============================================================

class ResumeProfile(BaseModel):

    name: Optional[str] = None

    contact_information: list[str] = Field(
        default_factory=list
    )

    education: list[str] = Field(
        default_factory=list
    )

    work_experience: list[str] = Field(
        default_factory=list
    )

    internships: list[str] = Field(
        default_factory=list
    )

    skills: list[str] = Field(
        default_factory=list
    )

    programming_languages: list[str] = Field(
        default_factory=list
    )

    frameworks: list[str] = Field(
        default_factory=list
    )

    databases: list[str] = Field(
        default_factory=list
    )

    cloud_tools: list[str] = Field(
        default_factory=list
    )

    certifications: list[str] = Field(
        default_factory=list
    )

    projects: list[str] = Field(
        default_factory=list
    )

    achievements: list[str] = Field(
        default_factory=list
    )

    links: list[str] = Field(
        default_factory=list
    )

    github_username: Optional[str] = None

    linkedin_url: Optional[str] = None

    target_role_if_mentioned: Optional[str] = None


# ============================================================
# OUTPUT
# ============================================================

class PlacementReport(BaseModel):

    role: str

    overall_summary: str

    matched_skills: list[str] = Field(
        default_factory=list
    )

    missing_skills: list[str] = Field(
        default_factory=list
    )

    weak_skills: list[str] = Field(
        default_factory=list
    )

    recommended_projects: list[str] = Field(
        default_factory=list
    )

    github_assessment: str = ""

    relevant_job_openings: list[str] = Field(
        default_factory=list
    )

    next_steps: list[str] = Field(
        default_factory=list
    )


# ============================================================
# TEXT CONVERTER
# ============================================================

def convert_to_text(value):

    if isinstance(value, str):
        return value

    if isinstance(value, list):

        parts = []

        for item in value:

            if isinstance(item, dict):

                if "text" in item:
                    parts.append(
                        str(item["text"])
                    )

                elif "content" in item:
                    parts.append(
                        str(item["content"])
                    )

                else:
                    parts.append(
                        str(item)
                    )

            else:
                parts.append(
                    str(item)
                )

        return "\n".join(parts)

    if isinstance(value, dict):

        if "text" in value:
            return str(value["text"])

        if "content" in value:
            return str(value["content"])

        return json.dumps(
            value,
            indent=2,
            default=str
        )

    return str(value)


# ============================================================
# CLEAN RESUME
# ============================================================

def clean_resume_text(text):

    text = text.replace("\x00", " ")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# RESUME PARSER
# ============================================================

structured_llm = llm.with_structured_output(
    ResumeProfile
)


def parse_resume(resume_text):

    prompt = f"""
You are an expert resume parser.

Analyze the resume below.

Extract information when available:

- name
- contact_information
- education
- work_experience
- internships
- skills
- programming_languages
- frameworks
- databases
- cloud_tools
- certifications
- projects
- achievements
- links
- github_username
- linkedin_url
- target_role_if_mentioned

The resume may use any format or headings.

Do not assume information that is not present.

If information is missing:

- use an empty list for list fields
- use null for optional string fields

IMPORTANT:
Do not invent information.

Resume:

{resume_text}
"""

    return structured_llm.invoke(prompt)


# ============================================================
# JOB SEARCH
# ============================================================

def search_jobs(role):

    try:

        if not TAVILY_API_KEY:

            return json.dumps({
                "status": "not_configured",
                "message": "TAVILY_API_KEY is not configured."
            })

        from langchain_community.tools.tavily_search import (
            TavilySearchResults
        )

        tavily = TavilySearchResults(
            max_results=5
        )

        query = (
            f"{role} jobs hiring India "
            f"entry level fresher 2026"
        )

        results = tavily.invoke({
            "query": query
        })

        return convert_to_text(results)

    except Exception as e:

        return json.dumps({
            "status": "error",
            "message": str(e)
        })


# ============================================================
# SKILL GAP
# ============================================================

def analyze_skill_gap(profile, role):

    prompt = f"""
You are an expert technical recruiter.

Target role:
{role}

Candidate resume:

{profile.model_dump_json(indent=2)}

Analyze the candidate.

Identify:

1. matched_skills
2. missing_skills
3. weak_skills
4. relevant_experience
5. overall_fit_score
6. explanation

Rules:

- Never invent candidate skills.
- Use projects and internships as evidence.
- Clearly distinguish missing and weak skills.
"""

    response = llm.invoke(prompt)

    return convert_to_text(
        response.content
    )


# ============================================================
# PROJECT RECOMMENDATION
# ============================================================

def recommend_projects(
    missing_skills,
    role
):

    missing_skills = convert_to_text(
        missing_skills
    )

    prompt = f"""
You are an expert career and project advisor.

Target role:
{role}

Missing or weak skills:
{missing_skills}

Recommend 3 to 5 practical,
resume-worthy projects.

For each project provide:

- project title
- skills covered
- description
- difficulty
- estimated duration
- why it improves the resume

Projects must address the candidate's
missing or weak skills.

Do not invent candidate experience.
"""

    response = llm.invoke(prompt)

    return convert_to_text(
        response.content
    )


# ============================================================
# GITHUB
# ============================================================

def check_github(github_username):

    if not github_username:

        return json.dumps({
            "status": "not_provided",
            "message": "GitHub username was not provided."
        })

    try:

        import requests

        username = github_username.strip()

        response = requests.get(
            f"https://api.github.com/users/{username}",
            timeout=15
        )

        if response.status_code != 200:

            return json.dumps({
                "status": "not_found",
                "username": username
            })

        user = response.json()

        repos_response = requests.get(
            f"https://api.github.com/users/{username}/repos",
            params={
                "sort": "updated",
                "per_page": 10
            },
            timeout=15
        )

        repos = []

        if repos_response.status_code == 200:
            repos = repos_response.json()

        result = {
            "status": "success",
            "username": username,
            "name": user.get("name"),
            "public_repositories":
                user.get("public_repos", 0),
            "followers":
                user.get("followers", 0),
            "recent_repositories": [
                {
                    "name": repo.get("name"),
                    "language": repo.get("language"),
                    "stars":
                        repo.get(
                            "stargazers_count",
                            0
                        ),
                    "updated_at":
                        repo.get("updated_at")
                }
                for repo in repos[:5]
            ]
        }

        return json.dumps(
            result,
            indent=2
        )

    except Exception as e:

        return json.dumps({
            "status": "error",
            "message": str(e)
        })


# ============================================================
# FINAL REPORT
# ============================================================

report_llm = llm.with_structured_output(
    PlacementReport
)


def create_report(
    profile,
    role,
    job_result,
    skill_result,
    project_result,
    github_result
):

    prompt = f"""
Create the final placement-readiness report.

Target role:
{role}

Resume:
{profile.model_dump_json(indent=2)}

Job search:
{job_result}

Skill gap:
{skill_result}

Projects:
{project_result}

GitHub:
{github_result}

Rules:

- Do not invent information.
- Use only information supported by the resume
  and analysis.
- Identify missing skills clearly.
- Give practical next steps.
- Keep recommendations relevant.
"""

    return report_llm.invoke(prompt)


# ============================================================
# MAIN PIPELINE
# ============================================================

def placement_pipeline(request):

    # Convert LangServe dictionary input
    # into the Pydantic ResumeRequest model
    request = ResumeRequest.model_validate(request)

    if not request.resume_text.strip():
        raise ValueError(
            "resume_text cannot be empty."
        )

    if not request.target_role.strip():
        raise ValueError(
            "target_role cannot be empty."
        )

    resume_text = clean_resume_text(
        request.resume_text
    )

    profile = parse_resume(
        resume_text
    )

    role = request.target_role.strip()

    github_username = (
        request.github_username
        or profile.github_username
    )

    job_result = search_jobs(
        role
    )

    skill_result = analyze_skill_gap(
        profile,
        role
    )

    project_result = recommend_projects(
        skill_result,
        role
    )

    github_result = check_github(
        github_username
    )

    final_report = create_report(
        profile,
        role,
        job_result,
        skill_result,
        project_result,
        github_result
    )

    return {
        "status": "success",
        "model": MODEL_NAME,
        "resume_profile":
            profile.model_dump(),
        "job_search":
            job_result,
        "skill_gap":
            skill_result,
        "project_recommendations":
            project_result,
        "github":
            github_result,
        "final_report":
            final_report.model_dump()
    }


# ============================================================
# FASTAPI + LANGSERVE
# ============================================================

app = FastAPI(
    title="Placement Ready AI Agent",
    version="1.0.0",
    description=(
        "AI resume analysis and placement assistant"
    )
)


placement_chain = RunnableLambda(
    placement_pipeline
).with_types(
    input_type=ResumeRequest
)


add_routes(
    app,
    placement_chain,
    path="/placement"
)


@app.get("/")
def root():

    return {
        "message":
            "Placement Ready AI Agent API",
        "status":
            "running",
        "endpoint":
            "/placement"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": MODEL_NAME
    }

# ============================================================
# LOCAL / DIRECT RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
