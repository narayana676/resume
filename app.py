import os
import re
import json
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from pypdf import PdfReader

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY environment variable is missing."
    )


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Placement Ready AI",
    version="2.0.0",
    description="AI Resume Analysis and Placement Assistant"
)


# ============================================================
# GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)


# ============================================================
# INPUT MODEL
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

    email: Optional[str] = None

    phone: Optional[str] = None

    location: Optional[str] = None

    summary: str = ""

    education: list[str] = Field(
        default_factory=list
    )

    experience: list[str] = Field(
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

    github_username: Optional[str] = None

    linkedin_url: Optional[str] = None


# ============================================================
# PLACEMENT REPORT
# ============================================================

class PlacementReport(BaseModel):

    placement_score: int = 0

    score_label: str = ""

    overall_summary: str = ""

    matched_skills: list[str] = Field(
        default_factory=list
    )

    missing_skills: list[str] = Field(
        default_factory=list
    )

    weak_skills: list[str] = Field(
        default_factory=list
    )

    strengths: list[str] = Field(
        default_factory=list
    )

    resume_improvements: list[str] = Field(
        default_factory=list
    )

    recommended_projects: list[str] = Field(
        default_factory=list
    )

    github_assessment: str = ""

    job_recommendations: list[str] = Field(
        default_factory=list
    )

    next_steps: list[str] = Field(
        default_factory=list
    )


# ============================================================
# HELPERS
# ============================================================

def convert_to_text(value):

    if isinstance(value, str):
        return value

    if isinstance(value, list):

        result = []

        for item in value:

            if isinstance(item, dict):

                if "text" in item:
                    result.append(
                        str(item["text"])
                    )

                else:
                    result.append(
                        str(item)
                    )

            else:
                result.append(
                    str(item)
                )

        return "\n".join(result)

    if isinstance(value, dict):

        if "text" in value:
            return str(value["text"])

        return json.dumps(
            value,
            indent=2,
            default=str
        )

    return str(value)


def clean_resume_text(text):

    text = text.replace(
        "\x00",
        " "
    )

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
# PDF EXTRACTION
# ============================================================

def extract_pdf_text(file_bytes):

    try:

        import io

        reader = PdfReader(
            io.BytesIO(file_bytes)
        )

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        extracted_text = "\n\n".join(
            pages
        )

        return clean_resume_text(
            extracted_text
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Could not read PDF: {str(e)}"
        )


# ============================================================
# RESUME PARSER
# ============================================================

structured_llm = llm.with_structured_output(
    ResumeProfile
)


def parse_resume(resume_text):

    prompt = f"""
You are an expert resume parser.

Analyze the following resume carefully.

Extract ONLY information that actually appears
in the resume.

Extract:

- name
- email
- phone
- location
- professional summary
- education
- work experience
- internships
- skills
- programming languages
- frameworks
- databases
- cloud tools
- certifications
- projects
- achievements
- GitHub username
- LinkedIn URL

Do NOT invent information.

If information is unavailable:
- use empty lists for list fields
- use null for optional fields

RESUME:

{resume_text}
"""

    return structured_llm.invoke(
        prompt
    )


# ============================================================
# SKILL ANALYSIS
# ============================================================

def analyze_skills(profile, role):

    prompt = f"""
You are an expert technical recruiter.

Analyze this candidate for the target role.

TARGET ROLE:
{role}

CANDIDATE:
{profile.model_dump_json(indent=2)}

Identify:

1. matched_skills
2. missing_skills
3. weak_skills
4. strengths
5. resume_improvements
6. placement_score

The placement score must be between 0 and 100.

Score meaning:

90-100 = Excellent match
75-89 = Very good match
60-74 = Good but needs improvement
40-59 = Needs significant improvement
0-39 = Not ready yet

Consider:

- technical skills
- projects
- experience
- education
- certifications
- role relevance

Do not invent candidate experience.
"""

    response = llm.invoke(
        prompt
    )

    return convert_to_text(
        response.content
    )


# ============================================================
# PROJECT RECOMMENDATIONS
# ============================================================

def recommend_projects(
    role,
    missing_skills,
    weak_skills
):

    prompt = f"""
You are a technical career mentor.

TARGET ROLE:
{role}

MISSING SKILLS:
{missing_skills}

WEAK SKILLS:
{weak_skills}

Recommend 3 practical projects.

Each project must include:

Project Name:
Why it is useful:
Skills to learn:
What to build:
Difficulty:

The projects should specifically help
the candidate become stronger for the target role.

Do not claim that the candidate has already
completed these projects.
"""

    response = llm.invoke(
        prompt
    )

    return convert_to_text(
        response.content
    )


# ============================================================
# GITHUB ANALYSIS
# ============================================================

def check_github(username):

    if not username:

        return (
            "GitHub username was not provided. "
            "Add your GitHub username to receive "
            "a GitHub profile assessment."
        )

    try:

        import requests

        username = username.strip()

        response = requests.get(
            f"https://api.github.com/users/{username}",
            timeout=15
        )

        if response.status_code != 200:

            return (
                f"GitHub profile '{username}' "
                "could not be found."
            )

        user = response.json()

        repos_response = requests.get(
            f"https://api.github.com/users/{username}/repos",
            params={
                "sort": "updated",
                "per_page": 10
            },
            timeout=15
        )

        repositories = []

        if repos_response.status_code == 200:

            repositories = (
                repos_response.json()
            )

        repo_names = []

        for repo in repositories[:5]:

            repo_names.append(
                f"{repo.get('name')} "
                f"({repo.get('language') or 'Unknown language'})"
            )

        return (
            f"GitHub Username: {username}\n"
            f"Public repositories: "
            f"{user.get('public_repos', 0)}\n"
            f"Followers: "
            f"{user.get('followers', 0)}\n"
            f"Recent repositories: "
            f"{', '.join(repo_names) if repo_names else 'None'}"
        )

    except Exception as e:

        return (
            "GitHub analysis could not be completed: "
            + str(e)
        )


# ============================================================
# JOB RECOMMENDATIONS
# ============================================================

def find_jobs(role):

    if not TAVILY_API_KEY:

        return [
            "Tavily search is not configured.",
            "Add TAVILY_API_KEY in Render environment variables "
            "to enable live job recommendations."
        ]

    try:

        from langchain_community.tools.tavily_search import (
            TavilySearchResults
        )

        tavily = TavilySearchResults(
            max_results=5
        )

        query = (
            f"{role} jobs India "
            f"fresher entry level hiring"
        )

        results = tavily.invoke(
            {
                "query": query
            }
        )

        text = convert_to_text(
            results
        )

        return [
            text
        ]

    except Exception as e:

        return [
            f"Job search error: {str(e)}"
        ]


# ============================================================
# FINAL REPORT
# ============================================================

def create_final_report(
    profile,
    role,
    skill_analysis,
    project_recommendations,
    github_assessment,
    jobs
):

    prompt = f"""
You are an expert placement advisor.

Create a clear final placement report.

TARGET ROLE:
{role}

RESUME PROFILE:
{profile.model_dump_json(indent=2)}

SKILL ANALYSIS:
{skill_analysis}

PROJECT RECOMMENDATIONS:
{project_recommendations}

GITHUB:
{github_assessment}

JOB INFORMATION:
{jobs}

Return:

placement_score:
Integer from 0 to 100.

score_label:
One of:
Excellent Match
Very Good Match
Good Match
Needs Improvement
Not Ready

overall_summary:
2-4 clear sentences.

matched_skills:
Skills from the resume that match the role.

missing_skills:
Important skills missing from the resume.

weak_skills:
Skills that appear weak or need stronger evidence.

strengths:
3-5 candidate strengths.

resume_improvements:
Specific changes to make the resume better.

recommended_projects:
3-5 practical projects.

github_assessment:
Clear assessment of GitHub.

job_recommendations:
Useful job search recommendations.

next_steps:
5 concrete actions the candidate should take.

Do NOT invent experience,
skills or qualifications.
"""

    final_llm = llm.with_structured_output(
        PlacementReport
    )

    return final_llm.invoke(
        prompt
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_resume(
    resume_text,
    target_role,
    github_username=None
):

    resume_text = clean_resume_text(
        resume_text
    )

    if not resume_text:

        raise HTTPException(
            status_code=400,
            detail="Resume text is empty."
        )

    if not target_role.strip():

        raise HTTPException(
            status_code=400,
            detail="Target role is required."
        )

    # ------------------------------
    # Parse resume
    # ------------------------------

    profile = parse_resume(
        resume_text
    )

    # ------------------------------
    # GitHub
    # ------------------------------

    username = (
        github_username
        or profile.github_username
    )

    github_assessment = check_github(
        username
    )

    # ------------------------------
    # Skill analysis
    # ------------------------------

    skill_analysis = analyze_skills(
        profile,
        target_role
    )

    # ------------------------------
    # Project recommendations
    # ------------------------------

    project_recommendations = (
        recommend_projects(
            target_role,
            skill_analysis,
            skill_analysis
        )
    )

    # ------------------------------
    # Job search
    # ------------------------------

    jobs = find_jobs(
        target_role
    )

    # ------------------------------
    # Final report
    # ------------------------------

    report = create_final_report(
        profile,
        target_role,
        skill_analysis,
        project_recommendations,
        github_assessment,
        jobs
    )

    return {
        "status": "success",
        "model": MODEL_NAME,
        "target_role": target_role,
        "resume_profile": profile.model_dump(),
        "report": report.model_dump()
    }


# ============================================================
# JSON API
# ============================================================

@app.post("/api/analyze")
def analyze_api(
    request: ResumeRequest
):

    return analyze_resume(
        request.resume_text,
        request.target_role,
        request.github_username
    )


# ============================================================
# FILE UPLOAD API
# ============================================================

@app.post("/api/analyze-file")
async def analyze_file(
    file: UploadFile = File(...),
    target_role: str = Form(...),
    github_username: str = Form("")
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Please upload a resume file."
        )

    filename = file.filename.lower()

    if not filename.endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Currently only PDF resume files are supported."
        )

    file_bytes = await file.read()

    if not file_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty."
        )

    resume_text = extract_pdf_text(
        file_bytes
    )

    if not resume_text:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not extract text from this PDF. "
                "Please upload a text-based PDF."
            )
        )

    return analyze_resume(
        resume_text,
        target_role,
        github_username
    )


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Placement Ready AI",
        "status": "running",
        "endpoint": "/placement"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": MODEL_NAME
    }


# ============================================================
# WEB APPLICATION
# ============================================================

@app.get(
    "/placement",
    response_class=HTMLResponse
)
def placement_page():

    return """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1">

<title>Placement Ready AI</title>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background:
        linear-gradient(
            135deg,
            #eff6ff,
            #f8fafc
        );

    color: #172033;
}

.container {

    max-width: 1100px;

    margin: 40px auto;

    padding: 20px;
}

.header {

    text-align: center;

    margin-bottom: 30px;
}

.header h1 {

    font-size: 42px;

    color: #2563eb;

    margin-bottom: 8px;
}

.header p {

    font-size: 18px;

    color: #64748b;
}

.card {

    background: white;

    border-radius: 18px;

    padding: 30px;

    margin-bottom: 25px;

    box-shadow:
        0 8px 30px
        rgba(15, 23, 42, 0.08);
}

.section-title {

    font-size: 22px;

    font-weight: bold;

    margin-bottom: 20px;

    color: #1e3a8a;
}

label {

    display: block;

    font-weight: bold;

    margin-top: 18px;

    margin-bottom: 8px;
}

textarea,
input {

    width: 100%;

    border: 1px solid #cbd5e1;

    border-radius: 10px;

    padding: 13px;

    font-size: 15px;

    outline: none;
}

textarea {

    min-height: 180px;

    resize: vertical;
}

textarea:focus,
input:focus {

    border-color: #2563eb;

    box-shadow:
        0 0 0 3px
        rgba(37, 99, 235, 0.1);
}

.upload-box {

    border: 2px dashed #93c5fd;

    border-radius: 12px;

    padding: 25px;

    text-align: center;

    background: #eff6ff;

    margin-bottom: 20px;
}

.upload-box input {

    border: none;

    background: transparent;
}

.or {

    text-align: center;

    color: #64748b;

    margin: 15px;
}

button {

    width: 100%;

    margin-top: 25px;

    padding: 15px;

    border: none;

    border-radius: 10px;

    background: #2563eb;

    color: white;

    font-size: 17px;

    font-weight: bold;

    cursor: pointer;
}

button:hover {

    background: #1d4ed8;
}

button:disabled {

    background: #94a3b8;

    cursor: not-allowed;
}

.status {

    text-align: center;

    margin-top: 20px;

    font-weight: bold;
}

.error {

    color: #dc2626;
}

.success {

    color: #15803d;
}


/* RESULT */

.result {

    display: none;
}

.score-box {

    display: flex;

    align-items: center;

    gap: 30px;

    padding: 25px;

    border-radius: 15px;

    background:
        linear-gradient(
            135deg,
            #eff6ff,
            #dbeafe
        );

    margin-bottom: 25px;
}

.score {

    width: 120px;

    height: 120px;

    border-radius: 50%;

    display: flex;

    align-items: center;

    justify-content: center;

    background: #2563eb;

    color: white;

    font-size: 32px;

    font-weight: bold;

    flex-shrink: 0;
}

.score-info h2 {

    margin: 0 0 8px;

    color: #1e40af;
}

.score-info p {

    margin: 0;

    color: #475569;
}

.summary {

    padding: 20px;

    background: #f8fafc;

    border-radius: 12px;

    line-height: 1.7;

    margin-bottom: 25px;
}

.grid {

    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(300px, 1fr)
        );

    gap: 20px;
}

.result-section {

    background: #f8fafc;

    border-radius: 12px;

    padding: 20px;

    margin-bottom: 20px;
}

.result-section h3 {

    margin-top: 0;

    color: #1e3a8a;
}

ul {

    padding-left: 22px;

    line-height: 1.7;
}

li {

    margin-bottom: 8px;
}

.match {

    border-left: 5px solid #16a34a;
}

.missing {

    border-left: 5px solid #dc2626;
}

.weak {

    border-left: 5px solid #f59e0b;
}

.strength {

    border-left: 5px solid #2563eb;
}

.improvement {

    border-left: 5px solid #7c3aed;
}

.project {

    border-left: 5px solid #0891b2;
}

.next {

    border-left: 5px solid #ea580c;
}

.profile-grid {

    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(250px, 1fr)
        );

    gap: 15px;
}

.profile-item {

    background: white;

    padding: 15px;

    border-radius: 10px;

    border: 1px solid #e2e8f0;
}

.profile-item strong {

    display: block;

    color: #475569;

    margin-bottom: 5px;
}

.file-name {

    margin-top: 8px;

    color: #2563eb;

    font-weight: bold;
}

</style>

</head>


<body>

<div class="container">


<div class="header">

<h1>Placement Ready AI</h1>

<p>
AI-powered Resume Analysis and Placement Assistant
</p>

</div>


<!-- INPUT -->

<div class="card">

<div class="section-title">
1. Upload or Paste Your Resume
</div>


<div class="upload-box">

<strong>
Upload Resume PDF
</strong>

<p>
Select your resume PDF from your computer
</p>

<input
    id="resumeFile"
    type="file"
    accept=".pdf,application/pdf"
    onchange="showFileName()"
>

<div
    id="fileName"
    class="file-name">
</div>

</div>


<div class="or">
OR
</div>


<label>
Paste Resume Text
</label>

<textarea
    id="resumeText"
    placeholder="Paste your resume text here if you do not want to upload a PDF..."
></textarea>


<label>
Target Job Role *
</label>

<input
    id="targetRole"
    type="text"
    placeholder="Example: Backend Developer"
>


<label>
GitHub Username (Optional)
</label>

<input
    id="githubUsername"
    type="text"
    placeholder="Example: octocat"
>


<button
    id="analyzeButton"
    onclick="analyzeResume()"
>
Analyze Resume
</button>


<div
    id="status"
    class="status">
</div>

</div>


<!-- RESULT -->

<div
    id="result"
    class="result">


<div class="card">

<div class="section-title">
2. Placement Readiness
</div>


<div class="score-box">

<div
    id="score"
    class="score">
0
</div>

<div class="score-info">

<h2 id="scoreLabel">
Analyzing...
</h2>

<p id="scoreSummary">
</p>

</div>

</div>


<div
    id="summary"
    class="summary">
</div>

</div>


<!-- PROFILE -->

<div class="card">

<div class="section-title">
3. Resume Profile
</div>

<div
    id="profile"
    class="profile-grid">
</div>

</div>


<!-- SKILLS -->

<div class="card">

<div class="section-title">
4. Skill Analysis
</div>


<div class="grid">


<div class="result-section match">

<h3>
✅ Matching Skills
</h3>

<div id="matchedSkills">
</div>

</div>


<div class="result-section missing">

<h3>
❌ Missing Skills
</h3>

<div id="missingSkills">
</div>

</div>


<div class="result-section weak">

<h3>
⚠️ Skills to Improve
</h3>

<div id="weakSkills">
</div>

</div>


<div class="result-section strength">

<h3>
💪 Your Strengths
</h3>

<div id="strengths">
</div>

</div>

</div>

</div>


<!-- RESUME IMPROVEMENTS -->

<div class="card">

<div class="section-title">
5. Resume Improvements
</div>

<div
    id="improvements"
    class="result-section improvement">
</div>

</div>


<!-- PROJECTS -->

<div class="card">

<div class="section-title">
6. Recommended Projects
</div>

<div
    id="projects"
    class="result-section project">
</div>

</div>


<!-- GITHUB -->

<div class="card">

<div class="section-title">
7. GitHub Assessment
</div>

<div
    id="github"
    class="result-section">
</div>

</div>


<!-- JOBS -->

<div class="card">

<div class="section-title">
8. Job Recommendations
</div>

<div
    id="jobs"
    class="result-section">
</div>

</div>


<!-- NEXT STEPS -->

<div class="card">

<div class="section-title">
9. What You Should Do Next
</div>

<div
    id="nextSteps"
    class="result-section next">
</div>

</div>


</div>


</div>


<script>


function showFileName() {

    const file =
        document.getElementById(
            "resumeFile"
        ).files[0];

    const name =
        document.getElementById(
            "fileName"
        );

    if (file) {

        name.innerText =
            "Selected: " + file.name;

        document.getElementById(
            "resumeText"
        ).value = "";

    } else {

        name.innerText = "";

    }

}


function escapeHtml(text) {

    if (text === null ||
        text === undefined) {

        return "";
    }

    return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


function createList(items) {

    if (!items ||
        !Array.isArray(items) ||
        items.length === 0) {

        return "<p>No information available.</p>";
    }

    return `
        <ul>
        ${items.map(
            item =>
                `<li>${escapeHtml(item)}</li>`
        ).join("")}
        </ul>
    `;
}


function displayProfile(profile) {

    const container =
        document.getElementById(
            "profile"
        );

    container.innerHTML = `

        <div class="profile-item">
            <strong>Name</strong>
            ${escapeHtml(profile.name || "Not found")}
        </div>

        <div class="profile-item">
            <strong>Email</strong>
            ${escapeHtml(profile.email || "Not found")}
        </div>

        <div class="profile-item">
            <strong>Phone</strong>
            ${escapeHtml(profile.phone || "Not found")}
        </div>

        <div class="profile-item">
            <strong>Education</strong>
            ${createList(profile.education)}
        </div>

        <div class="profile-item">
            <strong>Experience</strong>
            ${createList(profile.experience)}
        </div>

        <div class="profile-item">
            <strong>Skills</strong>
            ${createList(profile.skills)}
        </div>

        <div class="profile-item">
            <strong>Projects</strong>
            ${createList(profile.projects)}
        </div>

        <div class="profile-item">
            <strong>Certifications</strong>
            ${createList(profile.certifications)}
        </div>

    `;
}


async function analyzeResume() {

    const file =
        document.getElementById(
            "resumeFile"
        ).files[0];

    const text =
        document.getElementById(
            "resumeText"
        ).value.trim();

    const role =
        document.getElementById(
            "targetRole"
        ).value.trim();

    const github =
        document.getElementById(
            "githubUsername"
        ).value.trim();

    const button =
        document.getElementById(
            "analyzeButton"
        );

    const status =
        document.getElementById(
            "status"
        );

    const result =
        document.getElementById(
            "result"
        );


    if (!file && !text) {

        status.className =
            "status error";

        status.innerText =
            "Please upload your resume PDF or paste your resume text.";

        return;
    }


    if (!role) {

        status.className =
            "status error";

        status.innerText =
            "Please enter the target job role.";

        return;
    }


    button.disabled = true;

    status.className =
        "status";

    status.innerText =
        "Analyzing your resume. Please wait...";

    result.style.display = "none";


    try {

        let response;


        // ====================================================
        // PDF UPLOAD
        // ====================================================

        if (file) {

            const formData =
                new FormData();

            formData.append(
                "file",
                file
            );

            formData.append(
                "target_role",
                role
            );

            formData.append(
                "github_username",
                github
            );


            response = await fetch(
                "/api/analyze-file",
                {

                    method: "POST",

                    body: formData

                }
            );

        }


        // ====================================================
        // TEXT INPUT
        // ====================================================

        else {

            response = await fetch(
                "/api/analyze",
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        resume_text: text,

                        target_role: role,

                        github_username:
                            github || null

                    })

                }
            );

        }


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Analysis failed."
            );
        }


        const report =
            data.report;

        const profile =
            data.resume_profile;


        // ====================================================
        // SCORE
        // ====================================================

        document.getElementById(
            "score"
        ).innerText =
            (report.placement_score || 0)
            + "%";


        document.getElementById(
            "scoreLabel"
        ).innerText =
            report.score_label ||
            "Placement Readiness";


        document.getElementById(
            "scoreSummary"
        ).innerText =
            report.overall_summary || "";


        document.getElementById(
            "summary"
        ).innerText =
            report.overall_summary || "";


        // ====================================================
        // PROFILE
        // ====================================================

        displayProfile(
            profile
        );


        // ====================================================
        // SKILLS
        // ====================================================

        document.getElementById(
            "matchedSkills"
        ).innerHTML =
            createList(
                report.matched_skills
            );


        document.getElementById(
            "missingSkills"
        ).innerHTML =
            createList(
                report.missing_skills
            );


        document.getElementById(
            "weakSkills"
        ).innerHTML =
            createList(
                report.weak_skills
            );


        document.getElementById(
            "strengths"
        ).innerHTML =
            createList(
                report.strengths
            );


        // ====================================================
        // IMPROVEMENTS
        // ====================================================

        document.getElementById(
            "improvements"
        ).innerHTML =
            createList(
                report.resume_improvements
            );


        // ====================================================
        // PROJECTS
        // ====================================================

        document.getElementById(
            "projects"
        ).innerHTML =
            createList(
                report.recommended_projects
            );


        // ====================================================
        // GITHUB
        // ====================================================

        document.getElementById(
            "github"
        ).innerText =
            report.github_assessment ||
            "GitHub information not available.";


        // ====================================================
        // JOBS
        // ====================================================

        document.getElementById(
            "jobs"
        ).innerHTML =
            createList(
                report.job_recommendations
            );


        // ====================================================
        // NEXT STEPS
        // ====================================================

        document.getElementById(
            "nextSteps"
        ).innerHTML =
            createList(
                report.next_steps
            );


        result.style.display =
            "block";


        status.className =
            "status success";

        status.innerText =
            "Analysis completed successfully!";


        window.scrollTo({
            top:
                result.offsetTop - 20,
            behavior:
                "smooth"
        });


    } catch (error) {

        status.className =
            "status error";

        status.innerText =
            "Analysis failed.";

        result.style.display =
            "block";

        result.innerHTML = `

            <div class="card">

                <h2 style="color:#dc2626">
                    Error
                </h2>

                <p>
                    ${escapeHtml(error.message)}
                </p>

            </div>

        `;

    }


    button.disabled = false;

}

</script>


</body>

</html>
"""


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv(
            "PORT",
            "8000"
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
