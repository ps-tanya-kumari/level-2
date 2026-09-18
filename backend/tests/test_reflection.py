import pytest
from app.agents.reflection import ReflectionVerifier, VerificationResult

def test_extract_referenced_paths():
    text = (
        "The project entry point is in `backend/app/main.py`. "
        "It also uses `src/components/Navbar.jsx` and config in `package.json`."
    )
    paths = ReflectionVerifier.extract_referenced_paths(text)
    assert "backend/app/main.py" in paths
    assert "src/components/Navbar.jsx" in paths
    assert "package.json" in paths

def test_verify_answer_fully_grounded():
    text = "The application entry point is `backend/app/main.py` and tools are defined in `backend/app/tools.py`."
    repo_files = ["backend/app/main.py", "backend/app/tools.py", "README.md"]
    
    result: VerificationResult = ReflectionVerifier.verify_answer(
        candidate_answer=text,
        repository_files=repo_files
    )
    
    assert result.is_grounded is True
    assert result.confidence_score == 1.0
    assert len(result.unverified_paths) == 0
    assert len(result.limitations) == 0
    assert "Grounding & Reflection Report" not in result.final_answer

def test_verify_answer_detects_hallucination_and_annotates():
    text = (
        "The authentication flow is handled in `src/auth/cognito_handler.py`, "
        "while main application setup is in `backend/app/main.py`."
    )
    repo_files = ["backend/app/main.py", "backend/app/tools.py", "README.md"]
    
    result: VerificationResult = ReflectionVerifier.verify_answer(
        candidate_answer=text,
        repository_files=repo_files
    )
    
    assert result.is_grounded is False
    assert result.confidence_score == 0.5
    assert "src/auth/cognito_handler.py" in result.unverified_paths
    assert len(result.limitations) > 0
    assert "Grounding & Reflection Report" in result.final_answer
    assert "cognito_handler.py" in result.final_answer

def test_verify_answer_no_paths_referenced():
    text = "This repository implements a lightweight REST API for calculating fibonacci numbers."
    repo_files = ["app.py", "README.md"]
    
    result: VerificationResult = ReflectionVerifier.verify_answer(
        candidate_answer=text,
        repository_files=repo_files
    )
    
    assert result.is_grounded is True
    assert result.confidence_score == 1.0
    assert len(result.limitations) == 0
