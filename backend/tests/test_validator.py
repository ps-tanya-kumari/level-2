import pytest
from app.validator import (
    ToolValidator, 
    ToolValidationError, 
    sanitize_path, 
    ReadFileArgs, 
    SearchCodeArgs, 
    ListFilesArgs
)

def test_sanitize_path_valid():
    assert sanitize_path("backend/app/main.py") == "backend/app/main.py"
    assert sanitize_path("src\\components\\App.js") == "src/components/App.js"
    assert sanitize_path("/app/main.py") == "app/main.py"

def test_sanitize_path_traversal_rejection():
    with pytest.raises(ToolValidationError) as exc:
        sanitize_path("../../etc/passwd")
    assert "Path traversal attempt detected" in str(exc.value)

    with pytest.raises(ToolValidationError) as exc:
        sanitize_path("app/../secret.env")
    assert "Path traversal attempt detected" in str(exc.value)

def test_sanitize_path_null_byte_rejection():
    with pytest.raises(ToolValidationError) as exc:
        sanitize_path("main.py\x00.exe")
    assert "null byte character" in str(exc.value)

def test_sanitize_path_drive_letter_rejection():
    with pytest.raises(ToolValidationError) as exc:
        sanitize_path("C:/Windows/System32")
    assert "Absolute drive path rejected" in str(exc.value)

def test_validate_read_file_args():
    validated = ToolValidator.validate_arguments(
        "read_repository_file",
        {"owner": "octocat", "repo": "Spoon-Knife", "path": "src/index.js"}
    )
    assert validated["owner"] == "octocat"
    assert validated["repo"] == "Spoon-Knife"
    assert validated["path"] == "src/index.js"

def test_validate_search_code_args():
    validated = ToolValidator.validate_arguments(
        "search_repository_code",
        {"owner": "octocat", "repo": "Spoon-Knife", "query": "auth login", "n_results": 10}
    )
    assert validated["query"] == "auth login"
    assert validated["n_results"] == 10

def test_validate_search_code_bounds():
    with pytest.raises(ToolValidationError):
        ToolValidator.validate_arguments(
            "search_repository_code",
            {"owner": "octocat", "repo": "Spoon-Knife", "query": "auth", "n_results": 100}
        )

def test_validate_unknown_tool():
    with pytest.raises(ToolValidationError) as exc:
        ToolValidator.validate_arguments("drop_database_tables", {"owner": "test", "repo": "test"})
    assert "Unsupported tool" in str(exc.value)

def test_validate_invalid_repo_name():
    with pytest.raises(ToolValidationError) as exc:
        ToolValidator.validate_arguments(
            "list_repository_files", 
            {"owner": "octocat; rm -rf /", "repo": "repo"}
        )
    assert "Contains invalid characters" in str(exc.value)
