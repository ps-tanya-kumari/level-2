import os
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, ValidationError

# Regex to restrict owner and repository names to valid GitHub characters
REPO_OWNER_REGEX = re.compile(r"^[a-zA-Z0-9_.-]+$")

class ToolValidationError(ValueError):
    """Custom exception raised when tool argument validation fails."""
    def __init__(self, tool_name: str, message: str, field_errors: Optional[Dict[str, str]] = None):
        self.tool_name = tool_name
        self.message = message
        self.field_errors = field_errors or {}
        super().__init__(f"ToolValidationError for '{tool_name}': {message}")


def sanitize_path(path: str) -> str:
    """
    Validates and normalizes a repository relative file path.
    Prevents path traversal, absolute paths, null bytes, and Windows drive escapes.
    """
    if not path or not isinstance(path, str):
        raise ToolValidationError("read_repository_file", "File path must be a non-empty string.")
    
    # Check for null bytes
    if "\x00" in path:
        raise ToolValidationError("read_repository_file", "Path contains null byte character.")
    
    # Normalize backslashes to forward slashes
    normalized = path.replace("\\", "/").strip()
    
    # Reject path traversal (e.g. ../ or /..)
    parts = normalized.split("/")
    if any(p == ".." for p in parts):
        raise ToolValidationError("read_repository_file", f"Path traversal attempt detected in path: '{path}'.")
    
    # Reject leading slashes (absolute paths)
    if normalized.startswith("/"):
        normalized = normalized.lstrip("/")
        
    # Reject Windows drive letters (e.g. C:)
    if len(normalized) >= 2 and normalized[1] == ":" and normalized[0].isalpha():
        raise ToolValidationError("read_repository_file", f"Absolute drive path rejected: '{path}'.")
        
    if not normalized:
        raise ToolValidationError("read_repository_file", "Path resolved to an empty string.")
        
    return normalized


class BaseRepoArgs(BaseModel):
    owner: str = Field(..., description="GitHub repository owner username or organization")
    repo: str = Field(..., description="GitHub repository name")

    @field_validator("owner", "repo")
    @classmethod
    def validate_identifier(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Must not be empty.")
        if not REPO_OWNER_REGEX.match(v):
            raise ValueError(f"Contains invalid characters: '{v}'. Only alphanumeric, hyphens, underscores, and dots are allowed.")
        return v


class ListFilesArgs(BaseRepoArgs):
    pass


class GetStructureArgs(BaseRepoArgs):
    pass


class ReadFileArgs(BaseRepoArgs):
    path: str = Field(..., description="Relative file path within the repository")

    @field_validator("path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        return sanitize_path(v)


class SearchCodeArgs(BaseRepoArgs):
    query: str = Field(..., min_length=1, max_length=500, description="Search query string")
    n_results: int = Field(default=5, ge=1, le=25, description="Number of results to return (1-25)")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query string must not be empty.")
        return v


class GetMetadataArgs(BaseRepoArgs):
    pass


class GetProjectSummaryArgs(BaseRepoArgs):
    pass


# Mapping tool names to their corresponding Pydantic validation schemas
TOOL_ARG_SCHEMAS: Dict[str, type] = {
    "list_repository_files": ListFilesArgs,
    "get_repository_structure": GetStructureArgs,
    "read_repository_file": ReadFileArgs,
    "search_repository_code": SearchCodeArgs,
    "get_repository_metadata": GetMetadataArgs,
    "get_project_summary": GetProjectSummaryArgs,
}


class ToolValidator:
    """Central validator for all tool arguments and model outputs before execution."""

    @staticmethod
    def get_supported_tools() -> List[str]:
        return list(TOOL_ARG_SCHEMAS.keys())

    @staticmethod
    def is_tool_supported(tool_name: str) -> bool:
        return tool_name in TOOL_ARG_SCHEMAS

    @classmethod
    def validate_arguments(cls, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the arguments dictionary against the Pydantic schema for the tool.
        Returns the sanitized and validated dictionary.
        Raises ToolValidationError if invalid.
        """
        if not cls.is_tool_supported(tool_name):
            raise ToolValidationError(
                tool_name=tool_name,
                message=f"Unsupported tool '{tool_name}'. Available tools: {cls.get_supported_tools()}"
            )
        
        schema_cls = TOOL_ARG_SCHEMAS[tool_name]
        try:
            validated_obj = schema_cls(**arguments)
            return validated_obj.model_dump()
        except ValidationError as e:
            errors = {}
            for err in e.errors():
                loc = ".".join(str(p) for p in err.get("loc", []))
                msg = err.get("msg", "Invalid value")
                errors[loc] = msg
            
            error_details = ", ".join(f"{k}: {v}" for k, v in errors.items())
            raise ToolValidationError(
                tool_name=tool_name,
                message=f"Invalid arguments for '{tool_name}': {error_details}",
                field_errors=errors
            ) from e
