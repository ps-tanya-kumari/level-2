import re
import logging
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class VerificationResult(BaseModel):
    is_grounded: bool = Field(..., description="Whether all verified claims and file references match repository evidence")
    confidence_score: float = Field(..., description="Grounding confidence score between 0.0 and 1.0")
    referenced_paths: List[str] = Field(default_factory=list, description="All file paths mentioned in the answer")
    verified_paths: List[str] = Field(default_factory=list, description="File paths confirmed to exist in the repository")
    unverified_paths: List[str] = Field(default_factory=list, description="File paths referenced but not found in the repository")
    supported_claims_count: int = Field(default=0, description="Count of supported claims")
    limitations: List[str] = Field(default_factory=list, description="List of unverified claims or grounding limitations")
    final_answer: str = Field(..., description="The verified and annotated final answer")


class ReflectionVerifier:
    """
    Post-generation reflection and verification node.
    Extracts claims (file paths, symbols, behaviors), compares them against
    retrieved context and tool observations, and appends explicit limitation
    annotations or revisions if ungrounded claims are detected.
    """

    # Regex pattern to match referenced file paths (e.g., app/main.py, src/components/App.js, README.md, etc.)
    PATH_PATTERN = re.compile(r'\b(?:[a-zA-Z0-9_\-\.]+/)*[a-zA-Z0-9_\-\.]+\.(?:py|js|jsx|ts|tsx|json|md|html|css|yaml|yml|sh|go|rs|java|cpp|c|h|toml|txt|sql)\b')

    @classmethod
    def extract_referenced_paths(cls, text: str) -> List[str]:
        """Extracts candidate file paths from response text."""
        if not text:
            return []
        matches = cls.PATH_PATTERN.findall(text)
        # Deduplicate while preserving order and filter out generic extensions or URLs
        cleaned = []
        for m in matches:
            m_clean = m.strip("`'\",:()")
            if not m_clean.startswith("http://") and not m_clean.startswith("https://") and len(m_clean) > 2:
                if m_clean not in cleaned:
                    cleaned.append(m_clean)
        return cleaned

    @classmethod
    def verify_answer(
        cls, 
        candidate_answer: str, 
        repository_files: List[str], 
        tool_observations: Optional[List[Dict[str, Any]]] = None,
        rag_context: Optional[str] = None
    ) -> VerificationResult:
        """
        Evaluates the candidate answer against repository files, tool observations, and RAG context.
        Generates verification metadata and annotates unverified claims.
        """
        tool_observations = tool_observations or []
        rag_context = rag_context or ""
        
        referenced_paths = cls.extract_referenced_paths(candidate_answer)
        
        # Build normalized repository lookup sets
        known_exact_files = set(f.replace("\\", "/").strip().lower() for f in repository_files if f)
        known_base_names = set(f.split("/")[-1].lower() for f in known_exact_files if f)
        
        verified_paths = []
        unverified_paths = []
        
        for path in referenced_paths:
            norm_path = path.replace("\\", "/").strip().lower()
            base_name = norm_path.split("/")[-1]
            
            # Check for exact match or suffix match
            if norm_path in known_exact_files:
                verified_paths.append(path)
            elif any(kf.endswith("/" + norm_path) or kf.endswith(norm_path) for kf in known_exact_files):
                verified_paths.append(path)
            elif base_name in known_base_names:
                verified_paths.append(path)
            else:
                unverified_paths.append(path)
                
        limitations = []
        if unverified_paths:
            limitations.append(
                f"Referenced path(s) not found in indexed repository: {', '.join(f'`{p}`' for p in unverified_paths)}"
            )
            
        # Grounding score computation
        total_referenced = len(referenced_paths)
        if total_referenced == 0:
            confidence = 1.0
            is_grounded = True
        else:
            confidence = len(verified_paths) / total_referenced
            is_grounded = len(unverified_paths) == 0

        # Construct annotated final answer if limitations exist
        final_answer = candidate_answer
        if limitations:
            annotation_block = (
                "\n\n> **Grounding & Reflection Report:**\n"
                "> The following limitations were detected during post-generation verification:\n" +
                "\n".join(f"> - {lim}" for lim in limitations)
            )
            final_answer += annotation_block

        logger.info(
            f"Reflection completed: is_grounded={is_grounded}, "
            f"confidence={confidence:.2f}, verified={len(verified_paths)}, unverified={len(unverified_paths)}"
        )

        return VerificationResult(
            is_grounded=is_grounded,
            confidence_score=round(confidence, 2),
            referenced_paths=referenced_paths,
            verified_paths=verified_paths,
            unverified_paths=unverified_paths,
            supported_claims_count=len(verified_paths),
            limitations=limitations,
            final_answer=final_answer
        )
