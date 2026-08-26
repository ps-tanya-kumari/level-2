import os
from typing import List, Dict, Any

class CodeChunker:
    @staticmethod
    def chunk_file(
        github_username: str, 
        repo_name: str, 
        file_path: str, 
        content: str, 
        sha: str, 
        chunk_size: int = 1500, 
        chunk_overlap: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Chunks the content of a file into line-preserved blocks.
        Each chunk is mapped to standard metadata.
        """
        if not content or not content.strip():
            return []

        lines = content.splitlines()
        chunks = []
        current_chunk_lines = []
        current_length = 0
        
        file_name = file_path.split('/')[-1]
        _, file_type = os.path.splitext(file_name)
        
        chunk_index = 0
        for line in lines:
            current_chunk_lines.append(line)
            current_length += len(line) + 1 # count newline character
            
            if current_length >= chunk_size:
                chunk_content = "\n".join(current_chunk_lines)
                chunk_id = f"{github_username}/{repo_name}/{file_path}#chunk{chunk_index}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": chunk_content,
                    "metadata": {
                        "repo_name": repo_name,
                        "github_username": github_username,
                        "file_path": file_path,
                        "file_name": file_name,
                        "file_type": file_type or "unknown",
                        "chunk_id": chunk_id,
                        "sha": sha
                    }
                })
                chunk_index += 1
                
                # Create overlap by grabbing the last few lines that sum up to less than chunk_overlap
                overlap_lines = []
                overlap_length = 0
                for ol in reversed(current_chunk_lines):
                    if overlap_length + len(ol) + 1 <= chunk_overlap:
                        overlap_lines.insert(0, ol)
                        overlap_length += len(ol) + 1
                    else:
                        break
                current_chunk_lines = overlap_lines
                current_length = overlap_length
                
        # Append the remaining content if any
        if current_chunk_lines and len("\n".join(current_chunk_lines).strip()) > 0:
            chunk_content = "\n".join(current_chunk_lines)
            chunk_id = f"{github_username}/{repo_name}/{file_path}#chunk{chunk_index}"
            chunks.append({
                "chunk_id": chunk_id,
                "content": chunk_content,
                "metadata": {
                    "repo_name": repo_name,
                    "github_username": github_username,
                    "file_path": file_path,
                    "file_name": file_name,
                    "file_type": file_type or "unknown",
                    "chunk_id": chunk_id,
                    "sha": sha
                }
            })
            
        return chunks
