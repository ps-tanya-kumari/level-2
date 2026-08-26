import os
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

EXCLUDED_DIRS = {
    '.git', 'node_modules', '__pycache__', 'venv', '.venv', 'env', '.env',
    'build', 'dist', '.next', '.nuxt', 'out', '.vscode', '.idea', 'target',
    'bin', 'obj', 'vendor', 'bower_components', 'dist', 'tmp', 'cache'
}

EXCLUDED_FILES = {
    'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 'poetry.lock',
    'composer.lock', 'Gemfile.lock', 'cargo.lock', 'mix.lock',
    '.DS_Store', 'thumbs.db'
}

EXCLUDED_EXTENSIONS = {
    # Images
    '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.webp', '.tiff', '.bmp',
    # Binaries/Archives
    '.zip', '.tar', '.gz', '.rar', '.7z', '.exe', '.dll', '.so', '.bin', '.pdf', '.docx', '.xlsx', '.pptx',
    # Audio/Video
    '.mp3', '.mp4', '.avi', '.mov', '.flv', '.wav', '.webm',
    # Fonts
    '.woff', '.woff2', '.ttf', '.eot', '.otf',
    # Databases/Misc
    '.db', '.sqlite', '.sqlite3', '.pyc', '.pyd', '.pyo', '.class', '.o'
}

# Max file size to fetch (500 KB)
MAX_FILE_SIZE = 500 * 1024

class RepositoryParser:
    @staticmethod
    def should_analyze_file(path: str, size: int) -> bool:
        """Determines if a file should be parsed and analyzed."""
        # Check size
        if size > MAX_FILE_SIZE:
            return False
            
        parts = path.split('/')
        
        # Check if file is in an excluded directory
        for part in parts:
            if part in EXCLUDED_DIRS:
                return False
                
        # Check if file name is explicitly excluded
        filename = parts[-1]
        if filename in EXCLUDED_FILES:
            return False
            
        # Skip seed, test, and mock files to preserve API rate limit quota
        filename_lower = filename.lower()
        if 'seed' in filename_lower or 'test' in filename_lower or 'mock' in filename_lower:
            return False
            
        # Check if extension is excluded
        _, ext = os.path.splitext(filename.lower())
        if ext in EXCLUDED_EXTENSIONS:
            return False
            
        return True

    @staticmethod
    def filter_files(tree_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filters tree items to get a list of files that are worth analyzing."""
        analyzable_files = []
        for item in tree_items:
            if item.get("type") == "blob":
                path = item.get("path", "")
                size = item.get("size", 0)
                if RepositoryParser.should_analyze_file(path, size):
                    analyzable_files.append(item)
        return analyzable_files

    @staticmethod
    def build_tree_structure(tree_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Builds a nested dictionary representation of the file tree.
        Example output format for frontend:
        {
          "name": "root",
          "path": "",
          "type": "tree",
          "children": {
             "src": { "name": "src", "path": "src", "type": "tree", "children": {...} },
             "README.md": { "name": "README.md", "path": "README.md", "type": "blob", "sha": "..." }
          }
        }
        """
        root = {
            "name": "root",
            "path": "",
            "type": "tree",
            "children": {}
        }
        
        for item in tree_items:
            path = item.get("path", "")
            if not path:
                continue
                
            parts = path.split('/')
            current = root
            
            # Traversal for directories
            for i, part in enumerate(parts[:-1]):
                # If any directory part is in EXCLUDED_DIRS, we skip adding this subtree path
                if part in EXCLUDED_DIRS:
                    break
                    
                if part not in current["children"]:
                    current["children"][part] = {
                        "name": part,
                        "path": "/".join(parts[:i+1]),
                        "type": "tree",
                        "children": {}
                    }
                current = current["children"][part]
            else:
                # Add file or directory
                filename = parts[-1]
                # If directory
                if item.get("type") == "tree":
                    if filename not in EXCLUDED_DIRS:
                        if filename not in current["children"]:
                            current["children"][filename] = {
                                "name": filename,
                                "path": path,
                                "type": "tree",
                                "children": {}
                            }
                # If file (blob)
                elif item.get("type") == "blob":
                    # Check if file/extension is excluded
                    _, ext = os.path.splitext(filename.lower())
                    if filename not in EXCLUDED_FILES and ext not in EXCLUDED_EXTENSIONS:
                        current["children"][filename] = {
                            "name": filename,
                            "path": path,
                            "type": "blob",
                            "sha": item.get("sha"),
                            "size": item.get("size", 0)
                        }
        
        # Convert children dicts to sorted lists recursively
        def convert_to_list(node: Dict[str, Any]) -> Dict[str, Any]:
            if "children" in node:
                children_list = []
                for child in node["children"].values():
                    children_list.append(convert_to_list(child))
                # Sort: directories first, then files alphabetically
                children_list.sort(key=lambda x: (0 if x["type"] == "tree" else 1, x["name"].lower()))
                node["children"] = children_list
            return node

        return convert_to_list(root)
