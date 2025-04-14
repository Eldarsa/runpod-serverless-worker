import os
import traceback
from typing import Any, Dict

def list_filesystem(root_path: str = '/', max_depth: int = 10, current_depth: int = 0) -> Dict[str, Any]:
    """
    Recursively list the filesystem starting from the given path.
    
    Args:
        root_path: The starting directory path
        max_depth: Maximum recursion depth to prevent excessive listing
        current_depth: Current recursion depth

    Returns:
        Dictionary containing the filesystem structure
    """
    result = {}
    root_path = os.path.abspath(os.path.expanduser(root_path))
    
    if current_depth > max_depth:
        return {"error": "Max depth reached"}
    
    try:
        for item in os.listdir(root_path):
            full_path = os.path.join(root_path, item)
            
            if os.path.isdir(full_path):
                result[item] = {
                    "type": "directory",
                    "path": full_path,
                    "contents": list_filesystem(full_path, max_depth, current_depth + 1) if current_depth < max_depth else {"note": "Max depth reached"}
                }
            else:
                try:
                    size = os.path.getsize(full_path)
                    result[item] = {
                        "type": "file",
                        "path": full_path,
                        "size": size,
                        "size_human": f"{size / 1024:.2f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.2f} MB"
                    }
                except (FileNotFoundError, PermissionError) as e:
                    result[item] = {
                        "type": "file",
                        "path": full_path,
                        "error": str(e)
                    }
    except (PermissionError, FileNotFoundError) as e:
        return {"error": str(e)}
    
    return result 