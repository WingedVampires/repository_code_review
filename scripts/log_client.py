import requests
import json
from typing import Dict, Any, Union
import sys

def send_log_to_server(log_data: Dict[str, Any], server_url: str) -> Dict[str, Any]:
    """
    Send log data to the log server.
    
    Args:
        log_data: Dictionary containing the log data to send
        server_url: URL of the log server (e.g., "http://localhost:5000")
    
    Returns:
        Dictionary with response information
    """
    try:
        # Ensure server_url doesn't end with a slash
        if server_url.endswith('/'):
            server_url = server_url[:-1]
        
        # Send the log data to the server
        endpoint = f"{server_url}/log"
        headers = {'Content-Type': 'application/json'}
        
        response = requests.post(endpoint, json=log_data, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            return {"success": True, "response": response.json()}
        else:
            return {
                "success": False, 
                "error": f"Request failed with status code {response.status_code}",
                "response": response.text
            }
    
    except Exception as e:
        return {"success": False, "error": str(e)}

# Example usage if script is run directly
if __name__ == "__main__":
    # 发送日志
    result = send_log_to_server(
        {"message": "Test log", "level": "info"}, 
        "http://localhost:5000"
    )
    print(result)