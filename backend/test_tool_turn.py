import os
import httpx
from dotenv import load_dotenv

load_dotenv('backend/.env')

nvidia_key = os.getenv('NVIDIA_API_KEY')
headers = {'Authorization': f'Bearer {nvidia_key}', 'Content-Type': 'application/json'}

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'search_repository_code',
            'description': 'Search code in repository',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string', 'description': 'query text'}
                },
                'required': ['query']
            }
        }
    }
]

step1_msg = {
    'role': 'assistant',
    'content': '',
    'tool_calls': [{
        'id': 'call-7b055e86-98da-467e-98a4-6a44c00ac50d',
        'type': 'function',
        'function': {'name': 'search_repository_code', 'arguments': '{"query":"smart bottle"}'}
    }]
}
tool_msg = {
    'role': 'tool',
    'tool_call_id': 'call-7b055e86-98da-467e-98a4-6a44c00ac50d',
    'name': 'search_repository_code',
    'content': 'Found 2 chunks: chunk 1 smart bottle tracker...'
}

payload = {
    'model': 'nvidia/nemotron-3-super-120b-a12b',
    'messages': [
        {'role': 'user', 'content': 'tell me about the smart bottle water'},
        step1_msg,
        tool_msg
    ],
    'tools': tools
}

try:
    r = httpx.post('https://integrate.api.nvidia.com/v1/chat/completions', json=payload, headers=headers, timeout=20)
    print('Status:', r.status_code)
    print('Response:', r.text)
except Exception as e:
    print('Exception:', e)
