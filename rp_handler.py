import runpod
import requests
import time
import os
import traceback
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from requests.adapters import HTTPAdapter, Retry

from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger

from schemas.input import INPUT_SCHEMA
from schemas.api import API_SCHEMA
from schemas.img2img import IMG2IMG_SCHEMA
from schemas.txt2img import TXT2IMG_SCHEMA
from schemas.interrogate import INTERROGATE_SCHEMA
from schemas.sync import SYNC_SCHEMA
from schemas.download import DOWNLOAD_SCHEMA

# Import the filesystem utility
from utils.filesystem import list_filesystem

BASE_URI: str = 'http://127.0.0.1:3000'
TIMEOUT: int = 600
POST_RETRIES: int = 3

session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
logger = RunPodLogger()


# ---------------------------------------------------------------------------- #
#                               Utility Functions                              #
# ---------------------------------------------------------------------------- #
def wait_for_service(url: str) -> None:
    """
    Wait for a service to become available by repeatedly polling the URL.

    Args:
        url: The URL to check for service availability.
    """
    retries = 0

    while True:
        try:
            requests.get(url)
            return
        except requests.exceptions.RequestException:
            retries += 1

            # Only log every 15 retries so the logs don't get spammed
            if retries % 15 == 0:
                logger.info('Service not ready yet. Retrying...')
        except Exception as err:
            logger.error(f'Error: {err}')

        time.sleep(0.2)


def send_get_request(endpoint: str) -> requests.Response:
    """
    Send a GET request to the specified endpoint.

    Args:
        endpoint: The API endpoint to send the request to.

    Returns:
        The response from the server.
    """
    return session.get(
        url=f'{BASE_URI}/{endpoint}',
        timeout=TIMEOUT
    )


def send_post_request(endpoint: str, payload: Dict[str, Any], job_id: str, retry: int = 0) -> requests.Response:
    """
    Send a POST request to the specified endpoint with retries.

    Args:
        endpoint: The API endpoint to send the request to.
        payload: The data to send in the request body.
        job_id: The ID of the current job for logging.
        retry: Current retry attempt number.

    Returns:
        The response from the server.
    """
    response = session.post(
        url=f'{BASE_URI}/{endpoint}',
        json=payload,
        timeout=TIMEOUT
    )

    if response.status_code == 404 and retry < POST_RETRIES:
        retry += 1
        logger.warn(f'Received HTTP 404 from endpoint: {endpoint}, Retrying: {retry}', job_id)
        time.sleep(0.2)
        return send_post_request(endpoint, payload, job_id, retry)

    return response


def validate_input(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the input payload against the input schema.

    Args:
        job: The job dictionary containing the input to validate.

    Returns:
        The validated input or error dictionary.
    """
    return validate(job['input'], INPUT_SCHEMA)


def validate_api(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the API configuration against the API schema.

    Args:
        job: The job dictionary containing the API configuration to validate.

    Returns:
        The validated API configuration or error dictionary.
    """
    api = job['input']['api']
    api['endpoint'] = api['endpoint'].lstrip('/')

    return validate(api, API_SCHEMA)


def extract_scheduler(value: Union[str, int]) -> Tuple[Optional[str], str]:
    """
    Extract scheduler suffix from a sampler value if present.
    Preserves the original case of the value while checking for lowercase suffixes.

    Args:
        value: The sampler value to check for a scheduler suffix.

    Returns:
        A tuple containing the scheduler suffix (if found) and the cleaned value.
    """
    scheduler_suffixes = ['uniform', 'karras', 'exponential', 'polyexponential', 'sgm_uniform']
    value_str = str(value)
    value_lower = value_str.lower()

    for suffix in scheduler_suffixes:
        if value_lower.endswith(suffix):
            return suffix, value_str[:-(len(suffix))].rstrip()

    return None, value_str


def validate_payload(job: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """
    Validate the payload based on the endpoint and handle sampler/scheduler compatibility.

    Args:
        job: The job dictionary containing the payload to validate.

    Returns:
        A tuple containing the endpoint, method, and validated payload.
    """
    method = job['input']['api']['method']
    endpoint = job['input']['api']['endpoint']
    payload = job['input']['payload']
    validated_input = payload

    if endpoint in ['sdapi/v1/txt2img', 'sdapi/v1/img2img']:
        for field in ['sampler_index', 'sampler_name']:
            if field in payload:
                scheduler, cleaned_value = extract_scheduler(payload[field])
                if scheduler:
                    payload[field] = cleaned_value
                    payload['scheduler'] = scheduler

    if endpoint == 'v1/sync':
        logger.info(f'Validating /{endpoint} payload', job['id'])
        validated_input = validate(payload, SYNC_SCHEMA)
    elif endpoint == 'v1/download':
        logger.info(f'Validating /{endpoint} payload', job['id'])
        validated_input = validate(payload, DOWNLOAD_SCHEMA)
    elif endpoint == 'sdapi/v1/txt2img':
        logger.info(f'Validating /{endpoint} payload', job['id'])
        validated_input = validate(payload, TXT2IMG_SCHEMA)
    elif endpoint == 'sdapi/v1/img2img':
        logger.info(f'Validating /{endpoint} payload', job['id'])
        validated_input = validate(payload, IMG2IMG_SCHEMA)
    elif endpoint == 'sdapi/v1/interrogate' and method == 'POST':
        logger.info(f'Validating /{endpoint} payload', job['id'])
        validated_input = validate(payload, INTERROGATE_SCHEMA)

    return endpoint, job['input']['api']['method'], validated_input


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main handler function for processing RunPod jobs.
    """
    job_input = job['input']
    
    # Check if this is a filesystem listing request
    if job_input.get('list_filesystem'):
        start_path = job_input.get('start_path', '/')
        max_depth = job_input.get('max_depth', 10)
        logger.info(f"Listing filesystem from {start_path} with max depth {max_depth}", job['id'])
        
        try:
            result = list_filesystem(start_path, max_depth)
            return {
                "filesystem": result
            }
        except Exception as e:
            logger.error(f"Error listing filesystem: {str(e)}", job['id'])
            return {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    # Continue with regular API request handling
    try:
        # Validate the input
        validation_error = validate_input(job)
        if 'errors' in validation_error:
            return {
                "error": validation_error['errors']
            }
        
        # Validate the API configuration
        api_validation = validate_api(job)
        if 'errors' in api_validation:
            return {
                "error": api_validation['errors']
            }
        
        # Wait for the service to be available
        wait_for_service(f"{BASE_URI}/docs")
        
        # Process the request based on the API endpoint and method
        endpoint, method, payload = validate_payload(job)
        
        if 'errors' in payload:
            return {
                "error": payload['errors']
            }
        
        logger.info(f"Request: {method} /{endpoint}", job['id'])
        
        if method == 'GET':
            response = send_get_request(endpoint)
        else:
            response = send_post_request(endpoint, payload, job['id'])
        
        if response.status_code >= 400:
            logger.error(f"Error: HTTP {response.status_code}", job['id'])
            logger.error(f"Response: {response.text}", job['id'])
            return {
                "error": f"HTTP {response.status_code}",
                "response": response.text
            }
        
        # Return successful response
        try:
            return response.json()
        except ValueError:
            return {
                "content": response.text
            }
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", job['id'])
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Start the serverless function
runpod.start(handler)