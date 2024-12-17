import asyncio
import hashlib
import json
import logging
import os
import ssl
import time
from pprint import pprint
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, validator
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

from synth_sdk.tracing.abstractions import Dataset, SystemTrace
from synth_sdk.tracing.events.store import event_store
from synth_sdk.tracing.utils import get_system_id
from synth_sdk.tracing.local import (
    _local,
    system_name_var,
    system_id_var,
    system_instance_id_var,
)
load_dotenv()


# NOTE: This may cause memory issues in the future
def validate_json(data: dict) -> None:
    # Validate that a dictionary contains only JSON-serializable values.

    # Args:
    #    data: Dictionary to validate for JSON serialization

    # Raises:
    #    ValueError: If the dictionary contains non-serializable values

    try:
        json.dumps(data)
    except (TypeError, OverflowError) as e:
        raise ValueError(f"Contains non-JSON-serializable values: {e}. {data}")


def createPayload(dataset: Dataset, traces: List[SystemTrace]) -> Dict[str, Any]:
    payload = {
        "traces": [
            trace.to_dict() for trace in traces
        ],  # Convert SystemTrace objects to dicts
        "dataset": dataset.to_dict(),
    }
    return payload


class TLSAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False):
        """Create and initialize the urllib3 PoolManager."""
        ctx = ssl.create_default_context()
        ctx.set_ciphers("DEFAULT@SECLEVEL=1")
        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_version=ssl.PROTOCOL_TLSv1_2,
            ssl_context=ctx,
        )


def load_signed_url(signed_url: str, dataset: Dataset, traces: List[SystemTrace]):
    payload = createPayload(dataset, traces)
    validate_json(payload)

    session = requests.Session()
    adapter = TLSAdapter()
    session.mount("https://", adapter)

    try:
        response = session.put(
            signed_url, json=payload, headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {str(e)}")
        print(f"Request payload: {payload}")  # Add this for debugging
        raise

    if response.status_code != 200:
        raise ValueError(
            f"Failed to load signed URL Status Code: {response.status_code} Response: {response.text}, Signed URL: {signed_url}"
        )
    else:
        print(
            f"Successfully loaded signed URL Status Code: {response.status_code} Response: {response.text}, Signed URL: {signed_url}"
        )


def send_system_traces_s3(
    dataset: Dataset,
    traces: List[SystemTrace],
    base_url: str,
    api_key: str,
    system_instance_id: str,
    verbose: bool = False,
):
    # Create async function that contains all async operations
    async def _async_operations():
        upload_id, signed_url = await get_upload_id(
            base_url, api_key, system_instance_id, verbose
        )
        load_signed_url(signed_url, dataset, traces)

        token_url = f"{base_url}/v1/auth/token"
        token_response = requests.get(
            token_url, headers={"customer_specific_api_key": api_key}
        )
        token_response.raise_for_status()
        access_token = token_response.json()["access_token"]

        api_url = f"{base_url}/v1/uploads/process-upload/{upload_id}"
        data = {"signed_url": signed_url}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
        }

        try:
            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()

            upload_id = response.json()["upload_id"]
            signed_url = response.json()["signed_url"]
            status = response.json()["status"]

            if verbose:
                print(f"Status: {status}")
                print(f"Upload ID retrieved: {upload_id}")
                print(f"Signed URL: {signed_url}")

            return upload_id, signed_url
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP error occurred: {e}")
            raise
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise

    # Run the async operations in an event loop
    if not is_event_loop_running():
        # If no event loop is running, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_async_operations())
        finally:
            loop.close()
    else:
        # If an event loop is already running, use it
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_async_operations())


async def get_upload_id(
    base_url: str, api_key: str, system_instance_id, verbose: bool = False
):
    token_url = f"{base_url}/v1/auth/token"
    token_response = requests.get(
        token_url, headers={"customer_specific_api_key": api_key}
    )
    token_response.raise_for_status()
    access_token = token_response.json()["access_token"]

    api_url = f"{base_url}/v1/uploads/get-upload-id-signed-url?system_instance_id={system_instance_id}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        upload_id = response.json()["upload_id"]
        signed_url = response.json()["signed_url"]
        if verbose:
            print(f"Upload ID retrieved: {upload_id}")
            print(f"Signed URL retrieved: {signed_url}")
        return upload_id, signed_url
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error occurred: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


class UploadValidator(BaseModel):
    traces: List[Dict[str, Any]]
    dataset: Dict[str, Any]

    @validator("traces")
    def validate_traces(cls, traces):
        if not traces:
            raise ValueError("Traces list cannot be empty")

        for trace in traces:
            # Validate required fields in each trace
            if "system_instance_id" not in trace:
                raise ValueError("Each trace must have a system_instance_id")
            if "partition" not in trace:
                raise ValueError("Each trace must have a partition")

            # Validate metadata if present
            if "metadata" in trace and trace["metadata"] is not None:
                if not isinstance(trace["metadata"], dict):
                    raise ValueError("Metadata must be a dictionary")

            # Validate partition structure
            partition = trace["partition"]
            if not isinstance(partition, list):
                raise ValueError("Partition must be a list")

            for part in partition:
                if "partition_index" not in part:
                    raise ValueError(
                        "Each partition element must have a partition_index"
                    )
                if "events" not in part:
                    raise ValueError("Each partition element must have an events list")

                # Validate events
                events = part["events"]
                if not isinstance(events, list):
                    raise ValueError("Events must be a list")

                for event in events:
                    required_fields = [
                        "event_type",
                        "opened",
                        "closed",
                        "partition_index",
                    ]
                    missing_fields = [f for f in required_fields if f not in event]
                    if missing_fields:
                        raise ValueError(
                            f"Event missing required fields: {missing_fields}"
                        )

        return traces

    @validator("dataset")
    def validate_dataset(cls, dataset):
        required_fields = ["questions", "reward_signals"]
        missing_fields = [f for f in required_fields if f not in dataset]
        if missing_fields:
            raise ValueError(f"Dataset missing required fields: {missing_fields}")

        # Validate questions
        questions = dataset["questions"]
        if not isinstance(questions, list):
            raise ValueError("Questions must be a list")

        for question in questions:
            if "intent" not in question or "criteria" not in question:
                raise ValueError("Each question must have intent and criteria")

        # Validate reward signals
        reward_signals = dataset["reward_signals"]
        if not isinstance(reward_signals, list):
            raise ValueError("Reward signals must be a list")

        for signal in reward_signals:
            required_signal_fields = ["question_id", "system_instance_id", "reward"]
            missing_fields = [f for f in required_signal_fields if f not in signal]
            if missing_fields:
                raise ValueError(
                    f"Reward signal missing required fields: {missing_fields}"
                )

        return dataset


def validate_upload(traces: List[Dict[str, Any]], dataset: Dict[str, Any]):
    # Validate the upload format before sending to server.
    # Raises ValueError if validation fails.
    try:
        UploadValidator(traces=traces, dataset=dataset)
        return True
    except ValueError as e:
        raise ValueError(f"Upload validation failed: {str(e)}")


def is_event_loop_running():
    try:
        asyncio.get_running_loop()  # Check if there's a running event loop
        return True
    except RuntimeError:
        # This exception is raised if no event loop is running
        return False


def format_upload_output(dataset, traces):
    # Format questions array
    questions_data = [
        {"intent": q.intent, "criteria": q.criteria, "id": q.id}
        for q in dataset.questions
    ]

    # Format reward signals array with error handling
    reward_signals_data = [
        {
            "system_instance_id": rs.system_instance_id,
            "reward": rs.reward,
            "question_id": rs.question_id,
            "annotation": rs.annotation if hasattr(rs, "annotation") else None,
        }
        for rs in dataset.reward_signals
    ]

    # Format traces array
    traces_data = [
        {
            "system_instance_id": t.system_instance_id,
            "metadata": t.metadata if t.metadata else None,
            "partition": [
                {
                    "partition_index": p.partition_index,
                    "events": [e.to_dict() for e in p.events],
                }
                for p in t.partition
            ],
        }
        for t in traces
    ]

    return questions_data, reward_signals_data, traces_data


# Supports calls from both async and sync contexts
def upload(
    dataset: Dataset,
    traces: List[SystemTrace] = [],
    verbose: bool = False,
    show_payload: bool = False,
):
    """Upload all system traces and dataset to the server.
    Returns a tuple of (response, questions_json, reward_signals_json, traces_json)
    Note that you can directly upload questions, reward_signals, and traces to the server using the Website

    response is the response from the server.
    questions_json is the formatted questions array
    reward_signals_json is the formatted reward signals array
    traces_json is the formatted traces array"""

    return upload_helper(dataset, traces, verbose, show_payload)


def upload_helper(
    dataset: Dataset,
    traces: List[SystemTrace] = [],
    verbose: bool = False,
    show_payload: bool = False,
):
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        raise ValueError("SYNTH_API_KEY environment variable not set")

    # Initialize variables
    system_name = None
    system_id = None
    system_instance_id = None

    # Try to get variables from ContextVar (asynchronous context)
    try:
        system_name = system_name_var.get()
        system_id = system_id_var.get()
        system_instance_id = system_instance_id_var.get()
    except LookupError:
        pass  # Variables not set in ContextVar

    # If variables are not set, try to get them from thread-local storage (synchronous context)
    if not system_name and hasattr(_local, "system_name"):
        system_name = _local.system_name
    if not system_id and hasattr(_local, "system_id"):
        system_id = _local.system_id
    if not system_instance_id and hasattr(_local, "system_instance_id"):
        system_instance_id = _local.system_instance_id

    # Check if we were able to retrieve the required variables
    if not system_name or not system_id or not system_instance_id:
        raise ValueError(
            f"system_name, system_id, or system_instance_id not set in context or thread-local storage - {system_name}, {system_id}, {system_instance_id}"
        )
    # if not system_name:
    #     raise ValueError("system_name not set in context or thread-local storage")
    # if not system_id:
    #     raise ValueError("system_id not set in context or thread-local storage")
    # if not system_instance_id:
    #     raise ValueError("system_instance_id not set in context or thread-local storage")

    if verbose:
        print(f"Using system_name: {system_name}")
        print(f"Generated system_id: {system_id}")

    # End all active events before uploading
    if hasattr(_local, "active_events"):
        for event_type, event in _local.active_events.items():
            if event and event.closed is None:
                event.closed = time.time()
                if hasattr(event, "system_instance_id"):
                    try:
                        event_store.add_event(event.system_instance_id, event)
                        if verbose:
                            print(f"Closed and stored active event: {event_type}")
                    except Exception as e:
                        logging.error(f"Failed to store event {event_type}: {str(e)}")
                else:
                    logging.error(f"Event {event_type} has no system_instance_id")
        _local.active_events.clear()

    # Also close any unclosed events in existing traces
    logged_traces = event_store.get_system_traces()
    traces = logged_traces + traces
    # traces = event_store.get_system_traces() if len(traces) == 0 else traces
    current_time = time.time()
    for trace in traces:
        for partition in trace.partition:
            for event in partition.events:
                if event.closed is None:
                    event.closed = current_time
                    event_store.add_event(trace.system_instance_id, event)
                    if verbose:
                        print(f"Closed existing unclosed event: {event.event_type}")

    try:
        # Get traces and convert to dict format
        if len(traces) == 0:
            raise ValueError("No system traces found")
        traces_dict = [trace.to_dict() for trace in traces]
        dataset_dict = dataset.to_dict()

        # Validate upload format
        if verbose:
            print("Validating upload format...")
        validate_upload(traces_dict, dataset_dict)
        if verbose:
            print("Upload format validation successful")

        # Send to server
        response, payload = send_system_traces_s3(
            dataset=dataset,
            traces=traces,
            base_url="https://agent-learning.onrender.com",
            api_key=api_key,
            system_instance_id=system_id,
            verbose=verbose,
        )

        if verbose:
            print("Response status code:", response.status_code)
            if response.status_code == 202:
                print(f"Upload successful - sent {len(traces)} system traces.")
                print(
                    f"Dataset included {len(dataset.questions)} questions and {len(dataset.reward_signals)} reward signals."
                )

        if show_payload:
            print("Payload sent to server: ")
            pprint(payload)

        questions_json, reward_signals_json, traces_json = format_upload_output(
            dataset, traces
        )
        return response, questions_json, reward_signals_json, traces_json

    except ValueError as e:
        if verbose:
            print("Validation error:", str(e))
            print("\nTraces:")
            print(json.dumps(traces_dict, indent=2))
            print("\nDataset:")
            print(json.dumps(dataset_dict, indent=2))
        raise
    except requests.exceptions.HTTPError as e:
        if verbose:
            print("HTTP error occurred:", e)
            print("\nTraces:")
            print(json.dumps(traces_dict, indent=2))
            print("\nDataset:")
            print(json.dumps(dataset_dict, indent=2))
        raise