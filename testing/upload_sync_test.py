from zyk import LM
from synth_sdk.tracing.decorators import trace_system_sync, _local
from synth_sdk.tracing.trackers import SynthTrackerSync
from synth_sdk.tracing.upload import upload
from synth_sdk.tracing.upload import validate_json
from synth_sdk.tracing.upload import createPayload
from synth_sdk.tracing.abstractions import TrainingQuestion, RewardSignal, Dataset
from synth_sdk.tracing.events.store import event_store
from typing import Dict
#import asyncio
#import synth_sdk.config.settings
import time
#import json
import logging
import pytest
from unittest.mock import MagicMock, Mock, patch
import requests

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed from CRITICAL to DEBUG
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Unit Test Configuration: 
# ===============================  
questions = ["What's the capital of France?"]
mock_llm_response = "The capital of France is Paris."

# This function generates a payload from the data in the dataset to compare the sent payload against
def generate_payload_from_data(dataset: Dataset) -> Dict:
    traces = event_store.get_system_traces()

    payload = {
        "traces": [
            trace.to_dict() for trace in traces
        ],  # Convert SystemTrace objects to dicts
        "dataset": dataset.to_dict(),
    }
    return payload

def createPayload_wrapper(dataset: Dataset, base_url: str, api_key: str) -> Dict:
    payload = createPayload(dataset, event_store.get_system_traces())

    response = requests.Response()
    response.status_code = 200

    return response, payload

# ===============================

class TestAgent:
    def __init__(self):
        self.system_id = "test_agent_upload"
        logger.debug("Initializing TestAgent with system_id: %s", self.system_id)
        #self.lm = LM(model_name="gpt-4o-mini-2024-07-18", formatting_model_name="gpt-4o-mini-2024-07-18", temperature=1,)
        self.lm = MagicMock()
        self.lm.respond_sync.return_value = mock_llm_response
        logger.debug("LM initialized")

    @trace_system_sync(
        origin="agent",
        event_type="lm_call",
        manage_event="create",
        increment_partition=True,
        verbose=False,
    )
    def make_lm_call(self, user_message: str) -> str: # Calls an LLM to respond to a user message
        # Only pass the user message, not self
        SynthTrackerSync.track_input([user_message], variable_name="user_message", origin="agent")

        logger.debug("Starting LM call with message: %s", user_message)
        response = self.lm.respond_sync(
            system_message="You are a helpful assistant.", user_message=user_message
        )
        SynthTrackerSync.track_output(response, variable_name="response", origin="agent")

        logger.debug("LM response received: %s", response)
        #time.sleep(0.1)
        return response

    @trace_system_sync(
        origin="environment",
        event_type="environment_processing",
        manage_event="create",
        verbose=False,
    )
    def process_environment(self, input_data: str) -> dict:
        # Only pass the input data, not self
        SynthTrackerSync.track_input([input_data], variable_name="input_data", origin="environment")
        result = {"processed": input_data, "timestamp": time.time()}
        SynthTrackerSync.track_output(result, variable_name="result", origin="environment")
        return result

@pytest.mark.asyncio(loop_scope="session")
@patch("synth_sdk.tracing.upload.send_system_traces", side_effect=createPayload_wrapper)
async def test_upload(mock_send_system_traces):
    logger.info("Starting run_test")
    agent = TestAgent() # Create test agent

    logger.debug("Test questions initialized: %s", questions) # List of test questions

    # Make multiple LM calls with environment processing
    responses = []
    for i, question in enumerate(questions):
        logger.info("Processing question %d: %s", i, question)
        env_result = agent.process_environment(question)
        logger.debug("Environment processing result: %s", env_result)

        response = agent.make_lm_call(question)
        responses.append(response)
        logger.debug("Response received and stored: %s", response)

    logger.info("Creating dataset for upload")
    # Create dataset for upload
    dataset = Dataset(
        questions=[
            TrainingQuestion(
                intent="Test question",
                criteria="Testing tracing functionality",
                question_id=f"q{i}",
            )
            for i in range(len(questions))
        ],
        reward_signals=[
            RewardSignal(
                question_id=f"q{i}",
                system_id=agent.system_id,
                reward=1.0,
                annotation="Test reward",
            )
            for i in range(len(questions))
        ],
    )
    logger.debug(
        "Dataset created with %d questions and %d reward signals",
        len(dataset.questions),
        len(dataset.reward_signals),
    )

    # Upload traces
    logger.info("Attempting to upload traces")
    response, payload = await upload(dataset=dataset, verbose=True)
    logger.info("Upload successful!")

    # Pytest assertion
    assert payload == generate_payload_from_data(dataset)
