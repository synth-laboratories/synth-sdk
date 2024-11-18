from zyk import LM
from synth_sdk.tracing.decorators import trace_system, _local
from synth_sdk.tracing.trackers import SynthTracker
from synth_sdk.tracing.upload import upload, validate_json, createPayload
from synth_sdk.tracing.abstractions import (
    TrainingQuestion, RewardSignal, Dataset, SystemTrace, EventPartitionElement,
    MessageInputs, MessageOutputs, ArbitraryInputs, ArbitraryOutputs
)
from synth_sdk.tracing.events.store import event_store
from typing import Dict, List
import asyncio
import time
import logging
import pytest
from unittest.mock import MagicMock, patch
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

eventPartition_test = [
    EventPartitionElement(0, []),
    EventPartitionElement(1, [])
]

trace_test = [
    SystemTrace(
        system_id="test_agent_upload",
        partition=eventPartition_test,
        current_partition_index=1
    )
]

# This function generates a payload from the data in the dataset to compare the sent payload against
def generate_payload_from_data(dataset: Dataset, traces: List[SystemTrace]) -> Dict:
    payload = {
        "traces": [trace.to_dict() for trace in traces],
        "dataset": dataset.to_dict(),
    }
    return payload

def createPayload_wrapper(dataset: Dataset, traces: str, base_url: str, api_key: str) -> Dict:
    payload = createPayload(dataset, traces)
    response = requests.Response()
    response.status_code = 200
    return response, payload

# ===============================
# Utility Functions
def createUploadDataset(agent):
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
    return dataset

def ask_questions(agent):
    # Make multiple LM calls with environment processing
    responses = []
    for i, question in enumerate(questions):
        logger.info("Processing question %d: %s", i, question)
        env_result = agent.process_environment(question)
        logger.debug("Environment processing result: %s", env_result)
        response = agent.make_lm_call(question)
        responses.append(response)
        logger.debug("Response received and stored: %s", response)
    return responses

# ===============================

class TestAgent:
    def __init__(self):
        self.system_id = "test_agent_upload"
        logger.debug("Initializing TestAgent with system_id: %s", self.system_id)
        self.lm = MagicMock()
        self.lm.model_name = "gpt-4o-mini-2024-07-18"
        self.lm.respond_sync.return_value = mock_llm_response
        logger.debug("LM initialized")

    @trace_system(
        origin="agent",
        event_type="lm_call",
        manage_event="create",
        increment_partition=True,
        verbose=False,
    )
    def make_lm_call(self, user_message: str) -> str:
        # Create MessageInputs
        message_input = MessageInputs(messages=[{"role": "user", "content": user_message}])
        # Track LM interaction using the new SynthTracker form
        SynthTracker.track_lm(
            messages=message_input.messages,
            model_name=self.lm.model_name,
            finetune=False
        )

        logger.debug("Starting LM call with message: %s", user_message)
        response = self.lm.respond_sync(
            system_message="You are a helpful assistant.", user_message=user_message
        )

        # Create MessageOutputs
        message_output = MessageOutputs(messages=[{"role": "assistant", "content": response}])
        # Track state using the new SynthTracker form
        SynthTracker.track_state(
            variable_name="response",
            variable_value=message_output.messages,
            origin="agent",
            annotation="LLM response"
        )

        logger.debug("LM response received: %s", response)
        return response

    @trace_system(
        origin="environment",
        event_type="environment_processing",
        manage_event="create",
        verbose=False,
    )
    def process_environment(self, input_data: str) -> dict:
        # Create ArbitraryInputs
        arbitrary_input = ArbitraryInputs(inputs={"input_data": input_data})
        # Track state using the new SynthTracker form
        SynthTracker.track_state(
            variable_name="input_data",
            variable_value=arbitrary_input.inputs,
            origin="environment",
            annotation="Environment input data"
        )

        result = {"processed": input_data, "timestamp": time.time()}

        # Create ArbitraryOutputs
        arbitrary_output = ArbitraryOutputs(outputs=result)
        # Track state using the new SynthTracker form
        SynthTracker.track_state(
            variable_name="result",
            variable_value=arbitrary_output.outputs,
            origin="environment",
            annotation="Environment processing result"
        )
        return result

# Use the new SynthTracker finalize method appropriately
@patch("synth_sdk.tracing.upload.send_system_traces", side_effect=createPayload_wrapper)
def test_generate_traces_sync(mock_send_system_traces):
    logger.info("Starting test_generate_traces_sync")
    agent = TestAgent()  # Create test agent
    logger.debug("Test questions initialized: %s", questions)  # List of test questions

    # Ask questions
    responses = ask_questions(agent)

    logger.info("Creating dataset for upload")
    # Create dataset for upload
    dataset = createUploadDataset(agent)

    # Upload traces
    logger.info("Attempting to upload traces (sync version)")
    # Pytest assertion
    payload_ground_truth = generate_payload_from_data(dataset, trace_test)

    _, payload_default_trace = upload(dataset=dataset, verbose=True, show_payload=True)
    assert payload_ground_truth == payload_default_trace

    _, payload_trace_passed = upload(
        dataset=dataset,
        traces=trace_test,
        verbose=True,
        show_payload=True
    )
    assert payload_ground_truth == payload_trace_passed

    # Finalize the tracker
    SynthTracker.finalize()
    logger.info("Resetting event store 0")
    event_store.__init__()

@pytest.mark.asyncio
@patch("synth_sdk.tracing.upload.send_system_traces", side_effect=createPayload_wrapper)
async def test_generate_traces_async(mock_send_system_traces):
    logger.info("Starting test_generate_traces_async")
    agent = TestAgent()

    # Ask questions
    responses = ask_questions(agent)

    logger.info("Creating dataset for upload")
    # Create dataset for upload
    dataset = createUploadDataset(agent)

    # Upload traces
    logger.info("Attempting to upload traces (async version)")
    # Pytest assertion
    payload_ground_truth = generate_payload_from_data(dataset, trace_test)

    _, payload_default_trace = await upload(dataset=dataset, verbose=True, show_payload=True)
    assert payload_ground_truth == payload_default_trace

    _, payload_trace_passed = await upload(
        dataset=dataset,
        traces=trace_test,
        verbose=True,
        show_payload=True
    )
    assert payload_ground_truth == payload_trace_passed

    # Finalize the tracker
    SynthTracker.finalize()
    logger.info("Resetting event store 1")
    event_store.__init__()

# Run the tests
if __name__ == "__main__":
    logger.info("Starting main execution")
    asyncio.run(test_generate_traces_async())
    logger.info("Async test completed")
    print("=============================================")
    print("=============================================")
    test_generate_traces_sync()
    logger.info("Sync test completed")
    logger.info("Main execution completed")