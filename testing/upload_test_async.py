from zyk import LM
from synth_sdk.tracing.decorators import trace_system_async, _local
from synth_sdk.tracing.trackers import SynthTrackerAsync
from synth_sdk.tracing.upload import upload
from synth_sdk.tracing.abstractions import TrainingQuestion, RewardSignal, Dataset
from synth_sdk.tracing.events.store import event_store
from typing import Dict
import asyncio
import synth_sdk.config.settings
import time
import json
import logging
from pprint import pprint

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed from CRITICAL to DEBUG
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class TestAgent:
    def __init__(self):
        self.system_id = "test_agent_upload"
        logger.debug("Initializing TestAgent with system_id: %s", self.system_id)
        self.lm = LM(
            model_name="gpt-4o-mini-2024-07-18",
            formatting_model_name="gpt-4o-mini-2024-07-18",
            temperature=1,
        )
        logger.debug("LM initialized")

    @trace_system_async(
        origin="agent",
        event_type="lm_call",
        manage_event="create",
        increment_partition=True,
        verbose=True,
    )
    async def make_lm_call(self, user_message: str) -> str: # Calls an LLM to respond to a user message
        # Only pass the user message, not self
        SynthTrackerAsync.track_input([user_message], variable_name="user_message", origin="agent")

        logger.debug("Starting LM call with message: %s", user_message)
        response = await self.lm.respond_async(
            system_message="You are a helpful assistant.", user_message=user_message
        )

        SynthTrackerAsync.track_output(response, variable_name="response", origin="agent")

        logger.debug("LM response received: %s", response)
        time.sleep(0.1)
        return response

    @trace_system_async(
        origin="environment",
        event_type="environment_processing",
        manage_event="create",
        verbose=True,
    )
    async def process_environment(self, input_data: str) -> dict: # Doesn't really do anything?
        # Only pass the input data, not self
        SynthTrackerAsync.track_input([input_data], variable_name="input_data", origin="environment")

        result = {"processed": input_data, "timestamp": time.time()}

        SynthTrackerAsync.track_output(result, variable_name="result", origin="environment")
        return result


async def run_test():
    logger.info("Starting run_test")
    # Create test agent
    agent = TestAgent()

    try:
        # List of test questions
        questions = [
            "What's the capital of France?",
            "What's 2+2?",
            "Who wrote Romeo and Juliet?",
        ]
        logger.debug("Test questions initialized: %s", questions)

        # Make multiple LM calls with environment processing
        responses = []
        for i, question in enumerate(questions):
            logger.info("Processing question %d: %s", i, question)
            try:
                # First process in environment ==========================================================
                env_result = await agent.process_environment(question)
                logger.debug("Environment processing result: %s", env_result)

                # Then make LM call =====================================================================
                response = await agent.make_lm_call(question)
                responses.append(response)
                logger.debug("Response received and stored: %s", response)
            except Exception as e:
                logger.error("Error during processing: %s", str(e), exc_info=True)
                continue

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
        try:
            logger.info("Attempting to upload traces")
            response, payload = await upload(dataset=dataset, verbose=True)
            logger.info("Upload successful!")
            print("Upload successful!")
            pprint(payload)
        except Exception as e:
            logger.error("Upload failed: %s", str(e), exc_info=True)
            print(f"Upload failed: {str(e)}")

            # Print debug information
            traces = event_store.get_system_traces()
            logger.debug("Retrieved %d system traces", len(traces))
            print("\nTraces:")
            print(json.dumps([trace.to_dict() for trace in traces], indent=2))

            print("\nDataset:")
            print(json.dumps(dataset.to_dict(), indent=2))
    finally:
        logger.info("Starting cleanup")
        # Cleanup
        if hasattr(_local, "active_events"):
            for event_type, event in _local.active_events.items():
                logger.debug("Cleaning up event: %s", event_type)
                if event.closed is None:
                    event.closed = time.time()
                    if hasattr(_local, "system_id"):
                        try:
                            event_store.add_event(_local.system_id, event)
                            logger.debug(
                                "Successfully cleaned up event: %s", event_type
                            )
                        except Exception as e:
                            logger.error(
                                "Error during cleanup of event %s: %s",
                                event_type,
                                str(e),
                                exc_info=True,
                            )
                            print(
                                f"Error during cleanup of event {event_type}: {str(e)}"
                            )
        logger.info("Cleanup completed")

# Run a sample agent using the async decorator and tracker
if __name__ == "__main__":
    logger.info("Starting main execution")
    asyncio.run(run_test())
    logger.info("Main execution completed")