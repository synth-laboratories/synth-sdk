from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Literal
from pydantic import BaseModel
import logging
from synth_sdk.tracing.config import VALID_TYPES

logger = logging.getLogger(__name__)


@dataclass
class MessageInputs:
    messages: List[Dict[str, str]]  # {"role": "", "content": ""}


@dataclass
class ArbitraryInputs:
    inputs: Dict[str, Any]


@dataclass
class MessageOutputs:
    messages: List[Dict[str, str]]


@dataclass
class ArbitraryOutputs:
    outputs: Dict[str, Any]


@dataclass
class ComputeStep:
    event_order: int
    compute_ended: Any  # timestamp
    compute_began: Any  # timestamp
    compute_input: List[Any]
    compute_output: List[Any]

    def to_dict(self):
        # Serialize compute_input
        serializable_input = [
            input_item.__dict__ for input_item in self.compute_input
            if isinstance(input_item, (MessageInputs, ArbitraryInputs))
        ]

        # Serialize compute_output
        serializable_output = [
            output_item.__dict__ for output_item in self.compute_output
            if isinstance(output_item, (MessageOutputs, ArbitraryOutputs))
        ]

        # Warn about non-serializable inputs/outputs
        for item in self.compute_input:
            if not isinstance(item, (MessageInputs, ArbitraryInputs)):
                logger.warning(f"Skipping non-serializable input: {item}")
        for item in self.compute_output:
            if not isinstance(item, (MessageOutputs, ArbitraryOutputs)):
                logger.warning(f"Skipping non-serializable output: {item}")

        return {
            "event_order": self.event_order,
            "compute_ended": self.compute_ended,
            "compute_began": self.compute_began,
            "compute_input": serializable_input,
            "compute_output": serializable_output,
        }


@dataclass
class AgentComputeStep(ComputeStep):
    model_name: Optional[str] = None
    compute_input: List[Union[MessageInputs, ArbitraryInputs]]
    compute_output: List[Union[MessageOutputs, ArbitraryOutputs]]


@dataclass
class EnvironmentComputeStep(ComputeStep):
    compute_input: List[ArbitraryInputs]
    compute_output: List[ArbitraryOutputs]


@dataclass
class Event:
    system_id: str
    event_type: str
    opened: Any  # timestamp
    closed: Any  # timestamp
    partition_index: int
    agent_compute_steps: List[AgentComputeStep]
    environment_compute_steps: List[EnvironmentComputeStep]

    def to_dict(self):
        return {
            "event_type": self.event_type,
            "opened": self.opened,
            "closed": self.closed,
            "partition_index": self.partition_index,
            "agent_compute_steps": [
                step.to_dict() for step in self.agent_compute_steps
            ],
            "environment_compute_steps": [
                step.to_dict() for step in self.environment_compute_steps
            ],
        }


@dataclass
class EventPartitionElement:
    partition_index: int
    events: List[Event]

    def to_dict(self):
        return {
            "partition_index": self.partition_index,
            "events": [event.to_dict() for event in self.events],
        }


@dataclass
class SystemTrace:
    system_id: str
    partition: List[EventPartitionElement]
    current_partition_index: int = 0  # Track current partition

    def to_dict(self):
        return {
            "system_id": self.system_id,
            "partition": [element.to_dict() for element in self.partition],
            "current_partition_index": self.current_partition_index,
        }


class TrainingQuestion(BaseModel):
    intent: str
    criteria: str
    question_id: Optional[str] = None

    def to_dict(self):
        return {
            "intent": self.intent,
            "criteria": self.criteria,
            "question_id": self.question_id,
        }


class RewardSignal(BaseModel):
    question_id: Optional[str] = None
    system_id: str
    reward: Union[float, int, bool]
    annotation: Optional[str] = None

    def to_dict(self):
        return {
            "question_id": self.question_id,
            "system_id": self.system_id,
            "reward": self.reward,
            "annotation": self.annotation,
        }


class Dataset(BaseModel):
    questions: List[TrainingQuestion]
    reward_signals: List[RewardSignal]

    def to_dict(self):
        return {
            "questions": [question.to_dict() for question in self.questions],
            "reward_signals": [signal.to_dict() for signal in self.reward_signals],
        }
