# synth_sdk/tracing/decorators.py
import inspect
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Literal

from synth_sdk.tracing.abstractions import (
    AgentComputeStep,
    ArbitraryInputs,
    ArbitraryOutputs,
    EnvironmentComputeStep,
    Event,
    MessageInputs,
    MessageOutputs,
)
from synth_sdk.tracing.events.manage import set_current_event
from synth_sdk.tracing.events.store import event_store
from synth_sdk.tracing.local import (
    _local,
    active_events_var,
    logger,
    system_id_var,
    system_instance_id_var,
    system_name_var,
)
from synth_sdk.tracing.trackers import (
    synth_tracker_async,
    synth_tracker_sync,
)
from synth_sdk.tracing.utils import get_system_id

logger = logging.getLogger(__name__)


# # This decorator is used to trace synchronous functions
def trace_system_sync(
    origin: Literal["agent", "environment"],
    event_type: str,
    log_result: bool = False,
    manage_event: Literal["create", "end", None] = None,
    increment_partition: bool = False,
    verbose: bool = False,
    finetune_step: bool = True,
) -> Callable:
    """Decorator for tracing synchronous functions.

    Purpose is to keep track of inputs and outputs for compute steps for sync functions.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Determine the instance (self) if it's a method
            if not hasattr(func, "__self__") or not func.__self__:
                if not args:
                    raise ValueError(
                        "Instance method expected, but no arguments were passed."
                    )
                self_instance = args[0]
            else:
                self_instance = func.__self__

            if not hasattr(self_instance, "system_instance_id"):
                raise ValueError(
                    "Instance missing required system_instance_id attribute"
                )
            if not hasattr(self_instance, "system_name"):
                raise ValueError("Instance missing required system_name attribute")

            # Assign system_name to thread-local storage before checking
            _local.system_name = self_instance.system_name

            if not hasattr(_local, "system_name"):
                raise ValueError("System name not set in thread local storage")

            # Derive system_id from system_name
            self_instance.system_id = get_system_id(self_instance.system_name)
            _local.system_id = self_instance.system_id  # Store in thread local

            _local.system_instance_id = self_instance.system_instance_id
            # logger.debug(f"Set system_instance_id in thread local: {_local.system_instance_id}")

            # Initialize Trace
            synth_tracker_sync.initialize()

            # Initialize active_events if not present
            if not hasattr(_local, "active_events"):
                _local.active_events = {}
                # logger.debug("Initialized active_events in thread local storage")

            event = None
            compute_began = time.time()
            try:
                if manage_event == "create":
                    # logger.debug("Creating new event")
                    event = Event(
                        system_instance_id=_local.system_instance_id,
                        event_type=event_type,
                        opened=compute_began,
                        closed=None,
                        partition_index=0,
                        agent_compute_steps=[],
                        environment_compute_steps=[],
                    )
                    if increment_partition:
                        event.partition_index = event_store.increment_partition(
                            _local.system_instance_id
                        )
                        logger.debug(
                            f"Incremented partition to: {event.partition_index}"
                        )
                    set_current_event(event, decorator_type="sync")
                    logger.debug(f"Created and set new event: {event_type}")

                # Automatically trace function inputs
                bound_args = inspect.signature(func).bind(*args, **kwargs)
                bound_args.apply_defaults()
                for param, value in bound_args.arguments.items():
                    if param == "self":
                        continue
                    synth_tracker_sync.track_state(
                        variable_name=param, variable_value=value, origin=origin
                    )

                # Execute the function
                result = func(*args, **kwargs)

                # Automatically trace function output
                track_result(result, synth_tracker_sync, origin)

                # Collect traced inputs and outputs
                traced_inputs, traced_outputs = synth_tracker_sync.get_traced_data()

                compute_steps_by_origin: Dict[
                    Literal["agent", "environment"], Dict[str, List[Any]]
                ] = {
                    "agent": {"inputs": [], "outputs": []},
                    "environment": {"inputs": [], "outputs": []},
                }

                # Organize traced data by origin
                for item in traced_inputs:
                    var_origin = item["origin"]
                    if "variable_value" in item and "variable_name" in item:
                        # Standard variable input
                        compute_steps_by_origin[var_origin]["inputs"].append(
                            ArbitraryInputs(
                                inputs={item["variable_name"]: item["variable_value"]}
                            )
                        )
                    elif "messages" in item:
                        # Message input from track_lm
                        compute_steps_by_origin[var_origin]["inputs"].append(
                            MessageInputs(messages=item["messages"])
                        )
                        compute_steps_by_origin[var_origin]["inputs"].append(
                            ArbitraryInputs(inputs={"model_name": item["model_name"]})
                        )
                        finetune = item["finetune"] or finetune_step
                        compute_steps_by_origin[var_origin]["inputs"].append(
                            ArbitraryInputs(inputs={"finetune": finetune})
                        )
                    else:
                        logger.warning(f"Unhandled traced input item: {item}")

                for item in traced_outputs:
                    var_origin = item["origin"]
                    if "variable_value" in item and "variable_name" in item:
                        # Standard variable output
                        compute_steps_by_origin[var_origin]["outputs"].append(
                            ArbitraryOutputs(
                                outputs={item["variable_name"]: item["variable_value"]}
                            )
                        )
                    elif "messages" in item:
                        # Message output from track_lm
                        compute_steps_by_origin[var_origin]["outputs"].append(
                            MessageOutputs(messages=item["messages"])
                        )
                    else:
                        logger.warning(f"Unhandled traced output item: {item}")

                # Capture compute end time
                compute_ended = time.time()

                # Create compute steps grouped by origin
                for var_origin in ["agent", "environment"]:
                    inputs = compute_steps_by_origin[var_origin]["inputs"]
                    outputs = compute_steps_by_origin[var_origin]["outputs"]
                    if inputs or outputs:
                        event_order = (
                            len(event.agent_compute_steps)
                            + len(event.environment_compute_steps)
                            + 1
                            if event
                            else 1
                        )
                        compute_step = (
                            AgentComputeStep(
                                event_order=event_order,
                                compute_began=compute_began,
                                compute_ended=compute_ended,
                                compute_input=inputs,
                                compute_output=outputs,
                            )
                            if var_origin == "agent"
                            else EnvironmentComputeStep(
                                event_order=event_order,
                                compute_began=compute_began,
                                compute_ended=compute_ended,
                                compute_input=inputs,
                                compute_output=outputs,
                            )
                        )
                        if event:
                            if var_origin == "agent":
                                event.agent_compute_steps.append(compute_step)
                            else:
                                event.environment_compute_steps.append(compute_step)
                        # logger.debug(
                        #     f"Added compute step for {var_origin}: {compute_step.to_dict()}"
                        # )

                # Optionally log the function result
                if log_result:
                    logger.info(f"Function result: {result}")

                # Handle event management after function execution
                if manage_event == "end" and event_type in _local.active_events:
                    current_event = _local.active_events[event_type]
                    current_event.closed = compute_ended
                    # Store the event
                    if hasattr(_local, "system_instance_id"):
                        event_store.add_event(_local.system_instance_id, current_event)
                        # logger.debug(
                        #     f"Stored and closed event {event_type} for system {_local.system_instance_id}"
                        # )
                    del _local.active_events[event_type]

                return result
            except Exception as e:
                logger.error(f"Exception in traced function '{func.__name__}': {e}")
                raise
            finally:
                # synth_tracker_sync.finalize()
                if hasattr(_local, "system_instance_id"):
                    # logger.debug(f"Cleaning up system_instance_id: {_local.system_instance_id}")
                    delattr(_local, "system_instance_id")

        return wrapper

    return decorator


def trace_system_async(
    origin: Literal["agent", "environment"],
    event_type: str,
    log_result: bool = False,
    manage_event: Literal["create", "end", "lazy_end", None] = None,
    increment_partition: bool = False,
    verbose: bool = False,
    finetune_step: bool = True,
) -> Callable:
    """Decorator for tracing asynchronous functions.

    Purpose is to keep track of inputs and outputs for compute steps for async functions.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # logger.debug(f"Starting async_wrapper for {func.__name__}")
            # logger.debug(f"Args: {args}")
            # logger.debug(f"Kwargs: {kwargs}")

            # Automatically trace function inputs
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()
            # logger.debug(f"Bound args: {bound_args.arguments}")

            for param, value in bound_args.arguments.items():
                if param == "self":
                    continue
                # logger.debug(f"Tracking input param: {param} = {value}")
                synth_tracker_async.track_state(
                    variable_name=param,
                    variable_value=value,
                    origin=origin,
                    io_type="input",
                )

            # Determine the instance (self) if it's a method
            if not hasattr(func, "__self__") or not func.__self__:
                if not args:
                    raise ValueError(
                        "Instance method expected, but no arguments were passed."
                    )
                self_instance = args[0]
            else:
                self_instance = func.__self__

            if not hasattr(self_instance, "system_instance_id"):
                raise ValueError(
                    "Instance missing required system_instance_id attribute"
                )
            if not hasattr(self_instance, "system_name"):
                raise ValueError("Instance missing required system_name attribute")

            # Set system name and IDs in context variables
            system_name_var.set(self_instance.system_name)
            system_id = get_system_id(self_instance.system_name)
            system_id_var.set(system_id)
            system_instance_id_var.set(self_instance.system_instance_id)

            # Initialize AsyncTrace
            synth_tracker_async.initialize()

            # Initialize active_events if not present
            current_active_events = active_events_var.get()
            if not current_active_events:
                active_events_var.set({})
                # logger.debug("Initialized active_events in context vars")

            event = None
            compute_began = time.time()
            try:
                if manage_event == "create":
                    # logger.debug("Creating new event")
                    event = Event(
                        system_instance_id=self_instance.system_instance_id,
                        event_type=event_type,
                        opened=compute_began,
                        closed=None,
                        partition_index=0,
                        agent_compute_steps=[],
                        environment_compute_steps=[],
                    )
                    if increment_partition:
                        event.partition_index = event_store.increment_partition(
                            system_instance_id_var.get()
                        )
                        logger.debug(
                            f"Incremented partition to: {event.partition_index}"
                        )

                    set_current_event(event, decorator_type="async")
                    logger.debug(f"Created and set new event: {event_type}")

                # Automatically trace function inputs
                bound_args = inspect.signature(func).bind(*args, **kwargs)
                bound_args.apply_defaults()
                for param, value in bound_args.arguments.items():
                    if param == "self":
                        continue
                    synth_tracker_async.track_state(
                        variable_name=param,
                        variable_value=value,
                        origin=origin,
                        io_type="input",
                    )

                # Execute the coroutine
                result = await func(*args, **kwargs)

                # Automatically trace function output
                track_result(result, synth_tracker_async, origin)

                # Collect traced inputs and outputs
                traced_inputs, traced_outputs = synth_tracker_async.get_traced_data()

                compute_steps_by_origin: Dict[
                    Literal["agent", "environment"], Dict[str, List[Any]]
                ] = {
                    "agent": {"inputs": [], "outputs": []},
                    "environment": {"inputs": [], "outputs": []},
                }

                # Organize traced data by origin
                for item in traced_inputs:
                    var_origin = item["origin"]
                    if "variable_value" in item and "variable_name" in item:
                        # Standard variable input
                        compute_steps_by_origin[var_origin]["inputs"].append(
                            ArbitraryInputs(
                                inputs={item["variable_name"]: item["variable_value"]}
                            )
                        )
                    elif "messages" in item:
                        # Message input from track_lm
                        compute_steps_by_origin[var_origin]["inputs"].append(
                            MessageInputs(messages=item["messages"])
                        )
                        compute_steps_by_origin[var_origin]["inputs"].append(
                            ArbitraryInputs(inputs={"model_name": item["model_name"]})
                        )
                        finetune = finetune_step or item["finetune"]
                        compute_steps_by_origin[var_origin]["inputs"].append(
                            ArbitraryInputs(inputs={"finetune": finetune})
                        )
                    else:
                        logger.warning(f"Unhandled traced input item: {item}")

                for item in traced_outputs:
                    var_origin = item["origin"]
                    if "variable_value" in item and "variable_name" in item:
                        # Standard variable output
                        compute_steps_by_origin[var_origin]["outputs"].append(
                            ArbitraryOutputs(
                                outputs={item["variable_name"]: item["variable_value"]}
                            )
                        )
                    elif "messages" in item:
                        # Message output from track_lm
                        compute_steps_by_origin[var_origin]["outputs"].append(
                            MessageOutputs(messages=item["messages"])
                        )
                    else:
                        logger.warning(f"Unhandled traced output item: {item}")

                compute_ended = time.time()

                # Create compute steps grouped by origin
                for var_origin in ["agent", "environment"]:
                    inputs = compute_steps_by_origin[var_origin]["inputs"]
                    outputs = compute_steps_by_origin[var_origin]["outputs"]
                    if inputs or outputs:
                        event_order = (
                            len(event.agent_compute_steps)
                            + len(event.environment_compute_steps)
                            + 1
                            if event
                            else 1
                        )
                        compute_step = (
                            AgentComputeStep(
                                event_order=event_order,
                                compute_began=compute_began,
                                compute_ended=compute_ended,
                                compute_input=inputs,
                                compute_output=outputs,
                            )
                            if var_origin == "agent"
                            else EnvironmentComputeStep(
                                event_order=event_order,
                                compute_began=compute_began,
                                compute_ended=compute_ended,
                                compute_input=inputs,
                                compute_output=outputs,
                            )
                        )
                        if event:
                            if var_origin == "agent":
                                event.agent_compute_steps.append(compute_step)
                            else:
                                event.environment_compute_steps.append(compute_step)
                        # logger.debug(
                        #     f"Added compute step for {var_origin}: {compute_step.to_dict()}"
                        # )
                # Optionally log the function result
                if log_result:
                    logger.info(f"Function result: {result}")

                # Handle event management after function execution
                if manage_event == "end" and event_type in active_events_var.get():
                    current_event = active_events_var.get()[event_type]
                    current_event.closed = compute_ended
                    # Store the event
                    if system_instance_id_var.get():
                        event_store.add_event(
                            system_instance_id_var.get(), current_event
                        )
                        # logger.debug(
                        #     f"Stored and closed event {event_type} for system {system_instance_id_var.get()}"
                        # )
                    active_events = active_events_var.get()
                    del active_events[event_type]
                    active_events_var.set(active_events)

                return result
            except Exception as e:
                logger.error(f"Exception in traced function '{func.__name__}': {e}")
                raise
            finally:
                # synth_tracker_async.finalize()
                if hasattr(_local, "system_instance_id"):
                    # logger.debug(f"Cleaning up system_instance_id: {_local.system_instance_id}")
                    delattr(_local, "system_instance_id")

        return async_wrapper

    return decorator


def trace_system(
    origin: Literal["agent", "environment"],
    event_type: str,
    log_result: bool = False,
    manage_event: Literal["create", "end", "lazy_end", None] = None,
    increment_partition: bool = False,
    verbose: bool = False,
) -> Callable:
    """
    Decorator that chooses the correct tracing method (sync or async) based on
    whether the wrapped function is synchronous or asynchronous.

    Purpose is to keep track of inputs and outputs for compute steps for both sync and async functions.
    """

    def decorator(func: Callable) -> Callable:
        # Check if the function is async or sync
        if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
            # Use async tracing
            # logger.debug("Using async tracing")
            async_decorator = trace_system_async(
                origin,
                event_type,
                log_result,
                manage_event,
                increment_partition,
                verbose,
            )
            return async_decorator(func)
        else:
            # Use sync tracing
            # logger.debug("Using sync tracing")
            sync_decorator = trace_system_sync(
                origin,
                event_type,
                log_result,
                manage_event,
                increment_partition,
                verbose,
            )
            return sync_decorator(func)

    return decorator


def track_result(result, tracker, origin):
    # Helper function to track results, including tuple unpacking
    if isinstance(result, tuple):
        # Track each element of the tuple that matches valid types
        for i, item in enumerate(result):
            try:
                tracker.track_state(
                    variable_name=f"result_{i}", variable_value=item, origin=origin
                )
            except Exception as e:
                logger.warning(f"Could not track tuple element {i}: {str(e)}")
    else:
        # Track single result as before
        try:
            tracker.track_state(
                variable_name="result", variable_value=result, origin=origin
            )
        except Exception as e:
            logger.warning(f"Could not track result: {str(e)}")
