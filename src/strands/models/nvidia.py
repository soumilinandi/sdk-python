"""NVIDIA model provider.

- Docs: https://docs.nvidia.com/ai-foundation/
"""

import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator, AsyncIterable, Optional, Type, TypeVar, Union, cast

import httpx
from pydantic import BaseModel
from typing_extensions import TypedDict, Unpack, override

from ..types.content import ContentBlock, Messages
from ..types.exceptions import ModelThrottledException
from ..types.streaming import StreamEvent
from ..types.tools import ToolChoice, ToolSpec
from ._validation import validate_config_keys
from .model import Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class NvidiaConfig(TypedDict, total=False):
    """Configuration options for NVIDIA models.

    Attributes:
        api_key: NVIDIA API key for authentication.
            Can also be set via NVIDIA_API_KEY environment variable.
        base_url: Base URL for NVIDIA API endpoints.
            Defaults to "https://api.nvidia.com/v1".
        model_name: NVIDIA model name (e.g., "nvidia/chat-v1").
        temperature: Sampling temperature in [0, 1].
        max_tokens: Maximum number of tokens to generate.
        top_p: Top-p for distribution sampling.
        seed: Seed for deterministic results.
        stop: List of stop sequences.
        timeout: Request timeout in seconds.
    """

    api_key: str
    base_url: str
    model_name: str
    temperature: Optional[float]
    max_tokens: Optional[int]
    top_p: Optional[float]
    seed: Optional[int]
    stop: Optional[list[str]]
    timeout: Optional[int]


class NvidiaChatModel(Model):
    """NVIDIA Chat model provider implementation.

    Provides chat completion capabilities using NVIDIA's chat API endpoint.
    Supports streaming, tool calling, and structured output.

    Example:
        .. code-block:: python

            from strands.models.nvidia import NvidiaChatModel

            model = NvidiaChatModel(
                api_key="your-api-key",
                model_name="nvidia/chat-v1"
            )
            agent = Agent(model=model)
            response = agent("Hello!")
    """

    def __init__(self, **model_config: Unpack[NvidiaConfig]) -> None:
        """Initialize NVIDIA Chat model provider.

        Args:
            **model_config: Configuration options for the NVIDIA chat model.
                api_key: NVIDIA API key. Can also be set via NVIDIA_API_KEY environment variable.
                base_url: Base URL for NVIDIA API endpoints. Defaults to "https://api.nvidia.com/v1".
                model_name: NVIDIA model name (e.g., "nvidia/chat-v1").
                temperature: Sampling temperature in [0, 1].
                max_tokens: Maximum number of tokens to generate.
                top_p: Top-p for distribution sampling.
                seed: Seed for deterministic results.
                stop: List of stop sequences.
                timeout: Request timeout in seconds.

        API Key:
            The recommended way to provide the API key is through the `NVIDIA_API_KEY`
            environment variable. Alternatively, you can pass it directly as a parameter.
        """
        validate_config_keys(model_config, NvidiaConfig)
        self.config = dict(model_config)
        
        # Set defaults
        self.config.setdefault("base_url", "https://integrate.api.nvidia.com/v1")
        self.config.setdefault("timeout", 30)
        
        # Handle API key from environment variable or config
        api_key = self.config.get("api_key") or os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError(
                "NVIDIA API key is required. Set NVIDIA_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.config["api_key"] = api_key
        
        # Validate required fields
        if not self.config.get("model_name"):
            raise ValueError("model_name is required for NvidiaChatModel")
        
        logger.debug("config=<%s> | initializing", self.config)

    @override
    def update_config(self, **model_config: Unpack[NvidiaConfig]) -> None:
        """Update the NVIDIA model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        validate_config_keys(model_config, NvidiaConfig)
        self.config.update(model_config)

    @override
    def get_config(self) -> NvidiaConfig:
        """Get the NVIDIA model configuration.

        Returns:
            The NVIDIA model configuration.
        """
        return cast(NvidiaConfig, self.config)

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Stream chat completion from NVIDIA model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Formatted message chunks from the model.

        Raises:
            ModelThrottledException: When the model service is throttling requests.
        """
        async for event in self._stream_async(
            messages, tool_specs, system_prompt, tool_choice, **kwargs
        ):
            yield event

    @override
    async def structured_output(
        self, output_model: Type[T], prompt: Messages, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        """Get structured output from NVIDIA chat model.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Model events with the last being the structured output.

        Raises:
            ValidationException: The response format from the model does not match the output_model
        """
        async for event in self._structured_output_async(output_model, prompt, system_prompt, **kwargs):
            yield event

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for NVIDIA API requests.

        Returns:
            Dictionary of HTTP headers.
        """
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "strands-agents",
        }
        
        if api_key := self.config.get("api_key"):
            headers["Authorization"] = f"Bearer {api_key}"
            
        return headers

    def _get_base_url(self) -> str:
        """Get the base URL for NVIDIA API requests.

        Returns:
            The base URL for API requests.
        """
        return self.config["base_url"].rstrip("/")
    
    def _get_infer_url(self) -> str:
        """Get the inference URL for NVIDIA API requests.

        Returns:
            The inference URL for API requests.
        """
        return f"{self.config['base_url']}/chat/completions"

    def format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        """Format NVIDIA response events into standardized message chunks.

        Args:
            event: A response event from the NVIDIA model.

        Returns:
            The formatted chunk in Strands standard format.
        """
        match event.get("chunk_type"):
            case "message_start":
                return {"messageStart": {"role": "assistant"}}
            
            case "content_block_start":
                return {"contentBlockStart": {"start": {}}}
            
            case "content_block_delta":
                if event.get("data_type") == "text":
                    return {"contentBlockDelta": {"delta": {"text": event.get("data", "")}}}
                elif event.get("data_type") == "tool":
                    return {
                        "contentBlockDelta": {
                            "delta": {
                                "toolUse": {
                                    "input": event.get("data", {}).get("input", "")
                                }
                            }
                        }
                    }
            
            case "content_block_stop":
                return {"contentBlockStop": {}}
            
            case "message_stop":
                stop_reason_map = {
                    "stop": "end_turn",
                    "length": "max_tokens",
                    "tool_calls": "tool_use"
                }
                return {"messageStop": {"stopReason": stop_reason_map.get(event.get("data"), "end_turn")}}
            
            case "metadata":
                if event.get("data"):
                    return {
                        "metadata": {
                            "usage": {
                                "inputTokens": event["data"].get("prompt_tokens", 0),
                                "outputTokens": event["data"].get("completion_tokens", 0),
                                "totalTokens": event["data"].get("total_tokens", 0),
                            }
                        }
                    }
                return {"metadata": {"usage": {}}}
        
        # Default fallback
        return {}

    def _format_content_block(self, content: ContentBlock) -> dict[str, Any]:
        """Format a content block for NVIDIA API.

        Args:
            content: Content block to format.

        Returns:
            Formatted content block for NVIDIA API.
        """
        if "text" in content:
            return {"type": "text", "text": content["text"]}
        elif "image" in content:
            # Handle image content - convert to base64 data URL
            image_data = content["image"]["source"]["bytes"]
            mime_type = content["image"]["source"]["mediaType"]
            import base64
            encoded = base64.b64encode(image_data).decode("utf-8")
            return {
                "type": "image_url",
                "image_url": f"data:{mime_type};base64,{encoded}"
            }
        else:
            raise ValueError(f"Unsupported content block type: {content}")

    def _format_messages(self, messages: Messages) -> list[dict[str, Any]]:
        """Format messages for NVIDIA API.

        Args:
            messages: Messages to format.

        Returns:
            Formatted messages for NVIDIA API.
        """
        formatted_messages = []
        
        for message in messages:
            if message["role"] == "system":
                # NVIDIA doesn't have system messages, convert to user message
                # Extract text content from content blocks
                text_content = ""
                for content_block in message["content"]:
                    if content_block.get("text"):
                        text_content += content_block["text"]
                
                formatted_messages.append({
                    "role": "user",
                    "content": text_content
                })
            elif message["role"] in ["user", "assistant"]:
                # Extract text content from content blocks
                text_content = ""
                for content_block in message["content"]:
                    if content_block.get("text"):
                        text_content += content_block["text"]
                
                formatted_messages.append({
                    "role": message["role"],
                    "content": text_content
                })
            elif message["role"] == "tool":
                # Convert tool result to assistant message
                tool_result = message["content"][0]["toolResult"]
                formatted_messages.append({
                    "role": "assistant",
                    "content": f"Tool result: {tool_result}"
                })
        
        return formatted_messages

    def _prepare_chat_payload(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare payload for NVIDIA chat API.

        Args:
            messages: Messages to send to the model.
            tool_specs: Tool specifications for function calling.
            system_prompt: System prompt for the model.
            tool_choice: Tool choice strategy.
            **kwargs: Additional parameters.

        Returns:
            Payload dictionary for NVIDIA API.
        """
        # Format messages
        formatted_messages = self._format_messages(messages)
        
        # Add system prompt if provided
        if system_prompt:
            formatted_messages.insert(0, {
                "role": "system",
                "content": system_prompt  # Plain string for NVIDIA API
            })
        
        # Prepare base payload
        payload = {
            "model": self.config["model_name"],
            "messages": formatted_messages,
            "stream": kwargs.get("stream", True),
        }
        
        # Add optional parameters
        if temperature := self.config.get("temperature"):
            payload["temperature"] = temperature
        if max_tokens := self.config.get("max_tokens"):
            payload["max_tokens"] = max_tokens
        if top_p := self.config.get("top_p"):
            payload["top_p"] = top_p
        if seed := self.config.get("seed"):
            payload["seed"] = seed
        if stop := self.config.get("stop"):
            payload["stop"] = stop
            
        # Add tool specifications if provided
        if tool_specs:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool_spec["name"],
                        "description": tool_spec.get("description", ""),
                        "parameters": tool_spec.get("parameters", {})
                    }
                }
                for tool_spec in tool_specs
            ]
            
            # Add tool choice if specified
            if tool_choice:
                if isinstance(tool_choice, str):
                    if tool_choice == "auto":
                        payload["tool_choice"] = "auto"
                    elif tool_choice == "none":
                        payload["tool_choice"] = "none"
                    else:
                        payload["tool_choice"] = {
                            "type": "function",
                            "function": {"name": tool_choice}
                        }
                elif isinstance(tool_choice, dict):
                    payload["tool_choice"] = tool_choice
        
        return payload

    async def _stream_async(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Async implementation of stream method."""
        try:
            # Prepare request payload
            payload = self._prepare_chat_payload(
                messages, tool_specs, system_prompt, tool_choice, **kwargs
            )
            
            # Make streaming request to NVIDIA API
            async with httpx.AsyncClient(timeout=self.config["timeout"]) as client:
                async with client.stream(
                    "POST",
                    self._get_infer_url(),
                    headers=self._get_headers(),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    
                    # Emit message start
                    yield self.format_chunk({"chunk_type": "message_start"})
                    
                    # Track state
                    has_content = False
                    content_started = False
                    finish_reason = None
                    usage_data = None
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix
                            if data.strip() == "[DONE]":
                                break
                            
                            try:
                                chunk_data = json.loads(data)
                                
                                # Extract choice data
                                if "choices" in chunk_data and chunk_data["choices"]:
                                    choice = chunk_data["choices"][0]
                                    delta = choice.get("delta", {})
                                    finish_reason = choice.get("finish_reason") or finish_reason
                                    
                                    # Handle text content
                                    if "content" in delta and delta["content"]:
                                        if not content_started:
                                            yield self.format_chunk({"chunk_type": "content_block_start"})
                                            content_started = True
                                        
                                        yield self.format_chunk({
                                            "chunk_type": "content_block_delta",
                                            "data_type": "text",
                                            "data": delta["content"]
                                        })
                                        has_content = True
                                    
                                    # Handle tool calls
                                    if "tool_calls" in delta and delta["tool_calls"]:
                                        for tool_call in delta["tool_calls"]:
                                            if not content_started:
                                                yield self.format_chunk({"chunk_type": "content_block_start"})
                                                content_started = True
                                            
                                            tool_input = tool_call.get("function", {}).get("arguments", "")
                                            yield self.format_chunk({
                                                "chunk_type": "content_block_delta",
                                                "data_type": "tool",
                                                "data": {"input": tool_input}
                                            })
                                            has_content = True
                                
                                # Extract usage data if available
                                if "usage" in chunk_data:
                                    usage_data = chunk_data["usage"]
                                    
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse chunk: {data}")
                                continue
                    
                    # Emit content block stop if we had content
                    if content_started:
                        yield self.format_chunk({"chunk_type": "content_block_stop"})
                    
                    # Emit message stop
                    yield self.format_chunk({
                        "chunk_type": "message_stop",
                        "data": finish_reason or "stop"
                    })
                    
                    # Emit metadata if available
                    if usage_data:
                        yield self.format_chunk({
                            "chunk_type": "metadata",
                            "data": usage_data
                        })
                                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise ModelThrottledException("NVIDIA API rate limit exceeded")
            else:
                logger.error(f"NVIDIA API error: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error in NVIDIA chat stream: {e}")
            raise

    async def _structured_output_async(
        self, output_model: Type[T], prompt: Messages, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        """Async implementation of structured_output method."""
        try:
            # Prepare request payload with structured output schema
            payload = self._prepare_chat_payload(prompt, None, system_prompt, None, **kwargs)
            
            # Add structured output schema using NVIDIA's nvext format
            if hasattr(output_model, "model_json_schema"):
                payload["nvext"] = {
                    "guided_json": output_model.model_json_schema()
                }
            
            # Debug: Log the payload
            logger.debug(f"NVIDIA structured output payload: {json.dumps(payload, indent=2)}")
            
            # Make request to NVIDIA API
            async with httpx.AsyncClient(timeout=self.config["timeout"]) as client:
                async with client.stream(
                    "POST",
                    self._get_infer_url(),
                    headers=self._get_headers(),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                
                    # Collect streaming content for structured output
                    content_buffer = ""
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix
                            if data.strip() == "[DONE]":
                                break
                            
                            try:
                                chunk_data = json.loads(data)
                                if "choices" in chunk_data and chunk_data["choices"]:
                                    delta = chunk_data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        content_buffer += delta["content"]
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse chunk: {data}")
                                continue
                    
                    # Parse the complete JSON content
                    if not content_buffer.strip():
                        raise ValueError("Empty structured output from NVIDIA API")
                    
                    try:
                        parsed_data = json.loads(content_buffer)
                        structured_output = output_model(**parsed_data)
                        
                        yield {
                            "type": "structured_output",
                            "data": structured_output,
                            "raw_content": content_buffer
                        }
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in structured output: {content_buffer}") from e
                    except Exception as e:
                        raise ValueError(f"Response format does not match {output_model.__name__}: {content_buffer}") from e
                    
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise ModelThrottledException("NVIDIA API rate limit exceeded")
            else:
                logger.error(f"NVIDIA API error: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error in NVIDIA structured output: {e}")
            raise
