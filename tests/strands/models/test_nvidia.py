"""Tests for NVIDIA model provider."""

import os
import json
import unittest.mock
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import httpx
import pytest
import pydantic

from strands.models.nvidia import NvidiaChatModel, NvidiaConfig
from strands.types.exceptions import ModelThrottledException
from strands.types.tools import ToolSpec


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.AsyncClient for testing."""
    with patch("strands.models.nvidia.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client_cls.return_value.__aexit__.return_value = None
        yield mock_client


@pytest.fixture
def nvidia_model():
    """Create a test NVIDIA model instance."""
    return NvidiaChatModel(
        api_key="test-api-key",
        model_name="nvidia/chat-v1",
        temperature=0.7,
        max_tokens=1024
    )


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {
            "role": "user",
            "content": [{"text": "Hello, how are you?"}]
        }
    ]


@pytest.fixture
def sample_stream_response():
    """Sample streaming response from NVIDIA API."""
    return [
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"index\":0}]}\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\" there\"},\"index\":0}]}\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"!\"},\"index\":0}]}\n",
        "data: {\"choices\":[{\"finish_reason\":\"stop\",\"index\":0}]}\n",
        "data: [DONE]\n"
    ]


class TestNvidiaChatModel:
    """Test cases for NvidiaChatModel."""

    def test_init_with_api_key(self):
        """Test initialization with direct API key."""
        model = NvidiaChatModel(
            api_key="test-key",
            model_name="nvidia/chat-v1"
        )
        
        assert model.config["api_key"] == "test-key"
        assert model.config["model_name"] == "nvidia/chat-v1"
        assert model.config["base_url"] == "https://api.nvidia.com/v1"
        assert model.config["timeout"] == 30

    def test_init_with_environment_variable(self):
        """Test initialization with environment variable."""
        os.environ["NVIDIA_API_KEY"] = "env-api-key"
        
        try:
            model = NvidiaChatModel(model_name="nvidia/chat-v1")
            assert model.config["api_key"] == "env-api-key"
        finally:
            del os.environ["NVIDIA_API_KEY"]

    def test_init_direct_key_overrides_env(self):
        """Test that direct API key overrides environment variable."""
        os.environ["NVIDIA_API_KEY"] = "env-key"
        
        try:
            model = NvidiaChatModel(
                api_key="direct-key",
                model_name="nvidia/chat-v1"
            )
            assert model.config["api_key"] == "direct-key"
        finally:
            del os.environ["NVIDIA_API_KEY"]

    def test_init_missing_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        # Ensure no environment variable
        if "NVIDIA_API_KEY" in os.environ:
            del os.environ["NVIDIA_API_KEY"]
        
        with pytest.raises(ValueError, match="NVIDIA API key is required"):
            NvidiaChatModel(model_name="nvidia/chat-v1")

    def test_init_missing_model_name_raises_error(self):
        """Test that missing model name raises ValueError."""
        with pytest.raises(ValueError, match="model_name is required"):
            NvidiaChatModel(api_key="test-key")

    def test_update_config(self, nvidia_model):
        """Test configuration updates."""
        nvidia_model.update_config(temperature=0.9, max_tokens=512)
        
        assert nvidia_model.config["temperature"] == 0.9
        assert nvidia_model.config["max_tokens"] == 512
        assert nvidia_model.config["api_key"] == "test-api-key"  # Unchanged

    def test_get_config(self, nvidia_model):
        """Test getting configuration."""
        config = nvidia_model.get_config()
        
        assert isinstance(config, dict)
        assert config["api_key"] == "test-api-key"
        assert config["model_name"] == "nvidia/chat-v1"

    def test_get_headers(self, nvidia_model):
        """Test HTTP headers generation."""
        headers = nvidia_model._get_headers()
        
        assert headers["Content-Type"] == "application/json"
        assert headers["User-Agent"] == "strands-agents"
        assert headers["Authorization"] == "Bearer test-api-key"

    def test_get_base_url(self, nvidia_model):
        """Test base URL generation."""
        base_url = nvidia_model._get_base_url()
        assert base_url == "https://api.nvidia.com/v1"

    def test_format_content_block_text(self, nvidia_model):
        """Test formatting text content blocks."""
        content_block = {"text": "Hello world"}
        formatted = nvidia_model._format_content_block(content_block)
        
        assert formatted == {"type": "text", "text": "Hello world"}

    def test_format_content_block_image(self, nvidia_model):
        """Test formatting image content blocks."""
        content_block = {
            "image": {
                "source": {
                    "bytes": b"fake-image-data",
                    "mediaType": "image/png"
                }
            }
        }
        
        with patch("base64.b64encode") as mock_b64:
            mock_b64.return_value.decode.return_value = "fake-base64-data"
            formatted = nvidia_model._format_content_block(content_block)
            
            assert formatted["type"] == "image_url"
            assert "data:image/png;base64,fake-base64-data" in formatted["image_url"]

    def test_format_messages(self, nvidia_model):
        """Test message formatting."""
        messages = [
            {
                "role": "user",
                "content": [{"text": "Hello"}]
            },
            {
                "role": "assistant", 
                "content": [{"text": "Hi there!"}]
            }
        ]
        
        formatted = nvidia_model._format_messages(messages)
        
        assert len(formatted) == 2
        assert formatted[0]["role"] == "user"
        assert formatted[0]["content"][0]["text"] == "Hello"
        assert formatted[1]["role"] == "assistant"
        assert formatted[1]["content"][0]["text"] == "Hi there!"

    def test_format_messages_system_role(self, nvidia_model):
        """Test that system messages are converted to user messages."""
        messages = [
            {
                "role": "system",
                "content": [{"text": "You are a helpful assistant"}]
            }
        ]
        
        formatted = nvidia_model._format_messages(messages)
        
        assert len(formatted) == 1
        assert formatted[0]["role"] == "user"
        assert formatted[0]["content"][0]["text"] == "You are a helpful assistant"

    def test_prepare_chat_payload(self, nvidia_model, sample_messages):
        """Test chat payload preparation."""
        payload = nvidia_model._prepare_chat_payload(sample_messages)
        
        assert payload["model"] == "nvidia/chat-v1"
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"][0]["text"] == "Hello, how are you?"
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 1024
        assert payload["stream"] is True

    def test_prepare_chat_payload_with_tools(self, nvidia_model, sample_messages):
        """Test chat payload with tool specifications."""
        tool_specs = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {}}
            }
        ]
        
        payload = nvidia_model._prepare_chat_payload(sample_messages, tool_specs)
        
        assert "tools" in payload
        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["type"] == "function"
        assert payload["tools"][0]["function"]["name"] == "test_tool"

    def test_prepare_chat_payload_with_system_prompt(self, nvidia_model, sample_messages):
        """Test chat payload with system prompt."""
        payload = nvidia_model._prepare_chat_payload(
            sample_messages, 
            system_prompt="You are helpful"
        )
        
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"][0]["text"] == "You are helpful"

    def test_process_chunk(self, nvidia_model):
        """Test processing streaming chunks."""
        chunk_data = {
            "choices": [{
                "delta": {"content": "Hello"},
                "finish_reason": None
            }]
        }
        
        event = nvidia_model._process_chunk(chunk_data)
        
        assert event["type"] == "content_block_delta"
        assert event["content_block"]["text"] == "Hello"

    def test_process_chunk_with_tool_calls(self, nvidia_model):
        """Test processing chunks with tool calls."""
        chunk_data = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "id": "call_123",
                        "function": {
                            "name": "test_function",
                            "arguments": '{"arg": "value"}'
                        }
                    }]
                }
            }]
        }
        
        event = nvidia_model._process_chunk(chunk_data)
        
        assert event["type"] == "content_block_delta"
        assert "tool_use" in event
        assert event["tool_use"]["id"] == "call_123"
        assert event["tool_use"]["name"] == "test_function"

    def test_process_chunk_with_finish_reason(self, nvidia_model):
        """Test processing chunks with finish reason."""
        chunk_data = {
            "choices": [{
                "delta": {"content": ""},
                "finish_reason": "stop"
            }]
        }
        
        event = nvidia_model._process_chunk(chunk_data)
        
        assert event["stop_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_stream_async_success(self, nvidia_model, sample_messages, mock_httpx_client, sample_stream_response):
        """Test successful streaming."""
        # Mock the streaming response
        mock_response = AsyncMock()
        mock_response.aiter_lines.return_value = iter(sample_stream_response)
        mock_httpx_client.stream.return_value.__aenter__.return_value = mock_response
        
        events = []
        async for event in nvidia_model._stream_async(sample_messages):
            events.append(event)
        
        assert len(events) == 3  # Three content chunks
        assert events[0]["content_block"]["text"] == "Hello"
        assert events[1]["content_block"]["text"] == " there"
        assert events[2]["content_block"]["text"] == "!"

    @pytest.mark.asyncio
    async def test_stream_async_rate_limit_error(self, nvidia_model, sample_messages, mock_httpx_client):
        """Test streaming with rate limit error."""
        # Mock rate limit error
        mock_response = AsyncMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Rate limit exceeded", request=MagicMock(), response=MagicMock()
        )
        mock_response.status_code = 429
        mock_httpx_client.stream.return_value.__aenter__.return_value = mock_response
        
        with pytest.raises(ModelThrottledException, match="NVIDIA API rate limit exceeded"):
            async for _ in nvidia_model._stream_async(sample_messages):
                pass

    @pytest.mark.asyncio
    async def test_structured_output_async_success(self, nvidia_model, sample_messages, mock_httpx_client):
        """Test successful structured output."""
        # Mock structured output response
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"name": "John", "age": 30}'
                }
            }]
        }
        mock_httpx_client.post.return_value = mock_response
        
        # Define output model
        class Person(pydantic.BaseModel):
            name: str
            age: int
        
        events = []
        async for event in nvidia_model._structured_output_async(Person, sample_messages):
            events.append(event)
        
        assert len(events) == 1
        assert events[0]["type"] == "structured_output"
        assert isinstance(events[0]["data"], Person)
        assert events[0]["data"].name == "John"
        assert events[0]["data"].age == 30

    @pytest.mark.asyncio
    async def test_structured_output_async_invalid_json(self, nvidia_model, sample_messages, mock_httpx_client):
        """Test structured output with invalid JSON."""
        # Mock invalid JSON response
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "invalid json"
                }
            }]
        }
        mock_httpx_client.post.return_value = mock_response
        
        class Person(pydantic.BaseModel):
            name: str
            age: int
        
        with pytest.raises(ValueError, match="Response format does not match Person"):
            async for _ in nvidia_model._structured_output_async(Person, sample_messages):
                pass

    def test_stream_method(self, nvidia_model, sample_messages):
        """Test the public stream method."""
        with patch.object(nvidia_model, "_stream_async") as mock_stream_async:
            mock_stream_async.return_value = iter([])
            
            # Call the public stream method
            result = nvidia_model.stream(sample_messages)
            
            # Should call the async method
            mock_stream_async.assert_called_once_with(
                sample_messages, None, None, None
            )

    def test_structured_output_method(self, nvidia_model, sample_messages):
        """Test the public structured_output method."""
        with patch.object(nvidia_model, "_structured_output_async") as mock_structured_output_async:
            mock_structured_output_async.return_value = iter([])
            
            class Person(pydantic.BaseModel):
                name: str
            
            # Call the public structured_output method
            result = nvidia_model.structured_output(Person, sample_messages)
            
            # Should call the async method
            mock_structured_output_async.assert_called_once_with(
                Person, sample_messages, None
            )


class TestNvidiaConfig:
    """Test cases for NvidiaConfig TypedDict."""

    def test_config_validation(self):
        """Test that NvidiaConfig validates correctly."""
        config = NvidiaConfig(
            api_key="test-key",
            model_name="nvidia/chat-v1",
            temperature=0.7,
            max_tokens=1024
        )
        
        assert config["api_key"] == "test-key"
        assert config["model_name"] == "nvidia/chat-v1"
        assert config["temperature"] == 0.7
        assert config["max_tokens"] == 1024

    def test_config_optional_fields(self):
        """Test that optional fields work correctly."""
        config = NvidiaConfig(
            api_key="test-key",
            model_name="nvidia/chat-v1"
        )
        
        assert config["api_key"] == "test-key"
        assert config["model_name"] == "nvidia/chat-v1"
        assert config.get("temperature") is None
        assert config.get("max_tokens") is None
