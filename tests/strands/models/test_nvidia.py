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
        assert model.config["base_url"] == "https://integrate.api.nvidia.com/v1"
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

    def test_init_missing_api_key_raises_error_for_hosted(self):
        """Test that missing API key raises ValueError for hosted endpoints."""
        # Ensure no environment variable
        if "NVIDIA_API_KEY" in os.environ:
            del os.environ["NVIDIA_API_KEY"]
        
        # Should fail for hosted endpoints (nvidia.com in base_url)
        with pytest.raises(ValueError, match="NVIDIA API key is required for hosted endpoints"):
            NvidiaChatModel(
                model_name="nvidia/chat-v1",
                base_url="https://integrate.api.nvidia.com/v1"
            )
    
    def test_init_without_api_key_for_local_nim(self):
        """Test that API key is not required for local NIM deployments."""
        # Ensure no environment variable
        if "NVIDIA_API_KEY" in os.environ:
            del os.environ["NVIDIA_API_KEY"]
        
        # Should succeed for local endpoints (no nvidia.com in base_url)
        model = NvidiaChatModel(
            model_name="meta/llama-3.1-8b-instruct",
            base_url="http://localhost:8000/v1"
        )
        
        assert model.config["model_name"] == "meta/llama-3.1-8b-instruct"
        assert model.config["base_url"] == "http://localhost:8000/v1"
        assert "api_key" not in model.config

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

    def test_get_infer_url(self, nvidia_model):
        """Test inference URL generation."""
        infer_url = nvidia_model._get_infer_url()
        assert infer_url == "https://integrate.api.nvidia.com/v1/chat/completions"

    def test_format_content_block_text(self, nvidia_model):
        """Test formatting text content blocks."""
        content_block = {"text": "Hello world"}
        formatted = nvidia_model._format_content_block(content_block)
        
        assert formatted == {"type": "text", "text": "Hello world"}

    def test_format_content_block_image(self, nvidia_model):
        """Test formatting image content blocks."""
        content_block = {
            "image": {
                "format": "png",
                "source": {
                    "bytes": b"fake-image-data"
                }
            }
        }
        
        with patch("base64.b64encode") as mock_b64:
            mock_b64.return_value.decode.return_value = "fake-base64-data"
            formatted = nvidia_model._format_content_block(content_block)
            
            assert formatted["type"] == "image_url"
            assert formatted["image_url"]["url"] == "data:image/png;base64,fake-base64-data"

    def test_format_messages(self, nvidia_model):
        """Test message formatting - converts content blocks to plain strings for NVIDIA API."""
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
        assert formatted[0]["content"] == "Hello"  # Plain string for NVIDIA API
        assert formatted[1]["role"] == "assistant"
        assert formatted[1]["content"] == "Hi there!"  # Plain string for NVIDIA API

    def test_format_messages_system_role(self, nvidia_model):
        """Test that system messages are kept as system messages."""
        messages = [
            {
                "role": "system",
                "content": [{"text": "You are a helpful assistant"}]
            }
        ]
        
        formatted = nvidia_model._format_messages(messages)
        
        assert len(formatted) == 1
        assert formatted[0]["role"] == "system"
        assert formatted[0]["content"] == "You are a helpful assistant"  # Plain string for NVIDIA API

    def test_prepare_chat_payload(self, nvidia_model, sample_messages):
        """Test chat payload preparation."""
        payload = nvidia_model._prepare_chat_payload(sample_messages)
        
        assert payload["model"] == "nvidia/chat-v1"
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "Hello, how are you?"  # Plain string for NVIDIA API
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
        assert payload["messages"][0]["content"] == "You are helpful"  # Plain string for NVIDIA API

    def test_format_chunk_content_delta(self, nvidia_model):
        """Test formatting content delta chunks."""
        event = {
            "chunk_type": "content_block_delta",
            "data_type": "text",
            "data": "Hello"
        }
        
        formatted = nvidia_model.format_chunk(event)
        
        assert "contentBlockDelta" in formatted
        assert formatted["contentBlockDelta"]["delta"]["text"] == "Hello"

    def test_format_chunk_message_start(self, nvidia_model):
        """Test formatting message start chunks."""
        event = {"chunk_type": "message_start"}
        
        formatted = nvidia_model.format_chunk(event)
        
        assert "messageStart" in formatted
        assert formatted["messageStart"]["role"] == "assistant"

    def test_format_chunk_message_stop(self, nvidia_model):
        """Test formatting message stop chunks."""
        event = {"chunk_type": "message_stop", "data": "stop"}
        
        formatted = nvidia_model.format_chunk(event)
        
        assert "messageStop" in formatted
        assert formatted["messageStop"]["stopReason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_stream_async_success(self, nvidia_model, sample_messages):
        """Test successful streaming."""
        sample_stream_response = [
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"index\":0}]}\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" there\"},\"index\":0}]}\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"!\"},\"index\":0}]}\n",
            "data: {\"choices\":[{\"finish_reason\":\"stop\",\"index\":0}]}\n",
            "data: [DONE]\n"
        ]
        
        # Mock the streaming response
        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        
        async def mock_aiter_lines():
            for line in sample_stream_response:
                yield line
        
        mock_response.aiter_lines = mock_aiter_lines
        
        with patch("strands.models.nvidia.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_stream_ctx = AsyncMock()
            mock_stream_ctx.__aenter__.return_value = mock_response
            mock_stream_ctx.__aexit__.return_value = None
            mock_client.stream = MagicMock(return_value=mock_stream_ctx)
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_cls.return_value = mock_client
            
            events = []
            async for event in nvidia_model._stream_async(sample_messages):
                events.append(event)
            
            # Should have: message_start, content_start, 3x content_delta, content_stop, message_stop
            # At minimum: message_start, content_start, content_deltas, content_stop, message_stop
            assert len(events) >= 4
            # First event should be message start
            assert "messageStart" in events[0]
            # Should have content deltas
            content_events = [e for e in events if "contentBlockDelta" in e]
            assert len(content_events) >= 3  # At least 3 content chunks

    @pytest.mark.asyncio
    async def test_stream_async_rate_limit_error(self, nvidia_model, sample_messages):
        """Test streaming with rate limit error."""
        # Mock rate limit error
        mock_error_response = MagicMock()
        mock_error_response.status_code = 429
        
        def raise_status_error():
            raise httpx.HTTPStatusError(
                "Rate limit exceeded", 
                request=MagicMock(), 
                response=mock_error_response
            )
        
        mock_response = AsyncMock()
        mock_response.raise_for_status = raise_status_error
        
        with patch("strands.models.nvidia.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_stream_ctx = AsyncMock()
            mock_stream_ctx.__aenter__.return_value = mock_response
            mock_stream_ctx.__aexit__.return_value = None
            mock_client.stream = MagicMock(return_value=mock_stream_ctx)
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_cls.return_value = mock_client
            
            with pytest.raises(ModelThrottledException, match="NVIDIA API rate limit exceeded"):
                async for _ in nvidia_model._stream_async(sample_messages):
                    pass

    @pytest.mark.asyncio
    async def test_structured_output_async_success(self, nvidia_model, sample_messages):
        """Test successful structured output."""
        # Define output model
        class Person(pydantic.BaseModel):
            name: str
            age: int
        
        # Mock streaming response with JSON content
        sample_response = [
            'data: {"choices":[{"delta":{"content":"{\\"name\\""},\"index\":0}]}\n',
            'data: {"choices":[{"delta":{"content":": \\"John\\","},\"index\":0}]}\n',
            'data: {"choices":[{"delta":{"content":" \\"age\\": 30}"},\"index\":0}]}\n',
            'data: {"choices":[{"finish_reason":"stop","index":0}]}\n',
            'data: [DONE]\n'
        ]
        
        # Mock the streaming response
        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        
        async def mock_aiter_lines():
            for line in sample_response:
                yield line
        
        mock_response.aiter_lines = mock_aiter_lines
        
        with patch("strands.models.nvidia.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_stream_ctx = AsyncMock()
            mock_stream_ctx.__aenter__.return_value = mock_response
            mock_stream_ctx.__aexit__.return_value = None
            mock_client.stream = MagicMock(return_value=mock_stream_ctx)
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_cls.return_value = mock_client
            
            events = []
            async for event in nvidia_model._structured_output_async(Person, sample_messages):
                events.append(event)
            
            assert len(events) == 1
            assert "output" in events[0]
            assert isinstance(events[0]["output"], Person)
            assert events[0]["output"].name == "John"
            assert events[0]["output"].age == 30

    @pytest.mark.asyncio
    async def test_structured_output_async_invalid_json(self, nvidia_model, sample_messages):
        """Test structured output with invalid JSON."""
        class Person(pydantic.BaseModel):
            name: str
            age: int
        
        # Mock streaming response with invalid JSON
        sample_response = [
            'data: {"choices":[{"delta":{"content":"invalid json"},\"index\":0}]}\n',
            'data: {"choices":[{"finish_reason":"stop","index":0}]}\n',
            'data: [DONE]\n'
        ]
        
        # Mock the streaming response
        mock_response = AsyncMock()
        mock_response.raise_for_status = AsyncMock()
        
        async def mock_aiter_lines():
            for line in sample_response:
                yield line
        
        mock_response.aiter_lines = mock_aiter_lines
        
        with patch("strands.models.nvidia.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_stream_ctx = AsyncMock()
            mock_stream_ctx.__aenter__.return_value = mock_response
            mock_stream_ctx.__aexit__.return_value = None
            mock_client.stream = MagicMock(return_value=mock_stream_ctx)
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_cls.return_value = mock_client
            
            with pytest.raises(ValueError, match="Invalid JSON in structured output"):
                async for _ in nvidia_model._structured_output_async(Person, sample_messages):
                    pass

    @pytest.mark.asyncio
    async def test_stream_method(self, nvidia_model, sample_messages):
        """Test the public stream method."""
        with patch.object(nvidia_model, "_stream_async") as mock_stream_async:
            async def async_gen():
                yield {"messageStart": {"role": "assistant"}}
            
            mock_stream_async.return_value = async_gen()
            
            # Call the public stream method and consume it
            events = []
            async for event in nvidia_model.stream(sample_messages):
                events.append(event)
            
            # Should have called the async method
            mock_stream_async.assert_called_once_with(
                sample_messages, None, None, None
            )

    @pytest.mark.asyncio
    async def test_structured_output_method(self, nvidia_model, sample_messages):
        """Test the public structured_output method."""
        with patch.object(nvidia_model, "_structured_output_async") as mock_structured_output_async:
            class Person(pydantic.BaseModel):
                name: str
            
            async def async_gen():
                yield {"type": "structured_output", "data": Person(name="test")}
            
            mock_structured_output_async.return_value = async_gen()
            
            # Call the public structured_output method and consume it
            events = []
            async for event in nvidia_model.structured_output(Person, sample_messages):
                events.append(event)
            
            # Should have called the async method
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
