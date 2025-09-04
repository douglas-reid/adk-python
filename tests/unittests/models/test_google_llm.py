# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import sys
from typing import Optional
from unittest import mock
from unittest.mock import AsyncMock

from google.adk import version as adk_version
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.models.cache_metadata import CacheMetadata
from google.adk.models.gemini_llm_connection import GeminiLlmConnection
from google.adk.models.google_llm import _AGENT_ENGINE_TELEMETRY_ENV_VARIABLE_NAME
from google.adk.models.google_llm import _AGENT_ENGINE_TELEMETRY_TAG
from google.adk.models.google_llm import Gemini
from google.adk.models.google_llm import Gemma
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.utils.variant_utils import GoogleLLMVariant
from google.genai import types
from google.genai.types import Content
from google.genai.types import GenerateContentConfigOrDict
from google.genai.types import Part
import pytest


class MockAsyncIterator:
  """Mock for async iterator."""

  def __init__(self, seq):
    self.iter = iter(seq)

  def __aiter__(self):
    return self

  async def __anext__(self):
    try:
      return next(self.iter)
    except StopIteration as exc:
      raise StopAsyncIteration from exc

  async def aclose(self):
    pass


@pytest.fixture
def generate_content_response():
  return types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=Content(
                  role="model",
                  parts=[Part.from_text(text="Hello, how can I help you?")],
              ),
              finish_reason=types.FinishReason.STOP,
          )
      ]
  )


@pytest.fixture
def gemini_llm():
  return Gemini(model="gemini-1.5-flash")


@pytest.fixture
def llm_request():
  return LlmRequest(
      model="gemini-1.5-flash",
      contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
      config=types.GenerateContentConfig(
          temperature=0.1,
          response_modalities=[types.Modality.TEXT],
          system_instruction="You are a helpful assistant",
      ),
  )


@pytest.fixture
def cache_metadata():
  import time

  return CacheMetadata(
      cache_name="projects/test/locations/us-central1/cachedContents/test123",
      expire_time=time.time() + 3600,
      fingerprint="test_fingerprint",
      invocations_used=2,
      cached_contents_count=3,
      created_at=time.time() - 600,
  )


@pytest.fixture
def llm_request_with_cache(cache_metadata):
  return LlmRequest(
      model="gemini-1.5-flash",
      contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
      config=types.GenerateContentConfig(
          temperature=0.1,
          response_modalities=[types.Modality.TEXT],
          system_instruction="You are a helpful assistant",
      ),
      cache_config=ContextCacheConfig(
          cache_intervals=10, ttl_seconds=3600, min_tokens=100
      ),
      cache_metadata=cache_metadata,
  )


@pytest.fixture
def llm_request_with_computer_use():
  return LlmRequest(
      model="gemini-1.5-flash",
      contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
      config=types.GenerateContentConfig(
          temperature=0.1,
          response_modalities=[types.Modality.TEXT],
          system_instruction="You are a helpful assistant",
          tools=[
              types.Tool(
                  computer_use=types.ToolComputerUse(
                      environment=types.Environment.ENVIRONMENT_BROWSER
                  )
              )
          ],
      ),
  )


@pytest.fixture
def llm_request_with_duplicate_instruction():
  return LlmRequest(
      model="gemma-3-1b-it",
      contents=[
          types.Content(
              role="user",
              parts=[types.Part.from_text(text="Talk like a pirate.")],
          ),
          types.Content(
              role="user", parts=[types.Part.from_text(text="Hello")]
          ),
      ],
      config=types.GenerateContentConfig(
          response_modalities=[types.Modality.TEXT],
          system_instruction="Talk like a pirate.",
      ),
  )


@pytest.fixture
def llm_request_with_tools():
  return LlmRequest(
      model="gemma-3-1b-it",
      contents=[
          types.Content(role="user", parts=[types.Part.from_text(text="Hello")])
      ],
      config=types.GenerateContentConfig(
          tools=[
              types.Tool(
                  function_declarations=[
                      types.FunctionDeclaration(
                          name="search_web",
                          description="Search the web for a query.",
                          parameters=types.Schema(
                              type=types.Type.OBJECT,
                              properties={
                                  "query": types.Schema(type=types.Type.STRING)
                              },
                              required=["query"],
                          ),
                      ),
                      types.FunctionDeclaration(
                          name="get_current_time",
                          description="Gets the current time.",
                          parameters=types.Schema(
                              type=types.Type.OBJECT, properties={}
                          ),
                      ),
                  ]
              )
          ],
      ),
  )


@pytest.fixture
def mock_os_environ():
  initial_env = os.environ.copy()
  with mock.patch.dict(os.environ, initial_env, clear=False) as m:
    yield m


def test_supported_models():
  models = Gemini.supported_models()
  assert len(models) == 4
  assert models[0] == r"gemini-.*"
  assert models[1] == r"model-optimizer-.*"
  assert models[2] == r"projects\/.+\/locations\/.+\/endpoints\/.+"
  assert (
      models[3]
      == r"projects\/.+\/locations\/.+\/publishers\/google\/models\/gemini.+"
  )


def test_client_version_header():
  model = Gemini(model="gemini-1.5-flash")
  client = model.api_client

  # Check that ADK version and Python version are present in headers
  adk_version_string = f"google-adk/{adk_version.__version__}"
  python_version_string = f"gl-python/{sys.version.split()[0]}"

  x_goog_api_client_header = client._api_client._http_options.headers[
      "x-goog-api-client"
  ]
  user_agent_header = client._api_client._http_options.headers["user-agent"]

  # Verify ADK version is present
  assert adk_version_string in x_goog_api_client_header
  assert adk_version_string in user_agent_header

  # Verify Python version is present
  assert python_version_string in x_goog_api_client_header
  assert python_version_string in user_agent_header

  # Verify some Google SDK version is present (could be genai-sdk or vertex-genai-modules)
  assert any(
      sdk in x_goog_api_client_header
      for sdk in ["google-genai-sdk/", "vertex-genai-modules/"]
  )
  assert any(
      sdk in user_agent_header
      for sdk in ["google-genai-sdk/", "vertex-genai-modules/"]
  )


def test_client_version_header_with_agent_engine(mock_os_environ):
  os.environ[_AGENT_ENGINE_TELEMETRY_ENV_VARIABLE_NAME] = "my_test_project"
  model = Gemini(model="gemini-1.5-flash")
  client = model.api_client

  # Check that ADK version with telemetry tag and Python version are present in headers
  adk_version_with_telemetry = (
      f"google-adk/{adk_version.__version__}+{_AGENT_ENGINE_TELEMETRY_TAG}"
  )
  python_version_string = f"gl-python/{sys.version.split()[0]}"

  x_goog_api_client_header = client._api_client._http_options.headers[
      "x-goog-api-client"
  ]
  user_agent_header = client._api_client._http_options.headers["user-agent"]

  # Verify ADK version with telemetry tag is present
  assert adk_version_with_telemetry in x_goog_api_client_header
  assert adk_version_with_telemetry in user_agent_header

  # Verify Python version is present
  assert python_version_string in x_goog_api_client_header
  assert python_version_string in user_agent_header

  # Verify some Google SDK version is present (could be genai-sdk or vertex-genai-modules)
  assert any(
      sdk in x_goog_api_client_header
      for sdk in ["google-genai-sdk/", "vertex-genai-modules/"]
  )
  assert any(
      sdk in user_agent_header
      for sdk in ["google-genai-sdk/", "vertex-genai-modules/"]
  )


def test_maybe_append_user_content(gemini_llm, llm_request):
  # Test with user content already present
  gemini_llm._maybe_append_user_content(llm_request)
  assert len(llm_request.contents) == 1

  # Test with model content as the last message
  llm_request.contents.append(
      Content(role="model", parts=[Part.from_text(text="Response")])
  )
  gemini_llm._maybe_append_user_content(llm_request)
  assert len(llm_request.contents) == 3
  assert llm_request.contents[-1].role == "user"
  assert "Continue processing" in llm_request.contents[-1].parts[0].text


@pytest.mark.asyncio
async def test_generate_content_async(
    gemini_llm, llm_request, generate_content_response
):
  with mock.patch.object(gemini_llm, "api_client") as mock_client:
    # Create a mock coroutine that returns the generate_content_response
    async def mock_coro():
      return generate_content_response

    # Assign the coroutine to the mocked method
    mock_client.aio.models.generate_content.return_value = mock_coro()

    responses = [
        resp
        async for resp in gemini_llm.generate_content_async(
            llm_request, stream=False
        )
    ]

    assert len(responses) == 1
    assert isinstance(responses[0], LlmResponse)
    assert responses[0].content.parts[0].text == "Hello, how can I help you?"
    mock_client.aio.models.generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_generate_content_async_stream(gemini_llm, llm_request):
  with mock.patch.object(gemini_llm, "api_client") as mock_client:
    mock_responses = [
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model", parts=[Part.from_text(text="Hello")]
                    ),
                    finish_reason=None,
                )
            ]
        ),
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model", parts=[Part.from_text(text=", how")]
                    ),
                    finish_reason=None,
                )
            ]
        ),
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model",
                        parts=[Part.from_text(text=" can I help you?")],
                    ),
                    finish_reason=types.FinishReason.STOP,
                )
            ]
        ),
    ]

    # Create a mock coroutine that returns the MockAsyncIterator
    async def mock_coro():
      return MockAsyncIterator(mock_responses)

    # Set the mock to return the coroutine
    mock_client.aio.models.generate_content_stream.return_value = mock_coro()

    responses = [
        resp
        async for resp in gemini_llm.generate_content_async(
            llm_request, stream=True
        )
    ]

    # Assertions remain the same
    assert len(responses) == 4
    assert responses[0].partial is True
    assert responses[1].partial is True
    assert responses[2].partial is True
    assert responses[3].content.parts[0].text == "Hello, how can I help you?"
    mock_client.aio.models.generate_content_stream.assert_called_once()


@pytest.mark.asyncio
async def test_generate_content_async_stream_preserves_thinking_and_text_parts(
    gemini_llm, llm_request
):
  with mock.patch.object(gemini_llm, "api_client") as mock_client:
    response1 = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=Content(
                    role="model",
                    parts=[Part(text="Think1", thought=True)],
                ),
                finish_reason=None,
            )
        ]
    )
    response2 = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=Content(
                    role="model",
                    parts=[Part(text="Think2", thought=True)],
                ),
                finish_reason=None,
            )
        ]
    )
    response3 = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=Content(
                    role="model",
                    parts=[Part.from_text(text="Answer.")],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )

    async def mock_coro():
      return MockAsyncIterator([response1, response2, response3])

    mock_client.aio.models.generate_content_stream.return_value = mock_coro()

    responses = [
        resp
        async for resp in gemini_llm.generate_content_async(
            llm_request, stream=True
        )
    ]

    assert len(responses) == 4
    assert responses[0].partial is True
    assert responses[1].partial is True
    assert responses[2].partial is True
    assert responses[3].content.parts[0].text == "Think1Think2"
    assert responses[3].content.parts[0].thought is True
    assert responses[3].content.parts[1].text == "Answer."
    mock_client.aio.models.generate_content_stream.assert_called_once()


@pytest.mark.asyncio
async def test_connect(gemini_llm, llm_request):
  # Create a mock connection
  mock_connection = mock.MagicMock(spec=GeminiLlmConnection)

  # Create a mock context manager
  class MockContextManager:

    async def __aenter__(self):
      return mock_connection

    async def __aexit__(self, *args):
      pass

  # Mock the connect method at the class level
  with mock.patch(
      "google.adk.models.google_llm.Gemini.connect",
      return_value=MockContextManager(),
  ):
    async with gemini_llm.connect(llm_request) as connection:
      assert connection is mock_connection


@pytest.mark.asyncio
async def test_generate_content_async_with_custom_headers(
    gemini_llm, llm_request, generate_content_response
):
  """Test that tracking headers are updated when custom headers are provided."""
  # Add custom headers to the request config
  custom_headers = {"custom-header": "custom-value"}
  for key in gemini_llm._tracking_headers:
    custom_headers[key] = "custom " + gemini_llm._tracking_headers[key]
  llm_request.config.http_options = types.HttpOptions(headers=custom_headers)

  with mock.patch.object(gemini_llm, "api_client") as mock_client:
    # Create a mock coroutine that returns the generate_content_response
    async def mock_coro():
      return generate_content_response

    mock_client.aio.models.generate_content.return_value = mock_coro()

    responses = [
        resp
        async for resp in gemini_llm.generate_content_async(
            llm_request, stream=False
        )
    ]

    # Verify that the config passed to generate_content contains merged headers
    mock_client.aio.models.generate_content.assert_called_once()
    call_args = mock_client.aio.models.generate_content.call_args
    config_arg = call_args.kwargs["config"]

    for key, value in config_arg.http_options.headers.items():
      if key in gemini_llm._tracking_headers:
        assert value == gemini_llm._tracking_headers[key] + " custom"
      else:
        assert value == custom_headers[key]

    assert len(responses) == 1
    assert isinstance(responses[0], LlmResponse)


@pytest.mark.asyncio
async def test_generate_content_async_stream_with_custom_headers(
    gemini_llm, llm_request
):
  """Test that tracking headers are updated when custom headers are provided in streaming mode."""
  # Add custom headers to the request config
  custom_headers = {"custom-header": "custom-value"}
  llm_request.config.http_options = types.HttpOptions(headers=custom_headers)

  with mock.patch.object(gemini_llm, "api_client") as mock_client:
    mock_responses = [
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model", parts=[Part.from_text(text="Hello")]
                    ),
                    finish_reason=types.FinishReason.STOP,
                )
            ]
        )
    ]

    async def mock_coro():
      return MockAsyncIterator(mock_responses)

    mock_client.aio.models.generate_content_stream.return_value = mock_coro()

    responses = [
        resp
        async for resp in gemini_llm.generate_content_async(
            llm_request, stream=True
        )
    ]

    # Verify that the config passed to generate_content_stream contains merged headers
    mock_client.aio.models.generate_content_stream.assert_called_once()
    call_args = mock_client.aio.models.generate_content_stream.call_args
    config_arg = call_args.kwargs["config"]

    expected_headers = custom_headers.copy()
    expected_headers.update(gemini_llm._tracking_headers)
    assert config_arg.http_options.headers == expected_headers

    assert len(responses) == 2


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_generate_content_async_patches_tracking_headers(
    stream, gemini_llm, llm_request, generate_content_response
):
  """Tests that tracking headers are added to the request config."""
  # Set the request's config.http_options to None.
  llm_request.config.http_options = None

  with mock.patch.object(gemini_llm, "api_client") as mock_client:
    if stream:
      # Create a mock coroutine that returns the mock_responses.
      async def mock_coro():
        return MockAsyncIterator([generate_content_response])

      # Mock for streaming response.
      mock_client.aio.models.generate_content_stream.return_value = mock_coro()
    else:
      # Create a mock coroutine that returns the generate_content_response.
      async def mock_coro():
        return generate_content_response

      # Mock for non-streaming response.
      mock_client.aio.models.generate_content.return_value = mock_coro()

    # Call the generate_content_async method.
    responses = [
        resp
        async for resp in gemini_llm.generate_content_async(
            llm_request, stream=stream
        )
    ]

    # Assert that the config passed to the generate_content or
    # generate_content_stream method contains the tracking headers.
    if stream:
      mock_client.aio.models.generate_content_stream.assert_called_once()
      call_args = mock_client.aio.models.generate_content_stream.call_args
    else:
      mock_client.aio.models.generate_content.assert_called_once()
      call_args = mock_client.aio.models.generate_content.call_args

    final_config = call_args.kwargs["config"]

    assert final_config is not None
    assert final_config.http_options is not None
    assert (
        final_config.http_options.headers["x-goog-api-client"]
        == gemini_llm._tracking_headers["x-goog-api-client"]
    )

    assert len(responses) == 2 if stream else 1


def test_live_api_version_vertex_ai(gemini_llm):
  """Test that _live_api_version returns 'v1beta1' for Vertex AI backend."""
  with mock.patch.object(
      gemini_llm, "_api_backend", GoogleLLMVariant.VERTEX_AI
  ):
    assert gemini_llm._live_api_version == "v1beta1"


def test_live_api_version_gemini_api(gemini_llm):
  """Test that _live_api_version returns 'v1alpha' for Gemini API backend."""
  with mock.patch.object(
      gemini_llm, "_api_backend", GoogleLLMVariant.GEMINI_API
  ):
    assert gemini_llm._live_api_version == "v1alpha"


def test_live_api_client_properties(gemini_llm):
  """Test that _live_api_client is properly configured with tracking headers and API version."""
  with mock.patch.object(
      gemini_llm, "_api_backend", GoogleLLMVariant.VERTEX_AI
  ):
    client = gemini_llm._live_api_client

    # Verify that the client has the correct headers and API version
    http_options = client._api_client._http_options
    assert http_options.api_version == "v1beta1"

    # Check that tracking headers are included
    tracking_headers = gemini_llm._tracking_headers
    for key, value in tracking_headers.items():
      assert key in http_options.headers
      assert value in http_options.headers[key]


@pytest.mark.asyncio
async def test_connect_with_custom_headers(gemini_llm, llm_request):
  """Test that connect method updates tracking headers and API version when custom headers are provided."""
  # Setup request with live connect config and custom headers
  custom_headers = {"custom-live-header": "live-value"}
  llm_request.live_connect_config = types.LiveConnectConfig(
      http_options=types.HttpOptions(headers=custom_headers)
  )

  mock_live_session = mock.AsyncMock()

  # Mock the _live_api_client to return a mock client
  with mock.patch.object(gemini_llm, "_live_api_client") as mock_live_client:
    # Create a mock context manager
    class MockLiveConnect:

      async def __aenter__(self):
        return mock_live_session

      async def __aexit__(self, *args):
        pass

    mock_live_client.aio.live.connect.return_value = MockLiveConnect()

    async with gemini_llm.connect(llm_request) as connection:
      # Verify that the connect method was called with the right config
      mock_live_client.aio.live.connect.assert_called_once()
      call_args = mock_live_client.aio.live.connect.call_args
      config_arg = call_args.kwargs["config"]

      # Verify that tracking headers were merged with custom headers
      expected_headers = custom_headers.copy()
      expected_headers.update(gemini_llm._tracking_headers)
      assert config_arg.http_options.headers == expected_headers

      # Verify that API version was set
      assert config_arg.http_options.api_version == gemini_llm._live_api_version

      # Verify that system instruction and tools were set
      assert config_arg.system_instruction is not None
      assert config_arg.tools == llm_request.config.tools

      # Verify connection is properly wrapped
      assert isinstance(connection, GeminiLlmConnection)


@pytest.mark.asyncio
async def test_connect_without_custom_headers(gemini_llm, llm_request):
  """Test that connect method works properly when no custom headers are provided."""
  # Setup request with live connect config but no custom headers
  llm_request.live_connect_config = types.LiveConnectConfig()

  mock_live_session = mock.AsyncMock()

  with mock.patch.object(gemini_llm, "_live_api_client") as mock_live_client:

    class MockLiveConnect:

      async def __aenter__(self):
        return mock_live_session

      async def __aexit__(self, *args):
        pass

    mock_live_client.aio.live.connect.return_value = MockLiveConnect()

    async with gemini_llm.connect(llm_request) as connection:
      # Verify that the connect method was called with the right config
      mock_live_client.aio.live.connect.assert_called_once()
      call_args = mock_live_client.aio.live.connect.call_args
      config_arg = call_args.kwargs["config"]

      # Verify that http_options remains None since no custom headers were provided
      assert config_arg.http_options is None

      # Verify that system instruction and tools were still set
      assert config_arg.system_instruction is not None
      assert config_arg.tools == llm_request.config.tools

      assert isinstance(connection, GeminiLlmConnection)


@pytest.mark.parametrize(
    (
        "api_backend, "
        "expected_file_display_name, "
        "expected_inline_display_name, "
        "expected_labels"
    ),
    [
        (
            GoogleLLMVariant.GEMINI_API,
            None,
            None,
            None,
        ),
        (
            GoogleLLMVariant.VERTEX_AI,
            "My Test PDF",
            "My Test Image",
            {"key": "value"},
        ),
    ],
)
@pytest.mark.asyncio
async def test_preprocess_request_handles_backend_specific_fields(
    gemini_llm: Gemini,
    api_backend: GoogleLLMVariant,
    expected_file_display_name: Optional[str],
    expected_inline_display_name: Optional[str],
    expected_labels: Optional[str],
):
  """Tests that _preprocess_request correctly sanitizes fields based on the API backend.

  - For GEMINI_API, it should remove 'display_name' from file/inline data
    and remove 'labels' from the config.
  - For VERTEX_AI, it should leave these fields untouched.
  """
  # Arrange: Create a request with fields that need to be preprocessed.
  llm_request_with_files = LlmRequest(
      model="gemini-1.5-flash",
      contents=[
          Content(
              role="user",
              parts=[
                  Part(
                      file_data=types.FileData(
                          file_uri="gs://bucket/file.pdf",
                          mime_type="application/pdf",
                          display_name="My Test PDF",
                      )
                  ),
                  Part(
                      inline_data=types.Blob(
                          data=b"some_bytes",
                          mime_type="image/png",
                          display_name="My Test Image",
                      )
                  ),
              ],
          )
      ],
      config=types.GenerateContentConfig(labels={"key": "value"}),
  )

  # Mock the _api_backend property to control the test scenario
  with mock.patch.object(
      Gemini, "_api_backend", new_callable=mock.PropertyMock
  ) as mock_backend:
    mock_backend.return_value = api_backend

    # Act: Run the preprocessing method
    await gemini_llm._preprocess_request(llm_request_with_files)

    # Assert: Check if the fields were correctly processed
    file_part = llm_request_with_files.contents[0].parts[0]
    inline_part = llm_request_with_files.contents[0].parts[1]

    assert file_part.file_data.display_name == expected_file_display_name
    assert inline_part.inline_data.display_name == expected_inline_display_name
    assert llm_request_with_files.config.labels == expected_labels


@pytest.mark.asyncio
async def test_generate_content_async_stream_aggregated_content_regardless_of_finish_reason():
  """Test that aggregated content is generated regardless of finish_reason."""
  gemini_llm = Gemini(model="gemini-1.5-flash")
  llm_request = LlmRequest(
      model="gemini-1.5-flash",
      contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
      config=types.GenerateContentConfig(
          temperature=0.1,
          response_modalities=[types.Modality.TEXT],
          system_instruction="You are a helpful assistant",
      ),
  )

  with mock.patch.object(gemini_llm, "api_client") as mock_client:
    # Test with different finish reasons
    test_cases = [
        types.FinishReason.MAX_TOKENS,
        types.FinishReason.SAFETY,
        types.FinishReason.RECITATION,
        types.FinishReason.OTHER,
    ]

    for finish_reason in test_cases:
      mock_responses = [
          types.GenerateContentResponse(
              candidates=[
                  types.Candidate(
                      content=Content(
                          role="model", parts=[Part.from_text(text="Hello")]
                      ),
                      finish_reason=None,
                  )
              ]
          ),
          types.GenerateContentResponse(
              candidates=[
                  types.Candidate(
                      content=Content(
                          role="model", parts=[Part.from_text(text=" world")]
                      ),
                      finish_reason=finish_reason,
                      finish_message=f"Finished with {finish_reason}",
                  )
              ]
          ),
      ]

      async def mock_coro():
        return MockAsyncIterator(mock_responses)

      mock_client.aio.models.generate_content_stream.return_value = mock_coro()

      responses = [
          resp
          async for resp in gemini_llm.generate_content_async(
              llm_request, stream=True
          )
      ]

      # Should have 3 responses: 2 partial and 1 final aggregated
      assert len(responses) == 3
      assert responses[0].partial is True
      assert responses[1].partial is True

      # Final response should have aggregated content with error info
      final_response = responses[2]
      assert final_response.content.parts[0].text == "Hello world"
      # After the code changes, error_code and error_message are set for non-STOP finish reasons
      assert final_response.error_code == finish_reason
      assert final_response.error_message == f"Finished with {finish_reason}"


@pytest.mark.asyncio
async def test_generate_content_async_stream_with_thought_and_text_error_handling():
  """Test that aggregated content with thought and text preserves error information."""
  gemini_llm = Gemini(model="gemini-1.5-flash")
  llm_request = LlmRequest(
      model="gemini-1.5-flash",
      contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
      config=types.GenerateContentConfig(
          temperature=0.1,
          response_modalities=[types.Modality.TEXT],
          system_instruction="You are a helpful assistant",
      ),
  )

  with mock.patch.object(gemini_llm, "api_client") as mock_client:
    mock_responses = [
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model", parts=[Part(text="Think1", thought=True)]
                    ),
                    finish_reason=None,
                )
            ]
        ),
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model", parts=[Part.from_text(text="Answer")]
                    ),
                    finish_reason=types.FinishReason.MAX_TOKENS,
                    finish_message="Maximum tokens reached",
                )
            ]
        ),
    ]

    async def mock_coro():
      return MockAsyncIterator(mock_responses)

    mock_client.aio.models.generate_content_stream.return_value = mock_coro()

    responses = [
        resp
        async for resp in gemini_llm.generate_content_async(
            llm_request, stream=True
        )
    ]

    # Should have 3 responses: 2 partial and 1 final aggregated
    assert len(responses) == 3
    assert responses[0].partial is True
    assert responses[1].partial is True

    # Final response should have aggregated content with both thought and text
    final_response = responses[2]
    assert len(final_response.content.parts) == 2
    assert final_response.content.parts[0].text == "Think1"
    assert final_response.content.parts[0].thought is True
    assert final_response.content.parts[1].text == "Answer"
    # After the code changes, error_code and error_message are set for non-STOP finish reasons
    assert final_response.error_code == types.FinishReason.MAX_TOKENS
    assert final_response.error_message == "Maximum tokens reached"


@pytest.mark.asyncio
async def test_generate_content_async_stream_error_info_none_for_stop_finish_reason():
  """Test that error_code and error_message are None when finish_reason is STOP."""
  gemini_llm = Gemini(model="gemini-1.5-flash")
  llm_request = LlmRequest(
      model="gemini-1.5-flash",
      contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
      config=types.GenerateContentConfig(
          temperature=0.1,
          response_modalities=[types.Modality.TEXT],
          system_instruction="You are a helpful assistant",
      ),
  )

  with mock.patch.object(gemini_llm, "api_client") as mock_client:
    mock_responses = [
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model", parts=[Part.from_text(text="Hello")]
                    ),
                    finish_reason=None,
                )
            ]
        ),
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model", parts=[Part.from_text(text=" world")]
                    ),
                    finish_reason=types.FinishReason.STOP,
                    finish_message="Successfully completed",
                )
            ]
        ),
    ]

    async def mock_coro():
      return MockAsyncIterator(mock_responses)

    mock_client.aio.models.generate_content_stream.return_value = mock_coro()

    responses = [
        resp
        async for resp in gemini_llm.generate_content_async(
            llm_request, stream=True
        )
    ]

    # Should have 3 responses: 2 partial and 1 final aggregated
    assert len(responses) == 3
    assert responses[0].partial is True
    assert responses[1].partial is True

    # Final response should have aggregated content with error info None for STOP finish reason
    final_response = responses[2]
    assert final_response.content.parts[0].text == "Hello world"
    assert final_response.error_code is None
    assert final_response.error_message is None


@pytest.mark.asyncio
async def test_generate_content_async_stream_error_info_set_for_non_stop_finish_reason():
  """Test that error_code and error_message are set for non-STOP finish reasons."""
  gemini_llm = Gemini(model="gemini-1.5-flash")
  llm_request = LlmRequest(
      model="gemini-1.5-flash",
      contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
      config=types.GenerateContentConfig(
          temperature=0.1,
          response_modalities=[types.Modality.TEXT],
          system_instruction="You are a helpful assistant",
      ),
  )

  with mock.patch.object(gemini_llm, "api_client") as mock_client:
    mock_responses = [
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model", parts=[Part.from_text(text="Hello")]
                    ),
                    finish_reason=None,
                )
            ]
        ),
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model", parts=[Part.from_text(text=" world")]
                    ),
                    finish_reason=types.FinishReason.MAX_TOKENS,
                    finish_message="Maximum tokens reached",
                )
            ]
        ),
    ]

    async def mock_coro():
      return MockAsyncIterator(mock_responses)

    mock_client.aio.models.generate_content_stream.return_value = mock_coro()

    responses = [
        resp
        async for resp in gemini_llm.generate_content_async(
            llm_request, stream=True
        )
    ]

    # Should have 3 responses: 2 partial and 1 final aggregated
    assert len(responses) == 3
    assert responses[0].partial is True
    assert responses[1].partial is True

    # Final response should have aggregated content with error info set for non-STOP finish reason
    final_response = responses[2]
    assert final_response.content.parts[0].text == "Hello world"
    assert final_response.error_code == types.FinishReason.MAX_TOKENS
    assert final_response.error_message == "Maximum tokens reached"


@pytest.mark.asyncio
async def test_generate_content_async_stream_no_aggregated_content_without_text():
  """Test that no aggregated content is generated when there's no accumulated text."""
  gemini_llm = Gemini(model="gemini-1.5-flash")
  llm_request = LlmRequest(
      model="gemini-1.5-flash",
      contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
      config=types.GenerateContentConfig(
          temperature=0.1,
          response_modalities=[types.Modality.TEXT],
          system_instruction="You are a helpful assistant",
      ),
  )

  with mock.patch.object(gemini_llm, "api_client") as mock_client:
    # Mock response with no text content
    mock_responses = [
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model",
                        parts=[
                            Part(
                                function_call=types.FunctionCall(
                                    name="test", args={}
                                )
                            )
                        ],
                    ),
                    finish_reason=types.FinishReason.STOP,
                )
            ]
        ),
    ]

    async def mock_coro():
      return MockAsyncIterator(mock_responses)

    mock_client.aio.models.generate_content_stream.return_value = mock_coro()

    responses = [
        resp
        async for resp in gemini_llm.generate_content_async(
            llm_request, stream=True
        )
    ]

    # Should have only 1 response (no aggregated content generated)
    assert len(responses) == 1
    # Verify it's a function call, not text
    assert responses[0].content.parts[0].function_call is not None


@pytest.mark.asyncio
async def test_generate_content_async_stream_mixed_text_function_call_text():
  """Test streaming with pattern: [text, function_call, text] to verify proper aggregation."""
  gemini_llm = Gemini(model="gemini-1.5-flash")
  llm_request = LlmRequest(
      model="gemini-1.5-flash",
      contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
      config=types.GenerateContentConfig(
          temperature=0.1,
          response_modalities=[types.Modality.TEXT],
          system_instruction="You are a helpful assistant",
      ),
  )

  with mock.patch.object(gemini_llm, "api_client") as mock_client:
    # Create responses with pattern: text -> function_call -> text
    mock_responses = [
        # First text chunk
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model", parts=[Part.from_text(text="First text")]
                    ),
                    finish_reason=None,
                )
            ]
        ),
        # Function call interrupts the text flow
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model",
                        parts=[
                            Part(
                                function_call=types.FunctionCall(
                                    name="test_func", args={}
                                )
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ]
        ),
        # More text after function call
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model",
                        parts=[Part.from_text(text=" second text")],
                    ),
                    finish_reason=types.FinishReason.STOP,
                )
            ]
        ),
    ]

    async def mock_coro():
      return MockAsyncIterator(mock_responses)

    mock_client.aio.models.generate_content_stream.return_value = mock_coro()

    responses = [
        resp
        async for resp in gemini_llm.generate_content_async(
            llm_request, stream=True
        )
    ]

    # Should have multiple responses:
    # 1. Partial text "First text"
    # 2. Aggregated "First text" when function call interrupts
    # 3. Function call
    # 4. Partial text " second text"
    # 5. Final aggregated " second text"
    assert len(responses) == 5

    # First partial text
    assert responses[0].partial is True
    assert responses[0].content.parts[0].text == "First text"

    # Aggregated first text (when function call interrupts)
    assert responses[1].content.parts[0].text == "First text"
    assert (
        responses[1].partial is None
    )  # Aggregated responses don't have partial flag

    # Function call
    assert responses[2].content.parts[0].function_call is not None
    assert responses[2].content.parts[0].function_call.name == "test_func"

    # Second partial text
    assert responses[3].partial is True
    assert responses[3].content.parts[0].text == " second text"

    # Final aggregated text with error info
    assert responses[4].content.parts[0].text == " second text"
    assert (
        responses[4].error_code is None
    )  # STOP finish reason should have None error_code


@pytest.mark.asyncio
async def test_generate_content_async_stream_multiple_text_parts_in_single_response():
  """Test streaming with multiple text parts in a single response."""
  gemini_llm = Gemini(model="gemini-1.5-flash")
  llm_request = LlmRequest(
      model="gemini-1.5-flash",
      contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
      config=types.GenerateContentConfig(
          temperature=0.1,
          response_modalities=[types.Modality.TEXT],
          system_instruction="You are a helpful assistant",
      ),
  )

  with mock.patch.object(gemini_llm, "api_client") as mock_client:
    # Create a response with multiple text parts
    mock_responses = [
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model",
                        parts=[
                            Part.from_text(text="First part"),
                            Part.from_text(text=" second part"),
                        ],
                    ),
                    finish_reason=types.FinishReason.STOP,
                )
            ]
        ),
    ]

    async def mock_coro():
      return MockAsyncIterator(mock_responses)

    mock_client.aio.models.generate_content_stream.return_value = mock_coro()

    responses = [
        resp
        async for resp in gemini_llm.generate_content_async(
            llm_request, stream=True
        )
    ]

    # Should handle only the first text part in current implementation
    # Note: This test documents current behavior - the implementation only
    # looks at parts[0].text, so it would only process "First part"
    assert len(responses) >= 1
    assert responses[0].content.parts[0].text == "First part"


@pytest.mark.asyncio
async def test_generate_content_async_stream_complex_mixed_thought_text_function():
  """Test complex streaming with thought, text, and function calls mixed."""
  gemini_llm = Gemini(model="gemini-1.5-flash")
  llm_request = LlmRequest(
      model="gemini-1.5-flash",
      contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
      config=types.GenerateContentConfig(
          temperature=0.1,
          response_modalities=[types.Modality.TEXT],
          system_instruction="You are a helpful assistant",
      ),
  )

  with mock.patch.object(gemini_llm, "api_client") as mock_client:
    # Complex pattern: thought -> text -> function_call -> thought -> text
    mock_responses = [
        # Thought
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model",
                        parts=[Part(text="Thinking...", thought=True)],
                    ),
                    finish_reason=None,
                )
            ]
        ),
        # Regular text
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model",
                        parts=[Part.from_text(text="Here's my answer")],
                    ),
                    finish_reason=None,
                )
            ]
        ),
        # Function call
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model",
                        parts=[
                            Part(
                                function_call=types.FunctionCall(
                                    name="lookup", args={}
                                )
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ]
        ),
        # More thought
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model",
                        parts=[Part(text="More thinking...", thought=True)],
                    ),
                    finish_reason=None,
                )
            ]
        ),
        # Final text
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model",
                        parts=[Part.from_text(text=" and conclusion")],
                    ),
                    finish_reason=types.FinishReason.STOP,
                )
            ]
        ),
    ]

    async def mock_coro():
      return MockAsyncIterator(mock_responses)

    mock_client.aio.models.generate_content_stream.return_value = mock_coro()

    responses = [
        resp
        async for resp in gemini_llm.generate_content_async(
            llm_request, stream=True
        )
    ]

    # Should properly separate thought and regular text across aggregations
    assert len(responses) > 5  # Multiple partial + aggregated responses

    # Verify we get both thought and regular text parts in aggregated responses
    aggregated_responses = [
        r
        for r in responses
        if r.partial is None and r.content and len(r.content.parts) > 1
    ]
    assert (
        len(aggregated_responses) > 0
    )  # Should have at least one aggregated response with multiple parts

    # Final aggregated response should have both thought and text
    final_response = responses[-1]
    assert (
        final_response.error_code is None
    )  # STOP finish reason should have None error_code
    assert len(final_response.content.parts) == 2  # thought part + text part
    assert final_response.content.parts[0].thought is True
    assert "More thinking..." in final_response.content.parts[0].text
    assert final_response.content.parts[1].text == " and conclusion"


@pytest.mark.asyncio
async def test_generate_content_async_stream_two_separate_text_aggregations():
  """Test that [text, function_call, text] results in two separate text aggregations."""
  gemini_llm = Gemini(model="gemini-1.5-flash")
  llm_request = LlmRequest(
      model="gemini-1.5-flash",
      contents=[Content(role="user", parts=[Part.from_text(text="Hello")])],
      config=types.GenerateContentConfig(
          temperature=0.1,
          response_modalities=[types.Modality.TEXT],
          system_instruction="You are a helpful assistant",
      ),
  )

  with mock.patch.object(gemini_llm, "api_client") as mock_client:
    # Create responses: multiple text chunks -> function_call -> multiple text chunks
    mock_responses = [
        # First text accumulation (multiple chunks)
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model", parts=[Part.from_text(text="First")]
                    ),
                    finish_reason=None,
                )
            ]
        ),
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model", parts=[Part.from_text(text=" chunk")]
                    ),
                    finish_reason=None,
                )
            ]
        ),
        # Function call interrupts
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model",
                        parts=[
                            Part(
                                function_call=types.FunctionCall(
                                    name="divide", args={}
                                )
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ]
        ),
        # Second text accumulation (multiple chunks)
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model", parts=[Part.from_text(text="Second")]
                    ),
                    finish_reason=None,
                )
            ]
        ),
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        role="model", parts=[Part.from_text(text=" chunk")]
                    ),
                    finish_reason=types.FinishReason.STOP,
                )
            ]
        ),
    ]

    async def mock_coro():
      return MockAsyncIterator(mock_responses)

    mock_client.aio.models.generate_content_stream.return_value = mock_coro()

    responses = [
        resp
        async for resp in gemini_llm.generate_content_async(
            llm_request, stream=True
        )
    ]

    # Find the aggregated text responses (non-partial, text-only)
    aggregated_text_responses = [
        r
        for r in responses
        if (
            r.partial is None
            and r.content
            and r.content.parts
            and r.content.parts[0].text
            and not r.content.parts[0].function_call
        )
    ]

    # Should have two separate text aggregations: "First chunk" and "Second chunk"
    assert len(aggregated_text_responses) >= 2

    # First aggregation should contain "First chunk"
    first_aggregation = aggregated_text_responses[0]
    assert first_aggregation.content.parts[0].text == "First chunk"

    # Final aggregation should contain "Second chunk" and have error info
    final_aggregation = aggregated_text_responses[-1]
    assert final_aggregation.content.parts[0].text == "Second chunk"
    assert (
        final_aggregation.error_code is None
    )  # STOP finish reason should have None error_code

    # Verify the function call is preserved between aggregations
    function_call_responses = [
        r
        for r in responses
        if (r.content and r.content.parts and r.content.parts[0].function_call)
    ]
    assert len(function_call_responses) == 1
    assert (
        function_call_responses[0].content.parts[0].function_call.name
        == "divide"
    )


@pytest.mark.asyncio
async def test_computer_use_removes_system_instruction():
  """Test that system instruction is set to None when computer use is configured."""
  llm = Gemini()

  llm_request = LlmRequest(
      model="gemini-1.5-flash",
      contents=[
          types.Content(role="user", parts=[types.Part.from_text(text="Hello")])
      ],
      config=types.GenerateContentConfig(
          system_instruction="You are a helpful assistant",
          tools=[
              types.Tool(
                  computer_use=types.ToolComputerUse(
                      environment=types.Environment.ENVIRONMENT_BROWSER
                  )
              )
          ],
      ),
  )

  await llm._preprocess_request(llm_request)

  # System instruction should be set to None when computer use is configured
  assert llm_request.config.system_instruction is None


@pytest.mark.asyncio
async def test_computer_use_preserves_system_instruction_when_no_computer_use():
  """Test that system instruction is preserved when computer use is not configured."""
  llm = Gemini()

  original_instruction = "You are a helpful assistant"
  llm_request = LlmRequest(
      model="gemini-1.5-flash",
      contents=[
          types.Content(role="user", parts=[types.Part.from_text(text="Hello")])
      ],
      config=types.GenerateContentConfig(
          system_instruction=original_instruction,
          tools=[
              types.Tool(
                  function_declarations=[
                      types.FunctionDeclaration(name="test", description="test")
                  ]
              )
          ],
      ),
  )

  await llm._preprocess_request(llm_request)

  # System instruction should be preserved when no computer use
  assert llm_request.config.system_instruction == original_instruction


@pytest.mark.asyncio
async def test_computer_use_with_no_config():
  """Test that preprocessing works when config is None."""
  llm = Gemini()

  llm_request = LlmRequest(
      model="gemini-1.5-flash",
      contents=[
          types.Content(role="user", parts=[types.Part.from_text(text="Hello")])
      ],
  )

  # Should not raise an exception
  await llm._preprocess_request(llm_request)


@pytest.mark.asyncio
async def test_computer_use_with_no_tools():
  """Test that preprocessing works when config.tools is None."""
  llm = Gemini()

  original_instruction = "You are a helpful assistant"
  llm_request = LlmRequest(
      model="gemini-1.5-flash",
      contents=[
          types.Content(role="user", parts=[types.Part.from_text(text="Hello")])
      ],
      config=types.GenerateContentConfig(
          system_instruction=original_instruction,
          tools=None,
      ),
  )

  await llm._preprocess_request(llm_request)

  # System instruction should be preserved when no tools
  assert llm_request.config.system_instruction == original_instruction


@pytest.mark.asyncio
async def test_adapt_computer_use_tool_wait():
  """Test that _adapt_computer_use_tool correctly adapts wait to wait_5_seconds."""
  from google.adk.tools.computer_use.computer_use_tool import ComputerUseTool

  llm = Gemini()

  # Create a mock wait tool
  mock_wait_func = AsyncMock()
  mock_wait_func.return_value = "mock_result"

  original_wait_tool = ComputerUseTool(
      func=mock_wait_func,
      screen_size=(1920, 1080),
      virtual_screen_size=(1000, 1000),
  )

  llm_request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(),
  )

  # Add wait to tools_dict
  llm_request.tools_dict["wait"] = original_wait_tool

  # Call the adaptation method (now async)
  await llm._adapt_computer_use_tool(llm_request)

  # Verify wait was removed and wait_5_seconds was added
  assert "wait" not in llm_request.tools_dict
  assert "wait_5_seconds" in llm_request.tools_dict

  # Verify the new tool has correct properties
  wait_5_seconds_tool = llm_request.tools_dict["wait_5_seconds"]
  assert isinstance(wait_5_seconds_tool, ComputerUseTool)
  assert wait_5_seconds_tool._screen_size == (1920, 1080)
  assert wait_5_seconds_tool._coordinate_space == (1000, 1000)

  # Verify calling the new tool calls the original with 5 seconds
  result = await wait_5_seconds_tool.func()
  assert result == "mock_result"
  mock_wait_func.assert_awaited_once_with(5)


@pytest.mark.asyncio
async def test_adapt_computer_use_tool_no_wait():
  """Test that _adapt_computer_use_tool does nothing when wait is not present."""
  llm = Gemini()

  llm_request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(),
  )

  # Don't add any tools
  original_tools_dict = llm_request.tools_dict.copy()

  # Call the adaptation method (now async)
  await llm._adapt_computer_use_tool(llm_request)

  # Verify tools_dict is unchanged
  assert llm_request.tools_dict == original_tools_dict
  assert "wait_5_seconds" not in llm_request.tools_dict


@pytest.mark.asyncio
async def test_generate_content_async_with_cache_metadata_integration(
    gemini_llm, llm_request_with_cache, cache_metadata
):
  """Test integration between Google LLM and cache manager with proper parameter order.

  This test specifically validates that the cache manager's populate_cache_metadata_in_response
  method is called with the correct parameter order: (llm_response, cache_metadata).

  This test would have caught the parameter order bug where cache_metadata and llm_response
  were passed in the wrong order, causing 'CacheMetadata' object has no attribute 'usage_metadata' errors.
  """

  # Create a mock response with usage metadata including cached tokens
  generate_content_response = types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=Content(
                  role="model",
                  parts=[Part.from_text(text="Hello, how can I help you?")],
              ),
              finish_reason=types.FinishReason.STOP,
          )
      ],
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=1500,
          candidates_token_count=150,
          cached_content_token_count=800,  # This is the key field that was always 0 due to the bug
          total_token_count=1650,
      ),
  )

  with mock.patch.object(gemini_llm, "api_client") as mock_client:
    # Create a mock coroutine that returns the generate_content_response
    async def mock_coro():
      return generate_content_response

    mock_client.aio.models.generate_content.return_value = mock_coro()

    # Mock the cache manager module to verify correct method call
    with mock.patch(
        "google.adk.models.gemini_context_cache_manager.GeminiContextCacheManager"
    ) as MockCacheManagerClass:
      mock_cache_manager = MockCacheManagerClass.return_value
      # Configure cache manager to handle context caching
      mock_cache_manager.handle_context_caching = AsyncMock(
          return_value=cache_metadata
      )

      responses = [
          resp
          async for resp in gemini_llm.generate_content_async(
              llm_request_with_cache, stream=False
          )
      ]

      # Verify the response was processed
      assert len(responses) == 1
      response = responses[0]
      assert isinstance(response, LlmResponse)
      assert response.content.parts[0].text == "Hello, how can I help you?"

      # CRITICAL TEST: Verify populate_cache_metadata_in_response was called with correct parameter order
      mock_cache_manager.populate_cache_metadata_in_response.assert_called_once()
      call_args = (
          mock_cache_manager.populate_cache_metadata_in_response.call_args
      )

      # The first argument should be the LlmResponse (not CacheMetadata)
      first_arg = call_args[0][0]  # First positional argument
      second_arg = call_args[0][1]  # Second positional argument

      # Verify correct parameter order: (llm_response, cache_metadata)
      assert isinstance(first_arg, LlmResponse), (
          f"First parameter should be LlmResponse, got {type(first_arg)}. "
          "This indicates parameters are in wrong order."
      )
      assert isinstance(second_arg, CacheMetadata), (
          f"Second parameter should be CacheMetadata, got {type(second_arg)}. "
          "This indicates parameters are in wrong order."
      )

      # Verify the LlmResponse has the expected usage metadata
      assert first_arg.usage_metadata is not None
      assert first_arg.usage_metadata.cached_content_token_count == 800
      assert first_arg.usage_metadata.prompt_token_count == 1500
      assert first_arg.usage_metadata.candidates_token_count == 150

      # Verify cache metadata is preserved
      assert second_arg.cache_name == cache_metadata.cache_name
      assert second_arg.invocations_used == cache_metadata.invocations_used
async def test_not_gemma_model():
  llm = Gemma()
  llm_request_bad_model = LlmRequest(
      model="not-a-gemma-model",
  )
  with pytest.raises(AssertionError, match=r".*model.*"):
    async for _ in llm.generate_content_async(llm_request_bad_model):
      pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "llm_request",
    ["llm_request", "llm_request_with_duplicate_instruction"],
    indirect=True,
)
async def test_gemma_request_preprocess(llm_request):
  llm = Gemma()
  want_content_text = llm_request.config.system_instruction

  await llm._preprocess_request(llm_request=llm_request)

  # system instruction should be cleared
  assert not llm_request.config.system_instruction
  # should be two content bits now (deduped, if needed)
  assert len(llm_request.contents) == 2
  # first message in contents should be "user": <original sys instruction>
  assert llm_request.contents[0].role == "user"
  assert llm_request.contents[0].parts[0].text == want_content_text


def test_gemma_functions_before_model_callback(llm_request_with_tools):
  from google.adk.agents.callback_context import CallbackContext
  from google.adk.models.google_llm import gemma_functions_before_model_callback

  gemma_functions_before_model_callback(
      mock.MagicMock(spec=CallbackContext), llm_request_with_tools
  )

  assert not llm_request_with_tools.config.tools

  # The original user content should still be the first item
  assert llm_request_with_tools.contents[0].role == "user"
  assert llm_request_with_tools.contents[0].parts[0].text == "Hello"

  sys_instruct_text = llm_request_with_tools.config.system_instruction
  assert sys_instruct_text is not None
  assert "You have access to the following functions" in sys_instruct_text
  assert (
      """{"description":"Search the web for a query.","name":"search_web","""
      in sys_instruct_text
  )
  assert (
      """{"description":"Gets the current time.","name":"get_current_time","parameters":{"properties":{}"""
      in sys_instruct_text
  )


def test_gemma_functions_after_model_callback_valid_json_function_call():
  from google.adk.agents.callback_context import CallbackContext
  from google.adk.models.google_llm import gemma_functions_after_model_callback

  # Simulate a response from Gemma that should be converted to a FunctionCall
  json_function_call_str = (
      '{"name": "search_web", "parameters": {"query": "latest news"}}'
  )
  llm_response = LlmResponse(
      content=Content(
          role="model", parts=[Part.from_text(text=json_function_call_str)]
      )
  )

  gemma_functions_after_model_callback(
      mock.MagicMock(spec=CallbackContext), llm_response
  )

  # Assert that the content was transformed into a FunctionCall
  assert llm_response.content
  assert llm_response.content.parts
  assert len(llm_response.content.parts) == 1
  part = llm_response.content.parts[0]
  assert part.function_call is not None
  assert part.function_call.name == "search_web"
  assert part.function_call.args == {"query": "latest news"}
  # Assert that the entire part matches the expected structure
  expected_function_call = types.FunctionCall(
      name="search_web", args={"query": "latest news"}
  )
  expected_part = Part(function_call=expected_function_call)
  assert part == expected_part
  assert part.text is None  # Ensure text part is cleared


def test_gemma_functions_after_model_callback_invalid_json_text():
  from google.adk.agents.callback_context import CallbackContext
  from google.adk.models.google_llm import gemma_functions_after_model_callback

  # Simulate a response with plain text that is not JSON
  original_text = "This is a regular text response."
  llm_response = LlmResponse(
      content=Content(role="model", parts=[Part.from_text(text=original_text)])
  )

  gemma_functions_after_model_callback(
      mock.MagicMock(spec=CallbackContext), llm_response
  )

  # Assert that the content remains unchanged
  assert llm_response.content
  assert llm_response.content.parts
  assert len(llm_response.content.parts) == 1
  assert llm_response.content.parts[0].text == original_text
  assert llm_response.content.parts[0].function_call is None


def test_gemma_functions_after_model_callback_malformed_json():
  from google.adk.agents.callback_context import CallbackContext
  from google.adk.models.google_llm import gemma_functions_after_model_callback

  # Simulate a response with valid JSON but not in the function call format
  malformed_json_str = '{"not_a_function": "value", "another_field": 123}'
  llm_response = LlmResponse(
      content=Content(
          role="model", parts=[Part.from_text(text=malformed_json_str)]
      )
  )

  gemma_functions_after_model_callback(
      mock.MagicMock(spec=CallbackContext), llm_response
  )

  # Assert that the content remains unchanged because it doesn't match the expected schema
  assert llm_response.content
  assert llm_response.content.parts
  assert len(llm_response.content.parts) == 1
  assert llm_response.content.parts[0].text == malformed_json_str
  assert llm_response.content.parts[0].function_call is None


def test_gemma_functions_after_model_callback_empty_content_or_multiple_parts():
  from google.adk.agents.callback_context import CallbackContext
  from google.adk.models.google_llm import gemma_functions_after_model_callback

  # Test case 1: LlmResponse with no content
  llm_response_no_content = LlmResponse(content=None)
  gemma_functions_after_model_callback(
      mock.MagicMock(spec=CallbackContext), llm_response_no_content
  )
  assert llm_response_no_content.content is None

  # Test case 2: LlmResponse with empty parts list
  llm_response_empty_parts = LlmResponse(
      content=Content(role="model", parts=[])
  )
  gemma_functions_after_model_callback(
      mock.MagicMock(spec=CallbackContext), llm_response_empty_parts
  )
  assert llm_response_empty_parts.content
  assert not llm_response_empty_parts.content.parts

  # Test case 3: LlmResponse with multiple parts
  llm_response_multiple_parts = LlmResponse(
      content=Content(
          role="model",
          parts=[
              Part.from_text(text="part one"),
              Part.from_text(text="part two"),
          ],
      )
  )
  original_parts = list(
      llm_response_multiple_parts.content.parts
  )  # Copy for comparison
  gemma_functions_after_model_callback(
      mock.MagicMock(spec=CallbackContext), llm_response_multiple_parts
  )
  assert llm_response_multiple_parts.content
  assert (
      llm_response_multiple_parts.content.parts == original_parts
  )  # Should remain unchanged

  # Test case 4: LlmResponse with one part, but empty text
  llm_response_empty_text_part = LlmResponse(
      content=Content(role="model", parts=[Part.from_text(text="")])
  )
  gemma_functions_after_model_callback(
      mock.MagicMock(spec=CallbackContext), llm_response_empty_text_part
  )
  assert llm_response_empty_text_part.content
  assert llm_response_empty_text_part.content.parts
  assert llm_response_empty_text_part.content.parts[0].text == ""
  assert llm_response_empty_text_part.content.parts[0].function_call is None


def test_gemma_functions_before_model_callback_with_function_response():
  from google.adk.agents.callback_context import CallbackContext
  from google.adk.models.google_llm import gemma_functions_before_model_callback

  # Simulate an LlmRequest with a function response
  func_response_data = types.FunctionResponse(
      name="search_web", response={"results": [{"title": "ADK"}]}
  )
  llm_request = LlmRequest(
      model="gemma-3-1b-it",
      contents=[
          types.Content(
              role="model",
              parts=[types.Part(function_response=func_response_data)],
          )
      ],
      config=types.GenerateContentConfig(),
  )

  gemma_functions_before_model_callback(
      mock.MagicMock(spec=CallbackContext), llm_request
  )

  # Assertions: function response converted to user role text content
  assert llm_request.contents
  assert len(llm_request.contents) == 1
  assert llm_request.contents[0].role == "user"
  assert llm_request.contents[0].parts
  assert (
      llm_request.contents[0].parts[0].text
      == 'Invoking tool `search_web` produced: `{"results": [{"title":'
      ' "ADK"}]}`.'
  )
  assert llm_request.contents[0].parts[0].function_response is None
  assert llm_request.contents[0].parts[0].function_call is None


def test_gemma_functions_before_model_callback_with_function_call():
  from google.adk.agents.callback_context import CallbackContext
  from google.adk.models.google_llm import gemma_functions_before_model_callback

  func_call_data = types.FunctionCall(name="get_current_time", args={})
  llm_request = LlmRequest(
      model="gemma-3-1b-it",
      contents=[
          types.Content(
              role="user", parts=[types.Part(function_call=func_call_data)]
          )
      ],
  )

  gemma_functions_before_model_callback(
      mock.MagicMock(spec=CallbackContext), llm_request
  )

  assert len(llm_request.contents) == 1
  assert llm_request.contents[0].role == "model"
  expected_text = func_call_data.model_dump_json(exclude_none=True)
  assert llm_request.contents[0].parts
  got_part = llm_request.contents[0].parts[0]
  assert got_part.text == expected_text
  assert got_part.function_call is None
  assert got_part.function_response is None


def test_gemma_functions_before_model_callback_mixed_content():
  from google.adk.agents.callback_context import CallbackContext
  from google.adk.models.google_llm import gemma_functions_before_model_callback

  func_call = types.FunctionCall(name="get_weather", args={"city": "London"})
  func_response = types.FunctionResponse(
      name="get_weather", response={"temp": "15C"}
  )

  llm_request = LlmRequest(
      model="gemma-3-1b-it",
      contents=[
          types.Content(
              role="user", parts=[types.Part.from_text(text="Hello!")]
          ),
          types.Content(
              role="model", parts=[types.Part(function_call=func_call)]
          ),
          types.Content(
              role="some_function",
              parts=[types.Part(function_response=func_response)],
          ),
          types.Content(
              role="user", parts=[types.Part.from_text(text="How are you?")]
          ),
      ],
  )

  gemma_functions_before_model_callback(
      mock.MagicMock(spec=CallbackContext), llm_request
  )

  # Assertions
  assert len(llm_request.contents) == 4

  # First part: original user text
  assert llm_request.contents[0].role == "user"
  assert llm_request.contents[0].parts
  assert llm_request.contents[0].parts[0].text == "Hello!"

  # Second part: function call converted to model text
  assert llm_request.contents[1].role == "model"
  assert llm_request.contents[1].parts
  assert llm_request.contents[1].parts[0].text == func_call.model_dump_json(
      exclude_none=True
  )

  # Third part: function response converted to user text
  assert llm_request.contents[2].role == "user"
  assert llm_request.contents[2].parts
  assert (
      llm_request.contents[2].parts[0].text
      == 'Invoking tool `get_weather` produced: `{"temp": "15C"}`.'
  )

  # Fourth part: original user text
  assert llm_request.contents[3].role == "user"
  assert llm_request.contents[3].parts
  assert llm_request.contents[3].parts[0].text == "How are you?"


def test_gemma_functions_after_model_callback_markdown_json_block():
  from google.adk.agents.callback_context import CallbackContext
  from google.adk.models.google_llm import gemma_functions_after_model_callback

  # Simulate a response from Gemma with a JSON function call in a markdown block
  json_function_call_str = """
```json
{"name": "search_web", "parameters": {"query": "latest news"}}
```"""
  llm_response = LlmResponse(
      content=Content(
          role="model", parts=[Part.from_text(text=json_function_call_str)]
      )
  )

  gemma_functions_after_model_callback(
      mock.MagicMock(spec=CallbackContext), llm_response
  )

  assert llm_response.content
  assert llm_response.content.parts
  assert len(llm_response.content.parts) == 1
  part = llm_response.content.parts[0]
  assert part.function_call is not None
  assert part.function_call.name == "search_web"
  assert part.function_call.args == {"query": "latest news"}
  assert part.text is None


def test_gemma_functions_after_model_callback_markdown_tool_code_block():
  from google.adk.agents.callback_context import CallbackContext
  from google.adk.models.google_llm import gemma_functions_after_model_callback

  # Simulate a response from Gemma with a JSON function call in a 'tool_code' markdown block
  json_function_call_str = """
Some text before.
```tool_code
{"name": "get_current_time", "parameters": {}}
```
And some text after."""
  llm_response = LlmResponse(
      content=Content(
          role="model", parts=[Part.from_text(text=json_function_call_str)]
      )
  )

  gemma_functions_after_model_callback(
      mock.MagicMock(spec=CallbackContext), llm_response
  )

  assert llm_response.content
  assert llm_response.content.parts
  assert len(llm_response.content.parts) == 1
  part = llm_response.content.parts[0]
  assert part.function_call is not None
  assert part.function_call.name == "get_current_time"
  assert part.function_call.args == {}
  assert part.text is None


def test_gemma_functions_after_model_callback_embedded_json():
  from google.adk.agents.callback_context import CallbackContext
  from google.adk.models.google_llm import gemma_functions_after_model_callback

  # Simulate a response with valid JSON embedded in text
  embedded_json_str = (
      'Please call the tool: {"name": "search_web", "parameters": {"query":'
      ' "new features"}} thanks!'
  )
  llm_response = LlmResponse(
      content=Content(
          role="model", parts=[Part.from_text(text=embedded_json_str)]
      )
  )

  gemma_functions_after_model_callback(
      mock.MagicMock(spec=CallbackContext), llm_response
  )

  assert llm_response.content
  assert llm_response.content.parts
  assert len(llm_response.content.parts) == 1
  part = llm_response.content.parts[0]
  assert part.function_call is not None
  assert part.function_call.name == "search_web"
  assert part.function_call.args == {"query": "new features"}
  assert part.text is None


def test_gemma_functions_after_model_callback_flexible_parsing():
  from google.adk.agents.callback_context import CallbackContext
  from google.adk.models.google_llm import gemma_functions_after_model_callback

  # Test with "function" and "args" keys as supported by GemmaFunctionCallModel
  flexible_json_str = '{"function": "do_something", "args": {"value": 123}}'
  llm_response = LlmResponse(
      content=Content(
          role="model", parts=[Part.from_text(text=flexible_json_str)]
      )
  )

  gemma_functions_after_model_callback(
      mock.MagicMock(spec=CallbackContext), llm_response
  )

  assert llm_response.content
  assert llm_response.content.parts
  assert len(llm_response.content.parts) == 1
  part = llm_response.content.parts[0]
  assert part.function_call is not None
  assert part.function_call.name == "do_something"
  assert part.function_call.args == {"value": 123}
  assert part.text is None


def test_gemma_functions_after_model_callback_last_json_object():
  from google.adk.agents.callback_context import CallbackContext
  from google.adk.models.google_llm import gemma_functions_after_model_callback

  # Simulate a response with multiple JSON objects, ensuring the last valid one is picked
  multiple_json_str = (
      'I thought about {"name": "first_call", "parameters": {"a": 1}} but then'
      ' decided to call: {"name": "second_call", "parameters": {"b": 2}}'
  )
  llm_response = LlmResponse(
      content=Content(
          role="model", parts=[Part.from_text(text=multiple_json_str)]
      )
  )

  gemma_functions_after_model_callback(
      mock.MagicMock(spec=CallbackContext), llm_response
  )

  assert llm_response.content
  assert llm_response.content.parts
  assert len(llm_response.content.parts) == 1
  part = llm_response.content.parts[0]
  assert part.function_call is not None
  assert part.function_call.name == "second_call"
  assert part.function_call.args == {"b": 2}
  assert part.text is None
