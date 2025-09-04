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


from __future__ import annotations

import contextlib
from functools import cached_property
import json
import logging
import os
import re
import sys
from typing import Any
from typing import AsyncGenerator
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from google.adk.agents.callback_context import CallbackContext
from google.genai import Client
from google.genai import types
from google.genai.types import Content
from google.genai.types import FunctionDeclaration
from google.genai.types import Part
from pydantic import BaseModel
from pydantic import ValidationError
from pydantic.aliases import AliasChoices
from pydantic.fields import Field
from typing_extensions import override

from .. import version
from ..utils.context_utils import Aclosing
from ..utils.streaming_utils import StreamingResponseAggregator
from ..utils.variant_utils import GoogleLLMVariant
from .base_llm import BaseLlm
from .base_llm_connection import BaseLlmConnection
from .gemini_llm_connection import GeminiLlmConnection
from .llm_response import LlmResponse

if TYPE_CHECKING:
  from .llm_request import LlmRequest

logger = logging.getLogger('google_adk.' + __name__)


class GemmaFunctionCallModel(BaseModel):
  """Flexible Pydantic model for parsing inline Gemma function call responses."""

  name: str = Field(validation_alias=AliasChoices('name', 'function'))
  parameters: dict[str, Any] = Field(
      validation_alias=AliasChoices('parameters', 'args')
  )


_NEW_LINE = '\n'
_EXCLUDED_PART_FIELD = {'inline_data': {'data'}}
_AGENT_ENGINE_TELEMETRY_TAG = 'remote_reasoning_engine'
_AGENT_ENGINE_TELEMETRY_ENV_VARIABLE_NAME = 'GOOGLE_CLOUD_AGENT_ENGINE_ID'


class Gemini(BaseLlm):
  """Integration for Gemini models.

  Attributes:
    model: The name of the Gemini model.
  """

  model: str = 'gemini-2.5-flash'

  retry_options: Optional[types.HttpRetryOptions] = None
  """Allow Gemini to retry failed responses.

  Sample:
  ```python
  from google.genai import types

  # ...

  agent = Agent(
    model=Gemini(
      retry_options=types.HttpRetryOptions(initial_delay=1, attempts=2),
    )
  )
  ```
  """

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    """Provides the list of supported models.

    Returns:
      A list of supported models.
    """

    return [
        r'gemini-.*',
        # model optimizer pattern
        r'model-optimizer-.*',
        # fine-tuned vertex endpoint pattern
        r'projects\/.+\/locations\/.+\/endpoints\/.+',
        # vertex gemini long name
        r'projects\/.+\/locations\/.+\/publishers\/google\/models\/gemini.+',
    ]

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Sends a request to the Gemini model.

    Args:
      llm_request: LlmRequest, the request to send to the Gemini model.
      stream: bool = False, whether to do streaming call.

    Yields:
      LlmResponse: The model response.
    """
    await self._preprocess_request(llm_request)
    self._maybe_append_user_content(llm_request)

    # Handle context caching if configured
    cache_metadata = None
    cache_manager = None
    if llm_request.cache_config:
      from .gemini_context_cache_manager import GeminiContextCacheManager

      cache_manager = GeminiContextCacheManager(self.api_client)
      cache_metadata = await cache_manager.handle_context_caching(llm_request)

    logger.info(
        'Sending out request, model: %s, backend: %s, stream: %s',
        llm_request.model,
        self._api_backend,
        stream,
    )
    logger.debug(_build_request_log(llm_request))

    # Always add tracking headers to custom headers given it will override
    # the headers set in the api client constructor to avoid tracking headers
    # being dropped if user provides custom headers or overrides the api client.
    if llm_request.config:
      if not llm_request.config.http_options:
        llm_request.config.http_options = types.HttpOptions()
      llm_request.config.http_options.headers = self._merge_tracking_headers(
          llm_request.config.http_options.headers
      )

    if stream:
      responses = await self.api_client.aio.models.generate_content_stream(
          model=llm_request.model,
          contents=llm_request.contents,
          config=llm_request.config,
      )

      # for sse, similar as bidi (see receive method in gemini_llm_connection.py),
      # we need to mark those text content as partial and after all partial
      # contents are sent, we send an accumulated event which contains all the
      # previous partial content. The only difference is bidi rely on
      # complete_turn flag to detect end while sse depends on finish_reason.
      aggregator = StreamingResponseAggregator()
      async with Aclosing(responses) as agen:
        async for response in agen:
          logger.debug(_build_response_log(response))
          async with Aclosing(
              aggregator.process_response(response)
          ) as aggregator_gen:
            async for llm_response in aggregator_gen:
              yield llm_response
      if (close_result := aggregator.close()) is not None:
        # Populate cache metadata in the final aggregated response for streaming
        if cache_metadata:
          cache_manager.populate_cache_metadata_in_response(
              close_result, cache_metadata
          )
        yield close_result

    else:
      response = await self.api_client.aio.models.generate_content(
          model=llm_request.model,
          contents=llm_request.contents,
          config=llm_request.config,
      )
      logger.info('Response received from the model.')
      logger.debug(_build_response_log(response))

      llm_response = LlmResponse.create(response)
      if cache_metadata:
        cache_manager.populate_cache_metadata_in_response(
            llm_response, cache_metadata
        )
      yield llm_response

  @cached_property
  def api_client(self) -> Client:
    """Provides the api client.

    Returns:
      The api client.
    """
    return Client(
        http_options=types.HttpOptions(
            headers=self._tracking_headers,
            retry_options=self.retry_options,
        )
    )

  @cached_property
  def _api_backend(self) -> GoogleLLMVariant:
    return (
        GoogleLLMVariant.VERTEX_AI
        if self.api_client.vertexai
        else GoogleLLMVariant.GEMINI_API
    )

  @cached_property
  def _tracking_headers(self) -> dict[str, str]:
    framework_label = f'google-adk/{version.__version__}'
    if os.environ.get(_AGENT_ENGINE_TELEMETRY_ENV_VARIABLE_NAME):
      framework_label = f'{framework_label}+{_AGENT_ENGINE_TELEMETRY_TAG}'
    language_label = 'gl-python/' + sys.version.split()[0]
    version_header_value = f'{framework_label} {language_label}'
    tracking_headers = {
        'x-goog-api-client': version_header_value,
        'user-agent': version_header_value,
    }
    return tracking_headers

  @cached_property
  def _live_api_version(self) -> str:
    if self._api_backend == GoogleLLMVariant.VERTEX_AI:
      # use beta version for vertex api
      return 'v1beta1'
    else:
      # use v1alpha for using API KEY from Google AI Studio
      return 'v1alpha'

  @cached_property
  def _live_api_client(self) -> Client:
    return Client(
        http_options=types.HttpOptions(
            headers=self._tracking_headers, api_version=self._live_api_version
        )
    )

  @contextlib.asynccontextmanager
  async def connect(self, llm_request: LlmRequest) -> BaseLlmConnection:
    """Connects to the Gemini model and returns an llm connection.

    Args:
      llm_request: LlmRequest, the request to send to the Gemini model.

    Yields:
      BaseLlmConnection, the connection to the Gemini model.
    """
    # add tracking headers to custom headers and set api_version given
    # the customized http options will override the one set in the api client
    # constructor
    if (
        llm_request.live_connect_config
        and llm_request.live_connect_config.http_options
    ):
      if not llm_request.live_connect_config.http_options.headers:
        llm_request.live_connect_config.http_options.headers = {}
      llm_request.live_connect_config.http_options.headers.update(
          self._tracking_headers
      )
      llm_request.live_connect_config.http_options.api_version = (
          self._live_api_version
      )

    llm_request.live_connect_config.system_instruction = types.Content(
        role='system',
        parts=[
            types.Part.from_text(text=llm_request.config.system_instruction)
        ],
    )
    llm_request.live_connect_config.tools = llm_request.config.tools
    logger.info('Connecting to live with llm_request:%s', llm_request)
    async with self._live_api_client.aio.live.connect(
        model=llm_request.model, config=llm_request.live_connect_config
    ) as live_session:
      yield GeminiLlmConnection(live_session)

  async def _adapt_computer_use_tool(self, llm_request: LlmRequest) -> None:
    """Adapt the google computer use predefined functions to the adk computer use toolset."""

    from ..tools.computer_use.computer_use_toolset import ComputerUseToolset

    async def convert_wait_to_wait_5_seconds(wait_func):
      async def wait_5_seconds():
        return await wait_func(5)

      return wait_5_seconds

    await ComputerUseToolset.adapt_computer_use_tool(
        'wait', convert_wait_to_wait_5_seconds, llm_request
    )

  async def _preprocess_request(self, llm_request: LlmRequest) -> None:

    if self._api_backend == GoogleLLMVariant.GEMINI_API:
      # Using API key from Google AI Studio to call model doesn't support labels.
      if llm_request.config:
        llm_request.config.labels = None

      if llm_request.contents:
        for content in llm_request.contents:
          if not content.parts:
            continue
          for part in content.parts:
            _remove_display_name_if_present(part.inline_data)
            _remove_display_name_if_present(part.file_data)

    # Initialize config if needed
    if llm_request.config and llm_request.config.tools:
      # Check if computer use is configured
      for tool in llm_request.config.tools:
        if (
            isinstance(tool, (types.Tool, types.ToolDict))
            and hasattr(tool, 'computer_use')
            and tool.computer_use
        ):
          llm_request.config.system_instruction = None
          await self._adapt_computer_use_tool(llm_request)

  def _merge_tracking_headers(self, headers: dict[str, str]) -> dict[str, str]:
    """Merge tracking headers to the given headers."""
    headers = headers or {}
    for key, tracking_header_value in self._tracking_headers.items():
      custom_value = headers.get(key, None)
      if not custom_value:
        headers[key] = tracking_header_value
        continue

      # Merge tracking headers with existing headers and avoid duplicates.
      value_parts = tracking_header_value.split(' ')
      for custom_value_part in custom_value.split(' '):
        if custom_value_part not in value_parts:
          value_parts.append(custom_value_part)
      headers[key] = ' '.join(value_parts)
    return headers


def _build_function_declaration_log(
    func_decl: types.FunctionDeclaration,
) -> str:
  param_str = '{}'
  if func_decl.parameters and func_decl.parameters.properties:
    param_str = str({
        k: v.model_dump(exclude_none=True)
        for k, v in func_decl.parameters.properties.items()
    })
  return_str = ''
  if func_decl.response:
    return_str = '-> ' + str(func_decl.response.model_dump(exclude_none=True))
  return f'{func_decl.name}: {param_str} {return_str}'


def _build_request_log(req: LlmRequest) -> str:
  function_decls: list[types.FunctionDeclaration] = cast(
      list[types.FunctionDeclaration],
      req.config.tools[0].function_declarations if req.config.tools else [],
  )
  function_logs = (
      [
          _build_function_declaration_log(func_decl)
          for func_decl in function_decls
      ]
      if function_decls
      else []
  )
  contents_logs = [
      content.model_dump_json(
          exclude_none=True,
          exclude={
              'parts': {
                  i: _EXCLUDED_PART_FIELD for i in range(len(content.parts))
              }
          },
      )
      for content in req.contents
  ]

  return f"""
LLM Request:
-----------------------------------------------------------
System Instruction:
{req.config.system_instruction}
-----------------------------------------------------------
Contents:
{_NEW_LINE.join(contents_logs)}
-----------------------------------------------------------
Functions:
{_NEW_LINE.join(function_logs)}
-----------------------------------------------------------
"""


def _build_response_log(resp: types.GenerateContentResponse) -> str:
  function_calls_text = []
  if function_calls := resp.function_calls:
    for func_call in function_calls:
      function_calls_text.append(
          f'name: {func_call.name}, args: {func_call.args}'
      )
  return f"""
LLM Response:
-----------------------------------------------------------
Text:
{resp.text}
-----------------------------------------------------------
Function calls:
{_NEW_LINE.join(function_calls_text)}
-----------------------------------------------------------
Raw response:
{resp.model_dump_json(exclude_none=True)}
-----------------------------------------------------------
"""


def _remove_display_name_if_present(
    data_obj: Union[types.Blob, types.FileData, None],
):
  """Sets display_name to None for the Gemini API (non-Vertex) backend.

  This backend does not support the display_name parameter for file uploads,
  so it must be removed to prevent request failures.
  """
  if data_obj and data_obj.display_name:
    data_obj.display_name = None


class Gemma(Gemini):
  """Integration for Gemma models exposed via the Gemini API.

  Only Gemma 3 models are supported at this time.

  For full documentation, see: https://ai.google.dev/gemma/docs/core/

  NOTE: Gemma does **NOT** support system instructions. Any system instructions
  will be replaced with an initial *user* prompt in the LLM request. If system
  instructions change over the course of agent execution, the initial content
  **SHOULD** be replaced. Special care is warranted here.
  See: https://ai.google.dev/gemma/docs/core/prompt-structure#system-instructions

  NOTE: Gemma's function calling support is limited. It does not have full access to the
  same built-in tools as Gemini. It also does not have special API support for tools and
  functions. Rather, tools must be passed in via a `user` prompt, and extracted from model
  responses based on approximate shape. For agent developments, please use the provided
  `gemma_functions_before_model_callback` and `gemma_functions_after_model_callback` methods.

  NOTE: Vertex AI API support for Gemma is not currently included. This **ONLY** supports
  usage via the Gemini API.
  """

  model: str = 'gemma-3-27b-it'  # Others: [gemma-3-1b-it, gemma-3-4b-it, gemma-3-12b-it]

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    """Provides the list of supported models.

    Returns:
    A list of supported models.
    """

    return [
        r'gemma-3.*',
    ]

  @cached_property
  def _api_backend(self) -> GoogleLLMVariant:
    return GoogleLLMVariant.GEMINI_API

  @override
  async def _preprocess_request(self, llm_request: LlmRequest) -> None:
    if system_instruction := llm_request.config.system_instruction:
      contents = llm_request.contents
      instruction_content = Content(
          role='user', parts=[Part.from_text(text=system_instruction)]
      )

      # NOTE: if history is preserved, we must include the system instructions ONLY once at the beginning
      # of any chain of contents.
      if contents:
        if contents[0] != instruction_content:
          # only prepend if it hasn't already been done
          llm_request.contents = [instruction_content] + contents

      llm_request.config.system_instruction = None

    return await super()._preprocess_request(llm_request)

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Sends a request to the Gemma model.

    Args:
      llm_request: LlmRequest, the request to send to the Gemini model.
      stream: bool = False, whether to do streaming call.

    Yields:
      LlmResponse: The model response.
    """
    # print(f'{llm_request=}')
    assert llm_request.model.startswith('gemma-'), (
        f'Requesting a non-Gemma model ({llm_request.model}) with the Gemma LLM'
        ' is not supported.'
    )

    async for response in super().generate_content_async(llm_request, stream):
      yield response


def _convert_content_parts_for_gemma(
    content_item: Content,
) -> tuple[list[Part], bool, bool]:
  """Converts function call/response parts within a content item to text parts.

  Args:
    content_item: The original Content item.

  Returns:
    A tuple containing:
      - A list of new Part objects with function calls/responses converted to text.
      - A boolean indicating if any function response parts were found.
      - A boolean indicating if any function call parts were found.
  """
  new_parts: list[Part] = []
  has_function_response_part = False
  has_function_call_part = False

  for part in content_item.parts:
    if func_response := part.function_response:
      has_function_response_part = True
      response_text = (
          f'Invoking tool `{func_response.name}` produced:'
          f' `{json.dumps(func_response.response)}`.'
      )
      new_parts.append(Part.from_text(text=response_text))
    elif func_call := part.function_call:
      has_function_call_part = True
      new_parts.append(
          Part.from_text(text=func_call.model_dump_json(exclude_none=True))
      )
    else:
      new_parts.append(part)
  return new_parts, has_function_response_part, has_function_call_part


def _build_gemma_function_system_instruction(
    function_declarations: list[FunctionDeclaration],
) -> str:
  """Constructs the system instruction string for Gemma function calling."""
  if not function_declarations:
    return ''

  system_instruction_prefix = 'You have access to the following functions:\n['
  instruction_parts = []
  for func in function_declarations:
    instruction_parts.append(func.model_dump_json(exclude_none=True))

  separator = ',\n'
  system_instruction = (
      f'{system_instruction_prefix}{separator.join(instruction_parts)}\n]\n'
  )

  system_instruction += (
      'When you call a function, you MUST respond in the format of: '
      """{"name": function name, "parameters": dictionary of argument name and its value}\n"""
      'When you call a function, you MUST NOT include any other text in the'
      ' response.\n'
  )
  return system_instruction


def gemma_functions_before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
):
  """Translates function calls and responses to the Gemma-supported interaction model.

  NOTE: Gemma is **ONLY** able to handle external function declarations in a tool. It does NOT
  have access to the internal Gemini tools (including Google and Enterprise Search, URL Context, etc.).
  If the LLM Request includes those tools, they will be ignored and dropped from the request sent to
  the model.
  """

  if llm_request.model is None or not llm_request.model.startswith('gemma-3'):
    return

  # Iterate through the existing contents to find and convert function calls and responses
  # from text parts, as Gemma models don't directly support function calling.
  new_contents: list[Content] = []
  for content_item in llm_request.contents:
    (
        new_parts_for_content,
        has_function_response_part,
        has_function_call_part,
    ) = _convert_content_parts_for_gemma(content_item)

    if has_function_response_part:
      if new_parts_for_content:
        new_contents.append(Content(role='user', parts=new_parts_for_content))
    elif has_function_call_part:
      if new_parts_for_content:
        new_contents.append(Content(role='model', parts=new_parts_for_content))
    else:
      new_contents.append(content_item)

  llm_request.contents = new_contents

  if not llm_request.config.tools:
    return

  all_function_declarations: list[FunctionDeclaration] = []
  for tool_item in llm_request.config.tools:
    if isinstance(tool_item, types.Tool) and tool_item.function_declarations:
      all_function_declarations.extend(tool_item.function_declarations)

  if all_function_declarations:
    system_instruction = _build_gemma_function_system_instruction(
        all_function_declarations
    )
    llm_request.append_instructions([system_instruction])

  llm_request.config.tools = []


def gemma_functions_after_model_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
):
  """Translates function calls and responses to the Gemma-supported interaction model.

  Model function calls are attempted to be recognized in text responses and extracted into
  the objects that can be exploited by `Agents`. Some flexibility in parsing is provided
  in an attempt to improve model function in agentic systems.
  """
  if not llm_response.content:
    return

  if not llm_response.content.parts:
    return

  if len(llm_response.content.parts) > 1:
    return

  response_text = llm_response.content.parts[0].text
  if not response_text:
    return

  try:
    json_candidate = None

    markdown_code_block_pattern = re.compile(
        r'```(?:(json|tool_code))?\s*(.*?)\s*```', re.DOTALL
    )
    block_match = markdown_code_block_pattern.search(response_text)

    if block_match:
      json_candidate = block_match.group(2).strip()
    else:
      found, json_text = _get_last_valid_json_substring(response_text)
      if found:
        json_candidate = json_text

    if not json_candidate:
      return

    function_call_parsed = GemmaFunctionCallModel.model_validate_json(
        json_candidate
    )
    function_call = types.FunctionCall(
        name=function_call_parsed.name,
        args=function_call_parsed.parameters,
    )
    function_call_part = Part(function_call=function_call)
    llm_response.content.parts = [function_call_part]
  except (json.JSONDecodeError, ValidationError) as e:
    logger.debug(
        f'Error attempting to parse JSON into function call. Leaving as text'
        f' response. %s',
        e,
    )
  except Exception as e:
    logger.warning('Error processing Gemma function call response: %s', e)


def _get_last_valid_json_substring(text: str) -> tuple[bool, str | None]:
  """Attempts to find and return the last valid JSON object in a string.

  This function is designed to extract JSON that might be embedded in a larger
  text, potentially with introductory or concluding remarks. It will always chose
  the last block of valid json found within the supplied text (if it exists).

  Args:
    text: The input string to search for JSON objects.

  Returns:
    A tuple:
      - bool: True if a valid JSON substring was found, False otherwise.
      - str | None: The last valid JSON substring found, or None if none was
        found.
  """
  decoder = json.JSONDecoder()
  last_json_str = None
  start_pos = 0
  while start_pos < len(text):
    try:
      first_brace_index = text.index('{', start_pos)
      _, end_index = decoder.raw_decode(text[first_brace_index:])
      last_json_str = text[first_brace_index : first_brace_index + end_index]
      start_pos = first_brace_index + end_index
    except json.JSONDecodeError:
      start_pos = first_brace_index + 1
    except ValueError:
      break

  if last_json_str:
    return True, last_json_str
  return False, None
