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

from functools import cached_property
import json
import logging
from typing import Any
from typing import AsyncGenerator

from google.adk.models.google_llm import Gemini
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.utils.variant_utils import GoogleLLMVariant
from google.genai import types
from google.genai.types import Content
from google.genai.types import FunctionDeclaration
from google.genai.types import Part
import litellm
from pydantic import AliasChoices
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError
from typing_extensions import override

logger = logging.getLogger("google_adk." + __name__)


class Gemma3FunctionCallModel(BaseModel):
  """Flexible Pydantic model for parsing inline Gemma function call responses."""

  name: str = Field(validation_alias=AliasChoices("name", "function"))
  parameters: dict[str, Any] = Field(
      validation_alias=AliasChoices("parameters", "args")
  )


class Gemma3GeminiAPI(Gemini):
  """Integration for Gemma 3 models exposed via the Gemini API.

  Only the larger Gemma 3 model sizes are supported (12b, 27b) as
  function calling support on the smaller models is not consistent
  enough for basic agent usage.

  For full documentation, see: https://ai.google.dev/gemma/docs/core/

  NOTE: Gemma's function calling support is limited. It does not have full access to the
  same built-in tools as Gemini. It also does not have special API support for tools and
  functions. Rather, tools must be passed in via a `user` prompt, and extracted from model
  responses based on approximate shape.

  NOTE: Vertex AI API support for Gemma is not currently included. This **ONLY** supports
  usage via the Gemini API.
  """

  def __init__(self, model: str = "gemma-3-12b-it", **kwargs):
    if model not in self.supported_models():
      raise ValueError(
          f"Model '{model}' not supported. Use one of:"
          f" {self.supported_models()=}"
      )
    super().__init__(model=model, **kwargs)

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    """Provides the list of supported models.

    Returns:
    A list of supported models.
    """

    return [
        "gemma-3-12b-it",
        "gemma-3-27b-it",
    ]

  @cached_property
  def _api_backend(self) -> GoogleLLMVariant:
    return GoogleLLMVariant.GEMINI_API

  @override
  async def _preprocess_request(self, llm_request: LlmRequest) -> None:
    _move_function_calls_into_system_instruction(llm_request=llm_request)

    if system_instruction := llm_request.config.system_instruction:
      contents = llm_request.contents
      instruction_content = Content(
          role="user", parts=[Part.from_text(text=system_instruction)]
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
    """Sends a request to the Gemma model via Gemini API.

    Args:
      llm_request: LlmRequest, the request to send to the Gemini model.
      stream: bool = False, whether to do streaming call.

    Yields:
      LlmResponse: The model response.
    """
    async for response in super().generate_content_async(llm_request, stream):
      _extract_function_calls_from_response(response)
      yield response


class Gemma3Ollama(LiteLlm):
  """Integration for Gemma 3 models exposed via Ollama.

  Only the larger Gemma 3 model sizes are supported (12b, 27b) as
  function calling support on the smaller models is not consistent
  enough for basic agent usage.
  """

  def __init__(self, model: str = "ollama/gemma3:12b", **kwargs):
    if model not in self.supported_models():
      raise ValueError(
          f"Model '{model}' not supported. Use one of:"
          f" {self.supported_models()=}"
      )
    _register_gemma_prompt_template(model)
    super().__init__(model, **kwargs)

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    """Provides the list of supported models.

    Returns:
      A list of supported models.
    """
    return ["ollama/gemma3:12b", "ollama/gemma3:27b"]

  def _preprocess_request(self, llm_request: LlmRequest) -> None:
    _move_function_calls_into_system_instruction(llm_request=llm_request)

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Sends a request to the Gemma model hosted on Ollama via LiteLLM integration.

    Args:
      llm_request: LlmRequest, the request to send to the Gemini model.
      stream: bool = False, whether to do streaming call.

    Yields:
      LlmResponse: The model response.
    """
    self._preprocess_request(llm_request)
    async for response in super().generate_content_async(llm_request, stream):
      _extract_function_calls_from_response(response)
      yield response


def _move_function_calls_into_system_instruction(llm_request: LlmRequest):
  if llm_request.model is None or not (
      "gemma3" in llm_request.model or llm_request.model.startswith("gemma-3")
  ):
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
        new_contents.append(Content(role="user", parts=new_parts_for_content))
    elif has_function_call_part:
      if new_parts_for_content:
        new_contents.append(Content(role="model", parts=new_parts_for_content))
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


def _extract_function_calls_from_response(llm_response: LlmResponse):
  if llm_response.partial or (llm_response.turn_complete is True):
    return

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
    import instructor

    json_candidate = instructor.utils.extract_json_from_codeblock(response_text)

    if not json_candidate:
      return

    function_call_parsed = Gemma3FunctionCallModel.model_validate_json(
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
        f"Error attempting to parse JSON into function call. Leaving as text"
        f" response. %s",
        e,
    )
  except Exception as e:
    logger.warning("Error processing Gemma function call response: %s", e)


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
          f"Invoking tool `{func_response.name}` produced:"
          f" `{json.dumps(func_response.response)}`."
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
    return ""

  system_instruction_prefix = "You have access to the following functions:\n["
  instruction_parts = []
  for func in function_declarations:
    instruction_parts.append(func.model_dump_json(exclude_none=True))

  separator = ",\n"
  system_instruction = (
      f"{system_instruction_prefix}{separator.join(instruction_parts)}\n]\n"
  )

  system_instruction += (
      "When you call a function, you MUST respond in the format of: "
      """{"name": function name, "parameters": dictionary of argument name and its value}\n"""
      "When you call a function, you MUST NOT include any other text in the"
      " response.\n"
  )
  return system_instruction


def _register_gemma_prompt_template(model: str):
  litellm.register_prompt_template(
      model=model,
      roles={
          "system": {
              "pre_message": "<start_of_turn>user:\n",
              "post_message": "<end_of_turn>\n",
          },
          "user": {
              "pre_message": "<start_of_turn>user:\n",
              "post_message": "<end_of_turn>\n",
          },
          "assistant": {
              "pre_message": "<start_of_turn>model:\n",
              "post_message": "<end_of_turn>\n",
          },
      },
      final_prompt_value="<start_of_turn>model:\n",
  )
