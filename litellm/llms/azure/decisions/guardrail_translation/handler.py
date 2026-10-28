"""Translate Decisions text for unified guardrails without converting its wire schema."""

from typing import Any, List, Literal, Optional, Tuple, Union

import httpx
from fastapi import HTTPException

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.types.utils import GenericGuardrailAPIInputs


class AzureDecisionsHandler(BaseTranslation):
    @staticmethod
    def _collect_texts(
        value: Union[dict, list], texts: List[str], locations: List[Tuple[Any, Any]]
    ) -> None:
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for key, item in items:
            if isinstance(value, dict) and key == "type" and not isinstance(item, str):
                raise HTTPException(400, "Unsupported Decisions discriminator shape")
            if isinstance(item, str):
                texts.append(item)
                locations.append((value, key))
            elif isinstance(item, (dict, list)):
                AzureDecisionsHandler._collect_texts(item, texts, locations)
            elif item is not None and not isinstance(item, (bool, int, float)):
                raise HTTPException(400, "Unsupported Decisions guardrail content")

    async def _apply_texts(
        self,
        content: Union[dict, list],
        guardrail: CustomGuardrail,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any],
    ) -> bool:
        texts: List[str] = []
        locations: List[Tuple[Any, Any]] = []
        self._collect_texts(content, texts, locations)
        inputs = GenericGuardrailAPIInputs(texts=texts)
        if request_data.get("model"):
            inputs["model"] = request_data["model"]
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type=input_type,
            logging_obj=logging_obj,
        )
        replacements = result.get("texts") if isinstance(result, dict) else None
        if (
            not isinstance(replacements, list)
            or len(replacements) != len(locations)
            or any(not isinstance(text, str) for text in replacements)
        ):
            raise HTTPException(400, "Invalid Decisions guardrail text translation")
        changed = any(
            container[key] != text
            for (container, key), text in zip(locations, replacements)
        )
        for (container, key), text in zip(locations, replacements):
            # A policy may reject schema strings, but cannot change wire discriminators.
            if isinstance(container, dict) and key == "type" and container[key] != text:
                raise HTTPException(
                    400, "A Decisions guardrail cannot rewrite a discriminator"
                )
            container[key] = text
        return changed

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: CustomGuardrail,
        litellm_logging_obj: Optional[Any] = None,
    ) -> Any:
        # Non-text/multimodal input must not silently evade a selected text policy.
        if (
            not isinstance(data.get("input"), str)
            or not isinstance(data.get("questions"), list)
            or any(not isinstance(question, dict) for question in data["questions"])
        ):
            raise HTTPException(400, "Unsupported Decisions guardrail input shape")
        content = {"input": data["input"], "questions": data["questions"]}
        await self._apply_texts(
            content, guardrail_to_apply, data, "request", litellm_logging_obj
        )
        data.update(content)
        return data

    async def process_output_response(
        self,
        response: Any,
        guardrail_to_apply: CustomGuardrail,
        litellm_logging_obj: Optional[Any] = None,
        user_api_key_dict: Optional[Any] = None,
        request_data: Optional[dict] = None,
    ) -> Any:
        if not isinstance(response, httpx.Response):
            raise HTTPException(400, "Unsupported Decisions guardrail response shape")
        try:
            payload = response.json()
        except ValueError as error:
            raise HTTPException(
                400, "Invalid Decisions guardrail JSON response"
            ) from error
        if not isinstance(payload, dict) or not isinstance(
            payload.get("answers"), list
        ):
            raise HTTPException(400, "Unsupported Decisions guardrail answers shape")
        if any(not isinstance(answer, dict) for answer in payload["answers"]):
            raise HTTPException(400, "Unsupported Decisions guardrail answer shape")
        changed = await self._apply_texts(
            payload["answers"],
            guardrail_to_apply,
            request_data if request_data is not None else {},
            "response",
            litellm_logging_obj,
        )
        if not changed:
            return response
        # httpx decodes response.content; a rewritten body must drop stale encoding/length.
        headers = dict(response.headers)
        headers.pop("content-length", None)
        headers.pop("content-encoding", None)
        translated = httpx.Response(
            response.status_code,
            json=payload,
            headers=headers,
            request=getattr(response, "_request", None),
            extensions=response.extensions,
        )
        if hasattr(response, "_hidden_params"):
            setattr(translated, "_hidden_params", getattr(response, "_hidden_params"))
        return translated
