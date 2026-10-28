from typing import TYPE_CHECKING, List, Optional, Tuple

import httpx
from httpx import Response

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import Usage

AZURE_DECISIONS_ENDPOINT = "openai/v1/decisions"

if TYPE_CHECKING:
    from httpx import URL

    from litellm.types.utils import CostResponseTypes


class AzurePassthroughConfig(BasePassthroughConfig):
    def is_streaming_request(self, endpoint: str, request_data: dict) -> bool:
        if endpoint == AZURE_DECISIONS_ENDPOINT:
            return False
        return "stream" in request_data

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        endpoint: str,
        request_query_params: Optional[dict],
        litellm_params: dict,
    ) -> Tuple["URL", str]:
        base_target_url = self.get_api_base(api_base)

        if base_target_url is None:
            raise Exception("Azure api base not found")

        if endpoint == AZURE_DECISIONS_ENDPOINT:
            complete_url = BaseAzureLLM._get_base_azure_url(
                api_base=base_target_url, litellm_params={}, route=endpoint
            )
            return (
                httpx.URL(complete_url).copy_remove_param("api-version"),
                base_target_url,
            )

        litellm_metadata = litellm_params.get("litellm_metadata") or {}
        model_group = litellm_metadata.get("model_group")
        if model_group and model_group in endpoint:
            endpoint = endpoint.replace(model_group, model)

        complete_url = BaseAzureLLM._get_base_azure_url(
            api_base=base_target_url,
            litellm_params=litellm_params,
            route=endpoint,
            default_api_version=litellm_params.get("api_version"),
        )
        return (
            httpx.URL(complete_url),
            base_target_url,
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return BaseAzureLLM._base_validate_azure_environment(
            headers=headers,
            litellm_params=GenericLiteLLMParams(
                **{**litellm_params, "api_key": api_key}
            ),
        )

    @staticmethod
    def get_api_base(
        api_base: Optional[str] = None,
    ) -> Optional[str]:
        return api_base or get_secret_str("AZURE_API_BASE")

    @staticmethod
    def get_api_key(
        api_key: Optional[str] = None,
    ) -> Optional[str]:
        return api_key or get_secret_str("AZURE_API_KEY")

    @staticmethod
    def get_base_model(model: str) -> Optional[str]:
        return model

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        return super().get_models(api_key, api_base)

    def logging_non_streaming_response(
        self,
        model: str,
        custom_llm_provider: str,
        httpx_response: Response,
        request_data: dict,
        logging_obj: Logging,
        endpoint: str,
    ) -> Optional["CostResponseTypes"]:
        from litellm import encoding
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
        from litellm.types.utils import ModelResponse

        if endpoint == AZURE_DECISIONS_ENDPOINT:
            payload = httpx_response.json()
            usage = payload.get("usage") or {}
            input_details = dict(usage.get("input_tokens_details") or {})
            if "cache_write_tokens" in input_details:
                input_details["cache_creation_tokens"] = input_details.pop(
                    "cache_write_tokens"
                )
            return ModelResponse(
                model=payload.get("model", model),
                usage=Usage(
                    prompt_tokens=usage.get("input_tokens", 0),
                    completion_tokens=usage.get("output_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    prompt_tokens_details=input_details or None,
                    completion_tokens_details=usage.get("output_tokens_details"),
                ),
            )

        if "chat/completions" not in endpoint:
            return None

        openai_chat_config = OpenAIGPTConfig()

        litellm_model_response: ModelResponse = openai_chat_config.transform_response(
            model=model,
            messages=[{"role": "user", "content": "no-message-pass-through-endpoint"}],
            raw_response=httpx_response,
            model_response=ModelResponse(),
            logging_obj=logging_obj,
            optional_params={},
            litellm_params={},
            api_key="",
            request_data=request_data,
            encoding=encoding,
        )

        return litellm_model_response
