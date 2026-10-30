from collections.abc import Mapping
from typing import Final

import httpx

import litellm
from litellm.llms.azure.chat.gpt_5_transformation import AzureOpenAIGPT5Config
from litellm.llms.openai.decisions.transformation import OpenAIDecisionsConfig
from litellm.types.decisions import DecisionsIRResponse

_CALLER_AUTH_HEADERS: Final = frozenset({"api-key", "authorization"})


class AzureOpenAIDecisionsConfig(OpenAIDecisionsConfig):
    """Azure OpenAI resource deployments on the v1 Decisions surface; model is the deployment name.

    Same wire format as OpenAI's /v1/decisions. The v1 surface routes by the body's model, so neither a
    /deployments/ path nor an api-version query is sent, and the key goes in Azure's api-key header.
    """

    path = "/openai/v1/decisions"
    api_key_env = ("AZURE_API_KEY", "AZURE_OPENAI_API_KEY")
    api_base_env = ("AZURE_API_BASE", "AZURE_OPENAI_ENDPOINT")

    def get_default_api_base(self) -> str | None:
        return None

    def missing_api_base_message(self, custom_llm_provider: str) -> str:
        return "Missing AZURE_API_BASE - set AZURE_API_BASE or pass api_base, e.g. https://<resource>.openai.azure.com"

    def resolve_api_base(self, api_base: str | None) -> str | None:
        return api_base or self._first_secret(self.api_base_env)

    def resolve_api_key(self, api_key: str | None) -> str | None:
        return api_key or litellm.azure_key or self._first_secret(self.api_key_env)

    def canonical_model(self, model: str) -> str:
        # gpt5_series/ selects LiteLLM's GPT-5 handling; it is not part of the Azure deployment name.
        return model.removeprefix(AzureOpenAIGPT5Config.GPT5_SERIES_ROUTE)

    def pricing_model(self, response: DecisionsIRResponse, model: str) -> str:
        # Deployment names (gpt-6-luna-dz) are not cost-map keys; Azure reports the underlying model.
        return response.model or model

    def validate_environment(self, headers: Mapping[str, str], model: str, api_key: str | None) -> dict[str, str]:
        forwarded: Final = {name: value for name, value in headers.items() if name.lower() not in _CALLER_AUTH_HEADERS}
        return {
            **super().validate_environment(forwarded, model, None),
            **({"api-key": api_key} if api_key is not None else {}),
        }

    def get_complete_url(self, api_base: str, model: str) -> str:
        url: Final = httpx.URL(api_base)
        openai_path_start: Final = url.path.find("/openai")
        resource_path: Final = (url.path if openai_path_start == -1 else url.path[:openai_path_start]).rstrip("/")
        return str(url.copy_with(path=f"{resource_path}{self.path}", query=None, fragment=None))
