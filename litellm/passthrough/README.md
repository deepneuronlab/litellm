This makes it easier to pass through requests to the LLM APIs.

E.g. Route to VLLM's `/classify` endpoint:


## SDK (Basic)

```python
import litellm


response = litellm.llm_passthrough_route(
    model="hosted_vllm/papluca/xlm-roberta-base-language-detection",
    method="POST",
    endpoint="classify",
    api_base="http://localhost:8090",
    api_key=None,
    json={
        "model": "swapped-for-litellm-model",
        "input": "Hello, world!",
    }
)

print(response)
```

## SDK (Router)

```python
import asyncio
from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "roberta-base-language-detection",
            "litellm_params": {
                "model": "hosted_vllm/papluca/xlm-roberta-base-language-detection",
                "api_base": "http://localhost:8090", 
            }
        }
    ]
)

request_data = {
    "model": "roberta-base-language-detection",
    "method": "POST",
    "endpoint": "classify",
    "api_base": "http://localhost:8090",
    "api_key": None,
    "json": {
        "model": "roberta-base-language-detection",
        "input": "Hello, world!",
    }
}

async def main():
    response = await router.allm_passthrough_route(**request_data)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

## PROXY 

1. Setup config.yaml 

```yaml
model_list:
  - model_name: roberta-base-language-detection
    litellm_params:
      model: hosted_vllm/papluca/xlm-roberta-base-language-detection
      api_base: http://localhost:8090
```

2. Run the proxy

```bash
litellm proxy --config config.yaml

# RUNNING on http://localhost:4000
```

3. Use the proxy

```bash
curl -X POST http://localhost:4000/vllm/classify \
-H "Content-Type: application/json" \
-H "Authorization: Bearer <your-api-key>" \
-d '{"model": "roberta-base-language-detection", "input": "Hello, world!"}' \
```

## Native Azure Decisions API

The DNL fork serves `POST /decisions`, `/v1/decisions`, `/openai/decisions`, and
`/openai/v1/decisions` before the OpenAI catch-all. AzureOpenAI clients can use
the gateway as their Azure endpoint; OpenAI clients can use the gateway's `/v1`
base URL. Requests contain a configured router model group, `input`, and
`questions`, for example:

```bash
curl http://localhost:4000/v1/decisions \
  -H 'Authorization: Bearer sk-your-virtual-key' \
  -H 'Content-Type: application/json' \
  -d '{"model":"decisions-model","input":"Please refund the duplicate charge.","questions":[{"type":"predicate","name":"refund_requested","instructions":"Does the customer request a refund?"}]}'
```

Only Azure deployments supporting Decisions are accepted. This endpoint does
not translate Decisions requests to chat or Responses, and does not support
streaming (`stream: false` is accepted). Model availability and request-schema
validation remain Azure's responsibility. Unknown model groups and malformed
metadata are rejected with HTTP 400 before reaching Azure.

The handler uses the normal authenticated proxy pre-call, during-call, success,
failure, and response-header hooks. Global, key, router, and team-public model
aliases apply; another team's private deployment remains inaccessible.
Use `litellm_metadata` for proxy logging/session metadata; it is sanitized and
attributed to the authenticated key, and is not forwarded to Azure. Provider
request fields cannot override router credentials, providers, or destinations.
Administrator-configured team/project credential overrides still apply.

Unified `apply_guardrail` policies scan Decisions input, question instructions
and choices, and returned answer strings. Policies may reject or mask content;
numeric probabilities and usage are not rewritten. Unknown selected policies,
unsupported policy content shapes, and invalid text translations fail closed.

Router deployment ordering, retry policies, and fallbacks apply, including
DataZone order 1 followed by Global order 2 when configured. Authenticated
regional restrictions remain enforced through fallback, including non-chat
requests without optional token prechecks. Each attempt honors its effective
HTTP timeout without changing the shared client's default.

The outbound route is `/openai/v1/decisions`; incoming and deployment
`api-version` values are not forwarded. Azure response bodies, including answers
and usage, are returned unchanged unless a selected policy masks answer text.
Standard success logging preserves input/output totals, cache-read/cache-write
details, and reasoning/prediction token details for cost accounting. Azure HTTP
error statuses are preserved for routing and the client.

The underlying passthrough failure callbacks still receive raw
`httpx.HTTPStatusError` before the endpoint maps it to a status-aware exception.
As in the previous overlay, this can leave failure metric statuses unset and
does not establish per-deployment cooldown behavior. Order fallback and client
error statuses are covered separately by the regression suite.

Implementation and regression tests belong to this fork, not an infrastructure
image-layer patch. Build the native source using this repository's root
`Dockerfile`. Image build/publication remains an explicit manual operation:
publish a new immutable image tag to the deployment registry and record its
digest before changing any infrastructure image reference. This feature does
not add image-build automation or perform a cluster rollout.

The regression suites require Python with LiteLLM's proxy dependencies installed
(the native proxy image already contains them), but no database, provider
credentials, or external Azure connection. Deadline tests use a local TCP server:

```bash
PYTHONPATH=. LITELLM_LOCAL_MODEL_COST_MAP=True \
  python -m unittest discover -s tests/test_litellm/proxy -p 'test_decisions*.py' -v
```


# How to add a provider for passthrough

See [VLLMModelInfo](https://github.com/BerriAI/litellm/blob/main/litellm/llms/vllm/common_utils.py) for an example.

1. Inherit from BaseModelInfo

```python
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo

class VLLMModelInfo(BaseLLMModelInfo):
    pass
```

2. Register the provider in the ProviderConfigManager.get_provider_model_info

```python
from litellm.utils import ProviderConfigManager
from litellm.types.utils import LlmProviders

provider_config = ProviderConfigManager.get_provider_model_info(
    model="my-test-model", provider=LlmProviders.VLLM
)

print(provider_config)
```