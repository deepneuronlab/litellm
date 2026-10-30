"""DNL fork: Azure OpenAI deployments serve the Decisions API through the native proxy lifecycle.

Drives the real proxy app against a mock Azure transport; nothing leaves the process.
"""

import asyncio
import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import httpx

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.azure.decisions.transformation import AzureOpenAIDecisionsConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy import proxy_server

GATEWAY_KEY = "sk-gateway-master"
DECISION_REQUEST = {
    "model": "decisions-model",
    "input": "I was charged twice. Please refund the extra charge.",
    "questions": [
        {"type": "predicate", "name": "refund_requested", "instructions": "Does the customer ask for a refund?"},
        {
            "type": "choice",
            "name": "urgency",
            "instructions": "How urgent is this?",
            "choices": [{"value": "low"}, {"value": "high"}],
        },
    ],
}
AZURE_RESPONSE = {
    "model": "gpt-6-luna",
    "answers": [
        {"type": "predicate", "name": "refund_requested", "probability": 0.95},
        {
            "type": "choice",
            "name": "urgency",
            "choice": "high",
            "probabilities": [{"value": "low", "probability": 0.2}, {"value": "high", "probability": 0.8}],
            "confidence": 0.8,
        },
    ],
    "usage": {
        "input_tokens": 162,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": 3,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 165,
    },
}
ALIASES = (
    "/openai/v1/decisions?api-version=2025-03-01-preview",
    "/openai/decisions?api-version=2024-12-01-preview",
    "/v1/decisions",
    "/decisions",
)


def deployment(name: str, host: str, order: int) -> dict:
    return {
        "model_name": "decisions-model",
        "litellm_params": {
            "model": f"azure/gpt5_series/{name}",
            "api_base": f"https://{host}/",
            "api_key": f"key-{name}",
            "api_version": "2025-03-01-preview",
            "max_completion_tokens": 128000,
            "order": order,
        },
    }


class SuccessLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[dict] = []
        self.logged = asyncio.Event()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.payloads.append(kwargs["standard_logging_object"])
        self.logged.set()


class AzureDecisionsProxyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.azure_requests: list[httpx.Request] = []
        self.status_by_host: dict[str, int] = {}
        self.backend = httpx.AsyncClient(transport=httpx.MockTransport(self.azure))
        self.http_handler = AsyncHTTPHandler(timeout=3)
        await self.http_handler.client.aclose()
        self.http_handler.client = self.backend
        self.router = litellm.Router(
            model_list=[
                deployment("gpt-6-luna-dz", "datazone.openai.azure.com", 1),
                deployment("gpt-6-luna", "global.openai.azure.com", 2),
            ],
            num_retries=0,
            timeout=3,
        )
        self.logger = SuccessLogger()
        self.patches = [
            patch.object(proxy_server, "llm_router", self.router),
            patch.object(proxy_server, "master_key", GATEWAY_KEY),
            patch.object(proxy_server, "general_settings", {}),
            patch.object(proxy_server, "prisma_client", None),
            patch.object(litellm, "callbacks", [self.logger]),
            patch(
                "litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client",
                return_value=self.http_handler,
            ),
        ]
        for item in self.patches:
            item.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy_server.app),
            base_url="http://gateway",
            headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.backend.aclose()
        self.router.reset()
        for item in reversed(self.patches):
            item.stop()

    def azure(self, request: httpx.Request) -> httpx.Response:
        self.azure_requests.append(request)
        status = self.status_by_host.get(request.url.host, 200)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": f"mock Azure {status}", "code": str(status)}})
        return httpx.Response(200, json=AZURE_RESPONSE)

    async def test_every_alias_reaches_the_azure_v1_route_with_the_deployment_key(self):
        for path in ALIASES:
            with self.subTest(path=path):
                self.azure_requests.clear()
                response = await self.client.post(path, json=DECISION_REQUEST)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(
                    [answer["name"] for answer in response.json()["answers"]], ["refund_requested", "urgency"]
                )
                self.assertEqual(response.json()["answers"][0]["probability"], 0.95)
                (sent,) = self.azure_requests
                self.assertEqual(str(sent.url), "https://datazone.openai.azure.com/openai/v1/decisions")
                self.assertEqual(sent.headers["api-key"], "key-gpt-6-luna-dz")
                self.assertNotIn("authorization", sent.headers)
                body = json.loads(sent.content)
                self.assertEqual(body["model"], "gpt-6-luna-dz")
                self.assertEqual(body["input"], DECISION_REQUEST["input"])
                self.assertEqual([q["name"] for q in body["questions"]], ["refund_requested", "urgency"])

    async def test_aliases_require_the_gateway_key(self):
        for path in ALIASES[:2]:
            with self.subTest(path=path):
                response = await self.client.post(path, json=DECISION_REQUEST, headers={"Authorization": ""})
                self.assertEqual(response.status_code, 401, response.text)
        self.assertEqual(self.azure_requests, [])

    async def test_a_failing_order_one_deployment_falls_back_to_order_two(self):
        self.status_by_host["datazone.openai.azure.com"] = 429
        response = await self.client.post("/openai/v1/decisions", json=DECISION_REQUEST)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            [request.url.host for request in self.azure_requests],
            ["datazone.openai.azure.com", "global.openai.azure.com"],
        )
        self.assertEqual(json.loads(self.azure_requests[-1].content)["model"], "gpt-6-luna")

    async def test_azure_client_errors_keep_their_status(self):
        self.status_by_host["datazone.openai.azure.com"] = 400
        self.status_by_host["global.openai.azure.com"] = 400
        response = await self.client.post("/openai/v1/decisions", json=DECISION_REQUEST)
        self.assertEqual(response.status_code, 400, response.text)

    async def test_usage_and_cost_are_logged_against_the_underlying_model(self):
        response = await self.client.post("/openai/v1/decisions", json=DECISION_REQUEST)
        self.assertEqual(response.status_code, 200, response.text)
        await asyncio.wait_for(self.logger.logged.wait(), timeout=5)
        (payload,) = self.logger.payloads
        self.assertEqual((payload["prompt_tokens"], payload["completion_tokens"]), (162, 3))
        self.assertEqual(payload["model_group"], "decisions-model")
        self.assertEqual(payload["api_base"], "https://datazone.openai.azure.com/openai/v1/decisions")
        expected = (
            162 * litellm.model_cost["azure/gpt-6-luna"]["input_cost_per_token"]
            + 3 * (litellm.model_cost["azure/gpt-6-luna"]["output_cost_per_token"])
        )
        self.assertAlmostEqual(payload["response_cost"], expected)


class AzureDecisionsUrlTests(unittest.TestCase):
    def test_resource_url_variants_normalize_to_the_v1_route(self):
        config = AzureOpenAIDecisionsConfig()
        for api_base in (
            "https://res.openai.azure.com",
            "https://res.openai.azure.com/",
            "https://res.openai.azure.com/openai/v1/",
            "https://res.openai.azure.com/openai/deployments/x/chat/completions?api-version=2025-03-01-preview",
        ):
            with self.subTest(api_base=api_base):
                self.assertEqual(
                    config.get_complete_url(api_base, "gpt-6-luna"), "https://res.openai.azure.com/openai/v1/decisions"
                )

    def test_caller_credentials_never_replace_the_deployment_key(self):
        headers = AzureOpenAIDecisionsConfig().validate_environment(
            {"api-key": "caller", "Authorization": "Bearer caller", "x-trace": "1"}, "gpt-6-luna", "deployment"
        )
        self.assertEqual(headers["api-key"], "deployment")
        self.assertNotIn("Authorization", headers)
        self.assertEqual(headers["x-trace"], "1")

    def test_the_gpt5_series_selector_is_not_part_of_the_deployment_name(self):
        config = AzureOpenAIDecisionsConfig()
        self.assertEqual(config.canonical_model("gpt5_series/gpt-6-luna-dz"), "gpt-6-luna-dz")
        self.assertEqual(config.canonical_model("custom-gpt5_series/x"), "custom-gpt5_series/x")


if __name__ == "__main__":
    unittest.main()
