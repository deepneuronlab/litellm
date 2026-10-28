"""Consumer-visible Decisions pricing and real-HTTP deadline regressions."""

import asyncio
import json
import os
import time
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import httpx
import litellm
from litellm.llms.azure.passthrough.transformation import (
    AZURE_DECISIONS_ENDPOINT,
    AzurePassthroughConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy import proxy_server


class DecisionsUsageAccountingTests(unittest.TestCase):
    rates = {
        "input_cost_per_token": 0.002,
        "output_cost_per_token": 0.004,
        "cache_read_input_token_cost": 0.0002,
        "cache_creation_input_token_cost": 0.003,
    }

    def transform_usage(self, usage, model="gpt-5"):
        return AzurePassthroughConfig().logging_non_streaming_response(
            model=model,
            custom_llm_provider="azure",
            httpx_response=httpx.Response(
                200, json={"model": model, "answers": [], "usage": usage}
            ),
            request_data={},
            logging_obj=MagicMock(),
            endpoint=AZURE_DECISIONS_ENDPOINT,
        )

    def price(self, response):
        return litellm.completion_cost(
            completion_response=response,
            model="azure/gpt-5",
            custom_llm_provider="azure",
            custom_cost_per_token=self.rates,
        )

    def test_cached_input_is_billed_at_discounted_rate(self):
        usage = {"input_tokens": 1000, "output_tokens": 40, "total_tokens": 1040}
        uncached_cost = self.price(self.transform_usage(usage))
        usage["input_tokens_details"] = {"cached_tokens": 600, "cache_write_tokens": 0}
        cached_cost = self.price(self.transform_usage(usage))
        self.assertAlmostEqual(uncached_cost, 1000 * 0.002 + 40 * 0.004)
        self.assertAlmostEqual(cached_cost, 400 * 0.002 + 600 * 0.0002 + 40 * 0.004)
        self.assertLess(cached_cost, uncached_cost)

    def test_native_azure_prices_discount_cached_input(self):
        usage = {"input_tokens": 1000, "output_tokens": 0, "total_tokens": 1000}
        uncached_response = self.transform_usage(usage, model="gpt-5.5")
        usage["input_tokens_details"] = {"cached_tokens": 900, "cache_write_tokens": 0}
        cached_response = self.transform_usage(usage, model="gpt-5.5")
        uncached_cost = litellm.completion_cost(
            completion_response=uncached_response,
            model="azure/gpt-5.5",
            custom_llm_provider="azure",
        )
        cached_cost = litellm.completion_cost(
            completion_response=cached_response,
            model="azure/gpt-5.5",
            custom_llm_provider="azure",
        )
        self.assertAlmostEqual(uncached_cost, 0.005)
        self.assertAlmostEqual(cached_cost, 0.00095)
        self.assertLess(cached_cost, uncached_cost)

    def test_cache_writes_and_reasoning_use_native_usage_conventions(self):
        response = self.transform_usage(
            {
                "input_tokens": 1000,
                "input_tokens_details": {
                    "cached_tokens": 600,
                    "cache_write_tokens": 100,
                    "audio_tokens": 0,
                },
                "output_tokens": 40,
                "output_tokens_details": {
                    "reasoning_tokens": 30,
                    "accepted_prediction_tokens": 2,
                    "rejected_prediction_tokens": 1,
                },
                "total_tokens": 1040,
            }
        )
        self.assertAlmostEqual(
            self.price(response),
            300 * 0.002 + 600 * 0.0002 + 100 * 0.003 + 40 * 0.004,
        )
        self.assertEqual(
            response.usage.prompt_tokens_details.cache_creation_tokens, 100
        )
        self.assertEqual(response.usage.prompt_tokens_details.cached_tokens, 600)
        self.assertEqual(response.usage.completion_tokens_details.reasoning_tokens, 30)
        self.assertEqual(
            response.usage.completion_tokens_details.accepted_prediction_tokens, 2
        )
        self.assertEqual(
            response.usage.completion_tokens_details.rejected_prediction_tokens, 1
        )
        self.assertEqual(response.usage.total_tokens, 1040)

    def test_missing_or_null_details_preserve_zero_usage_defaults(self):
        for usage in (
            {},
            None,
            {"input_tokens_details": None, "output_tokens_details": None},
        ):
            with self.subTest(usage=usage):
                response = self.transform_usage(usage)
                self.assertEqual(response.usage.prompt_tokens, 0)
                self.assertEqual(response.usage.completion_tokens, 0)
                self.assertEqual(response.usage.total_tokens, 0)
                self.assertIsNone(response.usage.prompt_tokens_details)
                self.assertIsNone(response.usage.completion_tokens_details)


class DecisionsHTTPDeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.backend_requests = []
        self.connections = set()
        self.server = await asyncio.start_server(self.serve_azure, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        self.api_base = f"http://127.0.0.1:{port}/"
        self.http_handler = AsyncHTTPHandler(timeout=2)
        await self.http_handler.client.aclose()
        self.backend_client = httpx.AsyncClient(timeout=2, trust_env=False)
        self.http_handler.client = self.backend_client
        self.router = litellm.Router(
            model_list=[
                self.deployment("decision-dz", "decisions-model", 1, 0.15),
                self.deployment("decision-global", "decisions-model", 2, 0.8),
                self.deployment("decision-concurrent", "concurrent-model", 1, 0.8),
                self.deployment("decision-fast", "fast-model", 1, 0.1),
            ],
            routing_strategy="simple-shuffle",
            num_retries=0,
            retry_after=0,
            timeout=2,
        )
        self.patches = [
            patch.object(proxy_server, "llm_router", self.router),
            patch.object(proxy_server, "master_key", "deadline-gateway-key"),
            patch.object(proxy_server, "general_settings", {}),
            patch.object(proxy_server, "prisma_client", None),
            patch.object(litellm, "callbacks", []),
            patch.object(litellm, "model_alias_map", {}),
            patch.object(litellm, "module_level_aclient", self.http_handler),
        ]
        for item in self.patches:
            item.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy_server.app),
            base_url="http://gateway",
            headers={"Authorization": "Bearer deadline-gateway-key"},
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.backend_client.aclose()
        self.router.reset()
        for item in reversed(self.patches):
            item.stop()
        self.server.close()
        await self.server.wait_closed()
        connections = list(self.connections)
        for connection in connections:
            connection.cancel()
        await asyncio.gather(*connections, return_exceptions=True)

    def deployment(self, name, model_name, order, timeout):
        return {
            "model_name": model_name,
            "litellm_params": {
                "model": f"azure/gpt5_series/{name}",
                "api_base": self.api_base,
                "api_key": f"key-{name}",
                "order": order,
                "timeout": timeout,
            },
        }

    async def serve_azure(self, reader, writer):
        task = asyncio.current_task()
        self.connections.add(task)
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            content_length = next(
                int(line.split(b":", 1)[1])
                for line in headers.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            body = json.loads(await reader.readexactly(content_length))
            self.backend_requests.append((body["model"], time.monotonic()))
            delay = {"decision-dz": 1.2, "decision-concurrent": 0.3}.get(
                body["model"], 0
            )
            await asyncio.sleep(delay)
            payload = json.dumps(
                {
                    "model": body["model"],
                    "answers": [
                        {"type": "predicate", "name": "accepted", "probability": 0.95}
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
                + payload
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            self.connections.discard(task)

    async def request_decision(self, model):
        return await self.client.post(
            "/v1/decisions",
            json={
                "model": model,
                "input": "Should this request be accepted?",
                "questions": [
                    {
                        "type": "predicate",
                        "name": "accepted",
                        "instructions": "Accept it.",
                    }
                ],
            },
        )

    async def test_slow_datazone_times_out_and_global_fallback_preserves_concurrent_deadlines(
        self,
    ):
        original_timeout = self.backend_client.timeout
        original_values = original_timeout.as_dict()
        short_request = asyncio.create_task(self.request_decision("decisions-model"))
        long_request = asyncio.create_task(self.request_decision("concurrent-model"))
        fast_request = asyncio.create_task(self.request_decision("fast-model"))
        await asyncio.sleep(0.05)
        self.assertIs(self.backend_client.timeout, original_timeout)
        self.assertEqual(self.backend_client.timeout.as_dict(), original_values)
        short, long, fast = await asyncio.gather(
            short_request, long_request, fast_request
        )
        self.assertEqual(short.status_code, 200, short.text)
        self.assertEqual(short.json()["model"], "decision-global")
        self.assertEqual(long.status_code, 200, long.text)
        self.assertEqual(long.json()["model"], "decision-concurrent")
        self.assertEqual(fast.status_code, 200, fast.text)
        self.assertEqual(fast.json()["model"], "decision-fast")
        starts = dict(self.backend_requests)
        self.assertGreaterEqual(starts["decision-global"] - starts["decision-dz"], 0.12)
        self.assertLess(starts["decision-global"] - starts["decision-dz"], 1.0)
        self.assertIs(self.backend_client.timeout, original_timeout)
        self.assertEqual(self.backend_client.timeout.as_dict(), original_values)

    async def test_absent_timeout_uses_shared_client_default(self):
        self.backend_client.timeout = httpx.Timeout(0.05)
        original_timeout = self.backend_client.timeout
        with self.assertRaises(BaseLLMException):
            await litellm.allm_passthrough_route(
                method="POST",
                endpoint=AZURE_DECISIONS_ENDPOINT,
                model="decision-concurrent",
                custom_llm_provider="azure",
                api_base=self.api_base,
                api_key="key-concurrent",
                json={"model": "decision-concurrent", "input": "test", "questions": []},
            )
        self.assertIs(self.backend_client.timeout, original_timeout)

    async def test_unconfigured_router_timeout_inherits_shared_client_default(self):
        deployment = self.deployment("decision-concurrent", "concurrent-model", 1, None)
        del deployment["litellm_params"]["timeout"]
        with patch.object(litellm, "request_timeout", None):
            router = litellm.Router(
                model_list=[deployment],
                routing_strategy="simple-shuffle",
                num_retries=0,
                retry_after=0,
            )
        self.backend_client.timeout = httpx.Timeout(0.05)
        original_timeout = self.backend_client.timeout
        try:
            with patch.object(proxy_server, "llm_router", router):
                response = await self.request_decision("concurrent-model")
        finally:
            router.reset()
        self.assertEqual(response.status_code, 500, response.text)
        self.assertNotIn("answers", response.json())
        self.assertEqual(
            [model for model, _ in self.backend_requests], ["decision-concurrent"]
        )
        self.assertIs(self.backend_client.timeout, original_timeout)
        self.assertEqual(self.backend_client.timeout.read, 0.05)

    async def test_httpx_timeout_override_does_not_change_shared_client_default(self):
        self.backend_client.timeout = httpx.Timeout(0.05)
        original_timeout = self.backend_client.timeout
        response = await litellm.allm_passthrough_route(
            method="POST",
            endpoint=AZURE_DECISIONS_ENDPOINT,
            model="decision-concurrent",
            custom_llm_provider="azure",
            api_base=self.api_base,
            api_key="key-concurrent",
            json={"model": "decision-concurrent", "input": "test", "questions": []},
            timeout=httpx.Timeout(0.8, connect=0.4),
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "decision-concurrent")
        self.assertIs(self.backend_client.timeout, original_timeout)
        self.assertEqual(self.backend_client.timeout.read, 0.05)


if __name__ == "__main__":
    unittest.main()
