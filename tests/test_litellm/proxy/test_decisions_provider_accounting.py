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
        self.backend_arrivals = {
            name: asyncio.Event()
            for name in (
                "decision-dz",
                "decision-global",
                "decision-concurrent",
                "decision-fast",
            )
        }
        self.backend_barriers = {}
        self.server = await asyncio.start_server(self.serve_azure, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        self.api_base = f"http://127.0.0.1:{port}/"
        self.http_handler = AsyncHTTPHandler(timeout=2)
        await self.http_handler.client.aclose()
        self.outbound_requests = []
        self.backend_client = httpx.AsyncClient(
            timeout=2,
            trust_env=False,
            event_hooks={"request": [self.observe_outbound_request]},
        )
        self.http_handler.client = self.backend_client
        self.router = litellm.Router(
            model_list=[
                self.deployment(
                    "decision-dz", "decisions-model", 1, httpx.Timeout(10, read=0.15)
                ),
                self.deployment(
                    "decision-global", "decisions-model", 2, httpx.Timeout(10)
                ),
                self.deployment(
                    "decision-concurrent", "concurrent-model", 1, httpx.Timeout(10)
                ),
                self.deployment(
                    "decision-fast", "fast-model", 1, httpx.Timeout(10, read=0.5)
                ),
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
        self.server.close()
        connections = list(self.connections)
        for connection in connections:
            connection.cancel()
        results = await asyncio.gather(*connections, return_exceptions=True)
        await self.server.wait_closed()
        await self.client.aclose()
        await self.backend_client.aclose()
        self.router.reset()
        for item in reversed(self.patches):
            item.stop()
        for result in results:
            if isinstance(result, BaseException) and not isinstance(
                result, asyncio.CancelledError
            ):
                raise result

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

    async def observe_outbound_request(self, request):
        self.outbound_requests.append(
            {
                "model": json.loads(request.content)["model"],
                "url": str(request.url),
                "timeout": dict(request.extensions["timeout"]),
                "timeout_extension": request.extensions["timeout"],
                "shared_timeout": self.backend_client.timeout,
                "shared_values": self.backend_client.timeout.as_dict(),
            }
        )

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
            self.backend_arrivals[body["model"]].set()
            barrier = self.backend_barriers.get(body["model"])
            if barrier is not None:
                await barrier.wait()
            elif body["model"] == "decision-concurrent":
                await asyncio.sleep(0.3)
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
        self.backend_client.timeout = httpx.Timeout(10, read=None)
        original_timeout = self.backend_client.timeout
        original_values = original_timeout.as_dict()
        self.backend_barriers = {
            name: asyncio.Event()
            for name in ("decision-dz", "decision-global", "decision-concurrent")
        }
        requests = []
        try:
            long_request = asyncio.create_task(
                self.request_decision("concurrent-model")
            )
            requests.append(long_request)
            await asyncio.wait_for(
                self.backend_arrivals["decision-concurrent"].wait(), timeout=10
            )
            self.assertIs(self.backend_client.timeout, original_timeout)
            self.assertEqual(self.backend_client.timeout.as_dict(), original_values)
            short_request = asyncio.create_task(
                self.request_decision("decisions-model")
            )
            requests.append(short_request)
            await asyncio.wait_for(
                self.backend_arrivals["decision-dz"].wait(), timeout=10
            )
            self.assertIs(self.backend_client.timeout, original_timeout)
            self.assertEqual(self.backend_client.timeout.as_dict(), original_values)
            fast_request = asyncio.create_task(self.request_decision("fast-model"))
            requests.append(fast_request)
            fast = await asyncio.wait_for(asyncio.shield(fast_request), timeout=10)
            self.assertEqual(
                fast.status_code,
                200,
                f"{fast.text}; outbound={self.outbound_requests!r}",
            )
            self.assertEqual(fast.json()["model"], "decision-fast")
            await asyncio.wait_for(
                self.backend_arrivals["decision-global"].wait(), timeout=10
            )
            self.assertFalse(long_request.done())
            self.assertIs(self.backend_client.timeout, original_timeout)
            self.assertEqual(self.backend_client.timeout.as_dict(), original_values)
            self.backend_barriers["decision-concurrent"].set()
            self.backend_barriers["decision-global"].set()
            short, long = await asyncio.wait_for(
                asyncio.gather(short_request, long_request), timeout=10
            )
            self.assertEqual(short.status_code, 200, short.text)
            self.assertEqual(short.json()["model"], "decision-global")
            self.assertEqual(long.status_code, 200, long.text)
            self.assertEqual(long.json()["model"], "decision-concurrent")
            self.assertIs(self.backend_client.timeout, original_timeout)
            self.assertEqual(self.backend_client.timeout.as_dict(), original_values)
            expected_reads = {
                "decision-dz": 0.15,
                "decision-global": 10,
                "decision-concurrent": 10,
                "decision-fast": 0.5,
            }
            self.assertEqual(
                {request["model"] for request in self.outbound_requests},
                set(expected_reads),
            )
            for request in self.outbound_requests:
                self.assertEqual(
                    request["url"], f"{self.api_base}{AZURE_DECISIONS_ENDPOINT}"
                )
                self.assertEqual(
                    request["timeout"]["read"], expected_reads[request["model"]]
                )
                self.assertEqual(request["timeout"]["connect"], 10)
                self.assertIs(request["shared_timeout"], original_timeout)
                self.assertEqual(request["shared_values"], original_values)
            self.assertEqual(
                len(
                    {
                        id(request["timeout_extension"])
                        for request in self.outbound_requests
                    }
                ),
                len(self.outbound_requests),
            )
        finally:
            for request in requests:
                if not request.done():
                    request.cancel()
            results = await asyncio.gather(*requests, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    raise result

    async def test_absent_timeout_uses_shared_client_default(self):
        self.backend_client.timeout = httpx.Timeout(10, read=0.05)
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
        self.backend_client.timeout = httpx.Timeout(10, read=0.05)
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
        self.backend_client.timeout = httpx.Timeout(10, read=0)
        original_timeout = self.backend_client.timeout
        self.backend_barriers["decision-concurrent"] = asyncio.Event()
        with self.assertRaises(BaseLLMException):
            await asyncio.wait_for(
                litellm.allm_passthrough_route(
                    method="POST",
                    endpoint=AZURE_DECISIONS_ENDPOINT,
                    model="decision-concurrent",
                    custom_llm_provider="azure",
                    api_base=self.api_base,
                    api_key="key-concurrent",
                    json={
                        "model": "decision-concurrent",
                        "input": "test",
                        "questions": [],
                    },
                ),
                timeout=10,
            )
        self.assertIs(self.backend_client.timeout, original_timeout)
        self.assertEqual(self.outbound_requests[-1]["timeout"]["read"], 0)
        try:
            response = await asyncio.wait_for(
                litellm.allm_passthrough_route(
                    method="POST",
                    endpoint=AZURE_DECISIONS_ENDPOINT,
                    model="decision-fast",
                    custom_llm_provider="azure",
                    api_base=self.api_base,
                    api_key="key-fast",
                    json={"model": "decision-fast", "input": "test", "questions": []},
                    timeout=httpx.Timeout(10, read=None),
                ),
                timeout=10,
            )
        except BaseLLMException as exc:
            raise AssertionError(
                f"Explicit timeout override failed; outbound={self.outbound_requests!r}"
            ) from exc
        self.assertEqual(
            response.status_code, 200, f"outbound={self.outbound_requests!r}"
        )
        self.assertEqual(response.json()["model"], "decision-fast")
        self.assertIs(self.backend_client.timeout, original_timeout)
        self.assertEqual(self.backend_client.timeout.read, 0)
        self.assertIsNone(self.outbound_requests[-1]["timeout"]["read"])
        for request in self.outbound_requests:
            self.assertEqual(
                request["url"],
                f"{self.api_base}{AZURE_DECISIONS_ENDPOINT}",
                f"outbound={self.outbound_requests!r}",
            )
            self.assertIs(request["shared_timeout"], original_timeout)
            self.assertEqual(request["shared_values"]["read"], 0)


if __name__ == "__main__":
    unittest.main()
