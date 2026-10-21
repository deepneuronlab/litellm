"""Azure SDK /openai/responses alias and gpt5_series/ normalization, against a mock Azure transport.

Self-contained (stdlib unittest + httpx) so CI can also run it inside the built image.
"""

import copy
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import httpx
import litellm
from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.response_api_endpoints import endpoints
from litellm.router_utils.pre_call_checks.deployment_affinity_check import DeploymentAffinityCheck


AZURE_RESPONSE = {
    "id": "resp_test",
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": "gpt-5.2",
    "output": [
        {
            "id": "msg_test",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello", "annotations": []}],
        }
    ],
    "usage": {
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}


class NativeAzureResponsesTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.azure_requests = []
        self.azure_status = 200
        self.backend_client = httpx.AsyncClient(transport=httpx.MockTransport(self.azure_response))
        self.http_handler = AsyncHTTPHandler(timeout=3)
        await self.http_handler.client.aclose()
        self.http_handler.client = self.backend_client
        self.router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-5.2",
                    "litellm_params": {
                        "model": "azure/gpt5_series/gpt-5.2-dz",
                        "base_model": "gpt-5.2",
                        "api_base": "https://mock-azure.openai.azure.com/",
                        "api_key": "mock-azure-key",
                        "api_version": "2025-03-01-preview",
                    },
                }
            ],
            routing_strategy="simple-shuffle",
            optional_pre_call_checks=["session_affinity"],
            num_retries=0,
            timeout=3,
        )
        self.router.aresponses = AsyncMock(wraps=self.router.aresponses)
        self.patches = [
            patch.object(proxy_server, "llm_router", self.router),
            patch.object(proxy_server, "master_key", "mock-gateway-key"),
            patch.object(proxy_server, "general_settings", {}),
            patch.object(proxy_server, "prisma_client", None),
            patch.object(litellm, "callbacks", []),
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
            headers={"Authorization": "Bearer mock-gateway-key"},
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.backend_client.aclose()
        self.router.reset()
        for item in reversed(self.patches):
            item.stop()

    def azure_response(self, request):
        body = json.loads(request.content)
        self.azure_requests.append((str(request.url), body))
        if self.azure_status != 200:
            return httpx.Response(
                self.azure_status,
                json={
                    "error": {
                        "message": "mock Azure capacity exceeded",
                        "code": "RateLimitExceeded",
                        "type": "rate_limit_error",
                    }
                },
            )
        response = copy.deepcopy(AZURE_RESPONSE)
        response["model"] = body["model"]
        if body.get("stream"):
            events = [
                {"type": "response.created", "sequence_number": 0, "response": response},
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": "msg_test",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "hello",
                },
                {"type": "response.completed", "sequence_number": 2, "response": response},
            ]
            content = "".join(f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n' for event in events)
            return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=response)

    async def test_azure_url_routes_once_and_preserves_metadata(self):
        response = await self.client.post(
            "/openai/responses?api-version=2025-03-01-preview",
            json={
                "model": "gpt-5.2",
                "input": "hello",
                "litellm_metadata": {"session_id": "document-123"},
                "prompt_cache_key": "document-123",
                "caching": False,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.router.aresponses.await_count, 1)
        request_kwargs = self.router.aresponses.call_args.kwargs
        self.assertEqual(DeploymentAffinityCheck._get_session_id_from_request_kwargs(request_kwargs), "document-123")
        self.assertIs(request_kwargs["caching"], False)
        self.assertEqual(len(self.azure_requests), 1)
        url, body = self.azure_requests[0]
        self.assertEqual(url, "https://mock-azure.openai.azure.com/openai/responses?api-version=2025-03-01-preview")
        self.assertEqual(body["model"], "gpt-5.2-dz")
        self.assertEqual(body["prompt_cache_key"], "document-123")
        self.assertNotIn("litellm_metadata", body)

    async def test_streaming_uses_the_native_router(self):
        response = await self.client.post(
            "/openai/responses",
            json={
                "model": "gpt-5.2",
                "input": "hello",
                "stream": True,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn("response.output_text.delta", response.text)
        self.assertIn("hello", response.text)
        self.assertEqual(self.router.aresponses.await_count, 1)
        self.assertEqual(len(self.azure_requests), 1)

    async def test_azure_error_status_and_message_are_preserved(self):
        self.azure_status = 429
        response = await self.client.post("/openai/responses", json={"model": "gpt-5.2", "input": "hello"})
        self.assertEqual(response.status_code, 429, response.text)
        self.assertIn("mock Azure capacity exceeded", response.json()["error"]["message"])

    async def test_alias_requires_the_same_authentication(self):
        response = await self.client.post(
            "/openai/responses",
            headers={"Authorization": ""},
            json={
                "model": "gpt-5.2",
                "input": "hello",
            },
        )
        self.assertEqual(response.status_code, 401, response.text)
        self.assertFalse(self.azure_requests)

    def test_create_alias_uses_existing_handler_and_permissions(self):
        alias = next(r for r in endpoints.router.routes if r.path == "/openai/responses" and "POST" in r.methods)
        native = next(r for r in endpoints.router.routes if r.path == "/v1/responses" and "POST" in r.methods)
        self.assertIs(alias.endpoint, native.endpoint)
        self.assertEqual(
            [dependency.dependency for dependency in alias.dependencies],
            [dependency.dependency for dependency in native.dependencies],
        )
        self.assertTrue(RouteChecks.is_llm_api_route(alias.path))
        self.assertTrue(RouteChecks.should_call_route(alias.path, UserAPIKeyAuth(allowed_routes=["llm_api_routes"])))

    def test_prefix_normalization_preserves_real_deployment_names(self):
        config = AzureOpenAIResponsesAPIConfig()
        for model, expected in [
            ("gpt5_series/gpt-5.2-dz", "gpt-5.2-dz"),
            ("gpt5_series/gpt-6-luna-dz", "gpt-6-luna-dz"),
            ("responses/gpt-5.2", "gpt-5.2"),
            ("o_series/o3", "o3"),
            ("gpt-4.1", "gpt-4.1"),
            ("custom-gpt5_series/deployment", "custom-gpt5_series/deployment"),
        ]:
            with self.subTest(model=model):
                self.assertEqual(config.get_stripped_model_name(model), expected)


if __name__ == "__main__":
    unittest.main()
