"""DNL fork: general_settings.dnl_unauthenticated_llm_routes keeps the master key on the
Admin UI / management API while the model-serving surface stays anonymous."""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import httpx

import litellm
from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLMRoutes
from litellm.proxy.auth.auth_utils import dnl_route_is_unauthenticated_data_plane

FLAG_ON = {"dnl_unauthenticated_llm_routes": True}
CHAT = {"model": "chat", "messages": [{"role": "user", "content": "hi"}]}


class UnauthenticatedDataPlaneProxyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.router = litellm.Router(
            model_list=[{"model_name": "chat", "litellm_params": {"model": "openai/gpt-4o", "mock_response": "hello"}}],
            num_retries=0,
        )
        self.general_settings = dict(FLAG_ON)
        self.patches = [
            patch.object(proxy_server, "llm_router", self.router),
            patch.object(proxy_server, "master_key", "sk-gateway-master"),
            patch.object(proxy_server, "general_settings", self.general_settings),
            patch.object(proxy_server, "prisma_client", None),
            patch.object(proxy_server, "user_custom_auth", None),
            patch.object(litellm, "callbacks", []),
        ]
        for item in self.patches:
            item.start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://gateway")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.router.reset()
        for item in reversed(self.patches):
            item.stop()

    async def test_model_serving_route_needs_no_key_when_enabled(self):
        response = await self.client.post("/v1/chat/completions", json=CHAT)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "hello")

    async def test_management_routes_still_require_the_master_key(self):
        for method, path in (("GET", "/key/list"), ("POST", "/key/generate"), ("GET", "/user/list")):
            with self.subTest(path=path):
                response = await self.client.request(method, path, json={} if method == "POST" else None)
                self.assertEqual(response.status_code, 401, response.text)

    async def test_model_serving_route_requires_a_key_when_disabled(self):
        self.general_settings.clear()
        response = await self.client.post("/v1/chat/completions", json=CHAT)
        self.assertEqual(response.status_code, 401, response.text)


class UnauthenticatedDataPlanePredicateTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch.object(proxy_server, "user_custom_auth", None)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_parameterized_and_alias_serving_routes_match(self):
        for route in (
            "/responses/resp_123",
            "/openai/responses",
            "/openai/responses/resp_123",
            "/openai/responses/resp_123/input_items",
            "/openai/responses/resp_123/cancel",
            "/openai/responses/input_tokens",
            "/openai/v1/decisions",
            "/v1/embeddings",
        ):
            with self.subTest(route=route):
                self.assertTrue(dnl_route_is_unauthenticated_data_plane(route, FLAG_ON))

    def test_unlisted_openai_alias_paths_stay_authenticated(self):
        for route in ("/openai/responses/resp_123/extra", "/openai/responses/resp_123/input_items/x", "/openai/other"):
            with self.subTest(route=route):
                self.assertFalse(dnl_route_is_unauthenticated_data_plane(route, FLAG_ON))

    def test_upstream_route_list_growth_does_not_widen_the_allowlist(self):
        added_upstream = [
            route
            for route in ("/utils/model_info", "/claude-code/plugins", "/claude-code/marketplace.json")
            if route in LiteLLMRoutes.openai_routes.value + LiteLLMRoutes.anthropic_routes.value
        ]
        self.assertTrue(added_upstream, "fixture routes must exist in the upstream lists to prove anything")
        for route in added_upstream:
            with self.subTest(route=route):
                self.assertFalse(dnl_route_is_unauthenticated_data_plane(route, FLAG_ON))

    def test_other_authentication_modes_switch_it_off(self):
        for setting in ("enable_jwt_auth", "enable_oauth2_auth", "enable_oauth2_proxy_auth"):
            with self.subTest(setting=setting):
                self.assertFalse(
                    dnl_route_is_unauthenticated_data_plane("/v1/chat/completions", {**FLAG_ON, setting: True})
                )
        with patch.object(proxy_server, "user_custom_auth", object()):
            self.assertFalse(dnl_route_is_unauthenticated_data_plane("/v1/chat/completions", FLAG_ON))


if __name__ == "__main__":
    unittest.main()
