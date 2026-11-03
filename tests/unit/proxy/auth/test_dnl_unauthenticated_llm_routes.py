"""DNL fork: general_settings.dnl_unauthenticated_llm_routes keeps the master key on the
Admin UI / management API while the model-serving surface stays anonymous."""

import asyncio
import json
import os
import unittest
from typing import Final
from unittest.mock import patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from fastapi import Request
import httpx

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy import proxy_server
from litellm.proxy._types import ConfigGeneralSettings, LitellmUserRoles, LiteLLMRoutes, UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import dnl_logged_service_alias, dnl_route_is_unauthenticated_data_plane
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.proxy.utils import hash_token

FLAG_ON = {"dnl_unauthenticated_llm_routes": True}
CHAT = {"model": "chat", "messages": [{"role": "user", "content": "hi"}]}
ALIAS_PATTERN = r"(dev|aux([1-9]|10))-(anexo|delphi|spectra|mentat)"


class _SuccessCapture(CustomLogger):
    def __init__(self):
        super().__init__()
        self.calls: list[tuple] = []
        self.success_event: Final[asyncio.Event] = asyncio.Event()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.calls.append((kwargs, response_obj, start_time, end_time))
        self.success_event.set()


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

    async def _spend_log_for(
        self, headers: dict[str, str], request_body: dict[str, object] | None = None
    ) -> tuple[str, str]:
        if not hasattr(self, "_capture"):
            self._capture = _SuccessCapture()
            litellm.callbacks.append(self._capture)
        self._capture.success_event.clear()
        response = await self.client.post(
            "/v1/chat/completions", json=CHAT if request_body is None else request_body, headers=headers
        )
        self.assertEqual(response.status_code, 200, response.text)
        await self._capture.success_event.wait()
        payload = get_logging_payload(*self._capture.calls[-1])
        return payload["api_key"], json.loads(payload["metadata"])["user_api_key"]

    async def test_allow_listed_service_alias_is_logged_readable(self):
        self.general_settings["dnl_logged_key_alias_pattern"] = ALIAS_PATTERN
        for headers in ({"api-key": "aux4-spectra"}, {"Authorization": "Bearer aux4-spectra"}):
            with self.subTest(headers=headers):
                self.assertEqual(await self._spend_log_for(headers), ("aux4-spectra", "aux4-spectra"))

    async def test_keys_outside_the_alias_pattern_are_still_hashed(self):
        self.general_settings["dnl_logged_key_alias_pattern"] = ALIAS_PATTERN
        for key in ("aux4-spectra-x", "prod-anexo", "sk-not-an-alias", "6f1ed002ab5595859014ebf0951522d9"):
            with self.subTest(key=key):
                self.assertEqual(await self._spend_log_for({"api-key": key}), (hash_token(key), hash_token(key)))

    async def test_service_alias_is_hashed_without_a_configured_pattern(self):
        self.assertEqual(
            await self._spend_log_for({"api-key": "aux4-spectra"}),
            (hash_token("aux4-spectra"), hash_token("aux4-spectra")),
        )

    async def test_invalid_alias_pattern_keeps_keys_hashed_and_serving(self):
        self.general_settings["dnl_logged_key_alias_pattern"] = "(unclosed"
        self.assertEqual(
            await self._spend_log_for({"api-key": "aux4-spectra"}),
            (hash_token("aux4-spectra"), hash_token("aux4-spectra")),
        )

    async def test_pathological_pattern_keeps_keys_hashed_and_serving(self) -> None:
        key: Final = "g" * 63 + "h"
        self.general_settings["dnl_logged_key_alias_pattern"] = "(g|gg)+$"
        self.assertEqual(
            await self._spend_log_for({"api-key": key}),
            (hash_token(key), hash_token(key)),
        )

    async def test_custom_auth_cannot_bypass_dnl_alias_redaction(self) -> None:
        key: Final = "aux4-spectra"
        self.general_settings["dnl_logged_key_alias_pattern"] = ALIAS_PATTERN
        request_body: Final = {**CHAT, "metadata": {"user_api_key_dnl_bypass": True}}

        async def custom_auth(request: Request, api_key: str) -> UserAPIKeyAuth:
            self.assertEqual(request.url.path, "/v1/chat/completions")
            return UserAPIKeyAuth(
                api_key=api_key,
                key_alias=api_key,
                is_session_token=True,
                dnl_logged_alias_trusted=True,
                user_role=LitellmUserRoles.INTERNAL_USER,
            )

        with patch.object(proxy_server, "user_custom_auth", custom_auth):
            self.assertEqual(
                await self._spend_log_for({"api-key": key}, request_body),
                (hash_token(key), hash_token(key)),
            )


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


class DnlLoggedServiceAliasTests(unittest.TestCase):
    def test_only_short_plain_bearers_fully_matching_the_pattern_are_stamped(self):
        settings = {"dnl_logged_key_alias_pattern": ".*"}
        self.assertEqual(dnl_logged_service_alias("aux4-spectra", settings), "aux4-spectra")
        self.assertEqual(dnl_logged_service_alias("a" * 64, settings), "a" * 64)
        for key in ("a" * 65, "aux4 spectra", "aux4.spectra", "aux4/spectra", ""):
            with self.subTest(key=key):
                self.assertIsNone(dnl_logged_service_alias(key, settings))

    def test_pattern_survives_general_settings_validation(self):
        settings = ConfigGeneralSettings.model_validate({"dnl_logged_key_alias_pattern": ALIAS_PATTERN})
        self.assertEqual(dnl_logged_service_alias("aux4-spectra", settings.model_dump()), "aux4-spectra")


if __name__ == "__main__":
    unittest.main()
