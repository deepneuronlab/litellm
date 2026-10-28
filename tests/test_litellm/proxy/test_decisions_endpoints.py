"""Mock Azure transport coverage, runnable with unittest in the proxy image."""

import asyncio
import copy
import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import httpx
from fastapi import HTTPException

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import hash_token
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.guardrails import GuardrailEventHooks

DECISION_REQUEST = {
    "model": "decisions-model",
    "input": "I was charged twice. Please refund the extra charge.",
    "questions": [
        {
            "type": "predicate",
            "name": "refund_requested",
            "instructions": "Does the customer ask for a refund?",
        }
    ],
}
AZURE_RESPONSE = {
    "model": "decisions-model",
    "answers": [{"type": "predicate", "name": "refund_requested", "probability": 0.95}],
    "usage": {
        "input_tokens": 162,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": 3,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 165,
    },
}
DECISIONS_PATHS = (
    "/openai/decisions?api-version=2024-12-01-preview",
    "/openai/v1/decisions?api-version=2024-12-01-preview",
    "/v1/decisions",
    "/decisions",
)


class LifecycleLogger(CustomLogger):
    def __init__(self, reject_pre=False, reject_post=False, rewrite_input=None):
        super().__init__()
        self.reject_pre = reject_pre
        self.reject_post = reject_post
        self.rewrite_input = rewrite_input
        self.pre_calls = []
        self.post_calls = []
        self.success_payloads = []
        self.success_event = asyncio.Event()

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        self.pre_calls.append(
            {
                "auth": user_api_key_dict,
                "metadata": copy.deepcopy(data.get("metadata")),
                "litellm_metadata": copy.deepcopy(data.get("litellm_metadata")),
                "request_body": copy.deepcopy(data["proxy_server_request"]["body"]),
                "input": data.get("input"),
                "questions": copy.deepcopy(data.get("questions")),
                "call_type": call_type,
            }
        )
        if self.reject_pre:
            raise HTTPException(
                status_code=403, detail="Decisions blocked by pre-call policy"
            )
        if self.rewrite_input is not None:
            data["input"] = self.rewrite_input
        return data

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        self.post_calls.append(user_api_key_dict)
        if self.reject_post:
            raise HTTPException(
                status_code=403, detail="Decisions blocked by post-call policy"
            )
        return response

    async def async_post_call_response_headers_hook(
        self, data, user_api_key_dict, response, request_headers=None, **kwargs
    ):
        return {"x-decisions-policy": "checked"}

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.success_payloads.append(kwargs.get("standard_logging_object"))
        self.success_event.set()


class DuringCallPolicy(CustomGuardrail):
    def __init__(self, reject=False):
        super().__init__(
            guardrail_name="decisions-during-policy",
            event_hook=GuardrailEventHooks.during_call,
            default_on=True,
        )
        self.reject = reject
        self.calls = []

    async def async_moderation_hook(self, data, user_api_key_dict, call_type):
        self.calls.append(user_api_key_dict)
        if self.reject:
            raise HTTPException(
                status_code=403, detail="Decisions blocked by during-call policy"
            )


def deployment(name, host, order):
    return {
        "model_name": "decisions-model",
        "litellm_params": {
            "model": f"azure/gpt5_series/{name}",
            "api_base": f"https://{host}/",
            "api_key": f"key-{name}",
            "api_version": "2025-03-01-preview",
            "max_completion_tokens": 128000,
            "rpm": 1000,
            "tpm": 1000000,
            "order": order,
            "drop_params": True,
            "additional_drop_params": ["max_tokens"],
        },
    }


class NativeAzureDecisionsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.azure_requests = []
        self.azure_errors = {}
        self.azure_response_headers = {}
        self.backend_client = httpx.AsyncClient(
            transport=httpx.MockTransport(self.azure_response)
        )
        self.http_handler = AsyncHTTPHandler(timeout=3)
        await self.http_handler.client.aclose()
        self.http_handler.client = self.backend_client
        self.router = self.make_router(num_retries=2)
        self.original_overrides = dict(proxy_server.app.dependency_overrides)
        self.patches = [
            patch.object(proxy_server, "llm_router", self.router),
            patch.object(proxy_server, "master_key", "mock-gateway-key"),
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
            headers={"Authorization": "Bearer mock-gateway-key"},
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.backend_client.aclose()
        self.router.reset()
        proxy_server.app.dependency_overrides.clear()
        proxy_server.app.dependency_overrides.update(self.original_overrides)
        for item in reversed(self.patches):
            item.stop()

    def make_router(self, num_retries):
        return litellm.Router(
            model_list=[
                deployment("decision-dz", "mock-dz.openai.azure.com", 1),
                deployment("decision-global", "mock-global.openai.azure.com", 2),
            ],
            routing_strategy="simple-shuffle",
            num_retries=num_retries,
            retry_after=0,
            timeout=3,
        )

    def use_router(self, router):
        self.router.reset()
        self.router = router
        item = patch.object(proxy_server, "llm_router", router)
        item.start()
        self.patches.append(item)

    def use_virtual_principal(self, **kwargs):
        principal = UserAPIKeyAuth(
            api_key=hash_token("sk-virtual-decisions"),
            user_role=LitellmUserRoles.INTERNAL_USER,
            team_id="authenticated-team",
            user_id="authenticated-user",
            key_alias="decisions-key",
            models=["decisions-model"],
            allowed_routes=["llm_api_routes"],
            **kwargs,
        )
        proxy_server.app.dependency_overrides[user_api_key_auth] = lambda: principal
        return principal

    def azure_response(self, request):
        body = json.loads(request.content)
        self.azure_requests.append(
            (str(request.url), request.headers.get("api-key"), body)
        )
        if request.url.host in self.azure_errors:
            status, message = self.azure_errors[request.url.host]
            return httpx.Response(
                status,
                headers={"retry-after": "0"},
                json={
                    "error": {
                        "code": "429" if status == 429 else "BadRequest",
                        "message": message,
                        "type": "invalid_request_error",
                    }
                },
            )
        return httpx.Response(
            200,
            json=dict(AZURE_RESPONSE, model=body["model"]),
            headers=self.azure_response_headers,
        )

    async def test_every_sdk_path_reaches_azure_v1_decisions(self):
        for path in DECISIONS_PATHS:
            with self.subTest(path=path):
                self.azure_requests.clear()
                response = await self.client.post(path, json=DECISION_REQUEST)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(
                    response.json(), dict(AZURE_RESPONSE, model="decision-dz")
                )
                self.assertEqual(
                    self.azure_requests,
                    [
                        (
                            "https://mock-dz.openai.azure.com/openai/v1/decisions",
                            "key-decision-dz",
                            dict(DECISION_REQUEST, model="decision-dz"),
                        )
                    ],
                )
                self.assertIn("x-litellm-call-id", response.headers)

    async def test_explicit_stream_false_and_proxy_controls_stay_out_of_provider_body(
        self,
    ):
        response = await self.client.post(
            "/v1/decisions",
            json=dict(
                DECISION_REQUEST,
                stream=False,
                endpoint="hostile-path",
                callbacks=["hostile-callback"],
                custom_llm_provider="openai",
            ),
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["answers"], AZURE_RESPONSE["answers"])
        self.assertIn("application/json", response.headers["content-type"])
        sent = self.azure_requests[0][2]
        self.assertEqual(sent["model"], "decision-dz")
        for field in (
            "stream",
            "endpoint",
            "callbacks",
            "custom_llm_provider",
        ):
            self.assertNotIn(field, sent)

    async def test_upstream_response_headers_are_forwarded_and_entity_headers_rebuilt(
        self,
    ):
        self.azure_response_headers = {
            "x-ms-request-id": "azure-request-id",
            "x-ratelimit-remaining-requests": "9",
            "content-encoding": "identity",
            "content-length": "1",
        }
        response = await self.client.post("/v1/decisions", json=DECISION_REQUEST)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["x-ms-request-id"], "azure-request-id")
        self.assertEqual(response.headers["x-ratelimit-remaining-requests"], "9")
        self.assertNotIn("content-encoding", response.headers)
        self.assertEqual(
            int(response.headers["content-length"]),
            len(response.content),
        )

    async def test_pre_call_policy_rejects_before_any_provider_call(self):
        principal = self.use_virtual_principal()
        logger = LifecycleLogger(reject_pre=True)
        litellm.callbacks.append(logger)
        response = await self.client.post("/v1/decisions", json=DECISION_REQUEST)
        self.assertEqual(response.status_code, 403, response.text)
        self.assertIn("blocked by pre-call policy", response.text)
        self.assertEqual(len(logger.pre_calls), 1)
        self.assertIs(logger.pre_calls[0]["auth"], principal)
        self.assertEqual(logger.pre_calls[0]["input"], DECISION_REQUEST["input"])
        self.assertEqual(
            logger.pre_calls[0]["questions"], DECISION_REQUEST["questions"]
        )
        self.assertFalse(self.azure_requests)

    async def test_normal_during_and_success_lifecycle_and_headers(self):
        principal = self.use_virtual_principal()
        logger = LifecycleLogger(rewrite_input="Policy-sanitized input")
        during_policy = DuringCallPolicy()
        litellm.callbacks.append(during_policy)
        litellm.callbacks.append(logger)
        response = await self.client.post("/v1/decisions", json=DECISION_REQUEST)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(during_policy.calls, [principal])
        self.assertEqual(logger.post_calls, [principal])
        self.assertEqual(response.headers["x-decisions-policy"], "checked")
        self.assertEqual(self.azure_requests[0][2]["input"], "Policy-sanitized input")
        self.assertEqual(response.json()["usage"], AZURE_RESPONSE["usage"])

    async def test_during_call_guardrail_can_reject_provider_output(self):
        principal = self.use_virtual_principal()
        during_policy = DuringCallPolicy(reject=True)
        litellm.callbacks.append(during_policy)
        response = await self.client.post("/v1/decisions", json=DECISION_REQUEST)
        self.assertEqual(response.status_code, 403, response.text)
        self.assertIn("blocked by during-call policy", response.text)
        self.assertEqual(during_policy.calls, [principal])
        self.assertNotIn("answers", response.json())

    async def test_post_call_policy_can_reject_before_returning_provider_content(self):
        self.use_virtual_principal()
        logger = LifecycleLogger(reject_post=True)
        litellm.callbacks.append(logger)
        response = await self.client.post("/v1/decisions", json=DECISION_REQUEST)
        self.assertEqual(response.status_code, 403, response.text)
        self.assertIn("blocked by post-call policy", response.text)
        self.assertNotIn("answers", response.json())
        self.assertEqual(len(self.azure_requests), 1)
        self.assertEqual(len(logger.post_calls), 1)

    async def test_metadata_is_sanitized_and_attributed_to_authenticated_virtual_key(
        self,
    ):
        principal = self.use_virtual_principal()
        logger = LifecycleLogger()
        litellm.callbacks.append(logger)
        forged = {
            "user_api_key_team_id": "forged-team",
            "user_api_key_user_id": "forged-user",
            "user_api_key_hash": "forged-key",
            "user_api_key_auth": {"team_id": "forged-team"},
            "disable_global_guardrails": True,
            "_pipeline_managed_guardrails": ["policy"],
            "session_id": "document-123",
        }
        response = await self.client.post(
            "/v1/decisions",
            json=dict(
                DECISION_REQUEST,
                litellm_metadata=dict(forged),
                metadata=dict(forged),
            ),
        )
        self.assertEqual(response.status_code, 200, response.text)
        metadata = logger.pre_calls[0]["litellm_metadata"]
        self.assertEqual(metadata["user_api_key_team_id"], principal.team_id)
        self.assertEqual(metadata["user_api_key_user_id"], principal.user_id)
        self.assertEqual(metadata["user_api_key_hash"], principal.api_key)
        self.assertEqual(metadata["session_id"], "document-123")
        self.assertNotIn("disable_global_guardrails", metadata)
        self.assertNotIn("_pipeline_managed_guardrails", metadata)
        user_metadata = logger.pre_calls[0]["metadata"]
        self.assertEqual(user_metadata, {"session_id": "document-123"})
        logged_body = logger.pre_calls[0]["request_body"]
        self.assertNotEqual(
            logged_body["litellm_metadata"].get("user_api_key_team_id"), "forged-team"
        )
        self.assertNotIn("user_api_key_team_id", logged_body["metadata"])
        self.assertNotIn("litellm_metadata", self.azure_requests[0][2])
        self.assertEqual(self.azure_requests[0][2]["metadata"], user_metadata)
        await asyncio.wait_for(logger.success_event.wait(), timeout=3)
        payload = logger.success_payloads[0]
        self.assertEqual(payload["metadata"]["user_api_key_team_id"], principal.team_id)
        self.assertEqual(payload["metadata"]["user_api_key_hash"], principal.api_key)

    async def test_usage_logging_contains_standard_logging_payload(self):
        logger = LifecycleLogger()
        litellm.callbacks.append(logger)
        response = await self.client.post("/v1/decisions", json=DECISION_REQUEST)
        self.assertEqual(response.status_code, 200, response.text)
        await asyncio.wait_for(logger.success_event.wait(), timeout=3)
        self.assertEqual(len(logger.success_payloads), 1)
        payload = logger.success_payloads[0]
        self.assertIsNotNone(payload)
        self.assertEqual(payload["model_group"], "decisions-model")
        self.assertEqual(payload["prompt_tokens"], 162)
        self.assertEqual(payload["completion_tokens"], 3)
        self.assertEqual(payload["total_tokens"], 165)
        self.assertIn("mock-dz.openai.azure.com", payload["api_base"])

    async def test_model_aliases_resolve_after_authenticated_preprocessing(self):
        self.use_virtual_principal(aliases={"key-alias": "decisions-model"})
        litellm.model_alias_map["global-alias"] = "decisions-model"
        self.router.model_group_alias["router-alias"] = "decisions-model"
        for alias in ("key-alias", "global-alias", "router-alias"):
            with self.subTest(alias=alias):
                response = await self.client.post(
                    "/v1/decisions", json=dict(DECISION_REQUEST, model=alias)
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(self.azure_requests[-1][2]["model"], "decision-dz")

    async def test_falls_back_to_global_order_with_its_own_key_and_deployment(self):
        self.use_router(self.make_router(num_retries=0))
        self.azure_errors["mock-dz.openai.azure.com"] = (
            429,
            "DataZone capacity exceeded",
        )
        response = await self.client.post("/openai/v1/decisions", json=DECISION_REQUEST)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            [(url, key, body["model"]) for url, key, body in self.azure_requests],
            [
                (
                    "https://mock-dz.openai.azure.com/openai/v1/decisions",
                    "key-decision-dz",
                    "decision-dz",
                ),
                (
                    "https://mock-global.openai.azure.com/openai/v1/decisions",
                    "key-decision-global",
                    "decision-global",
                ),
            ],
        )
        self.assertEqual(response.json(), dict(AZURE_RESPONSE, model="decision-global"))

    async def test_provider_error_status_and_message_survive_router_mapping(self):
        for status in (400, 401, 403, 404, 422, 429, 500, 503):
            with self.subTest(status=status):
                self.use_router(self.make_router(num_retries=0))
                self.azure_requests.clear()
                for host in (
                    "mock-dz.openai.azure.com",
                    "mock-global.openai.azure.com",
                ):
                    self.azure_errors[host] = (status, f"mock Azure error {status}")
                response = await self.client.post(
                    "/v1/decisions", json=DECISION_REQUEST
                )
                self.assertEqual(response.status_code, status, response.text)
                self.assertIn(
                    f"mock Azure error {status}", response.json()["error"]["message"]
                )

    async def test_client_errors_do_not_retry_within_an_order(self):
        for host in ("mock-dz.openai.azure.com", "mock-global.openai.azure.com"):
            self.azure_errors[host] = (400, "questions[0].type is invalid")
        response = await self.client.post("/v1/decisions", json=DECISION_REQUEST)
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(len(self.azure_requests), 2)

    async def test_non_azure_model_group_is_rejected_without_network_io(self):
        self.use_router(
            litellm.Router(
                model_list=[
                    {
                        "model_name": "decisions-model",
                        "litellm_params": {
                            "model": "openai/gpt-4o",
                            "api_key": "not-an-azure-key",
                            "api_base": "https://not-azure.invalid/",
                        },
                    }
                ],
                num_retries=0,
            )
        )
        response = await self.client.post("/v1/decisions", json=DECISION_REQUEST)
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("requires an Azure deployment", response.text)
        self.assertFalse(self.azure_requests)

    async def test_provider_control_fields_never_become_router_kwargs(self):
        self.use_virtual_principal()
        logger = LifecycleLogger()
        litellm.callbacks.append(logger)
        response = await self.client.post(
            "/v1/decisions",
            json=dict(
                DECISION_REQUEST,
                api_key="client-key",
                api_base="https://attacker.invalid/",
                custom_llm_provider="openai",
                endpoint="attacker-path",
                callbacks=[],
                disable_global_guardrails=True,
            ),
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(logger.pre_calls), 1)
        url, api_key, _ = self.azure_requests[0]
        self.assertEqual(url, "https://mock-dz.openai.azure.com/openai/v1/decisions")
        self.assertEqual(api_key, "key-decision-dz")

    async def test_malformed_body_metadata_stream_and_model_return_typed_400(self):
        for body in (
            [],
            17,
            None,
            {"input": "missing model"},
            dict(DECISION_REQUEST, model="not-a-deployment"),
            dict(DECISION_REQUEST, model=17),
            dict(DECISION_REQUEST, litellm_metadata=17),
            dict(DECISION_REQUEST, litellm_metadata=["invalid"]),
            dict(DECISION_REQUEST, litellm_metadata="invalid"),
            dict(DECISION_REQUEST, metadata=17),
            dict(DECISION_REQUEST, stream=True),
            dict(DECISION_REQUEST, stream="false"),
        ):
            with self.subTest(body=body):
                response = await self.client.post(
                    "/v1/decisions",
                    content=json.dumps(body),
                    headers={"content-type": "application/json"},
                )
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(
                    response.json()["error"]["type"], "invalid_request_error"
                )
        self.assertFalse(self.azure_requests)

    async def test_routes_require_authentication(self):
        for path in DECISIONS_PATHS:
            response = await self.client.post(
                path, headers={"Authorization": ""}, json=DECISION_REQUEST
            )
            self.assertEqual(response.status_code, 401, response.text)
        self.assertFalse(self.azure_requests)

    def test_llm_route_permissions_accept_and_reject_decisions_paths(self):
        for path in (
            "/openai/decisions",
            "/openai/v1/decisions",
            "/v1/decisions",
            "/decisions",
        ):
            with self.subTest(path=path):
                self.assertTrue(
                    RouteChecks.should_call_route(
                        path, UserAPIKeyAuth(allowed_routes=["llm_api_routes"])
                    )
                )
                with self.assertRaises(HTTPException) as rejected:
                    RouteChecks.should_call_route(
                        path, UserAPIKeyAuth(allowed_routes=["/v1/embeddings"])
                    )
                self.assertEqual(rejected.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
