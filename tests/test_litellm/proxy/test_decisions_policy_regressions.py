"""Real proxy/Router policy regressions with only the Azure HTTP transport replaced."""

import asyncio
import copy
import unittest
from unittest.mock import patch

import httpx
import test_decisions_endpoints as fixtures
from fastapi import HTTPException

import litellm
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
    ContentFilterGuardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
    UnifiedLLMGuardrails,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CredentialItem


class DecisionsPolicyRegressionTests(unittest.IsolatedAsyncioTestCase):
    # Reuse the existing HTTP fixture without inheriting/rerunning its 17 tests.
    make_router = fixtures.NativeAzureDecisionsTests.make_router
    use_router = fixtures.NativeAzureDecisionsTests.use_router
    use_virtual_principal = fixtures.NativeAzureDecisionsTests.use_virtual_principal

    async def asyncSetUp(self):
        self._baseline_tasks = asyncio.all_tasks()
        self.azure_payload = copy.deepcopy(fixtures.AZURE_RESPONSE)
        await fixtures.NativeAzureDecisionsTests.asyncSetUp(self)

    async def asyncTearDown(self):
        try:
            current = asyncio.current_task()
            loop = asyncio.get_running_loop()
            while True:
                # The logging consumer is long-lived; drain its finite work through
                # flush(), rather than cancelling wrappers holding unawaited coroutines.
                pending = asyncio.all_tasks() - self._baseline_tasks - {current}
                pending.discard(GLOBAL_LOGGING_WORKER._worker_task)
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                    continue
                if GLOBAL_LOGGING_WORKER._bound_loop is loop:
                    await GLOBAL_LOGGING_WORKER.flush()
                pending = asyncio.all_tasks() - self._baseline_tasks - {current}
                pending.discard(GLOBAL_LOGGING_WORKER._worker_task)
                if not pending:
                    break
            if (
                GLOBAL_LOGGING_WORKER._bound_loop is loop
                and GLOBAL_LOGGING_WORKER._worker_task not in self._baseline_tasks
            ):
                await GLOBAL_LOGGING_WORKER.stop()
        finally:
            await fixtures.NativeAzureDecisionsTests.asyncTearDown(self)

    def azure_response(self, request):
        response = fixtures.NativeAzureDecisionsTests.azure_response(self, request)
        if response.status_code == 200:
            payload = copy.deepcopy(self.azure_payload)
            payload["model"] = response.json()["model"]
            return httpx.Response(200, json=payload)
        return response

    def add_filter(self, event_hook, action="BLOCK", default_on=True):
        policy = ContentFilterGuardrail(
            guardrail_name="decisions-content-policy",
            blocked_words=[{"keyword": "classified", "action": action}],
            keyword_redaction_tag="[REDACTED]",
            event_hook=event_hook,
            default_on=default_on,
        )
        litellm.callbacks.append(policy)
        return policy

    def regional_deployment(self, name, region, order, group="decisions-model"):
        result = fixtures.deployment(name, f"{name}.openai.azure.com", order)
        result["model_name"] = group
        result["litellm_params"]["region_name"] = region
        return result

    async def test_chat_input_policy_cannot_be_redirected_by_decisions_body_fields(
        self,
    ):
        self.use_virtual_principal()
        self.add_filter(GuardrailEventHooks.pre_call)
        response = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": "decisions-model",
                "messages": [{"role": "user", "content": "classified chat content"}],
                "endpoint": "openai/v1/decisions",
                "input": "benign decoy",
                "questions": [],
                "proxy_server_request": {"url": "http://gateway/v1/decisions"},
            },
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("blocked", response.text.lower())
        self.assertFalse(self.azure_requests)

    async def test_chat_output_policy_ignores_spoofed_decisions_endpoint(self):
        principal = self.use_virtual_principal(request_route="/v1/chat/completions")
        for action in ("BLOCK", "MASK"):
            with self.subTest(action=action):
                policy = self.add_filter(GuardrailEventHooks.post_call, action=action)
                data = {
                    "model": "decisions-model",
                    "endpoint": "openai/v1/decisions",
                    "input": "benign decoy",
                    "questions": [],
                    "proxy_server_request": {
                        "url": "http://gateway/v1/chat/completions"
                    },
                    "guardrail_to_apply": policy,
                }
                response = litellm.ModelResponse(
                    model="decisions-model",
                    choices=[
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "classified chat answer",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                )
                if action == "BLOCK":
                    with self.assertRaises(HTTPException) as rejected:
                        await UnifiedLLMGuardrails().async_post_call_success_hook(
                            data=data, user_api_key_dict=principal, response=response
                        )
                    self.assertEqual(rejected.exception.status_code, 400)
                    self.assertIn("blocked", str(rejected.exception.detail).lower())
                else:
                    translated = (
                        await UnifiedLLMGuardrails().async_post_call_success_hook(
                            data=data, user_api_key_dict=principal, response=response
                        )
                    )
                    self.assertEqual(
                        translated.choices[0].message.content,
                        "[REDACTED] chat answer",
                    )
        self.assertFalse(self.azure_requests)

    async def test_actual_apply_guardrail_blocks_each_provider_prompt_surface(self):
        self.use_virtual_principal()
        self.add_filter(GuardrailEventHooks.pre_call)
        for surface in ("input", "instructions", "choice"):
            with self.subTest(surface=surface):
                body = copy.deepcopy(fixtures.DECISION_REQUEST)
                if surface == "input":
                    body["input"] = "classified material"
                elif surface == "instructions":
                    body["questions"][0]["instructions"] = "Read classified material"
                else:
                    body["questions"] = [
                        {
                            "type": "choice",
                            "name": "category",
                            "instructions": "Choose",
                            "choices": [
                                {"value": "restricted", "description": "classified"}
                            ],
                        }
                    ]
                response = await self.client.post("/v1/decisions", json=body)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn("blocked", response.text.lower())
                self.assertFalse(self.azure_requests)

    async def test_actual_apply_guardrail_masks_input_question_and_choice_text(self):
        self.use_virtual_principal()
        self.add_filter(GuardrailEventHooks.pre_call, action="MASK")
        body = copy.deepcopy(fixtures.DECISION_REQUEST)
        body["input"] = "classified customer"
        body["questions"] = [
            {
                "type": "choice",
                "name": "classified-question",
                "instructions": "Find classified",
                "choices": [
                    {
                        "value": "classified-choice",
                        "description": "classified option",
                    }
                ],
            }
        ]
        response = await self.client.post("/v1/decisions", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        sent = self.azure_requests[0][2]
        self.assertEqual(sent["input"], "[REDACTED] customer")
        self.assertEqual(sent["questions"][0]["instructions"], "Find [REDACTED]")
        self.assertEqual(
            sent["questions"][0]["choices"][0]["description"], "[REDACTED] option"
        )
        self.assertEqual(sent["questions"][0]["type"], "choice")
        self.assertEqual(sent["questions"][0]["name"], "classified-question")
        self.assertEqual(
            sent["questions"][0]["choices"][0]["value"], "classified-choice"
        )
        self.assertEqual(response.json()["usage"], fixtures.AZURE_RESPONSE["usage"])

    async def test_actual_apply_guardrail_post_call_blocks_returned_answer(self):
        self.use_virtual_principal()
        self.add_filter(GuardrailEventHooks.post_call)
        self.azure_payload["answers"][0]["text"] = "classified answer"
        response = await self.client.post(
            "/v1/decisions", json=fixtures.DECISION_REQUEST
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("blocked", response.text.lower())
        self.assertNotIn("answers", response.json())
        self.assertEqual(len(self.azure_requests), 1)

    async def test_actual_apply_guardrail_post_masks_text_not_schema_identifiers(self):
        self.use_virtual_principal()
        self.add_filter(GuardrailEventHooks.post_call, action="MASK")
        self.azure_payload["answers"] = [
            {
                "type": "choice",
                "name": "classified-question",
                "choice": "classified-choice",
                "confidence": 0.95,
                "probabilities": [{"value": "classified-choice", "probability": 0.95}],
                "text": "classified answer",
            }
        ]
        response = await self.client.post(
            "/v1/decisions", json=fixtures.DECISION_REQUEST
        )
        self.assertEqual(response.status_code, 200, response.text)
        answer = response.json()["answers"][0]
        self.assertEqual(answer["text"], "[REDACTED] answer")
        self.assertEqual(answer["name"], "classified-question")
        self.assertEqual(answer["choice"], "classified-choice")
        self.assertEqual(answer["probabilities"][0]["value"], "classified-choice")
        self.assertEqual(answer["probabilities"][0]["probability"], 0.95)
        self.assertEqual(answer["type"], "choice")
        self.assertEqual(response.json()["usage"], self.azure_payload["usage"])
        self.assertEqual(response.json()["model"], "decision-dz")

    async def test_actual_apply_guardrail_during_call_can_reject(self):
        self.use_virtual_principal()
        self.add_filter(GuardrailEventHooks.during_call)
        response = await self.client.post(
            "/v1/decisions",
            json=dict(
                fixtures.DECISION_REQUEST,
                input="classified",
            ),
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("blocked", response.text.lower())
        self.assertNotIn("answers", response.json())

    async def test_selected_post_policy_fails_closed_on_unsupported_answers(self):
        self.use_virtual_principal()
        self.add_filter(GuardrailEventHooks.post_call)
        for answers in (None, {"text": "classified"}, ["classified"]):
            with self.subTest(answers=answers):
                self.azure_payload["answers"] = answers
                response = await self.client.post(
                    "/v1/decisions", json=fixtures.DECISION_REQUEST
                )
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn("Unsupported Decisions guardrail", response.text)
                self.assertNotIn("answers", response.json())

    async def test_explicitly_selected_apply_guardrail_is_not_ignored(self):
        self.use_virtual_principal()
        self.add_filter(GuardrailEventHooks.pre_call, default_on=False)
        response = await self.client.post(
            "/v1/decisions",
            json=dict(
                fixtures.DECISION_REQUEST,
                input="classified",
                guardrails=["decisions-content-policy"],
            ),
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("blocked", response.text.lower())
        self.assertFalse(self.azure_requests)
        response = await self.client.post(
            "/v1/decisions",
            json=dict(
                fixtures.DECISION_REQUEST,
                guardrails=["decisions-content-policy"],
            ),
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn("guardrails", self.azure_requests[0][2])

    async def test_unknown_selected_guardrail_fails_closed(self):
        self.use_virtual_principal()
        for selection, expected_error in (
            ({"guardrails": ["missing-policy"]}, "Unknown guardrail"),
            (
                {"metadata": {"guardrails": ["missing-policy"]}},
                "metadata.guardrails is reserved",
            ),
        ):
            with self.subTest(selection=selection):
                response = await self.client.post(
                    "/v1/decisions", json=dict(fixtures.DECISION_REQUEST, **selection)
                )
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn(expected_error, response.text)
                self.assertFalse(self.azure_requests)

    async def test_malformed_dynamic_guardrail_configuration_returns_typed_400(self):
        self.use_virtual_principal()
        self.add_filter(GuardrailEventHooks.pre_call, default_on=False)
        for configuration in (
            None,
            [],
            {"extra_body": None},
            {"extra_body": []},
        ):
            with self.subTest(configuration=configuration):
                response = await self.client.post(
                    "/v1/decisions",
                    json=dict(
                        fixtures.DECISION_REQUEST,
                        guardrails=[{"decisions-content-policy": configuration}],
                    ),
                )
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(
                    response.json()["error"]["type"], "invalid_request_error"
                )
                self.assertEqual(response.json()["error"]["param"], "guardrails")
                self.assertFalse(self.azure_requests)

    async def test_selected_policy_rejects_unsupported_multimodal_input(self):
        self.use_virtual_principal()
        self.add_filter(GuardrailEventHooks.pre_call)
        response = await self.client.post(
            "/v1/decisions",
            json=dict(
                fixtures.DECISION_REQUEST,
                input=[
                    {
                        "type": "input_image",
                        "image_url": "https://example.invalid/x.png",
                    }
                ],
            ),
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("Unsupported Decisions guardrail input", response.text)
        self.assertFalse(self.azure_requests)

    async def test_eu_region_is_authoritative_without_optional_token_prechecks(self):
        self.use_virtual_principal(allowed_model_region="eu")
        self.use_router(
            litellm.Router(
                model_list=[
                    self.regional_deployment("us-preferred", "us", 1),
                    self.regional_deployment("eu-allowed", "eu", 2),
                ],
                enable_pre_call_checks=False,
                num_retries=0,
                retry_after=0,
            )
        )
        response = await self.client.post(
            "/v1/decisions",
            json=dict(
                fixtures.DECISION_REQUEST,
                allowed_model_region="us",
            ),
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(self.azure_requests), 1)
        self.assertIn("eu-allowed.openai.azure.com", self.azure_requests[0][0])

    async def test_region_survives_order_and_model_group_fallback(self):
        self.use_virtual_principal(allowed_model_region="eu")
        self.use_router(
            litellm.Router(
                model_list=[
                    self.regional_deployment("eu-primary", "eu", 1),
                    self.regional_deployment("us-order-fallback", "us", 2),
                    self.regional_deployment("us-group-preferred", "us", 1, "backup"),
                    self.regional_deployment("eu-group-fallback", "eu", 2, "backup"),
                ],
                fallbacks=[{"decisions-model": ["backup"]}],
                enable_pre_call_checks=False,
                num_retries=0,
                retry_after=0,
            )
        )
        self.azure_errors["eu-primary.openai.azure.com"] = (429, "EU capacity exceeded")
        response = await self.client.post(
            "/v1/decisions", json=fixtures.DECISION_REQUEST
        )
        self.assertEqual(response.status_code, 200, response.text)
        attempted = [body["model"] for _, _, body in self.azure_requests]
        self.assertEqual(attempted[0], "eu-primary")
        self.assertEqual(attempted[-1], "eu-group-fallback")
        self.assertTrue(set(attempted) <= {"eu-primary", "eu-group-fallback"})
        self.assertEqual(response.json()["model"], "eu-group-fallback")
        self.assertEqual(response.json()["answers"], self.azure_payload["answers"])
        for url, _, _ in self.azure_requests:
            self.assertIn(
                httpx.URL(url).host,
                {"eu-primary.openai.azure.com", "eu-group-fallback.openai.azure.com"},
            )

    async def test_no_eligible_region_never_contacts_us_or_unknown_deployments(self):
        self.use_virtual_principal(allowed_model_region="eu")
        self.use_router(
            litellm.Router(
                model_list=[
                    self.regional_deployment("us-only", "us", 1),
                    fixtures.deployment(
                        "unknown-region", "unknown.openai.azure.com", 2
                    ),
                ],
                enable_pre_call_checks=False,
                num_retries=0,
                retry_after=0,
            )
        )
        response = await self.client.post(
            "/v1/decisions", json=fixtures.DECISION_REQUEST
        )
        self.assertNotEqual(response.status_code, 200, response.text)
        self.assertFalse(self.azure_requests)

    async def test_public_chat_selection_enforces_eu_unrestricted_and_us_regions(self):
        deployments = []
        for region in ("eu", "us"):
            model = self.regional_deployment(f"chat-{region}", region, 1)
            model["litellm_params"]["model"] = "azure/gpt-4o"
            model["model_info"] = {"id": f"chat-{region}", "base_model": "gpt-4o"}
            deployments.append(model)
        self.use_router(
            litellm.Router(
                model_list=deployments,
                enable_pre_call_checks=True,
                routing_strategy="simple-shuffle",
                num_retries=0,
            )
        )
        messages = [{"role": "user", "content": "Hello"}]
        for region, expected_ids in (
            ("eu", {"chat-eu"}),
            (None, {"chat-eu", "chat-us"}),
            ("us", {"chat-us"}),
        ):
            with self.subTest(region=region):
                healthy = await self.router.async_get_healthy_deployments(
                    model="decisions-model",
                    messages=messages,
                    request_kwargs={"allowed_model_region": region},
                )
                self.assertEqual(len(healthy), len(expected_ids))
                self.assertEqual(
                    {deployment["model_info"]["id"] for deployment in healthy},
                    expected_ids,
                )
                selected = self.router.get_available_deployment(
                    model="decisions-model",
                    messages=messages,
                    request_kwargs={"allowed_model_region": region},
                )
                self.assertIn(selected["model_info"]["id"], expected_ids)
        self.assertFalse(self.azure_requests)

    async def test_router_specific_nonchat_deployment_cannot_escape_region(self):
        model = self.regional_deployment("us-specific", "us", 1)
        model["model_info"] = {"id": "us-deployment-id"}
        self.use_router(
            litellm.Router(
                model_list=[model], enable_pre_call_checks=False, num_retries=0
            )
        )
        with self.assertRaises(litellm.PermissionDeniedError) as rejected:
            await self.router.async_get_available_deployment(
                model="us-deployment-id", request_kwargs={"allowed_model_region": "eu"}
            )
        self.assertEqual(rejected.exception.status_code, 403)
        self.assertEqual(rejected.exception.response.status_code, 403)
        with self.assertRaises(litellm.PermissionDeniedError) as rejected:
            self.router.get_available_deployment(
                model="us-deployment-id", request_kwargs={"allowed_model_region": "eu"}
            )
        self.assertEqual(rejected.exception.status_code, 403)
        self.assertEqual(rejected.exception.response.status_code, 403)

    async def test_opaque_team_public_model_resolves_only_authenticated_team(self):
        self.use_virtual_principal()
        deployments = []
        for team, name in (
            ("authenticated-team", "own-team"),
            ("other-team", "other-team"),
        ):
            model = fixtures.deployment(name, f"{name}.openai.azure.com", 1)
            model["model_name"] = f"opaque-{team}-deployment"
            model["model_info"] = {
                "id": f"{team}-id",
                "team_id": team,
                "team_public_model_name": "decisions-model",
            }
            deployments.append(model)
        self.use_router(litellm.Router(model_list=deployments, num_retries=0))
        response = await self.client.post(
            "/v1/decisions", json=fixtures.DECISION_REQUEST
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(self.azure_requests), 1)
        self.assertIn("own-team.openai.azure.com", self.azure_requests[0][0])
        self.assertEqual(self.azure_requests[0][1], "key-own-team")
        self.azure_requests.clear()
        for forbidden in (
            "opaque-other-team-deployment",
            "other-team-id",
            "not-configured",
        ):
            with self.subTest(forbidden=forbidden):
                response = await self.client.post(
                    "/v1/decisions",
                    json=dict(
                        fixtures.DECISION_REQUEST,
                        model=forbidden,
                        litellm_metadata={"user_api_key_team_id": "other-team"},
                    ),
                )
                self.assertNotEqual(response.status_code, 200, response.text)
                if forbidden == "other-team-id":
                    self.assertEqual(response.status_code, 403, response.text)
                self.assertFalse(self.azure_requests)

    async def test_trusted_team_and_project_credentials_override_actual_azure_transport(
        self,
    ):
        credentials = [
            CredentialItem(
                credential_name=f"{owner}-decisions-resource",
                credential_info={},
                credential_values={
                    "api_base": f"https://{owner}-resource.openai.azure.com/",
                    "api_key": f"trusted-{owner}-key",
                },
            )
            for owner in ("team", "project")
        ]
        team_metadata = {
            "model_config": {
                "defaultconfig": {
                    "azure": {"litellm_credentials": "team-decisions-resource"},
                }
            }
        }
        project_metadata = {
            "model_config": {
                "decisions-model": {
                    "azure": {"litellm_credentials": "project-decisions-resource"},
                }
            }
        }
        with (
            patch.object(litellm, "credential_list", credentials),
            patch.object(litellm, "enable_model_config_credential_overrides", True),
        ):
            for owner in ("team", "project"):
                with self.subTest(owner=owner):
                    self.azure_requests.clear()
                    self.use_virtual_principal(
                        team_metadata=team_metadata,
                        project_metadata=project_metadata if owner == "project" else {},
                    )
                    response = await self.client.post(
                        "/v1/decisions",
                        json=dict(
                            fixtures.DECISION_REQUEST,
                            api_base="https://attacker.invalid/",
                            api_key="hostile-client-key",
                            custom_llm_provider="openai",
                            endpoint="hostile-path",
                        ),
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(len(self.azure_requests), 1)
                    url, key, body = self.azure_requests[0]
                    self.assertEqual(
                        url,
                        f"https://{owner}-resource.openai.azure.com/openai/v1/decisions",
                    )
                    self.assertEqual(key, f"trusted-{owner}-key")
                    self.assertEqual(body["model"], "decision-dz")


if __name__ == "__main__":
    unittest.main()
