"""Authenticated and selected-deployment Decisions policy composition over real HTTP."""

import copy
import unittest
from unittest.mock import patch

import litellm
import test_decisions_endpoints as transport_fixtures
import test_decisions_policy_regressions as policy_fixtures
from litellm.proxy import proxy_server
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
    ContentFilterGuardrail,
)
from litellm.types.guardrails import GuardrailEventHooks


class ObservedContentFilter(ContentFilterGuardrail):
    """Observe real scanner calls without replacing filtering or proxy hooks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scans = []

    async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
        metadata = request_data.get("litellm_metadata") or {}
        self.scans.append(
            {
                "input_type": input_type,
                "texts": list(inputs.get("texts") or []),
                "team_id": metadata.get("user_api_key_team_id"),
                "user_id": metadata.get("user_api_key_user_id"),
                "principal": metadata.get("user_api_key_auth"),
            }
        )
        return await super().apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type=input_type,
            logging_obj=logging_obj,
        )


class DecisionsPolicyCompositionTests(unittest.IsolatedAsyncioTestCase):
    asyncTearDown = policy_fixtures.DecisionsPolicyRegressionTests.asyncTearDown
    azure_response = policy_fixtures.DecisionsPolicyRegressionTests.azure_response
    make_router = policy_fixtures.DecisionsPolicyRegressionTests.make_router
    use_router = policy_fixtures.DecisionsPolicyRegressionTests.use_router
    use_virtual_principal = (
        policy_fixtures.DecisionsPolicyRegressionTests.use_virtual_principal
    )

    async def asyncSetUp(self):
        await policy_fixtures.DecisionsPolicyRegressionTests.asyncSetUp(self)
        entitlement = patch.object(proxy_server, "premium_user", True)
        entitlement.start()
        self.patches.append(entitlement)

    def add_policy(
        self, name, event, action="BLOCK", keyword="classified", default_on=False
    ):
        policy = ObservedContentFilter(
            guardrail_name=name,
            event_hook=event,
            default_on=default_on,
            blocked_words=[{"keyword": keyword, "action": action}],
            keyword_redaction_tag="[REDACTED]",
        )
        litellm.callbacks.append(policy)
        return policy

    def model(self, name, policies, order=1, group="decisions-model"):
        model = transport_fixtures.deployment(name, f"{name}.openai.azure.com", order)
        model["model_name"] = group
        model["model_info"] = {"id": f"{name}-id"}
        model["litellm_params"]["guardrails"] = policies
        return model

    def attach(self, owner, policy):
        field = {
            "key": "metadata",
            "team": "team_metadata",
            "project": "project_metadata",
        }[owner]
        return self.use_virtual_principal(
            **{field: {"guardrails": [policy.guardrail_name]}}
        )

    async def test_provider_metadata_cannot_shadow_key_team_or_project_policies(self):
        for owner in ("key", "team", "project"):
            for event in (GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call):
                with self.subTest(owner=owner, event=event):
                    policy = self.add_policy(f"required-{owner}-{event.value}", event)
                    self.attach(owner, policy)
                    self.azure_payload = copy.deepcopy(
                        transport_fixtures.AZURE_RESPONSE
                    )
                    body = copy.deepcopy(transport_fixtures.DECISION_REQUEST)
                    if event == GuardrailEventHooks.pre_call:
                        body["input"] = "classified request"
                    else:
                        self.azure_payload["answers"][0]["text"] = "classified answer"
                    self.azure_requests.clear()
                    normal = await self.client.post("/v1/decisions", json=body)
                    self.assertEqual(normal.status_code, 400, normal.text)
                    self.assertIn("blocked", normal.text.lower())
                    self.assertEqual(len(policy.scans), 1)
                    self.assertEqual(
                        len(self.azure_requests),
                        0 if event == GuardrailEventHooks.pre_call else 1,
                    )
                    for forged in ([], ["benign-policy"]):
                        with self.subTest(forged=forged):
                            self.azure_requests.clear()
                            attack = await self.client.post(
                                "/v1/decisions",
                                json=dict(
                                    body,
                                    metadata={
                                        "guardrails": forged,
                                        "document_id": "invoice-123",
                                    },
                                    guardrails=[policy.guardrail_name],
                                ),
                            )
                            self.assertEqual(attack.status_code, 400, attack.text)
                            self.assertEqual(
                                attack.json()["error"]["type"], "invalid_request_error"
                            )
                            self.assertEqual(
                                attack.json()["error"]["param"], "metadata.guardrails"
                            )
                            self.assertFalse(self.azure_requests)
                            self.assertEqual(len(policy.scans), 1)

    async def test_explicit_root_policy_cannot_be_shadowed_by_provider_metadata(self):
        self.use_virtual_principal()
        for event in (GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call):
            with self.subTest(event=event):
                policy = self.add_policy(f"explicit-{event.value}", event)
                body = copy.deepcopy(transport_fixtures.DECISION_REQUEST)
                body["guardrails"] = [policy.guardrail_name]
                if event == GuardrailEventHooks.pre_call:
                    body["input"] = "classified request"
                else:
                    self.azure_payload["answers"][0]["text"] = "classified answer"
                self.azure_requests.clear()
                response = await self.client.post("/v1/decisions", json=body)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn("blocked", response.text.lower())
                self.assertEqual(len(policy.scans), 1)
                for forged in ([], [policy.guardrail_name]):
                    self.azure_requests.clear()
                    response = await self.client.post(
                        "/v1/decisions",
                        json=dict(
                            body,
                            metadata={"guardrails": forged},
                        ),
                    )
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertEqual(
                        response.json()["error"]["param"], "metadata.guardrails"
                    )
                    self.assertFalse(self.azure_requests)

    async def test_provider_model_info_cannot_inject_another_deployments_policy(self):
        required = self.add_policy("required-output", GuardrailEventHooks.post_call)
        unrelated = self.add_policy(
            "unrelated-output",
            GuardrailEventHooks.post_call,
            keyword="not-in-the-answer",
        )
        self.attach("key", required)
        self.use_router(
            litellm.Router(
                model_list=[
                    self.model("selected", []),
                    self.model(
                        "unrelated", [unrelated.guardrail_name], group="other-model"
                    ),
                ],
                num_retries=0,
                retry_after=0,
            )
        )
        self.azure_payload["answers"][0]["text"] = "classified answer"
        metadata = {"model_info": {"id": "unrelated-id"}, "document_id": "invoice-123"}
        response = await self.client.post(
            "/v1/decisions",
            json=dict(
                transport_fixtures.DECISION_REQUEST,
                metadata=metadata,
            ),
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("blocked", response.text.lower())
        self.assertEqual(len(required.scans), 1)
        self.assertFalse(unrelated.scans)
        self.assertEqual(len(self.azure_requests), 1)
        # Existing proxy pricing isolation strips reserved client model_info.
        # It must neither select another deployment nor replace the required policy.
        sent = self.azure_requests[0][2]["metadata"]
        self.assertNotIn("model_info", sent)
        self.assertEqual(sent["document_id"], metadata["document_id"])

    async def test_legitimate_provider_metadata_remains_wire_data(self):
        policy = self.add_policy("required-input", GuardrailEventHooks.pre_call)
        self.attach("team", policy)
        metadata = {
            "document_id": "invoice-123",
            "document_type": "invoice",
        }
        response = await self.client.post(
            "/v1/decisions",
            json=dict(
                transport_fixtures.DECISION_REQUEST,
                metadata=metadata,
            ),
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(policy.scans), 1)
        sent = self.azure_requests[0][2]["metadata"]
        self.assertEqual(sent["document_id"], metadata["document_id"])
        self.assertEqual(sent["document_type"], metadata["document_type"])
        self.assertEqual(sent, metadata)
        self.assertNotIn("guardrails", sent)

    async def test_selected_deployment_real_input_and_output_block_and_mask(self):
        principal = self.use_virtual_principal()
        for event in (
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
        ):
            for action in ("BLOCK", "MASK"):
                with self.subTest(event=event, action=action):
                    policy = self.add_policy(
                        f"deployment-{event.value}-{action}", event, action=action
                    )
                    self.use_router(
                        litellm.Router(
                            model_list=[
                                self.model("selected", [policy.guardrail_name]),
                            ],
                            num_retries=0,
                            retry_after=0,
                        )
                    )
                    self.azure_requests.clear()
                    self.azure_payload = copy.deepcopy(
                        transport_fixtures.AZURE_RESPONSE
                    )
                    body = copy.deepcopy(transport_fixtures.DECISION_REQUEST)
                    if event != GuardrailEventHooks.post_call:
                        body["input"] = "classified customer"
                        body["questions"][0][
                            "instructions"
                        ] = "Find classified information"
                    else:
                        self.azure_payload["answers"][0]["text"] = "classified answer"
                    response = await self.client.post("/v1/decisions", json=body)
                    self.assertEqual(
                        response.status_code,
                        400 if action == "BLOCK" else 200,
                        response.text,
                    )
                    self.assertEqual(len(policy.scans), 1)
                    self.assertEqual(policy.scans[0]["team_id"], principal.team_id)
                    self.assertEqual(policy.scans[0]["user_id"], principal.user_id)
                    self.assertEqual(
                        policy.scans[0]["principal"].team_id, principal.team_id
                    )
                    if action == "BLOCK":
                        self.assertIn("blocked", response.text.lower())
                        self.assertNotIn("answers", response.json())
                        self.assertEqual(
                            len(self.azure_requests),
                            1 if event == GuardrailEventHooks.post_call else 0,
                        )
                    elif event != GuardrailEventHooks.post_call:
                        self.assertEqual(
                            self.azure_requests[0][2]["input"], "[REDACTED] customer"
                        )
                        self.assertEqual(
                            self.azure_requests[0][2]["questions"][0]["instructions"],
                            "Find [REDACTED] information",
                        )
                    else:
                        self.assertEqual(
                            response.json()["answers"][0]["text"], "[REDACTED] answer"
                        )
                        self.assertEqual(
                            response.json()["answers"][0]["probability"], 0.95
                        )
                        self.assertEqual(
                            response.json()["usage"], self.azure_payload["usage"]
                        )

    async def test_unselected_deployment_policy_is_not_aggregated(self):
        self.use_virtual_principal()
        selected = self.add_policy(
            "selected-mask", GuardrailEventHooks.pre_call, action="MASK"
        )
        unselected = self.add_policy(
            "unselected-block", GuardrailEventHooks.pre_call, keyword="customer"
        )
        self.use_router(
            litellm.Router(
                model_list=[
                    self.model("selected", [selected.guardrail_name], order=1),
                    self.model("unselected", [unselected.guardrail_name], order=2),
                ],
                num_retries=0,
                retry_after=0,
            )
        )
        response = await self.client.post(
            "/v1/decisions",
            json=dict(
                transport_fixtures.DECISION_REQUEST,
                input="classified customer",
            ),
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(selected.scans), 1)
        self.assertFalse(unselected.scans)
        self.assertEqual(len(self.azure_requests), 1)
        self.assertIn("selected.openai.azure.com", self.azure_requests[0][0])
        self.assertEqual(self.azure_requests[0][2]["input"], "[REDACTED] customer")

    async def test_fallback_runs_its_own_policies_on_an_independent_attempt_payload(
        self,
    ):
        self.use_virtual_principal()
        primary_pre = self.add_policy(
            "primary-input",
            GuardrailEventHooks.pre_call,
            action="MASK",
            keyword="primary-secret",
        )
        primary_post = self.add_policy(
            "primary-output",
            GuardrailEventHooks.post_call,
            keyword="fallback-answer-secret",
        )
        fallback_pre = self.add_policy(
            "fallback-input",
            GuardrailEventHooks.pre_call,
            action="MASK",
            keyword="fallback-secret",
        )
        fallback_post = self.add_policy(
            "fallback-output",
            GuardrailEventHooks.post_call,
            action="MASK",
            keyword="fallback-answer-secret",
        )
        self.use_router(
            litellm.Router(
                model_list=[
                    self.model(
                        "primary",
                        [primary_pre.guardrail_name, primary_post.guardrail_name],
                    ),
                    self.model(
                        "fallback",
                        [fallback_pre.guardrail_name, fallback_post.guardrail_name],
                        group="backup",
                    ),
                ],
                fallbacks=[{"decisions-model": ["backup"]}],
                num_retries=0,
                retry_after=0,
            )
        )
        self.azure_errors["primary.openai.azure.com"] = (
            429,
            "Primary capacity exceeded",
        )
        self.azure_payload["answers"][0]["text"] = "fallback-answer-secret"
        body = copy.deepcopy(transport_fixtures.DECISION_REQUEST)
        body["input"] = "primary-secret fallback-secret"
        body["questions"][0]["instructions"] = "primary-secret fallback-secret"
        response = await self.client.post("/v1/decisions", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        primary_calls = [
            payload
            for _, _, payload in self.azure_requests
            if payload["model"] == "primary"
        ]
        fallback_calls = [
            payload
            for _, _, payload in self.azure_requests
            if payload["model"] == "fallback"
        ]
        self.assertTrue(primary_calls)
        self.assertTrue(fallback_calls)
        self.assertEqual(len(primary_pre.scans), len(primary_calls))
        self.assertEqual(len(fallback_pre.scans), len(fallback_calls))
        for payload in primary_calls:
            self.assertEqual(payload["input"], "[REDACTED] fallback-secret")
            self.assertEqual(
                payload["questions"][0]["instructions"], "[REDACTED] fallback-secret"
            )
        for payload in fallback_calls:
            self.assertEqual(payload["input"], "primary-secret [REDACTED]")
            self.assertEqual(
                payload["questions"][0]["instructions"], "primary-secret [REDACTED]"
            )
        self.assertFalse(primary_post.scans)
        self.assertEqual(len(fallback_post.scans), 1)
        self.assertEqual(response.json()["answers"][0]["text"], "[REDACTED]")
        self.assertEqual(response.json()["model"], "fallback")

    async def test_proxy_required_policy_is_not_scanned_twice_at_deployment_boundary(
        self,
    ):
        both = [GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call]
        for owner in ("key", "team", "project", "default"):
            with self.subTest(owner=owner):
                required = self.add_policy(
                    f"required-{owner}",
                    both,
                    action="MASK",
                    keyword="global-secret",
                    default_on=owner == "default",
                )
                local = self.add_policy(
                    f"local-{owner}", both, action="MASK", keyword="local-secret"
                )
                if owner == "default":
                    self.use_virtual_principal()
                else:
                    self.attach(owner, required)
                self.use_router(
                    litellm.Router(
                        model_list=[
                            self.model(
                                "selected",
                                [required.guardrail_name, local.guardrail_name],
                            ),
                        ],
                        num_retries=0,
                        retry_after=0,
                    )
                )
                self.azure_requests.clear()
                self.azure_payload["answers"][0]["text"] = "global-secret local-secret"
                response = await self.client.post(
                    "/v1/decisions",
                    json=dict(
                        transport_fixtures.DECISION_REQUEST,
                        input="global-secret local-secret",
                    ),
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(
                    [scan["input_type"] for scan in required.scans],
                    ["request", "response"],
                )
                self.assertEqual(
                    [scan["input_type"] for scan in local.scans],
                    ["request", "response"],
                )
                self.assertEqual(
                    self.azure_requests[0][2]["input"], "[REDACTED] [REDACTED]"
                )
                self.assertEqual(
                    response.json()["answers"][0]["text"], "[REDACTED] [REDACTED]"
                )

    async def test_every_selected_post_policy_receives_prior_redactions(self):
        self.use_virtual_principal()
        redact = self.add_policy(
            "first-output-policy",
            GuardrailEventHooks.post_call,
            action="MASK",
            keyword="first-secret",
        )
        reject = self.add_policy("second-output-policy", GuardrailEventHooks.post_call)
        self.use_router(
            litellm.Router(
                model_list=[
                    self.model(
                        "selected", [redact.guardrail_name, reject.guardrail_name]
                    )
                ],
                num_retries=0,
                retry_after=0,
            )
        )
        self.azure_payload["answers"][0]["text"] = "first-secret classified"
        response = await self.client.post(
            "/v1/decisions", json=transport_fixtures.DECISION_REQUEST
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("blocked", response.text.lower())
        self.assertEqual(len(redact.scans), 1)
        self.assertEqual(len(reject.scans), 1)
        self.assertIn("[REDACTED] classified", reject.scans[0]["texts"])
        self.assertNotIn("first-secret classified", reject.scans[0]["texts"])
        self.assertEqual(len(self.azure_requests), 1)
        self.assertNotIn("answers", response.json())

    async def test_unknown_selected_deployment_policy_fails_closed(self):
        self.use_virtual_principal()
        self.use_router(
            litellm.Router(
                model_list=[self.model("selected", ["missing-policy"])],
                num_retries=0,
                retry_after=0,
            )
        )
        response = await self.client.post(
            "/v1/decisions", json=transport_fixtures.DECISION_REQUEST
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("Unknown guardrail", response.text)
        self.assertFalse(self.azure_requests)


if __name__ == "__main__":
    unittest.main()
