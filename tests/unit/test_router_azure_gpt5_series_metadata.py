"""`gpt5_series/` is a routing selector: Router metadata must ignore it, requests must keep it."""

import unittest
from unittest.mock import patch

import litellm

DEPLOYMENT = {
    "api_base": "https://mock-azure.openai.azure.com/",
    "api_key": "mock-azure-key",
    "api_version": "2025-03-01-preview",
}


def _router(model: str) -> litellm.Router:
    return litellm.Router(
        model_list=[{"model_name": "m", "litellm_params": {"model": model, **DEPLOYMENT}}],
        num_retries=0,
    )


class AzureGpt5SeriesRouterMetadataTests(unittest.TestCase):
    def test_selector_resolves_the_real_model_limits(self):
        router = _router("azure/gpt5_series/gpt-5.2")
        self.addCleanup(router.reset)
        info = router.get_router_model_info(deployment=router.model_list[0], received_model_name="m")
        self.assertEqual(info["max_input_tokens"], litellm.get_model_info("azure/gpt-5.2")["max_input_tokens"])

    def test_selector_cannot_bypass_the_input_context_limit(self):
        router = _router("azure/gpt5_series/gpt-5.2")
        self.addCleanup(router.reset)
        limit = litellm.get_model_info("azure/gpt-5.2")["max_input_tokens"]
        messages = [{"role": "user", "content": "x"}]
        with patch.object(litellm, "token_counter", return_value=limit):
            router._pre_call_checks(model="m", healthy_deployments=router.model_list, messages=messages)
        with patch.object(litellm, "token_counter", return_value=limit + 1):
            with self.assertRaises(litellm.ContextWindowExceededError):
                router._pre_call_checks(model="m", healthy_deployments=router.model_list, messages=messages)

    def test_unknown_model_after_selector_is_not_given_other_limits(self):
        router = _router("azure/gpt5_series/not-a-mapped-model")
        self.addCleanup(router.reset)
        # v1.106.x Router registers every deployment name as a zeroed cost-map stub,
        # so an unknown model resolves to that stub rather than raising.
        info = router.get_router_model_info(deployment=router.model_list[0], received_model_name="m")
        self.assertIsNone(info.get("max_input_tokens"))

    def test_only_a_leading_selector_is_stripped(self):
        router = _router("azure/custom-gpt5_series/gpt-5.2")
        self.addCleanup(router.reset)
        info = router.get_router_model_info(deployment=router.model_list[0], received_model_name="m")
        self.assertIsNone(info.get("max_input_tokens"))


class UnmappedGpt5SeriesDeploymentTests(unittest.TestCase):
    """A deployment name that is not a cost-map key (e.g. a DataZone `-dz` name) keeps its own registered metadata."""

    def test_unmapped_deployment_does_not_log_a_router_error(self):
        router = _router("azure/gpt5_series/gpt-6-luna-dz")
        self.addCleanup(router.reset)
        with self.assertNoLogs("LiteLLM Router", level="ERROR"):
            info = router.get_router_model_info(deployment=router.model_list[0], received_model_name="m")
        self.assertIsNone(info.get("max_input_tokens"))

    def test_unmapped_deployment_uses_its_explicit_model_info(self):
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "m",
                    "litellm_params": {"model": "azure/gpt5_series/gpt-6-luna-dz", **DEPLOYMENT},
                    "model_info": {"max_input_tokens": 1234},
                }
            ],
            num_retries=0,
        )
        self.addCleanup(router.reset)
        info = router.get_router_model_info(deployment=router.model_list[0], received_model_name="m")
        self.assertEqual(info["max_input_tokens"], 1234)

    def test_unmapped_name_without_the_selector_still_reports_an_error(self):
        router = _router("azure/my-unmapped-deployment")
        self.addCleanup(router.reset)
        with self.assertLogs("LiteLLM Router", level="ERROR") as logs:
            router.get_router_model_info(deployment=router.model_list[0], received_model_name="m")
        self.assertIn("Could not identify azure model", logs.output[0])


if __name__ == "__main__":
    unittest.main()
