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
        with self.assertRaises(Exception):
            router.get_router_model_info(deployment=router.model_list[0], received_model_name="m")

    def test_only_a_leading_selector_is_stripped(self):
        router = _router("azure/custom-gpt5_series/gpt-5.2")
        self.addCleanup(router.reset)
        info = router.get_router_model_info(deployment=router.model_list[0], received_model_name="m")
        self.assertIsNone(info.get("max_input_tokens"))



if __name__ == "__main__":
    unittest.main()
