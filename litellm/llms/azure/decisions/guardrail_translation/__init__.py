from litellm.types.utils import CallTypes

from .handler import AzureDecisionsHandler

guardrail_translation_mappings = {
    CallTypes.adecisions: AzureDecisionsHandler,
}

__all__ = ["guardrail_translation_mappings", "AzureDecisionsHandler"]
