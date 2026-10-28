from typing import Optional

import httpx
from fastapi import APIRouter, Depends, Request, Response

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.azure.passthrough.transformation import AZURE_DECISIONS_ENDPOINT
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.router import Router

_AZURE_HTTP_ERRORS = {
    400: litellm.BadRequestError,
    401: litellm.AuthenticationError,
    403: litellm.PermissionDeniedError,
    404: litellm.NotFoundError,
    422: litellm.UnprocessableEntityError,
    429: litellm.RateLimitError,
    500: litellm.InternalServerError,
    502: litellm.BadGatewayError,
    503: litellm.ServiceUnavailableError,
}


def _invalid_request(message: str, param: str) -> ProxyException:
    return ProxyException(
        message=message, type="invalid_request_error", param=param, code=400
    )


async def _validate_decisions_request(request: Request) -> dict:
    data = await _read_request_body(request=request)
    if not isinstance(data, dict):
        raise _invalid_request(
            "The Decisions request body must be a JSON object", "body"
        )
    for field in ("metadata", "litellm_metadata"):
        if data.get(field) is not None and not isinstance(data[field], dict):
            raise _invalid_request(f"{field} must be a JSON object", field)
    if not isinstance(data.get("model"), str) or not data["model"]:
        raise _invalid_request(
            "A router model is required for the Decisions API", "model"
        )
    if "stream" in data and data["stream"] is not False:
        raise _invalid_request("The Decisions API does not support streaming", "stream")
    return data


router = APIRouter(dependencies=[Depends(_validate_decisions_request)])


async def _adecisions(
    *, model: str, json: dict, custom_llm_provider: Optional[str] = None, **kwargs
):
    deployment_team = (kwargs.get("model_info") or {}).get("team_id")
    request_team = (kwargs.get("litellm_metadata") or {}).get("user_api_key_team_id")
    if deployment_team is not None and deployment_team != request_team:
        raise litellm.PermissionDeniedError(
            message="The Decisions deployment belongs to another team",
            model=model,
            llm_provider=custom_llm_provider or "azure",
            response=httpx.Response(
                403, request=httpx.Request("POST", "http://litellm-router/decisions")
            ),
        )
    deployment, provider, _, _ = litellm.get_llm_provider(
        model=model, custom_llm_provider=custom_llm_provider
    )
    if provider != "azure":
        raise litellm.BadRequestError(
            message="The Decisions API requires an Azure deployment",
            model=model,
            llm_provider=provider,
        )
    deployment = deployment.removeprefix("gpt5_series/")
    kwargs.pop("api_version", None)
    try:
        return await litellm.allm_passthrough_route(
            method="POST",
            endpoint=AZURE_DECISIONS_ENDPOINT,
            model=deployment,
            custom_llm_provider=provider,
            json={**json, "model": deployment},
            **kwargs,
        )
    except httpx.HTTPStatusError as e:
        error_class = _AZURE_HTTP_ERRORS.get(e.response.status_code)
        if error_class is not None:
            raise error_class(
                message=f"Azure Decisions API - {e.response.text}",
                model=deployment,
                llm_provider=provider,
                response=e.response,
            ) from e
        raise litellm.APIError(
            status_code=e.response.status_code,
            message=f"Azure Decisions API - {e.response.text}",
            model=deployment,
            llm_provider=provider,
            request=e.request,
        ) from e


def _validate_selected_guardrails(data: dict, llm_router: Router) -> None:
    from litellm.proxy.utils import _check_and_merge_model_level_guardrails

    guardrail_data = _check_and_merge_model_level_guardrails(data, llm_router)
    selected = CustomGuardrail().get_guardrail_from_metadata(guardrail_data)
    if not isinstance(selected, list):
        raise _invalid_request("guardrails must be a list", "guardrails")
    configured = {
        callback.guardrail_name
        for callback in litellm.callbacks
        if isinstance(callback, CustomGuardrail)
    }
    for selection in selected:
        names = selection if isinstance(selection, dict) else [selection]
        for name in names:
            if not isinstance(name, str):
                raise _invalid_request("guardrail names must be strings", "guardrails")
            if name not in configured:
                raise _invalid_request(
                    f"Unknown guardrail for the Decisions API: {name!r}", "guardrails"
                )


async def route_decisions_request(data: dict, llm_router: Optional[Router]):
    if llm_router is None:
        raise _invalid_request(
            "The Decisions API requires a configured router", "model"
        )
    model = data.get("model")
    team_id = (data.get("litellm_metadata") or {}).get("user_api_key_team_id")
    team_model = llm_router.map_team_model(model, team_id) if team_id else None
    model = team_model or model
    _validate_selected_guardrails(data, llm_router)
    if not isinstance(model, str) or (
        team_model is None
        and model not in llm_router.model_names
        and model not in llm_router.model_group_alias
        and not llm_router.has_model_id(model)
    ):
        raise _invalid_request(
            f"Invalid model for the Decisions API: {model!r}", "model"
        )

    body = dict(data["json"])
    for field in ("input", "questions", "metadata"):
        if field in data:
            body[field] = data[field]
        else:
            body.pop(field, None)
    kwargs = {
        key: data[key]
        for key in (
            "litellm_metadata",
            "litellm_logging_obj",
            "litellm_call_id",
            "litellm_trace_id",
            "proxy_server_request",
            "secret_fields",
            "user",
            "request_timeout",
            "timeout",
            "allowed_model_region",
            "api_base",
            "api_key",
            "api_version",
        )
        if key in data
    }
    overrides = data.get("router_settings_override") or {}
    for key in (
        "fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "num_retries",
        "timeout",
        "model_group_retry_policy",
    ):
        if key in overrides:
            kwargs[key] = overrides[key]
    return await llm_router._ageneric_api_call_with_fallbacks(
        model=model, original_function=_adecisions, json=body, **kwargs
    )


@router.post(
    "/openai/v1/decisions",
    dependencies=[Depends(user_api_key_auth)],
    tags=["decisions"],
)
@router.post(
    "/openai/decisions", dependencies=[Depends(user_api_key_auth)], tags=["decisions"]
)
@router.post(
    "/v1/decisions", dependencies=[Depends(user_api_key_auth)], tags=["decisions"]
)
@router.post(
    "/decisions", dependencies=[Depends(user_api_key_auth)], tags=["decisions"]
)
async def decisions(
    request: Request,
    fastapi_response: Response,
    body: dict = Depends(_validate_decisions_request),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        version,
    )

    # Only authenticated preprocessing may add provider credentials at the root.
    data = {
        "model": body["model"],
        "endpoint": AZURE_DECISIONS_ENDPOINT,
        "json": {
            key: value
            for key, value in body.items()
            if key not in ("litellm_metadata", "guardrails")
        },
        "litellm_metadata": body.get("litellm_metadata") or {},
    }
    for field in ("input", "questions", "metadata", "guardrails"):
        if field in body:
            data[field] = body[field]
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        response = await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="allm_passthrough_route",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            version=version,
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(fastapi_response.headers),
            media_type=response.headers.get("content-type", "application/json"),
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
