"""
Polls LiteLLM_ManagedObjectTable to check if the batch job is complete, and if the cost has been tracked.
"""

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.constants import (
    MANAGED_OBJECT_STALENESS_CUTOFF_DAYS,
    MAX_OBJECTS_PER_POLL_CYCLE,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient, ProxyLogging
    from litellm.router import Router


CHECK_BATCH_COST_USER_AGENT = "LiteLLM Proxy/CheckBatchCost"


def _get_passthrough_batch_retrieve_kwargs(model_id: str) -> dict:
    """Build litellm.aretrieve_batch kwargs using passthrough env credentials."""
    from litellm.proxy.proxy_server import passthrough_endpoint_router
    from litellm.secret_managers.main import get_secret_str

    custom_llm_provider = "openai"
    if model_id.startswith("azure/") or "azure" in model_id.lower():
        custom_llm_provider = "azure"

    retrieve_kwargs: dict = {"custom_llm_provider": custom_llm_provider}

    if passthrough_endpoint_router is not None:
        api_key = passthrough_endpoint_router.get_credentials(
            custom_llm_provider=custom_llm_provider,
            region_name=None,
        )
        if api_key:
            retrieve_kwargs["api_key"] = api_key

    if custom_llm_provider == "azure":
        api_base = get_secret_str("AZURE_API_BASE")
        if api_base:
            retrieve_kwargs["api_base"] = api_base
        if model_id.startswith("azure/"):
            retrieve_kwargs["model"] = model_id.split("/", 1)[1]

    return retrieve_kwargs


def _extract_raw_output_file_id(output_file_id: str) -> str:
    """Resolve a batch output file ID down to the raw provider file ID.

    Handles both managed-ID encodings before falling back to the input:
    - passthrough managed IDs: litellm_proxy:passthrough;...;raw_id,<id>
    - native unified file IDs: base64(litellm_proxy;...;llm_output_file_id,<id>)
    """
    from litellm.proxy.openai_files_endpoints.common_utils import (
        _is_base64_encoded_unified_file_id,
    )
    from litellm.proxy.pass_through_endpoints.managed_id_codec import (
        decode as decode_passthrough_managed_id,
    )

    passthrough_payload = decode_passthrough_managed_id(output_file_id)
    if passthrough_payload is not None:
        return passthrough_payload.raw_provider_id

    decoded = _is_base64_encoded_unified_file_id(output_file_id)
    if decoded:
        try:
            return decoded.split("llm_output_file_id,")[1].split(";")[0]
        except (IndexError, AttributeError):
            pass

    return output_file_id


def _resolve_provider_and_model(
    deployment_info: Optional[Any],
    passthrough_retrieve_kwargs: Optional[dict],
) -> Tuple[Optional[str], Optional[str], dict]:
    """Resolve ``(llm_provider, model_name, deployment_model_info)`` for cost calc.

    Native (router) batches resolve the deployment's configured provider/model
    plus any custom batch pricing in ``model_info``.  Passthrough batches have no
    deployment, so the provider comes from the env-cred kwargs and ``model_name``
    is left ``None`` so that cost is derived per-line from the batch output JSONL.
    """
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    if deployment_info is not None:
        model_name, llm_provider, _, _ = get_llm_provider(
            model=deployment_info.litellm_params.model,
            custom_llm_provider=deployment_info.litellm_params.custom_llm_provider,
        )
        deployment_model_info = (
            deployment_info.model_info.model_dump()
            if deployment_info.model_info
            else {}
        )
        return llm_provider, model_name, deployment_model_info

    llm_provider = (passthrough_retrieve_kwargs or {}).get(
        "custom_llm_provider", "openai"
    )
    return llm_provider, None, {}


class CheckBatchCost:
    def __init__(
        self,
        proxy_logging_obj: "ProxyLogging",
        prisma_client: "PrismaClient",
        llm_router: "Router",
    ):
        from litellm.proxy.utils import PrismaClient, ProxyLogging
        from litellm.router import Router

        self.proxy_logging_obj: ProxyLogging = proxy_logging_obj
        self.prisma_client: PrismaClient = prisma_client
        self.llm_router: Router = llm_router
        # Cached after the first poll cycle. Once we know the column is absent we skip
        # the guaranteed-failing primary query on every subsequent cycle.
        self._has_batch_processed_column: bool = True

    async def _get_user_info(self, batch_id, user_id) -> dict:
        """
        Look up user email and key alias by user_id for enriching the S3 callback metadata.
        Returns a dict with user_api_key_user_email and user_api_key_alias (both may be None).
        """
        try:
            user_row = await self.prisma_client.db.litellm_usertable.find_unique(
                where={"user_id": user_id}
            )
            if user_row is None:
                return {}
            return {
                "user_api_key_user_email": getattr(user_row, "user_email", None),
                "user_api_key_alias": getattr(user_row, "user_alias", None),
            }
        except Exception as e:
            verbose_proxy_logger.error(f"CheckBatchCost: could not look up user {user_id} for batch {batch_id}: {e}")
            return {}

    async def _cleanup_stale_managed_objects(self) -> None:
        """
        Mark managed objects older than MANAGED_OBJECT_STALENESS_CUTOFF_DAYS days
        in non-terminal states as 'stale_expired'. These will never complete and
        should not be polled.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=MANAGED_OBJECT_STALENESS_CUTOFF_DAYS)
        result = await self.prisma_client.db.litellm_managedobjecttable.update_many(
            where={
                "file_purpose": "batch",
                "status": {"not_in": ["completed", "complete", "failed", "expired", "cancelled", "stale_expired"]},
                "created_at": {"lt": cutoff},
            },
            data={"status": "stale_expired"},
        )
        if result > 0:
            verbose_proxy_logger.warning(
                f"CheckBatchCost: marked {result} stale managed objects "
                f"(older than {MANAGED_OBJECT_STALENESS_CUTOFF_DAYS} days) as stale_expired"
            )

    async def _fallback_find_jobs(self) -> list:
        """Query batch jobs without the batch_processed filter (for older schemas)."""
        return await self.prisma_client.db.litellm_managedobjecttable.find_many(
            where={
                "file_purpose": "batch",
                "status": {
                    "not_in": [
                        "failed",
                        "expired",
                        "cancelled",
                        "complete",
                        "completed",
                        "stale_expired",
                    ]
                },
            },
            take=MAX_OBJECTS_PER_POLL_CYCLE,
            order={"created_at": "asc"},
        )

    async def check_batch_cost(self):
        """
        Check if the batch JOB has been tracked.
        - get all status="validating" and file_purpose="batch" jobs
        - check if batch is now complete
        - if not, return False
        - if so, return True
        """
        from litellm.batches.batch_utils import (
            _get_file_content_as_dictionary,
            calculate_batch_cost_and_usage,
        )
        from litellm.files.main import afile_content
        from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
        from litellm.proxy.openai_files_endpoints.common_utils import (
            _is_base64_encoded_unified_file_id,
            get_batch_id_from_unified_batch_id,
            get_model_id_from_unified_batch_id,
            is_passthrough_unified_batch_id,
        )

        try:
            from litellm.integrations.prometheus import PrometheusLogger
            prom_logger = PrometheusLogger.get_instance()
        except Exception as e:
            verbose_proxy_logger.error(f"CheckBatchCost: could not get Prometheus logger: {e}")
            prom_logger = None

        processed_models: List[Tuple[Optional[str], Optional[str]]] = []

        try:
            await self._cleanup_stale_managed_objects()
        except Exception as cleanup_err:
            verbose_proxy_logger.warning(
                f"CheckBatchCost: stale cleanup failed (poll will continue): {cleanup_err}"
            )

        # Look for all batches that have not yet been processed by CheckBatchCost.
        # self._has_batch_processed_column is cached after the first probe so that
        # older schemas don't pay a guaranteed-failing primary query + warning on
        # every subsequent poll cycle.
        if self._has_batch_processed_column:
            try:
                # Include "complete"/"completed" batches: the retrieve_batch
                # endpoint may transition a batch to "complete" before
                # CheckBatchCost runs.  The batch_processed=False filter
                # already prevents reprocessing finished batches.
                jobs = await self.prisma_client.db.litellm_managedobjecttable.find_many(
                    where={
                        "file_purpose": "batch",
                        "batch_processed": False,
                        "status": {
                            "not_in": [
                                "failed",
                                "expired",
                                "cancelled",
                                "stale_expired",
                            ]
                        },
                    },
                    take=MAX_OBJECTS_PER_POLL_CYCLE,
                    order={"created_at": "asc"},
                )
            except Exception as query_err:
                if "batch_processed" not in str(query_err).lower() and "unknown column" not in str(query_err).lower() and "does not exist" not in str(query_err).lower():
                    raise
                # Permanent schema gap — cache the result so future cycles skip straight to fallback
                self._has_batch_processed_column = False
                verbose_proxy_logger.warning(
                    "CheckBatchCost: batch_processed column not found, querying without it"
                )
                jobs = await self._fallback_find_jobs()
        else:
            jobs = await self._fallback_find_jobs()
        for job in jobs:
            # get the model from the job
            unified_object_id = job.unified_object_id
            decoded_unified_object_id = _is_base64_encoded_unified_file_id(
                unified_object_id
            )
            if not decoded_unified_object_id:
                verbose_proxy_logger.info(
                    f"Skipping job {unified_object_id} because it is not a valid unified object id"
                )
                if prom_logger:
                    prom_logger.record_check_batch_cost_error("invalid_unified_id")
                continue
            else:
                unified_object_id = decoded_unified_object_id

            model_id = get_model_id_from_unified_batch_id(unified_object_id)
            batch_id = get_batch_id_from_unified_batch_id(unified_object_id)
            is_passthrough_batch = is_passthrough_unified_batch_id(
                unified_object_id
            )

            if model_id is None or not batch_id:
                verbose_proxy_logger.info(
                    f"Skipping job {unified_object_id} because it is not a valid "
                    f"batch id (model_id={model_id!r}, batch_id={batch_id!r})"
                )
                if prom_logger:
                    prom_logger.record_check_batch_cost_error("invalid_model_id")
                continue

            verbose_proxy_logger.info(
                f"Querying model ID: {model_id} for cost and usage of batch ID: {batch_id}"
            )

            use_router_deployment = self.llm_router.has_model_id(model_id)
            response = None
            passthrough_retrieve_kwargs: Optional[dict] = None

            if use_router_deployment:
                try:
                    response = await self.llm_router.aretrieve_batch(
                        model=model_id,
                        batch_id=batch_id,
                        litellm_metadata={
                            "user_api_key_user_id": job.created_by or "default-user-id",
                            "batch_ignore_default_logging": True,
                        },
                    )
                except Exception as e:
                    verbose_proxy_logger.info(
                        f"Skipping job {unified_object_id} because of error querying model ID: {model_id} for cost and usage of batch ID: {batch_id}: {e}"
                    )
                    if prom_logger:
                        prom_logger.record_check_batch_cost_error(
                            "provider_retrieval_error"
                        )
                    continue
            else:
                import litellm

                passthrough_retrieve_kwargs = _get_passthrough_batch_retrieve_kwargs(
                    model_id
                )
                try:
                    verbose_proxy_logger.info(
                        f"CheckBatchCost: no proxy deployment for model_id={model_id!r}, "
                        f"using passthrough env credentials"
                    )
                    response = await litellm.aretrieve_batch(
                        batch_id=batch_id,
                        litellm_metadata={
                            "user_api_key_user_id": job.created_by
                            or "default-user-id",
                            "batch_ignore_default_logging": True,
                        },
                        **passthrough_retrieve_kwargs,
                    )
                except Exception as e:
                    verbose_proxy_logger.info(
                        f"Skipping job {unified_object_id} because of error querying batch ID: {batch_id} via passthrough env creds: {e}"
                    )
                    if prom_logger:
                        prom_logger.record_check_batch_cost_error(
                            "provider_retrieval_error"
                        )
                    continue

            ## RETRIEVE THE BATCH JOB OUTPUT FILE
            if (
                response.status == "completed"
                and response.output_file_id is not None
            ):
                verbose_proxy_logger.info(
                    f"Batch ID: {batch_id} is complete, tracking cost and usage"
                )

                # aretrieve_batch is called with the raw provider batch ID, so response.id
                # is the raw provider value (e.g. "batch_20260223-0518.234"). We need the
                # unified base64 ID in the S3 log so downstream consumers can correlate it
                # back to the batch they submitted via the proxy.
                #
                # CheckBatchCost builds its own LiteLLMLogging object (logging_obj below) and
                # calls async_success_handler(result=response) directly. That handler calls
                # _build_standard_logging_payload(response, ...) which reads response.id at
                # that point — so setting response.id here is sufficient.
                #
                # The HTTP endpoint does this substitution via the managed files hook
                # (async_post_call_success_hook). CheckBatchCost bypasses that hook entirely,
                # so we do it explicitly here.
                response.id = job.unified_object_id

                # This background job runs as default_user_id, so going through the HTTP endpoint
                # would trigger check_managed_file_id_access and get 403. Instead, extract the raw
                # provider file ID and call afile_content directly with deployment credentials.
                raw_output_file_id = _extract_raw_output_file_id(
                    response.output_file_id
                )

                credentials = (
                    self.llm_router.get_deployment_credentials_with_provider(model_id)
                    if use_router_deployment
                    else (passthrough_retrieve_kwargs or {})
                ) or {}
                _file_content = await afile_content(
                    file_id=raw_output_file_id,
                    **credentials,
                )

                # Access content - handle both direct attribute and method call
                if hasattr(_file_content, 'content'):
                    content_bytes = _file_content.content  # type: ignore[union-attr]
                elif hasattr(_file_content, 'read'):
                    content_bytes = await _file_content.read()  # type: ignore[misc]
                else:
                    content_bytes = _file_content  # type: ignore[assignment]

                file_content_as_dict = _get_file_content_as_dictionary(
                    content_bytes  # type: ignore[arg-type]
                )

                # Record output file size
                if prom_logger and content_bytes:
                    try:
                        prom_logger.record_managed_file_size(
                            size_bytes=len(content_bytes),  # type: ignore
                            purpose="batch",
                            file_type="output",
                            model=model_id,
                        )
                    except Exception:
                        pass

                deployment_info = (
                    self.llm_router.get_deployment(model_id=model_id)
                    if use_router_deployment
                    else None
                )
                if use_router_deployment and deployment_info is None:
                    verbose_proxy_logger.info(
                        f"Skipping job {unified_object_id} because it is not a valid deployment info"
                    )
                    if prom_logger:
                        prom_logger.record_check_batch_cost_error("deployment_not_found")
                    continue

                # Native batches resolve provider/model from the deployment;
                # passthrough batches fall back to env-cred provider with the
                # model name derived per-line from the output JSONL.
                llm_provider, model_name, deployment_model_info = (
                    _resolve_provider_and_model(
                        deployment_info=deployment_info,
                        passthrough_retrieve_kwargs=passthrough_retrieve_kwargs,
                    )
                )

                batch_cost, batch_usage, batch_models = (
                    await calculate_batch_cost_and_usage(
                        file_content_dictionary=file_content_as_dict,
                        custom_llm_provider=llm_provider,  # type: ignore
                        model_name=model_name,
                        model_info=deployment_model_info,  # type: ignore[arg-type]
                    )
                )
                if is_passthrough_batch and not batch_models:
                    verbose_proxy_logger.info(
                        f"Skipping job {unified_object_id}: no models found in "
                        f"batch output for provider={model_id!r}"
                    )
                    if prom_logger:
                        prom_logger.record_check_batch_cost_error(
                            "batch_models_not_found"
                        )
                    continue

                # CheckBatchCost bypasses async_post_call_success_hook, so convert raw
                # output/error file IDs to managed base64 IDs before the DB write here.
                managed_files_hook = self.proxy_logging_obj.get_proxy_hook("managed_files")
                if managed_files_hook is not None:
                    from litellm.proxy._types import UserAPIKeyAuth
                    _minimal_auth = UserAPIKeyAuth(
                        user_id=job.created_by or "default-user-id",
                        team_id=getattr(job, "team_id", None),
                    )
                    _resolved_model_name = (
                        str(model_name)
                        if model_name
                        else (
                            deployment_info.model_name
                            if deployment_info is not None
                            else batch_models[0]
                        )
                    )
                    for _file_attr in ["output_file_id", "error_file_id"]:
                        _raw_file_id = getattr(response, _file_attr, None)
                        if _raw_file_id and not _is_base64_encoded_unified_file_id(_raw_file_id):
                            try:
                                _unified_file_id = managed_files_hook.get_unified_output_file_id(
                                    output_file_id=_raw_file_id,
                                    model_id=model_id,
                                    model_name=_resolved_model_name,
                                )
                                await managed_files_hook.store_unified_file_id(
                                    file_id=_unified_file_id,
                                    file_object=None,
                                    litellm_parent_otel_span=None,
                                    model_mappings={model_id: _raw_file_id},
                                    user_api_key_dict=_minimal_auth,
                                )
                                setattr(response, _file_attr, _unified_file_id)
                                verbose_proxy_logger.info(
                                    f"CheckBatchCost: converted {_file_attr} "
                                    f"{_raw_file_id!r} -> managed ID for batch {batch_id}"
                                )
                            except Exception as _e:
                                verbose_proxy_logger.warning(
                                    f"CheckBatchCost: failed to create managed file ID for "
                                    f"{_file_attr}={_raw_file_id!r}: {_e}"
                                )
                logging_obj = LiteLLMLogging(
                    model=batch_models[0],
                    messages=[{"role": "user", "content": "<retrieve_batch>"}],
                    stream=False,
                    call_type="aretrieve_batch",
                    start_time=datetime.now(),
                    litellm_call_id=str(uuid.uuid4()),
                    function_id=str(uuid.uuid4()),
                )

                creator_user_id = job.created_by
                user_info = await self._get_user_info(batch_id, job.created_by)

                logging_obj.update_environment_variables(
                    litellm_params={
                        # set the user-agent header so that S3 callback consumers can easily identify CheckBatchCost callbacks
                        "proxy_server_request": {
                            "headers": {
                                "user-agent": CHECK_BATCH_COST_USER_AGENT,
                            }
                        },
                        "metadata": {
                            "user_api_key_user_id": creator_user_id,
                            **user_info,
                        },
                    },
                    optional_params={},
                )

                await logging_obj.async_success_handler(
                    result=response,
                    batch_cost=batch_cost,
                    batch_usage=batch_usage,
                    batch_models=batch_models,
                )

                # Record batch duration (completed_at - created_at)
                if prom_logger and response.completed_at and response.created_at:
                    duration_seconds = float(response.completed_at - response.created_at)
                    if duration_seconds >= 0:
                        prom_logger.record_managed_batch_duration(
                            duration_seconds=duration_seconds,
                            model=model_name,
                            api_provider=str(llm_provider) if llm_provider else None,
                        )

                # Track this job for the final metrics summary
                processed_models.append((model_name, str(llm_provider) if llm_provider else None))

                # mark the job as complete
                try:
                    update_data: dict = {
                        "status": "complete",
                        "file_object": response.model_dump_json(),
                    }
                    if self._has_batch_processed_column:
                        update_data["batch_processed"] = True
                    await self.prisma_client.db.litellm_managedobjecttable.update(
                        where={"id": job.id},
                        data=update_data,
                    )
                except Exception as db_err:
                    verbose_proxy_logger.error(
                        f"CheckBatchCost: failed to mark job {job.id} complete in DB: {db_err}"
                    )

        # Record polling run metrics (always, even if nothing was processed)
        if prom_logger:
            prom_logger.record_check_batch_cost_run(
                jobs_polled=len(jobs),
                processed_models=processed_models if processed_models else None,
            )
