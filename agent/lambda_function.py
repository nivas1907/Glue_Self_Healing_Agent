# AGENT_TOKEN
# my-secret-token-123
# AGENT_URL
# http://13.204.79.18:8080/process-error

# AWSLambdaBasicExecutionRole
# {
#     "Version": "2012-10-17",
#     "Statement": [
#         {
#             "Effect": "Allow",
#             "Action": "logs:CreateLogGroup",
#             "Resource": "arn:aws:logs:us-east-1:997525378140:*"
#         },
#         {
#             "Effect": "Allow",
#             "Action": [
#                 "logs:CreateLogStream",
#                 "logs:PutLogEvents"
#             ],
#             "Resource": [
#                 "arn:aws:logs:us-east-1:997525378140:log-group:/aws/lambda/glue-error-forwarder:*"
#             ]
#         }
#     ]
# }
# LambdaGlueLogReadAccess 
# {
# 	"Version": "2012-10-17",
# 	"Statement": [
# 		{
# 			"Effect": "Allow",
# 			"Action": [
# 				"glue:GetJobRun"
# 			],
# 			"Resource": "*"
# 		},
# 		{
# 			"Effect": "Allow",
# 			"Action": [
# 				"logs:FilterLogEvents",
# 				"logs:GetLogEvents",
# 				"logs:DescribeLogStreams",
# 				"logs:DescribeLogGroups"
# 			],
# 			"Resource": "*"
# 		}
# 	]
# }



import json
import os
import urllib3
import boto3
from botocore.config import Config
from datetime import datetime, timezone

http = urllib3.PoolManager()

AGENT_URL = os.environ["AGENT_URL"]
API_TOKEN = os.environ.get("AGENT_TOKEN", "")

boto_cfg = Config(retries={"max_attempts": 3, "mode": "standard"})
# Global default clients (your current pattern)
glue = boto3.client("glue", config=boto_cfg)
logs = boto3.client("logs", config=boto_cfg)

MAX_INLINE_BYTES = 64 * 1024  # 64 KB
DEFAULT_LOOKBACK_MIN = 60
TOP_ERROR_SHARDS = 8
ERROR_TERMS = [
    "ERROR", "Exception", "RESOURCE_NOT_FOUND_ERROR", "Entity Not Found",
    "getCatalogSource", "Status Code: 400"
]

def _epoch_millis(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def _safe_join_lines(lines, max_bytes):
    out, total = [], 0
    for ln in lines:
        b = (ln + "\n").encode("utf-8", errors="ignore")
        if total + len(b) > max_bytes:
            break
        out.append(ln)
        total += len(b)
    return "\n".join(out), total

def _mk_clients_if_event_region(event_region: str):
    """
    Optional override: if EventBridge provides a region, prefer regional
    clients for this invocation. Otherwise, fall back to globals.
    """
    if not event_region:
        return None, None
    try:
        regional_glue = boto3.client("glue", region_name=event_region, config=boto_cfg)
        regional_logs = boto3.client("logs", region_name=event_region, config=boto_cfg)
        return regional_glue, regional_logs
    except Exception as e:
        print(f"[region-override] failed to create clients for {event_region}: {e}")
        return None, None

def _get_job_run(glue_client, job_name, job_run_id):
    try:
        resp = glue_client.get_job_run(JobName=job_name, RunId=job_run_id, PredecessorsIncluded=False)
        return resp.get("JobRun", {}) or {}
    except Exception as e:
        print(f"get_job_run failed: {e}")
        return {}

def _get_run_window_ms(job_run: dict, default_end_ms: int):
    """
    If Glue returns StartedOn/CompletedOn, use them; else fall back to last DEFAULT_LOOKBACK_MIN window.
    """
    start = job_run.get("StartedOn")
    end = job_run.get("CompletedOn") or job_run.get("LastModifiedOn")
    try:
        if hasattr(start, "timestamp"):
            start_ms = _epoch_millis(start)
        else:
            start_ms = default_end_ms - DEFAULT_LOOKBACK_MIN * 60 * 1000
        if hasattr(end, "timestamp"):
            end_ms = _epoch_millis(end)
        else:
            end_ms = default_end_ms
        return start_ms, end_ms
    except Exception:
        return default_end_ms - DEFAULT_LOOKBACK_MIN * 60 * 1000, default_end_ms

def _get_log_events(logs_client, group, stream, start_ms, end_ms):
    events = []
    try:
        next_token = None
        while True:
            kwargs = {
                "logGroupName": group,
                "logStreamName": stream,
                "startTime": start_ms,
                "endTime": end_ms,
                "startFromHead": True,
            }
            if next_token:
                kwargs["nextToken"] = next_token
            resp = logs_client.get_log_events(**kwargs)
            for ev in resp.get("events", []):
                msg = ev.get("message", "")
                if msg:
                    events.append((ev.get("timestamp", 0), msg.rstrip("\n")))
            nt = resp.get("nextForwardToken")
            if not next_token or nt != next_token:
                next_token = nt
            else:
                break
    except logs_client.exceptions.ResourceNotFoundException:
        print(f"[get_log_events] not found: {group}/{stream}")
    except Exception as e:
        print(f"[get_log_events] error for {group}/{stream}: {e}")
    return events

def _describe_streams(logs_client, group, prefix, limit):
    names = []
    try:
        paginator = logs_client.get_paginator("describe_log_streams")
        for page in paginator.paginate(logGroupName=group, logStreamNamePrefix=prefix):
            for s in page.get("logStreams", []):
                name = s.get("logStreamName")
                if name:
                    names.append(name)
                    if len(names) >= limit:
                        return names
    except logs_client.exceptions.ResourceNotFoundException:
        print(f"[describe_streams] group not found: {group}")
    except Exception as e:
        print(f"[describe_streams] error: group={group} prefix={prefix} err={e}")
    return names

def _filter_by_run_id(logs_client, group, run_id, start_ms, end_ms):
    lines = []
    try:
        next_tok = None
        while True:
            kwargs = {
                "logGroupName": group,
                "startTime": start_ms,
                "endTime": end_ms,
                "filterPattern": run_id
            }
            if next_tok:
                kwargs["nextToken"] = next_tok
            resp = logs_client.filter_log_events(**kwargs)
            for ev in resp.get("events", []):
                msg = ev.get("message", "")
                if msg:
                    lines.append(msg.rstrip("\n"))
            next_tok = resp.get("nextToken")
            if not next_tok:
                break
    except logs_client.exceptions.ResourceNotFoundException:
        print(f"[filter] group not found: {group}")
    except Exception as e:
        print(f"[filter] error on {group}: {e}")
    return lines

def _collect_glue_logs(glue_client, logs_client, job_name: str, job_run_id: str, end_time_ms: int) -> str:
    """
    Robust collector tuned to your account:
      - output group: stream == <job_run_id>
      - error group: streams == <job_run_id>_g-*
      - fallback: filter across error/output/logs-v2
      - includes Glue ErrorMessage
    """
    lines = []

    jr = _get_job_run(glue_client, job_name, job_run_id)
    glue_error = jr.get("ErrorMessage") or jr.get("Error", "")
    start_ms, end_ms = _get_run_window_ms(jr, end_time_ms)
    print(f"[debug] window(ms) {start_ms}..{end_ms} for run {job_run_id}")

    # 1) Direct fetches based on patterns you confirmed via CloudShell
    # /aws-glue/jobs/output → stream is exactly run id
    out_group = "/aws-glue/jobs/output"
    out_stream = job_run_id
    out_evs = _get_log_events(logs_client, out_group, out_stream, start_ms, end_ms)
    if out_evs:
        print(f"[collector] output events: {len(out_evs)} from {out_stream}")
        lines.extend([m for _, m in sorted(out_evs, key=lambda t: t[0])])

    # /aws-glue/jobs/error → multiple shards <run_id>_g-*
    err_group = "/aws-glue/jobs/error"
    shard_prefix = f"{job_run_id}_g-"
    shard_names = _describe_streams(logs_client, err_group, shard_prefix, TOP_ERROR_SHARDS)
    print(f"[collector] error shard candidates: {shard_names}")
    for s in shard_names:
        err_evs = _get_log_events(logs_client, err_group, s, start_ms, end_ms)
        if err_evs:
            print(f"[collector] error events: {len(err_evs)} from {s}")
            lines.extend([m for _, m in sorted(err_evs, key=lambda t: t[0])])

    # 2) Fallback: filter by run id across common groups (including logs-v2 if enabled)
    if not lines:
        for g in [err_group, out_group, "/aws-glue/jobs/logs-v2"]:
            fl = _filter_by_run_id(logs_client, g, job_run_id, start_ms, end_ms)
            if fl:
                print(f"[collector] filter hits in {g}: {len(fl)}")
                lines.extend(fl)

    # De-dup while preserving order
    seen, unique_lines = set(), []
    for ln in lines:
        if ln not in seen:
            unique_lines.append(ln)
            seen.add(ln)

    # Prioritize error-ish lines, but keep context fallback
    final_lines = unique_lines
    hot = [ln for ln in unique_lines if (job_run_id in ln) or any(t in ln for t in ERROR_TERMS)]
    if hot:
        final_lines = hot
    elif unique_lines:
        final_lines = unique_lines[-200:]

    parts = []
    if glue_error:
        parts.append(f"[GLUE_ERROR] {glue_error}")
    if final_lines:
        parts.append("\n".join(final_lines))

    if not parts:
        return f"(no log lines collected for jobRunId={job_run_id}; check CW region/time window)"
    return "\n".join(parts)

def lambda_handler(event, context):
    print("Received Event:", json.dumps(event))

    detail = event.get("detail", {}) or {}
    job_name   = detail.get("jobName")
    job_run_id = detail.get("jobRunId")
    state      = detail.get("state")
    event_region = event.get("region") or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")

    print(f"[debug] region={event_region} job={job_name} run={job_run_id} state={state}")
    print(f"[debug] posting to AGENT_URL={AGENT_URL}")

    if not job_name or not job_run_id:
        return {"status": "ignored", "reason": "missing job_name/job_run_id"}
    if state not in ("FAILED", "STOPPED", "TIMEOUT"):
        return {"status": "ignored", "reason": f"state={state}"}

    # Prefer event-scoped clients if event.region is present; else use globals
    g_client, l_client = _mk_clients_if_event_region(event_region)
    glue_client = g_client or glue
    logs_client = l_client or logs

    now = datetime.now(timezone.utc)
    end_ms = _epoch_millis(now)

    full_text = _collect_glue_logs(glue_client, logs_client, job_name, job_run_id, end_ms)
    snippet, _ = _safe_join_lines(full_text.splitlines(), MAX_INLINE_BYTES)

    payload = {
        "job_name": job_name,
        "job_run_id": job_run_id,
        "state": state,
        "message": "Glue job failure received",
        "log_snippet": snippet,
        "s3_log_url": None
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-token": API_TOKEN
    }

    try:
        response = http.request(
            "POST",
            AGENT_URL,
            body=json.dumps(payload),
            headers=headers,
            timeout=20.0
        )
        print(f"[debug] agent_status={response.status}")
        return {
            "status": "success",
            "agent_status": response.status,
            "agent_response": (response.data.decode() if response.data else ""),
            "log_snippet_bytes": len(snippet.encode("utf-8"))
        }
    except Exception as e:
        print("Error posting to agent:", str(e))
        return {"status": "error", "error": str(e)}
