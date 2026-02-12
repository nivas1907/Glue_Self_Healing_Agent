# app.py
import os
import json
import argparse
import re
import uuid
import time
import smtplib
import logging
from typing import List, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import imaplib
import email as pyemail
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from kb_s3 import S3KB
from semantic_safety import semantic_safety_check, blast_radius_check, safety_score

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
log = logging.getLogger("glue-agent")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# ---------------------------------------------------------------------
# KB configuration — STATIC as requested
# ---------------------------------------------------------------------
KB_BUCKET = "self-heal-kb-396034748887-ap-south-1"  # <- static, do not change at runtime
KB_PREFIX = ""                                      # <- static
DISABLE_KB = os.getenv("DISABLE_KB", "0") == "1"    # optional flag; still keeps bucket static

# Lazy singleton for KB
_KB_INSTANCE: Optional[S3KB] = None

def get_kb() -> Optional[S3KB]:
    """
    Lazily initialize the S3KB. Returns None if disabled or initialization fails.
    """
    global _KB_INSTANCE
    if DISABLE_KB:
        if _KB_INSTANCE is not None:
            return _KB_INSTANCE  # already set (even if None)
        log.warning("KB disabled via DISABLE_KB=1; continuing without KB.")
        _KB_INSTANCE = None
        return _KB_INSTANCE

    if _KB_INSTANCE is not None:
        return _KB_INSTANCE

    try:
        # STATIC bucket/prefix as per your request
        _KB_INSTANCE = S3KB(s3_bucket=KB_BUCKET, s3_prefix=KB_PREFIX)
        log.info("KB initialized from S3 (static bucket).")
    except ClientError as e:
        log.warning("KB initialization failed due to S3 access error; continuing without KB. err=%s", e)
        _KB_INSTANCE = None
    except Exception as e:
        log.warning("KB initialization failed; continuing without KB. err=%s", e)
        _KB_INSTANCE = None

    return _KB_INSTANCE

# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
app = FastAPI()
AGENT_TOKEN = "my-secret-token-123"

@app.on_event("startup")
async def init_kb_on_startup():
    """
    Initialize KB at server startup, but do not block the app if it fails.
    """
    get_kb()  # best-effort; logs on failure


@app.post("/process-error")
async def process_error(payload: dict, background_tasks: BackgroundTasks, x_api_token: str = Header(None)):
    if x_api_token != AGENT_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    print("Received from Lambda:", {
        "job_name": payload.get("job_name"),
        "job_run_id": payload.get("job_run_id"),
        "state": payload.get("state"),
        "has_log_snippet": bool(payload.get("log_snippet")),
    })

    log_text = payload.get("log_snippet") or "(no log snippet received)"

    # (Optional) If you want to route through KB first in the API path, uncomment:
    # try:
    #     kb = get_kb()
    #     if kb:
    #         kb_structured = kb.try_build_plan(log_text, payload or {})
    #         if kb_structured:
    #             print("\n===== KB MATCH FOUND — USING KB FIX PLAN (API) =====\n")
    #             print(json.dumps(kb_structured, indent=2))
    #             # You could short-circuit here if desired...
    # except Exception as e:
    #     log.warning("[API:KB] KB lookup failed: %s", e)

    background_tasks.add_task(agent_flow, log_text, payload)
    return {"status": "accepted", "message": "Processing started"}


# If Strands isn't installed yet, guide the user.
try:
    from strands import Agent
except ImportError:
    raise SystemExit("Strands not installed. Run: pip install -r agent/requirements.txt")


SYSTEM_PROMPT = """
You are **Glue Doctor**, an expert at diagnosing AWS Glue job failures and producing an executable fix plan.
You will receive a chunk of log lines (possibly from CloudWatch) that include a stack trace or error text.

### Your Goals
1) Identify the **root cause** (precise and short).
2) Provide a clear **human-readable explanation** for a software engineer new to this pipeline.
3) Produce a **machine-executable plan** that this agent (or a follow-up runner) can actually perform.
4) Plans MUST be **safe-first**, **idempotent**, and **reversible** where possible.
5) Any destructive or privilege-changing action MUST be flagged as **requires_approval=true**.

### Capabilities (you MUST limit fix steps to these)
Only propose actions from this list. Do NOT propose anything outside this list.

- `s3.test_path_exists` — check if a path or partition exists (S3 ListObjectsV2/HeadObject).
- `glue.get_job` / `glue.get_job_run` — read job configuration / last run info (precheck).
- `glue.update_job_default_args` — set or patch job default arguments (e.g., enable/disable pushdown, add missing args).
- `glue.start_job_run` — trigger a Glue job run with safe overrides.
- `glue.update_job_script_ref` — switch script S3 URI to a known-good location (non-destructive).
- `iam.propose_policy_patch` — OUTPUT ONLY a least-privilege policy JSON for review; DO NOT attach or apply (marked requires_approval=true).
- `s3.propose_bucket_policy_patch` — OUTPUT ONLY a bucket policy statement to unblock access; DO NOT apply (requires_approval=true).
- `notify.email` — send a notification with human-readable steps and decision token.
- `github.create_pr` — open a PR that updates a config file (non-prod by default); requires inputs (owner, repo, path, branch, content). Mark requires_approval=true unless targeting a sandbox branch.
- `observability.create_cloudwatch_alarm` — propose an alarm definition as JSON (OUTPUT ONLY unless executor supports applying it).

If an operation requires credentials or permissions the runner may not have, **output the action as a proposal with `requires_approval=true`** and include the exact JSON patch the executor should apply after approval.

### Output Format — STRICT
Return TWO parts exactly:

(A) Human-friendly markdown for engineers:
- Sections:
  - **Root Cause**
  - **What will be executed (if approved)** — a numbered list of steps taken from your plan, referencing action ids.
  - **Assumptions**
  - **Pre-checks**
  - **Post-checks / Verification**
  - **Rollback**

(B) A JSON object with fields (VALID JSON):
{
  "root_cause": str,
  "explanation": str,
  "suggested_fix": {
    "type": "config|code|infra|data",
    "steps": [str]
  },
  "confidence": float,
  "next_steps": [str],
  "tags": [str],
  "actions": [
    {
      "id": "a1",
      "capability": "s3.test_path_exists|glue.get_job|glue.update_job_default_args|glue.start_job_run|glue.update_job_script_ref|iam.propose_policy_patch|s3.propose_bucket_policy_patch|notify.email|github.create_pr|observability.create_cloudwatch_alarm",
      "params": { },
      "requires_approval": false,
      "prechecks": [str],
      "postchecks": [str],
      "rollback": { "capability": "...", "params": { } }
    }
  ]
}
### Requirements & Guardrails
- Do not suggest manual steps; only propose actions from the Capabilities list above.
- Prefer safe, non-destructive changes first.
- Permissions-related issues -> only policy JSON proposals with requires_approval=true.
- Include at least one precheck and one postcheck for mutating actions.
- Idempotent and rollback where meaningful.
- If uncertain, propose what additional logs/metrics to fetch next.
- Keep the human markdown concise but complete.

### Input
You will receive:
### LOG SNIPPET START
<log text>
### LOG SNIPPET END

### Respond with EXACTLY:
---BEGIN-HUMAN---
<markdown for engineers>
---END-HUMAN---
---BEGIN-JSON---
<valid JSON per schema>
---END-JSON---
"""

def refine_plan_until_safe(log_text, payload, max_attempts=3):
    job_name = payload.get("job_name")
    attempt = 1

    kb_structured = None
    try:
        kb = get_kb()
        if kb:
            kb_structured = kb.try_build_plan(log_text, {})
    except Exception as e:
        print(f"[CLI:KB] Error while loading/using KB: {e}")
        kb_structured = None

    # -------------------------------------------------------------------------
    # If KB matched, use KB plan and skip LLM
    # -------------------------------------------------------------------------
    if kb_structured:
        attempt = max_attempts  # skip regeneration loop since we're using KB plan
        print("\n===== KB MATCH FOUND — USING KB FIX PLAN =====\n")
        human = f"""
**Root Cause**
- {kb_structured.get("root_cause")}

**What will be executed (if approved)**  
{chr(10).join([f"- [{a['id']}] {a['capability']}" for a in kb_structured.get('actions', [])])}

(This plan came from S3 Knowledge Base, not the LLM.)
""".strip()

        structured = kb_structured

    else:
        print("\n===== NO KB MATCH — RUNNING LLM =====\n")
        raw = run_agent(log_text)
        human, structured = parse_sections(raw)


    while attempt <= max_attempts:
        if not isinstance(structured, dict) or not structured.get("actions"):
            sem_errors = ["Plan missing 'actions' array or failed to parse JSON."]
            blast_errors = []
        else:
            sem_errors, _ = semantic_safety_check(structured)
            blast_errors = blast_radius_check(structured, job_name)

        print(f"\n[SFT PASS {attempt}/{max_attempts}] semantic errors={sem_errors}, blast errors={blast_errors}")

        if not sem_errors and not blast_errors:
            return human, structured, sem_errors, blast_errors, attempt, True

        repair_prompt = build_repair_context(job_name, sem_errors, blast_errors)
        raw = run_agent(log_text, repair_context=repair_prompt)
        human, structured = parse_sections(raw)
        attempt += 1

    if not isinstance(structured, dict) or not structured.get("actions"):
        sem_errors = ["Plan missing 'actions' array or failed to parse JSON."]
        blast_errors = []
    else:
        sem_errors, _ = semantic_safety_check(structured)
        blast_errors = blast_radius_check(structured, job_name)

    return human, structured, sem_errors, blast_errors, attempt-1, False

def agent_flow(log_text, payload=None):
    to_email = "pasumarthynivas5@gmail.com"

    # 1) Try safe regeneration loop
    human, structured, sem_errors, blast_errors, attempts, is_safe = refine_plan_until_safe(
        log_text,
        payload or {},
        max_attempts=3
    )

    # 2) Compute safety score
    job_name = payload.get("job_name") if payload else None
    score, penalty_components = safety_score(structured, job_name_from_payload=job_name)

    print("\n===== SAFETY SCORE =====")
    print("Score:", score)
    print("Penalty breakdown:", penalty_components)

    print("\n===== FINAL PLAN (AFTER SAFETY LOOP) =====\n")
    print(human)
    print("\n===== STRUCTURED JSON =====\n")
    print(json.dumps(structured, indent=2))

    # 2.b Build Lambda artifacts
    lambda_steps = build_lambda_steps(structured)
    instructions = build_lambda_instructions(structured, job_name_from_payload=job_name)

    # (Optional) expose lambda_steps inside structured for debugging/visibility
    structured["lambda_steps"] = lambda_steps

    print("\n===== STEPS FOR LAMBDA (Action/Resource) =====")
    print(json.dumps(lambda_steps, indent=2))

    print("\n===== GENERAL LAMBDA INSTRUCTIONS =====")
    print(json.dumps(instructions, indent=2))

    # Diagnostics for unsafe plans
    diagnostics_md = ""
    if not is_safe:
        diagnostics_md = (
            "\n\n---\n### Safety Diagnostics\n"
            f"- Regeneration attempts: {attempts}\n"
            f"- Semantic errors:\n  - " + ("\n  - ".join(sem_errors) if sem_errors else "(none)") + "\n"
            f"- Blast radius errors:\n  - " + ("\n  - ".join(blast_errors) if blast_errors else "(none)")
        )

    # 3) Safety SCORE block in email
    safety_block = (
        "\n\n---\n### Safety Score\n"
        f"- Score: **{score}/100**\n"
        f"- Penalty breakdown: `{json.dumps(penalty_components)}`\n"
    )

    # 4) Generate approval token
    token = uuid.uuid4().hex[:8]
    subject_with_token = (
        ("[SAFETY REVIEW REQUIRED] " if not is_safe else "")
        + f"glue error fix steps [Token:{token}] — Reply SUBJECT with APPROVE {token} or DENY {token}"
    )

    instructions_txt = (
        f"\n\nPlease reply by editing the SUBJECT to:\n"
        f"  APPROVE {token}\n"
        f"  DENY {token}\n\n"
    )

    # 5) Full email body
    body_to_send = human + diagnostics_md + safety_block + instructions_txt

    print(f"\n📧 Sending email to {to_email}...")
    normalized_body = body_to_send.replace("\\n", "\n")
    sent = send_email(to_email, subject_with_token, normalized_body, html=False)

    if not sent:
        print("Email failed — abort")
        return

    # Wait for approval
    decision = wait_for_email_response(
        recipient_email=to_email,
        token=token,
        poll_seconds=30,
        timeout_seconds=900
    )

    if decision == "approve":
        # === Call execution Lambda with ONLY the actions array
        result = invoke_execution_lambda(structured)
        print("\n===== EXECUTION LAMBDA RESULT (agent_flow) =====")
        print(json.dumps(result, indent=2))
    elif decision == "deny":
        print("\n❌ Denied — stopping workflow")
    else:
        print("\n⏱️ Timeout — no response")


# -----------------------------
# Email (SMTP) - sender side
# -----------------------------
def send_email(to_email: str, subject: str, body_text: str, *, html: bool = False):
    """
    Send an email using SMTP (Gmail).
    TODO: Move credentials to env/Secrets Manager and rotate immediately.
    """
    sender_email = "nivaspasu@gmail.com"
    sender_password = "qukj nrnn fgcr qffg"

    if not sender_email or not sender_password:
        print("⚠️ EMAIL_USER or EMAIL_PASS not set. Skipping email.")
        return False

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject

    subtype = "html" if html else "plain"
    msg.attach(MIMEText(body_text, subtype))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())
        print(f"✅ Email sent successfully to {to_email}")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False


# -----------------------------
# Email (IMAP) - listener side
# -----------------------------
def wait_for_email_response(recipient_email: str, token: str, poll_seconds: int = 30, timeout_seconds: int = 900) -> str:
    """
    Poll Gmail via IMAP for a reply FROM `recipient_email` whose SUBJECT contains `token`.
    Returns 'approve', 'deny', or 'timeout'.
    Decision is parsed ONLY from the SUBJECT with word-boundary regex.
    """
    imap_user = "nivaspasu@gmail.com"
    imap_pass = "qukj nrnn fgcr qffg"
    if not imap_user or not imap_pass:
        print("⚠️ EMAIL_USER or EMAIL_PASS not set. Cannot listen for replies.")
        return "timeout"

    end_time = time.time() + timeout_seconds
    try:
        M = imaplib.IMAP4_SSL("imap.gmail.com", 993)
        M.login(imap_user, imap_pass)
    except Exception as e:
        print(f"❌ IMAP login failed: {e}")
        return "timeout"

    print(f"👂 Listening for reply from {recipient_email} with token {token} (timeout {timeout_seconds}s)...")

    approve_re = re.compile(r"\bapprove\b", re.IGNORECASE)
    deny_re    = re.compile(r"\bdeny\b|\bdecline\b|\breject(?:ed)?\b", re.IGNORECASE)
    token_re   = re.compile(re.escape(token), re.IGNORECASE)

    try:
        while time.time() < end_time:
            M.select("INBOX")

            status, data = M.search(None, f'(FROM "{recipient_email}")')
            if status != "OK":
                time.sleep(poll_seconds)
                continue

            ids = (data[0] or b"").split()
            for num in reversed(ids):  # newest first
                try:
                    status, msg_data = M.fetch(num, "(RFC822)")
                    if status != "OK" or not msg_data:
                        continue

                    msg_bytes = msg_data[0][1]
                    msg = pyemail.message_from_bytes(msg_bytes)

                    frm = pyemail.utils.parseaddr(msg.get("From", ""))[1]
                    subj = msg.get("Subject", "") or ""

                    if frm.lower() != recipient_email.lower():
                        continue

                    if not token_re.search(subj):
                        continue

                    has_deny = bool(deny_re.search(subj))
                    has_approve = bool(approve_re.search(subj))

                    if has_deny and has_approve:
                        print("⚠️ Both approve and deny found in subject; treating as DENY.")
                        M.store(num, "+FLAGS", "\\Seen")
                        M.logout()
                        return "deny"

                    if has_deny:
                        print("❌ Denial detected in subject.")
                        M.store(num, "+FLAGS", "\\Seen")
                        M.logout()
                        return "deny"

                    if has_approve:
                        print("✅ Approval detected in subject.")
                        M.store(num, "+FLAGS", "\\Seen")
                        M.logout()
                        return "approve"

                except Exception:
                    continue

            time.sleep(poll_seconds)

        M.logout()
        print("⏱️ Timeout waiting for email decision.")
        return "timeout"

    except Exception as e:
        print(f"❌ IMAP error: {e}")
        try:
            M.logout()
        except Exception:
            pass
        return "timeout"

# -----------------------------
# Bedrock / Strands Agent
# -----------------------------
def run_agent(input_text: str, repair_context: str = "") -> str:
    """
    Run the Strands Agent reasoning against the provided log input.
    Optional repair_context is appended when regenerating safer plans.
    """
    region = os.getenv("AWS_REGION", "us-east-1")
    _ = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(read_timeout=3600, connect_timeout=3600, retries={'max_attempts': 1})
    )

    model_id = os.getenv("BEDROCK_MODEL", "apac.amazon.nova-lite-v1:0")

    agent = Agent(
        system_prompt=SYSTEM_PROMPT,
        model=model_id,
    )

    prompt = f"""
### LOG SNIPPET START
{input_text}
### LOG SNIPPET END

{repair_context}

Please respond with:

---BEGIN-HUMAN---
<markdown for engineers>
---END-HUMAN---

---BEGIN-JSON---
<valid JSON per schema>
---END-JSON---
"""

    result = agent(prompt)
    raw = getattr(result, "lastMessage", None) or getattr(result, "message", None) or str(result)
    return raw

import html

def parse_sections(raw: str):
    """
    Extract the human-readable markdown and the JSON object from the agent's response.
    Works for both raw text and wrapped structures like:
      {"role":"assistant","content":[{"text":"..."}]}
    Returns (human_md: str, data: dict).
    """
    # Normalize to str
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8", errors="replace")
        except Exception:
            raw = str(raw)
    elif not isinstance(raw, str):
        try:
            raw = json.dumps(raw)
        except Exception:
            raw = str(raw)

    # 1) Try to unwrap if the model returned a structured message
    text_for_parsing = raw
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "content" in obj:
            pieces = []

            def collect_text(node):
                if isinstance(node, dict):
                    if "text" in node and isinstance(node["text"], str):
                        pieces.append(node["text"])
                    for v in node.values():
                        collect_text(v)
                elif isinstance(node, list):
                    for item in node:
                        collect_text(item)

            collect_text(obj["content"])
            if pieces:
                text_for_parsing = "\n".join(pieces)
    except Exception:
        pass

    # 2) Unescape HTML entities (< > etc.)
    raw_unescaped = html.unescape(text_for_parsing)

    # 3) Robust regex — match anything between markers
    human_match = re.search(r"---BEGIN-HUMAN---\s*([\s\S]*?)\s*---END-HUMAN---", raw_unescaped)
    json_match  = re.search(r"---BEGIN-JSON---\s*([\s\S]*?)\s*---END-JSON---", raw_unescaped)

    # 4) Human section cleanup
    human_md = human_match.group(1) if human_match else raw_unescaped
    human_md = (
        human_md
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\'", "'")
    )

    # 5) JSON parsing
    data = {}
    if json_match:
        json_txt = json_match.group(1).strip()
        try:
            data = json.loads(json_txt)
        except Exception as e:
            print("⚠️ JSON parse error in agent output:", e)
            print("JSON snippet start >>>")
            print(json_txt[:2000])
            print("<<< JSON snippet end")
            data = {}
    else:
        print("⚠️ JSON block not found between markers. Raw (first 1000 chars):")
        print(raw_unescaped[:1000])

    return human_md.strip(), data


# -----------------------------
# Lambda payload builders
# -----------------------------
def build_lambda_instructions(structured: dict, *, job_name_from_payload: Optional[str] = None) -> List[Dict]:
    if not isinstance(structured, dict):
        return []

    out: List[Dict] = []
    for a in (structured.get("actions") or []):
        capability = (a.get("capability") or "").strip()
        params = a.get("params") or {}
        action_id = a.get("id")

        if capability.startswith("glue.") and job_name_from_payload and "job_name" not in params:
            params = {**params, "job_name": job_name_from_payload}

        out.append({
            "id": action_id,
            "capability": capability,
            "params": params,
            "requires_approval": bool(a.get("requires_approval", False))
        })
    return out


def build_lambda_steps(structured: dict) -> List[Dict]:
    if not isinstance(structured, dict):
        return []

    actions = structured.get("actions") or []
    lambda_steps: List[Dict] = []

    def normalize_to_list(x):
        if x is None:
            return []
        if isinstance(x, list):
            return x
        return [x]

    def parse_s3_uri(uri: str):
        if not uri or not isinstance(uri, str):
            return (None, None)
        if not uri.lower().startswith("s3://"):
            return (None, None)
        rest = uri[5:]
        parts = rest.split("/", 1)
        bucket = parts[0].strip()
        key = parts[1].strip() if len(parts) > 1 else None
        return (bucket or None), (key or None)

    for a in actions:
        capability = (a.get("capability") or "").strip()

        if capability == "iam.propose_policy_patch":
            policy = (a.get("params") or {}).get("policy") or {}
            statements = normalize_to_list(policy.get("Statement"))
            for st in statements:
                if not isinstance(st, dict):
                    continue
                actions_field = st.get("Action")
                resources_field = st.get("Resource")

                actions_list = normalize_to_list(actions_field)
                resources_list = normalize_to_list(resources_field)

                for action_val in actions_list:
                    for resource_val in resources_list:
                        if action_val and resource_val:
                            lambda_steps.append({
                                "Action": action_val,
                                "Resource": resource_val
                            })

        elif capability == "s3.propose_bucket_policy_patch":
            params = a.get("params") or {}
            statements = []
            if "statement" in params:
                statements = normalize_to_list(params.get("statement"))
            elif "policy" in params and isinstance(params["policy"], dict):
                statements = normalize_to_list(params["policy"].get("Statement"))

            for st in statements:
                if not isinstance(st, dict):
                    continue
                actions_list = normalize_to_list(st.get("Action"))
                resources_list = normalize_to_list(st.get("Resource"))
                for action_val in actions_list:
                    for resource_val in resources_list:
                        if action_val and resource_val:
                            lambda_steps.append({
                                "Action": action_val,
                                "Resource": resource_val
                            })

        elif capability == "s3.test_path_exists":
            params = a.get("params") or {}
            path = params.get("path")
            bucket, key = parse_s3_uri(path)
            if bucket:
                lambda_steps.append({
                    "Action": "s3:ListBucket",
                    "Resource": f"arn:aws:s3:::{bucket}"
                })
                if key:
                    lambda_steps.append({
                        "Action": "s3:HeadObject",
                        "Resource": f"arn:aws:s3:::{bucket}/{key}"
                    })

    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for item in lambda_steps:
        key = (item["Action"], item["Resource"])
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def build_repair_context(job_name: str, sem_errors, blast_errors):
    error_list = ""
    if sem_errors:
        error_list += "\n- Semantic issues:\n  - " + "\n  - ".join(sem_errors)
    if blast_errors:
        error_list += "\n- Blast radius issues:\n  - " + "\n  - ".join(blast_errors)

    return f"""
### REPAIR CONTEXT (YOU MUST FIX THIS PLAN)

Your previous plan was unsafe or violated mandatory constraints.
Regenerate an improved and SAFE plan by following ALL rules below:

MANDATORY RULES:
1. Blast radius: All Glue mutating actions MUST use job_name="{job_name}".
2. No destructive actions.
3. Safe-first ordering.
4. Rollback required for all config actions.
5. Idempotent.
6. Use ONLY allowed capabilities.

Issues you MUST correct:
{error_list or "- No specific errors recorded, but previous plan unsafe. Regenerate a safer one."}

Regenerate the plan now, keeping all format requirements EXACTLY.
"""

# -----------------------------
# Lambda invocation helper
# -----------------------------
def invoke_execution_lambda(structured: dict) -> Dict:
    """
    Invoke the execution Lambda synchronously with ONLY the actions array as payload.

    - FunctionName: Glue_Execution_Agent
    - Payload: JSON array = structured["actions"]

    Returns the parsed Lambda JSON response (or best-effort dict on parse error).
    """
    # 1) Extract only the actions array
    actions = []
    try:
        if isinstance(structured, dict):
            actions = structured.get("actions") or []
        if not isinstance(actions, list):
            print("⚠️ 'actions' is present but not a list. Coercing to empty list.")
            actions = []
    except Exception as e:
        print(f"⚠️ Error extracting actions from structured plan: {e}")
        actions = []

    if not actions:
        print("⚠️ No actions to send. Skipping Lambda invocation.")
        return {"status": "skipped", "reason": "no_actions"}

    # 2) Invoke Lambda
    function_name = "Glue_Execution_Agent"  # <- your Lambda name
    region =  "us-east-1"
    account_b_id = "997525378140"  # <-- replace with real Account B ID
    function_arn = f"arn:aws:lambda:us-east-1:{account_b_id}:function:Glue_Execution_Agent"

    print(f"🚀 Invoking Lambda '{function_name}' in region '{region}' with ONLY the actions array...")
    try:
        client = boto3.client("lambda", region_name=region)
        payload = {"actions": actions}
        resp = client.invoke(
            FunctionName=function_arn,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload).encode("utf-8")
        )
        payload_stream = resp.get("Payload")
        payload_text = payload_stream.read().decode("utf-8", errors="replace") if hasattr(payload_stream, "read") else ""
        try:
            result = json.loads(payload_text)
        except Exception:
            # Fallback if the Lambda double-encoded JSON or returned a string
            try:
                result = json.loads(json.loads(payload_text))
            except Exception:
                print("⚠️ Could not parse Lambda result JSON. Raw:", payload_text[:1000])
                result = {"raw": payload_text}
        return result
    except Exception as e:
        print(f"❌ Lambda invocation failed: {e}")
        return {"status": "failed", "error": str(e)}

# -----------------------------
# Fixer Lambda invocation (kept)
# -----------------------------
def run_approval_agent():
    print("\n===== APPROVAL AGENT =====")
    print("Plan approved ✅")

# -----------------------------
# CLI / Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to a file containing the dummy error log")
    parser.add_argument("--to", type=str, required=True, help="Recipient email address (who will approve/deny)")
    parser.add_argument("--subject", type=str, default="AWS Glue Job Failure - Fix Steps", help="Email subject")
    parser.add_argument("--html-email", action="store_true", help="Send email as HTML")
    parser.add_argument("--poll-seconds", type=int, default=30, help="IMAP poll interval in seconds")
    parser.add_argument("--timeout-seconds", type=int, default=900, help="IMAP decision timeout in seconds (default 15 min)")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        raise SystemExit(f"File not found: {args.file}")

    with open(args.file, "r", encoding="utf-8") as f:
        log_text = f.read()

    # -------------------------------------------------------------------------
    # KB FIRST (lazy & non-fatal)
    # -------------------------------------------------------------------------
    kb_structured = None
    try:
        kb = get_kb()
        if kb:
            kb_structured = kb.try_build_plan(log_text, {})
    except Exception as e:
        print(f"[CLI:KB] Error while loading/using KB: {e}")
        kb_structured = None

    # -------------------------------------------------------------------------
    # If KB matched, use KB plan and skip LLM
    # -------------------------------------------------------------------------
    if kb_structured:
        print("\n===== KB MATCH FOUND — USING KB FIX PLAN =====\n")
        human = f"""
**Root Cause**
- {kb_structured.get("root_cause")}

**What will be executed (if approved)**  
{chr(10).join([f"- [{a['id']}] {a['capability']}" for a in kb_structured.get('actions', [])])}

(This plan came from S3 Knowledge Base, not the LLM.)
""".strip()

        structured = kb_structured

    else:
        print("\n===== NO KB MATCH — RUNNING LLM =====\n")
        raw = run_agent(log_text)
        human, structured = parse_sections(raw)

    print("\n===== HUMAN-READABLE EXPLANATION =====\n")
    print(human)

    print("\n===== STRUCTURED JSON =====\n")
    print(json.dumps(structured, indent=2))

    lambda_steps = build_lambda_steps(structured)
    instructions = build_lambda_instructions(structured, job_name_from_payload=None)

    print("\n===== STEPS FOR LAMBDA (Action/Resource) =====")
    print(json.dumps(lambda_steps, indent=2))

    if not lambda_steps:
        reason_bits = []
        actions = structured.get("actions") or []
        has_policy = any(a.get("capability") in ("iam.propose_policy_patch", "s3.propose_bucket_policy_patch") for a in actions)
        has_s3_test = any(a.get("capability") == "s3.test_path_exists" for a in actions)
        has_s3_test_with_path = any(
            a.get("capability") == "s3.test_path_exists" and (a.get("params") or {}).get("path")
            for a in actions
        )
        if not has_policy:
            reason_bits.append("no iam/s3 policy proposals present")
        if has_s3_test and not has_s3_test_with_path:
            reason_bits.append("s3.test_path_exists is present but params.path is missing")
        print("ℹ️ lambda_steps is empty because " + (", ".join(reason_bits) or "no extractable actions found"))

    print("\n===== GENERAL LAMBDA INSTRUCTIONS =====")
    print(json.dumps(instructions, indent=2))

    token = uuid.uuid4().hex[:8]
    subject_with_token = f"{args.subject} [Token:{token}] — Reply SUBJECT with APPROVE {token} or DENY {token}"

    instructions_txt = (
        f"\n\nPlease reply by editing the SUBJECT to one of:\n"
        f"  APPROVE {token}\n"
        f"  DENY {token}\n\n"
        f"Example: APPROVE {token}\n"
    )
    body_to_send = human + instructions_txt

    print(f"\n📧 Sending email to {args.to}...")
    if args.html_email:
        html_body = f"<html><body><pre style='font-family:Consolas,Monaco,monospace'>{body_to_send}</pre></body></html>"
        sent = send_email(args.to, subject_with_token, html_body, html=True)
    else:
        normalized_body = body_to_send.replace("\\n", "\n")
        sent = send_email(args.to, subject_with_token, normalized_body, html=False)

    if not sent:
        raise SystemExit("Email send failed; cannot proceed to approval listening.")

    decision = wait_for_email_response(
        recipient_email=args.to,
        token=token,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds
    )

    if decision == "approve":
        print("\nDecision: Approved ✅. Invoking execution Lambda...")
        # === Call execution Lambda with ONLY the actions array
        result = invoke_execution_lambda(structured)
        print("\n===== EXECUTION LAMBDA RESULT (CLI) =====")
        print(json.dumps(result, indent=2))
    elif decision == "deny":
        print("\nDecision: Denied ❌. Skipping follow-up agent.")
    else:
        print("\nDecision: Timeout ⏱️. No reply received within allotted time.")

if __name__ == "__main__":
    main()