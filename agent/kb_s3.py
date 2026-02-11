# kb_s3.py
import re
import json
import time
import yaml
import boto3
from typing import Dict, List, Optional, Tuple
from botocore.exceptions import ClientError

S3_URI_RE = re.compile(r"s3://[a-z0-9.\-_/]+", re.IGNORECASE)
KMS_ARN_RE = re.compile(r"arn:aws:kms:[a-z0-9\-]+:\d{12}:key/[a-f0-9\-]+", re.IGNORECASE)

def _now() -> float:
    return time.time()

class S3KB:
    """
    S3-backed knowledge base for Glue self-healing.

    Required layout in S3 (under the provided s3_bucket/s3_prefix):
      - index.json     # { error_key: { slug, aliases[] }, ... }
      - index.yaml     # glue_error_index: { error_key: "errors/<file>.yaml", ... }
      - errors/*.yaml  # per-error KB with safe_fixes etc.
    """

    # NOTE: s3_bucket is REQUIRED and STATIC (no env fallback).
    def __init__(self, s3_bucket: str, s3_prefix: str = ""):
        if not s3_bucket:
            raise ValueError("s3_bucket is required")
        self.s3_bucket = s3_bucket
        self.s3_prefix = (s3_prefix or "").lstrip("/")

        # Cache TTL in seconds (static default 300). You can tune via code if needed.
        self.cache_ttl = 300
        self.s3 = boto3.client("s3")

        self._index_json: Dict = {}
        self._index_yaml: Dict = {}
        self._alias_map: List[Tuple[re.Pattern, str]] = []
        self._obj_cache: Dict[str, Dict] = {}  # key -> {"etag": str, "ts": float, "body": str}

        self.reload()

    # ----------------------------
    # S3 helpers with ETag+TTL cache
    # ----------------------------
    def _s3_key(self, name: str) -> str:
        return f"{self.s3_prefix}{name}" if self.s3_prefix else name

    def _get_text(self, key: str) -> str:
        """
        Read S3 object (cached by ETag and TTL).
        Gracefully handle cases where HeadObject is forbidden by falling back to GetObject.
        Includes light retry/backoff for transient errors.
        """
        cache = self._obj_cache.get(key)
        if cache and (self.cache_ttl <= 0 or (_now() - cache["ts"] < self.cache_ttl)):
            return cache["body"]

        etag = None
        # Try HeadObject to leverage ETag (if permitted)
        try:
            head = self.s3.head_object(Bucket=self.s3_bucket, Key=key)
            etag = (head.get("ETag") or "").strip('"')
        except ClientError as e:
            code = (e.response.get("Error", {}) or {}).get("Code", "")
            # If head is forbidden or not found, we'll still try GetObject
            if code not in ("403", "404", "AccessDenied", "NoSuchKey"):
                # Unexpected ClientError -> re-raise
                raise

        # Light retry loop for GetObject (handles transient 5xx)
        last_exc = None
        for attempt in range(1, 4):
            try:
                obj = self.s3.get_object(Bucket=self.s3_bucket, Key=key)
                body = obj["Body"].read().decode("utf-8", errors="replace")
                self._obj_cache[key] = {"etag": etag or "NA", "ts": _now(), "body": body}
                return body
            except ClientError as e:
                last_exc = e
                code = (e.response.get("Error", {}) or {}).get("Code", "")
                # Immediate fail on access errors
                if code in ("403", "AccessDenied"):
                    raise
                # Backoff for transient errors
                time.sleep(0.3 * attempt)
            except Exception as e:
                last_exc = e
                time.sleep(0.3 * attempt)

        # If we get here, retries exhausted
        raise last_exc if last_exc else RuntimeError(f"Failed to read s3://{self.s3_bucket}/{key}")

    # ----------------------------
    # KB lifecycle
    # ----------------------------
    def reload(self):
        """
        Load/refresh index.json and index.yaml from S3 and rebuild alias map.
        """
        idx_json_key = self._s3_key("index.json")
        idx_json_txt = self._get_text(idx_json_key)
        self._index_json = json.loads(idx_json_txt)

        idx_yaml_key = self._s3_key("index.yaml")
        idx_yaml_txt = self._get_text(idx_yaml_key)
        self._index_yaml = yaml.safe_load(idx_yaml_txt)

        # Build alias regex map
        self._alias_map = []
        for error_key, idx_entry in (self._index_json or {}).items():
            aliases = (idx_entry or {}).get("aliases", [])
            for alias in aliases:
                pat = re.compile(re.escape(alias), re.IGNORECASE)
                self._alias_map.append((pat, error_key))

    def _match_error_key(self, log_text: str, explicit_error_key: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        Match using payload.error_type/error_code (if present), then alias regex on logs.
        Returns (error_key, matched_alias_pattern_or_None)
        """
        if explicit_error_key and explicit_error_key in self._index_json:
            return explicit_error_key, None

        text = log_text or ""
        for pat, error_key in self._alias_map:
            if pat.search(text):
                return error_key, pat.pattern
        return None, None

    def _kb_yaml_key_for_error_key(self, error_key: str) -> Optional[str]:
        """
        Resolve error_key -> errors/<file>.yaml using index.yaml: glue_error_index
        """
        mapping = (self._index_yaml or {}).get("glue_error_index") or {}
        rel_path = mapping.get(error_key)
        if not rel_path:
            return None
        return self._s3_key(rel_path)

    @staticmethod
    def _extract_context(log_text: str) -> Dict[str, List[str]]:
        s3_uris = list(dict.fromkeys(S3_URI_RE.findall(log_text or "")))  # de-dup
        kms_arns = list(dict.fromkeys(KMS_ARN_RE.findall(log_text or "")))
        return {"s3_uris": s3_uris, "kms_arns": kms_arns}

    # ----------------------------
    # Public: KB → structured plan
    # ----------------------------
    def try_build_plan(self, log_text: str, payload: Dict) -> Optional[Dict]:
        """
        If a KB entry matches, return structured JSON (your agent schema).
        Else return None.
        """
        explicit_error_key = payload.get("error_type") or payload.get("error_code")
        error_key, matched_alias = self._match_error_key(log_text, explicit_error_key)
        if not error_key:
            return None

        kb_yaml_key = self._kb_yaml_key_for_error_key(error_key)
        if not kb_yaml_key:
            return None

        kb_yaml_txt = self._get_text(kb_yaml_key)
        kb_doc = yaml.safe_load(kb_yaml_txt) or {}

        job_name = payload.get("job_name")
        ctx = self._extract_context(log_text)

        actions: List[Dict] = []
        aid = 1

        # 1) Prechecks (read-only)
        if job_name:
            actions.append({
                "id": f"a{aid}",
                "capability": "glue.get_job",
                "params": {"job_name": job_name},
                "requires_approval": False,
                "prechecks": [],
                "postchecks": ["Glue job definition retrieved"],
                "rollback": None
            }); aid += 1

        for s3_uri in (ctx["s3_uris"] or [])[:5]:  # cap to avoid explosion
            actions.append({
                "id": f"a{aid}",
                "capability": "s3.test_path_exists",
                "params": {"path": s3_uri},
                "requires_approval": False,
                "prechecks": [],
                "postchecks": [f"S3 path validated: {s3_uri}"],
                "rollback": None
            }); aid += 1

        # 2) Convert safe_fixes -> IAM policy proposals (approval required)
        fixes = kb_doc.get("safe_fixes") or []
        policy_statements: List[Dict] = []

        def _s3_uri_to_arn_patterns(u: str) -> List[str]:
            # s3://bucket/prefix -> narrow ARNs
            u = u[5:]  # strip s3://
            parts = u.split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            if key:
                return [f"arn:aws:s3:::{bucket}/{key}*", f"arn:aws:s3:::{bucket}/{key}/*"]
            return [f"arn:aws:s3:::{bucket}/*"]

        for fix in fixes:
            allowed = (fix or {}).get("allowed_iam_actions") or []
            if not allowed:
                continue

            resources: List[str] = []

            if any(a.lower().startswith("s3:") for a in allowed):
                if ctx["s3_uris"]:
                    for u in ctx["s3_uris"]:
                        resources.extend(_s3_uri_to_arn_patterns(u))
                else:
                    resources.append("arn:aws:s3:::REPLACE_BUCKET/REPLACE_PREFIX*")

            if any(a.lower().startswith("kms:") for a in allowed):
                if ctx["kms_arns"]:
                    resources.extend(ctx["kms_arns"])
                else:
                    resources.append("arn:aws:kms:REGION:ACCOUNT_ID:key/REPLACE_KEY_ID")

            # de-dup while preserving order
            seen = set()
            dedup = []
            for r in resources:
                if r not in seen:
                    seen.add(r)
                    dedup.append(r)

            if dedup:
                policy_statements.append({
                    "Sid": f"KB_{fix.get('fix_id', 'fix')}",
                    "Effect": "Allow",
                    "Action": allowed,
                    "Resource": dedup
                })

        if policy_statements:
            actions.append({
                "id": f"a{aid}",
                "capability": "iam.propose_policy_patch",
                "params": {
                    "policy": {
                        "Version": "2012-10-17",
                        "Statement": policy_statements
                    },
                    "note": f"Proposed by KB: {error_key}. Apply to the Glue job role after approval."
                },
                "requires_approval": True,
                "prechecks": ["Confirm target role ARN from glue.get_job before apply"],
                "postchecks": ["Re-validate S3/KMS access", "Re-run job in non-prod first"],
                "rollback": None
            }); aid += 1

        # 3) Safe re-run (non-destructive)
        if job_name:
            actions.append({
                "id": f"a{aid}",
                "capability": "glue.start_job_run",
                "params": {"job_name": job_name, "arguments": {"--self_healer": "true"}},
                "requires_approval": False,
                "prechecks": ["All required policies have been applied (if any)"],
                "postchecks": ["JobRun state transitions to RUNNING or SUCCEEDED"],
                "rollback": None
            }); aid += 1

        if not actions:
            return None

        root_cause = kb_doc.get("description") or error_key
        severity = (kb_doc.get("severity") or "").lower()

        structured = {
            "root_cause": root_cause,
            "explanation": (
                f"Matched KB entry '{error_key}'"
                + (f" via alias pattern {matched_alias!r}" if matched_alias else "")
                + ". Applied KB safe_fixes as least-privilege IAM policy proposals."
            ),
            "suggested_fix": {
                "type": "infra",
                "steps": [
                    "Run read-only prechecks",
                    "Propose IAM policy patch (approval required)",
                    f"Safe re-run of job {job_name or '<unknown>'}"
                ]
            },
            "confidence": 0.85,
            "next_steps": [
                "Review and approve IAM proposal",
                "Apply to Glue job role",
                "Re-run job and monitor"
            ],
            "tags": [
                "kb-match",
                error_key,
                (kb_doc.get("category") or "unknown").lower(),
                f"severity:{severity or 'unknown'}"
            ],
            "actions": actions
        }
        return structured