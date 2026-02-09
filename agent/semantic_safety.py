import re

# -----------------------------
# 1. Forbidden keywords — absolutely not allowed
# -----------------------------
FORBIDDEN_KEYWORDS = {
    "delete bucket",
    "delete s3 bucket",
    "remove bucket",
    "drop table",
    "drop database",
    "delete object",
    "truncate",
    "rm -rf",
    "wipe",
    "disable encryption",
    "kms:disable",
    "detach policy",
    "remove permission",
    "revoke permissions",
    "replace script with empty",
}

# -----------------------------
# 2. Logical ordering (safe-first)
# -----------------------------
SAFE_ACTIONS = {
    "s3.test_path_exists",
    "glue.get_job",
    "glue.get_job_run"
}

LOW_RISK_ACTIONS = {
    "glue.update_job_default_args",
    "glue.update_job_script_ref",
}

LAST_RESORT = {
    "iam.propose_policy_patch",
    "s3.propose_bucket_policy_patch",
}

# -----------------------------
# 3. Semantic Safety
# -----------------------------
def semantic_safety_check(plan):
    """
    Makes sure the fix plan:
    - is safe
    - avoids destructive actions
    - follows safe-first ordering
    - avoids nonsense fix steps
    """

    errors = []
    warnings = []

    # Text-level scan
    text = str(plan).lower()
    for bad in FORBIDDEN_KEYWORDS:
        if bad in text:
            errors.append(f"Forbidden dangerous instruction detected: '{bad}'")

    actions = plan.get("actions", [])
        # NEW: basic shape requirement — ensure we actually have actions
    if not isinstance(actions, list) or len(actions) == 0:
        errors.append("Plan has no actions — a valid fix plan must include at least one action.")
        return errors, warnings
    # Ordering: safe actions MUST come before IAM/S3 privileged proposals
    saw_last_resort = False
    for a in actions:
        cap = a.get("capability")

        if cap in LAST_RESORT:
            saw_last_resort = True

        if saw_last_resort and cap in SAFE_ACTIONS:
            errors.append("Privileged fix appears before safe checks — bad ordering.")

    # If last-resort actions exist → safe actions MUST exist
    if any(a.get("capability") in LAST_RESORT for a in actions):
        if not any(a.get("capability") in SAFE_ACTIONS for a in actions):
            errors.append("IAM/S3 policy patches cannot be suggested without safe checks first.")

    # Mutating actions MUST have rollback
    for a in actions:
        cap = a.get("capability")
        if cap in LOW_RISK_ACTIONS:
            if not a.get("rollback"):
                errors.append(f"Mutating action '{cap}' must include rollback.")

    return errors, warnings


# -----------------------------
# 4. Blast Radius Check
# -----------------------------
def blast_radius_check(plan, job_name_from_payload, allowed_bucket=None):
    """
    Ensures all Glue actions refer ONLY to the failing job
    Ensures S3 actions refer ONLY to expected buckets (optional)
    """

    errors = []

    for a in plan.get("actions", []):
        cap = a.get("capability")

        # Glue directional blast radius
        if cap.startswith("glue."):
            target_job = (
                a.get("params", {}).get("job_name") or
                a.get("params", {}).get("JobName")
            )
            if target_job and job_name_from_payload and target_job != job_name_from_payload:
                errors.append(
                    f"Blast radius violation: action '{cap}' targets job '{target_job}', "
                    f"but failing job is '{job_name_from_payload}'."
                )

        # Optional S3 bucket radius
        if allowed_bucket:
            if "s3" in cap:
                uri = a.get("params", {}).get("s3_uri") or ""
                if uri.startswith("s3://"):
                    bucket = uri.split("/")[2]
                    if bucket != allowed_bucket:
                        errors.append(
                            f"Blast radius violation: S3 action '{cap}' touches bucket '{bucket}', "
                            f"but allowed bucket is '{allowed_bucket}'."
                        )

    return errors

def safety_score(plan, job_name_from_payload=None):
    """
    Compute a numeric safety score (0–100) based on semantic and blast-radius violations.
    Higher score = safer plan.
    """
    base_score = 100

    # Run existing checks
    sem_errors, _ = semantic_safety_check(plan)
    br_errors = blast_radius_check(plan, job_name_from_payload)

    penalties = {
        "forbidden_keywords": 0,
        "ordering_issues": 0,
        "missing_safe_checks": 0,
        "missing_rollback": 0,
        "no_actions_or_parse": 0,
        "blast_radius": 0,
    }

    # Penalties for semantic errors
    for err in sem_errors:
        text = err.lower()

        if "forbidden" in text:
            penalties["forbidden_keywords"] += 30

        elif "bad ordering" in text:
            penalties["ordering_issues"] += 15

        elif "cannot be suggested without safe checks" in text:
            penalties["missing_safe_checks"] += 10

        elif "must include rollback" in text:
            penalties["missing_rollback"] += 8

        elif "no actions" in text or "missing 'actions'" in text:
            penalties["no_actions_or_parse"] += 40

        else:
            penalties["ordering_issues"] += 5

    # Penalties for blast-radius violations
    if br_errors:
        penalties["blast_radius"] += min(40, len(br_errors) * 15)

    # Compute final score
    score = base_score - sum(penalties.values())
    score = max(0, min(100, score))

    return score, penalties