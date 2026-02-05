"""
Splitwise MCP Server (Async) — Multi-tenant via per-user Splitwise API Key (HTTP header)

Auth model:
- Server holds app credentials (consumer key/secret) in env:
    SPLITWISE_CONSUMER_KEY
    SPLITWISE_CONSUMER_SECRET
- Each request provides the user's Splitwise API key in headers:
    - Authorization: Bearer <splitwise_api_key>   (preferred)
      OR
    - x-splitwise-api-key: <splitwise_api_key>

Safety model for write tools:
- If confirm=False: returns preview + confirmation_token; does NOT write.
- If confirm=True: requires confirmation_token from prior preview with same payload & same owner.

Notes:
- This server is stateless w.r.t user secrets: it does NOT store API keys.
- In-memory token caches are per-process. If you run multiple replicas, use Redis or sticky sessions.
"""

import os
import asyncio
import time
import json
import hashlib
import uuid
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers

from splitwise import Splitwise
from splitwise.expense import Expense
from splitwise.user import ExpenseUser

load_dotenv()
mcp = FastMCP("Splitwise MCP")

# =============================================================================
# In-memory caches (idempotency + confirmations)
# =============================================================================

_DEDUPE_TTL_SEC = 120          # de-dupe window for identical creates
_CONFIRM_TTL_SEC = 15 * 60     # confirmation token lifetime (15 minutes)


class _DedupeEntry(NamedTuple):
    ts: float
    response: Dict[str, Any]


class _ConfirmEntry(NamedTuple):
    ts: float
    action: str
    payload: Dict[str, Any]
    owner: str                 # bind confirmation to a specific requester


_recent_creates: Dict[str, _DedupeEntry] = {}
_pending_confirms: Dict[str, _ConfirmEntry] = {}


def _now() -> float:
    return time.time()


def _prune_cache() -> None:
    # prune create de-dupe
    d_cutoff = _now() - _DEDUPE_TTL_SEC
    dead = [k for k, v in _recent_creates.items() if v.ts < d_cutoff]
    for k in dead:
        _recent_creates.pop(k, None)

    # prune confirmation tokens
    c_cutoff = _now() - _CONFIRM_TTL_SEC
    dead2 = [k for k, v in _pending_confirms.items() if v.ts < c_cutoff]
    for k in dead2:
        _pending_confirms.pop(k, None)


def _stable_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _make_confirmation(action: str, payload: Dict[str, Any], owner: str) -> str:
    _prune_cache()
    token = str(uuid.uuid4())
    _pending_confirms[token] = _ConfirmEntry(ts=_now(), action=action, payload=payload, owner=owner)
    return token


def _require_confirmation(action: str, token: Optional[str], payload: Dict[str, Any], owner: str) -> Tuple[bool, Optional[str]]:
    """
    Returns (ok, error_message).
    ok=True means token exists, is not expired, matches action, owner, AND payload matches.
    """
    _prune_cache()
    if not token:
        return False, "Missing confirmation_token. Call preview first to obtain a token."

    entry = _pending_confirms.get(token)
    if not entry:
        return False, "Invalid or expired confirmation_token. Call preview again."

    if entry.owner != owner:
        return False, "confirmation_token belongs to a different client. Call preview again."

    if entry.action != action:
        return False, f"confirmation_token is for a different action ({entry.action}). Call preview again."

    # Require payload match so the token cannot be reused to commit something else.
    if _stable_hash(entry.payload) != _stable_hash(payload):
        return False, "confirmation_token payload mismatch. Call preview again."

    return True, None


# =============================================================================
# Per-request credentials from headers (API-key-only)
# =============================================================================

def _get_splitwise_api_key_from_headers() -> Optional[str]:
    """
    Reads the user's Splitwise API key from either:
    - Authorization: Bearer <key>
    - x-splitwise-api-key: <key>
    """
    h = get_http_headers() or {}

    api_key = h.get("x-splitwise-api-key")

    auth = h.get("authorization")
    if (not api_key) and auth and auth.lower().startswith("bearer "):
        api_key = auth.split(" ", 1)[1].strip()

    return api_key


def _owner_from_api_key(api_key: str) -> str:
    # Never store raw secrets as "owner"; hash only.
    return hashlib.sha256(f"api:{api_key}".encode("utf-8")).hexdigest()


def _require_api_key_or_error(api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    if api_key:
        return None
    return {
        "ok": False,
        "error": "Missing Splitwise API key. Provide Authorization: Bearer <key> or x-splitwise-api-key: <key>."
    }


# =============================================================================
# Splitwise client
# =============================================================================

def _client(api_key: str) -> Splitwise:
    """Create a configured Splitwise client using host consumer key/secret + per-request user api_key."""
    consumer_key = os.environ["SPLITWISE_CONSUMER_KEY"]
    consumer_secret = os.environ["SPLITWISE_CONSUMER_SECRET"]
    return Splitwise(consumer_key, consumer_secret, api_key=api_key)


def _client_from_request() -> Tuple[Optional[Splitwise], Optional[str], Optional[Dict[str, Any]]]:
    """
    Returns (client, owner, error_response).
    """
    api_key = _get_splitwise_api_key_from_headers()
    err = _require_api_key_or_error(api_key)
    if err:
        return None, None, err
    owner = _owner_from_api_key(api_key)  # type: ignore[arg-type]
    return _client(api_key=api_key), owner, None  # type: ignore[arg-type]


# =============================================================================
# Helpers: normalization / lookup
# =============================================================================

def _norm(s: str) -> str:
    """Normalize strings for robust matching."""
    return " ".join((s or "").strip().lower().split())


def _user_to_dict(u: Any) -> Dict[str, Any]:
    """Convert Splitwise user-like objects to JSON-friendly dict."""
    return {
        "id": getattr(u, "getId", lambda: None)(),
        "first_name": getattr(u, "getFirstName", lambda: None)(),
        "last_name": getattr(u, "getLastName", lambda: None)(),
        "email": getattr(u, "getEmail", lambda: None)(),
    }


def _find_user_id_by_name(users: List[Any], name: str) -> Optional[int]:
    """Match by first name OR full name."""
    target = _norm(name)
    if not target:
        return None

    for u in users:
        first = _norm(getattr(u, "getFirstName", lambda: "")() or "")
        last = _norm(getattr(u, "getLastName", lambda: "")() or "")
        full = _norm(f"{first} {last}".strip())

        if target == first or target == full:
            return getattr(u, "getId", lambda: None)()

    return None


def _find_group_by_name(groups: List[Any], group_name: str) -> Optional[Any]:
    """Find group by name (case-insensitive)."""
    target = _norm(group_name)
    if not target:
        return None

    for g in groups:
        if _norm(getattr(g, "getName", lambda: "")() or "") == target:
            return g

    return None


async def _get_me_friends_groups(s: Splitwise) -> Tuple[Any, List[Any], List[Any]]:
    """Fetch current user, friends, groups."""
    me = await asyncio.to_thread(s.getCurrentUser)
    friends = await asyncio.to_thread(s.getFriends)
    groups = await asyncio.to_thread(s.getGroups)
    return me, friends, groups


async def _get_group_members(s: Splitwise, group_id: int) -> List[Any]:
    """Fetch group details and return members."""
    g = await asyncio.to_thread(s.getGroup, int(group_id))
    return getattr(g, "getMembers", lambda: [])() or []


def _d2(x: Decimal) -> Decimal:
    """Quantize to 2 decimals (currency cents) using HALF_UP rounding."""
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# =============================================================================
# AUTH / SANITY TOOL (read-only)
# =============================================================================

@mcp.tool()
async def splitwise_auth_check() -> Dict[str, Any]:
    """
    Validate that the provided Splitwise API key works.
    Returns the current user's profile if successful.
    """
    s, _, err = _client_from_request()
    if err:
        return err
    try:
        u = await asyncio.to_thread(s.getCurrentUser)
        return {"ok": True, "valid": True, "user": _user_to_dict(u)}
    except Exception as e:
        # Keep errors generic so we don't leak anything sensitive.
        return {"ok": False, "valid": False, "error": "Authentication failed with provided API key.", "detail": str(e)}


# =============================================================================
# READ TOOLS
# =============================================================================

@mcp.tool()
async def splitwise_current_user() -> Dict[str, Any]:
    """Return the authenticated user's profile."""
    s, _, err = _client_from_request()
    if err:
        return err
    u = await asyncio.to_thread(s.getCurrentUser)
    return {"ok": True, **_user_to_dict(u)}


@mcp.tool()
async def splitwise_friends() -> Dict[str, Any]:
    """List friends with balances."""
    s, _, err = _client_from_request()
    if err:
        return err

    friends = await asyncio.to_thread(s.getFriends)

    out: List[Dict[str, Any]] = []
    for f in friends:
        item = _user_to_dict(f)
        balances = []
        for b in (getattr(f, "getBalances", lambda: [])() or []):
            balances.append(
                {
                    "currency": getattr(b, "getCurrencyCode", lambda: None)(),
                    "amount": getattr(b, "getAmount", lambda: None)(),
                }
            )
        item["balances"] = balances
        out.append(item)

    return {"ok": True, "friends": out}


@mcp.tool()
async def splitwise_groups() -> Dict[str, Any]:
    """List groups."""
    s, _, err = _client_from_request()
    if err:
        return err
    groups = await asyncio.to_thread(s.getGroups)
    return {"ok": True, "groups": [{"id": g.getId(), "name": g.getName()} for g in groups]}


@mcp.tool()
async def splitwise_expenses(
    limit: int = 20,
    offset: int = 0,
    group_id: Optional[int] = None,
    dated_after: Optional[str] = None,
    dated_before: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch expenses (light projection)."""
    s, _, err = _client_from_request()
    if err:
        return err

    def _fetch():
        return s.getExpenses(
            offset=offset,
            limit=limit,
            group_id=group_id,
            dated_after=dated_after,
            dated_before=dated_before,
        )

    expenses = await asyncio.to_thread(_fetch)
    return {
        "ok": True,
        "expenses": [
            {
                "id": e.getId(),
                "group_id": e.getGroupId(),
                "description": e.getDescription(),
                "cost": e.getCost(),
                "currency_code": e.getCurrencyCode(),
                "date": e.getDate(),
            }
            for e in expenses
        ],
    }


# =============================================================================
# SINGLE CREATE TOOL (shares-based) — preview+token, then commit+token
# =============================================================================

@mcp.tool()
async def splitwise_create_expense_shares(
    description: str,
    cost: float,
    currency_code: Optional[str] = None,

    # Group targeting (optional): provide either group_id OR group_name
    group_id: Optional[int] = None,
    group_name: Optional[str] = None,

    # Default payer (optional): if paid_share not provided, payer pays 100%
    paid_by: str = "me",

    # Participants for equal split (optional). If owed_share/owed_percent are not provided,
    # we will split equally among these names.
    participants: Optional[List[str]] = None,

    # Splits list (optional): each item supports name + owed_share OR owed_percent,
    # and optionally paid_share.
    splits: Optional[List[Dict[str, Any]]] = None,

    # Confirmation gate
    confirm: bool = False,
    confirmation_token: Optional[str] = None,

    # Idempotency / de-dupe
    request_id: Optional[str] = None,
    force_create: bool = False,
) -> Dict[str, Any]:
    """
    Create ONE Splitwise expense using shares.

    Safety:
    - If confirm=False: returns preview + confirmation_token; does NOT write.
    - If confirm=True: requires confirmation_token from a prior preview for same payload (and same owner).
    """
    if cost <= 0:
        return {"ok": False, "error": "cost must be > 0"}

    s, owner, err = _client_from_request()
    if err:
        return err

    total_cost = _d2(Decimal(str(cost)))

    me, friends, groups = await _get_me_friends_groups(s)

    my_id = getattr(me, "getId", lambda: None)()
    if not my_id:
        return {"ok": False, "error": "Could not determine current user id."}

    # Resolve group id by name if needed
    resolved_group_id = group_id
    if resolved_group_id is None and group_name:
        g = _find_group_by_name(groups, group_name)
        if not g:
            available = [getattr(x, "getName", lambda: "")() for x in groups]
            return {"ok": False, "errors": [f"Group not found: {group_name}", "Available: " + ", ".join(available)]}
        resolved_group_id = getattr(g, "getId", lambda: None)()

    # Use group members for name -> id if group specified; else use friends
    members = friends
    if resolved_group_id is not None:
        members = await _get_group_members(s, int(resolved_group_id))

    def resolve_id(name: str) -> Optional[int]:
        nn = _norm(name)
        if nn in ("me", "myself", "i"):
            return int(my_id)
        return _find_user_id_by_name(members, name) or _find_user_id_by_name(friends, name)

    # Build split entries
    split_entries: List[Dict[str, Any]] = []

    if splits:
        unresolved: List[str] = []
        resolved: List[Dict[str, Any]] = []

        for item in splits:
            name = item.get("name")
            if not name:
                return {"ok": False, "error": "Each split must include 'name'"}
            uid = resolve_id(str(name))
            if uid is None:
                unresolved.append(str(name))
                continue
            resolved.append({"id": int(uid), **item})

        if unresolved:
            member_names = [
                f"{getattr(m, 'getFirstName', lambda: '')() or ''} {getattr(m, 'getLastName', lambda: '')() or ''}".strip()
                for m in members
            ]
            return {
                "ok": False,
                "errors": [
                    "Could not resolve these names: " + ", ".join(unresolved),
                    "Members I can see: " + ", ".join([n for n in member_names if n]),
                ],
            }

        uses_percent = any("owed_percent" in x or "percent" in x for x in resolved)
        uses_amount = any("owed_share" in x or "owedShare" in x for x in resolved)
        if uses_percent and uses_amount:
            return {"ok": False, "error": "Use either owed_percent OR owed_share, not both mixed."}

        if uses_percent:
            pct_total = Decimal("0")
            owed_list: List[Decimal] = []
            for x in resolved:
                pct = Decimal(str(x.get("owed_percent", x.get("percent", 0))))
                pct_total += pct
                owed_list.append(_d2(total_cost * pct / Decimal("100")))

            if abs(pct_total - Decimal("100")) > Decimal("0.01"):
                return {"ok": False, "errors": [f"owed_percent must sum to 100. Got {pct_total}."]}

            drift = total_cost - sum(owed_list, Decimal("0"))
            if drift != Decimal("0"):
                owed_list[-1] = _d2(owed_list[-1] + drift)

            for x, owed in zip(resolved, owed_list):
                split_entries.append(
                    {
                        "id": int(x["id"]),
                        "owed_share": owed,
                        "paid_share": _d2(Decimal(str(x.get("paid_share", x.get("paidShare", 0))))) if ("paid_share" in x or "paidShare" in x) else None,
                    }
                )
        elif uses_amount:
            for x in resolved:
                owed = Decimal(str(x.get("owed_share", x.get("owedShare", 0))))
                split_entries.append(
                    {
                        "id": int(x["id"]),
                        "owed_share": _d2(owed),
                        "paid_share": _d2(Decimal(str(x.get("paid_share", x.get("paidShare", 0))))) if ("paid_share" in x or "paidShare" in x) else None,
                    }
                )
        else:
            return {"ok": False, "error": "Each split must include owed_share or owed_percent."}
    else:
        if not participants:
            return {"ok": False, "error": "Provide either splits or participants."}

        ids: List[int] = []
        unresolved: List[str] = []
        for name in participants:
            uid = resolve_id(name)
            if uid is None:
                unresolved.append(name)
            else:
                ids.append(int(uid))

        if unresolved:
            member_names = [
                f"{getattr(m, 'getFirstName', lambda: '')() or ''} {getattr(m, 'getLastName', lambda: '')() or ''}".strip()
                for m in members
            ]
            return {
                "ok": False,
                "errors": [
                    "Could not resolve these participant names: " + ", ".join(unresolved),
                    "Members I can see: " + ", ".join([n for n in member_names if n]),
                ],
            }

        n = len(ids)
        owed_each = _d2(total_cost / Decimal(n))
        owed_list = [owed_each] * n
        drift = total_cost - sum(owed_list, Decimal("0"))
        if drift != Decimal("0"):
            owed_list[-1] = _d2(owed_list[-1] + drift)

        for uid, owed in zip(ids, owed_list):
            split_entries.append({"id": uid, "owed_share": owed, "paid_share": None})

    # Decide paid shares
    any_paid_provided = any(e["paid_share"] is not None for e in split_entries)
    if not any_paid_provided:
        payer_id = resolve_id(paid_by)
        if payer_id is None:
            return {"ok": False, "errors": [f"Could not resolve paid_by name: {paid_by}"]}

        for e in split_entries:
            e["paid_share"] = Decimal("0.00")

        found = False
        for e in split_entries:
            if int(e["id"]) == int(payer_id):
                e["paid_share"] = total_cost
                found = True
                break

        if not found:
            split_entries.append({"id": int(payer_id), "owed_share": Decimal("0.00"), "paid_share": total_cost})

    for e in split_entries:
        e["paid_share"] = _d2(Decimal(str(e["paid_share"])))

    # Validate sums and fix tiny drift
    owed_total = sum((e["owed_share"] for e in split_entries), Decimal("0"))
    owed_drift = total_cost - owed_total
    if owed_drift != Decimal("0") and split_entries:
        split_entries[-1]["owed_share"] = _d2(split_entries[-1]["owed_share"] + owed_drift)

    paid_total = sum((e["paid_share"] for e in split_entries), Decimal("0"))
    paid_drift = total_cost - paid_total
    if paid_drift != Decimal("0") and split_entries:
        split_entries[-1]["paid_share"] = _d2(split_entries[-1]["paid_share"] + paid_drift)

    owed_total = sum((e["owed_share"] for e in split_entries), Decimal("0"))
    paid_total = sum((e["paid_share"] for e in split_entries), Decimal("0"))
    if abs(owed_total - total_cost) > Decimal("0.01") or abs(paid_total - total_cost) > Decimal("0.01"):
        return {
            "ok": False,
            "errors": [f"Shares invalid after rounding: owed_total={owed_total}, paid_total={paid_total}, cost={total_cost}"],
        }

    preview_payload = {
        "description": description,
        "cost": f"{total_cost:.2f}",
        "currency_code": currency_code,
        "group_id": resolved_group_id,
        "paid_by": paid_by,
        "splits": [
            {"id": int(e["id"]), "paid_share": f"{e['paid_share']:.2f}", "owed_share": f"{e['owed_share']:.2f}"}
            for e in split_entries
        ],
    }

    # PREVIEW path (always safe)
    if not confirm:
        token = _make_confirmation("create", preview_payload, owner=owner)
        return {
            "ok": True,
            "preview": True,
            "message": "Preview only. Re-run with confirm=True and the confirmation_token to create this expense.",
            "confirmation_token": token,
            **preview_payload,
        }

    # COMMIT path: require token
    ok, err_msg = _require_confirmation("create", confirmation_token, preview_payload, owner=owner)
    if not ok:
        token = _make_confirmation("create", preview_payload, owner=owner)
        return {
            "ok": True,
            "preview": True,
            "blocked": True,
            "message": f"Write blocked: {err_msg}",
            "confirmation_token": token,
            **preview_payload,
        }

    # De-dupe / idempotency (owner-bound)
    _prune_cache()
    dedupe_key = f"{owner}|rid:{request_id}" if request_id else f"{owner}|sig:{_stable_hash(preview_payload)}"
    if (not force_create) and dedupe_key in _recent_creates:
        prev = _recent_creates[dedupe_key].response
        return {**prev, "deduped": True}

    # Create expense
    expense = Expense()
    expense.setDescription(description)
    expense.setCost(f"{total_cost:.2f}")
    if resolved_group_id is not None:
        expense.setGroupId(int(resolved_group_id))
    if currency_code:
        expense.setCurrencyCode(currency_code)

    users: List[ExpenseUser] = []
    for e in split_entries:
        eu = ExpenseUser()
        eu.setId(int(e["id"]))
        eu.setPaidShare(f"{e['paid_share']:.2f}")
        eu.setOwedShare(f"{e['owed_share']:.2f}")
        users.append(eu)

    expense.setUsers(users)

    created, errors = await asyncio.to_thread(s.createExpense, expense)
    if errors:
        return {"ok": False, "errors": errors}

    response = {
        "ok": True,
        "expense_id": created.getId(),
        "group_id": resolved_group_id,
        "currency_code": currency_code,
        "splits": [
            {"id": int(e["id"]), "paid_share": f"{e['paid_share']:.2f}", "owed_share": f"{e['owed_share']:.2f}"}
            for e in split_entries
        ],
    }

    _recent_creates[dedupe_key] = _DedupeEntry(ts=_now(), response=response)
    return response


# =============================================================================
# Other write tools — require preview token, then commit token (owner-bound)
# =============================================================================

@mcp.tool()
async def splitwise_update_expense(
    expense_id: int,
    description: Optional[str] = None,
    cost: Optional[float] = None,
    confirm: bool = False,
    confirmation_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Update an existing expense. NO write unless confirmed with a valid token."""
    s, owner, err = _client_from_request()
    if err:
        return err

    preview_payload = {
        "expense_id": int(expense_id),
        "description": description,
        "cost": f"{float(cost):.2f}" if cost is not None else None,
    }

    if not confirm:
        token = _make_confirmation("update", preview_payload, owner=owner)
        return {
            "ok": True,
            "preview": True,
            "message": "Preview only. Re-run with confirm=True and the confirmation_token to apply this update.",
            "confirmation_token": token,
            **preview_payload,
        }

    ok, err_msg = _require_confirmation("update", confirmation_token, preview_payload, owner=owner)
    if not ok:
        token = _make_confirmation("update", preview_payload, owner=owner)
        return {
            "ok": True,
            "preview": True,
            "blocked": True,
            "message": f"Write blocked: {err_msg}",
            "confirmation_token": token,
            **preview_payload,
        }

    e = Expense()
    e.id = int(expense_id)
    if description is not None:
        e.setDescription(description)
    if cost is not None:
        e.setCost(f"{float(cost):.2f}")

    updated, errors = await asyncio.to_thread(s.updateExpense, e)
    if errors:
        return {"ok": False, "errors": errors}
    return {"ok": True, "expense_id": updated.getId()}


@mcp.tool()
async def splitwise_delete_expense(
    expense_id: int,
    confirm: bool = False,
    confirmation_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Delete an expense. NO delete unless confirmed with a valid token."""
    s, owner, err = _client_from_request()
    if err:
        return err

    preview_payload = {"expense_id": int(expense_id)}

    if not confirm:
        token = _make_confirmation("delete", preview_payload, owner=owner)
        return {
            "ok": True,
            "preview": True,
            "message": "Preview only. Re-run with confirm=True and the confirmation_token to delete this expense.",
            "confirmation_token": token,
            **preview_payload,
        }

    ok, err_msg = _require_confirmation("delete", confirmation_token, preview_payload, owner=owner)
    if not ok:
        token = _make_confirmation("delete", preview_payload, owner=owner)
        return {
            "ok": True,
            "preview": True,
            "blocked": True,
            "message": f"Write blocked: {err_msg}",
            "confirmation_token": token,
            **preview_payload,
        }

    success, errors = await asyncio.to_thread(s.deleteExpense, int(expense_id))
    return {"ok": bool(success), "errors": errors}


@mcp.tool()
async def splitwise_add_comment(
    expense_id: int,
    content: str,
    confirm: bool = False,
    confirmation_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Add a comment. NO write unless confirmed with a valid token."""
    s, owner, err = _client_from_request()
    if err:
        return err

    preview_payload = {"expense_id": int(expense_id), "content": content}

    if not confirm:
        token = _make_confirmation("comment", preview_payload, owner=owner)
        return {
            "ok": True,
            "preview": True,
            "message": "Preview only. Re-run with confirm=True and the confirmation_token to post this comment.",
            "confirmation_token": token,
            **preview_payload,
        }

    ok, err_msg = _require_confirmation("comment", confirmation_token, preview_payload, owner=owner)
    if not ok:
        token = _make_confirmation("comment", preview_payload, owner=owner)
        return {
            "ok": True,
            "preview": True,
            "blocked": True,
            "message": f"Write blocked: {err_msg}",
            "confirmation_token": token,
            **preview_payload,
        }

    comment, errors = await asyncio.to_thread(s.createComment, int(expense_id), content)
    if errors:
        return {"ok": False, "errors": errors}
    return {"ok": True, "comment_id": comment.getId(), "content": comment.getContent()}


# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    # IMPORTANT: run behind HTTPS in production (reverse proxy / ingress)
    mcp.run(transport="http", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
