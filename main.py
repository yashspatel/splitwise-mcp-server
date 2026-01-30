"""
Splitwise MCP Server (Async) — Public Multi-User (BYO Splitwise creds) + Auth0 + Redis

✅ What this solves:
- Each user (Raj, etc.) uses THEIR OWN Splitwise credentials.
- Expenses created by Raj go ONLY to Raj’s Splitwise.
- Your Splitwise account is never touched unless YOU saved YOUR creds.

✅ Why Redis is used:
- Store each user's Splitwise creds (encrypted).
- Store Auth0 OAuth proxy client registrations + tokens (encrypted) so ChatGPT "Refresh actions"
  keeps working even if the server restarts / scales.

───────────────────────────────────────────────────────────────────────────────
Required env vars

Auth0:
- AUTH0_CONFIG_URL            e.g. https://YOUR_DOMAIN/.well-known/openid-configuration
- AUTH0_CLIENT_ID
- AUTH0_CLIENT_SECRET
- AUTH0_AUDIENCE
- PUBLIC_BASE_URL             e.g. https://visioner.fastmcp.app

Redis:
- REDIS_HOST                  e.g. redis-xxxx.cloud.redislabs.com
- REDIS_PORT                  e.g. 11683
- REDIS_PASSWORD              (from Redis Cloud)
- REDIS_SSL                   "true" or "false" (try true first)

OAuth proxy persistence (VERY IMPORTANT for ChatGPT Refresh actions):
- JWT_SIGNING_KEY             any long random string (keep stable forever)
- STORAGE_ENCRYPTION_KEY      Fernet key (encrypts OAuth proxy data at rest)

Splitwise creds encryption:
- FERNET_KEY                  Fernet key (encrypts Splitwise creds at rest)
  (If you don't set FERNET_KEY, we will reuse STORAGE_ENCRYPTION_KEY.)

───────────────────────────────────────────────────────────────────────────────
requirements.txt (recommended):

fastmcp==2.14.4
python-dotenv>=1.0.0
splitwise>=3.0.0
py-key-value-aio[redis]>=0.3.0
cryptography>=42.0.0
"""

import os
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.auth0 import Auth0Provider
from fastmcp.server.dependencies import get_access_token

from cryptography.fernet import Fernet
from key_value.aio.stores.redis import RedisStore
from key_value.aio.wrappers.encryption import FernetEncryptionWrapper

from splitwise import Splitwise
from splitwise.expense import Expense
from splitwise.user import ExpenseUser

load_dotenv()


# =============================================================================
# Helpers: env + redis url/store
# =============================================================================

def _bool_env(name: str, default: bool = False) -> bool:
    v = (os.getenv(name, "") or "").strip().lower()
    if v == "":
        return default
    return v in ("1", "true", "yes", "y", "on")


def _redis_url() -> str:
    """
    Build redis:// or rediss:// URL.
    We prefer URL because it’s the cleanest way to represent TLS usage.
    """
    host = os.environ["REDIS_HOST"]
    port = int(os.environ.get("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD")
    use_ssl = _bool_env("REDIS_SSL", True)

    scheme = "rediss" if use_ssl else "redis"
    if password:
        return f"{scheme}://:{password}@{host}:{port}/0"
    return f"{scheme}://{host}:{port}/0"


def _make_redis_store() -> RedisStore:
    """
    Create RedisStore robustly.
    Some versions support RedisStore(url=...), others prefer host/port/password.
    We try url first, then fallback.
    """
    try:
        return RedisStore(url=_redis_url())
    except TypeError:
        # Fallback if this version doesn't accept `url=`
        host = os.environ["REDIS_HOST"]
        port = int(os.environ.get("REDIS_PORT", "6379"))
        password = os.getenv("REDIS_PASSWORD")
        return RedisStore(host=host, port=port, password=password)


# =============================================================================
# Auth0 OAuth proxy persistence (fixes ChatGPT "Error refreshing actions")
# =============================================================================

def _oauth_client_storage():
    """
    Storage for OAuth proxy registrations/tokens (NOT your Splitwise creds).
    Must be shared & persistent for cloud deployments.
    """
    redis_store = _make_redis_store()
    fernet_key = os.environ["STORAGE_ENCRYPTION_KEY"]
    return FernetEncryptionWrapper(key_value=redis_store, fernet=Fernet(fernet_key))


# =============================================================================
# FastMCP server with Auth0
# =============================================================================

auth_provider = Auth0Provider(
    config_url=os.environ["AUTH0_CONFIG_URL"],
    client_id=os.environ["AUTH0_CLIENT_ID"],
    client_secret=os.environ["AUTH0_CLIENT_SECRET"],
    audience=os.environ["AUTH0_AUDIENCE"],
    base_url=os.environ["PUBLIC_BASE_URL"].rstrip("/"),

    # ✅ critical for production + ChatGPT connector stability
    jwt_signing_key=os.environ["JWT_SIGNING_KEY"],
    client_storage=_oauth_client_storage(),
)

mcp = FastMCP("Splitwise MCP (Public BYO)", auth=auth_provider)


# =============================================================================
# Splitwise creds store (per Auth0 user, encrypted at rest)
# =============================================================================

def _splitwise_creds_store():
    redis_store = _make_redis_store()

    # If you didn't set FERNET_KEY, reuse STORAGE_ENCRYPTION_KEY
    fernet_key = os.getenv("FERNET_KEY") or os.environ["STORAGE_ENCRYPTION_KEY"]
    return FernetEncryptionWrapper(key_value=redis_store, fernet=Fernet(fernet_key))

creds_store = _splitwise_creds_store()


def _auth0_sub() -> str:
    """
    Get Auth0 user id (sub) from the access token.
    This is the key that keeps each user's data separate.
    """
    token = get_access_token()
    sub = (token.claims or {}).get("sub")
    if not sub:
        raise ValueError("Missing Auth0 subject (sub) in access token.")
    return str(sub)


async def _get_creds(sub: str) -> Optional[Dict[str, Any]]:
    # We store dicts directly; wrapper handles serialization/encryption.
    return await creds_store.get(key=f"splitwise:{sub}:creds")


async def _set_creds(sub: str, creds: Dict[str, Any]) -> None:
    await creds_store.put(key=f"splitwise:{sub}:creds", value=creds)


async def _delete_creds(sub: str) -> None:
    await creds_store.delete(key=f"splitwise:{sub}:creds")


# =============================================================================
# Splitwise client per-user
# =============================================================================

def _client_from_creds(creds: Dict[str, Any]) -> Splitwise:
    consumer_key = (creds.get("consumer_key") or "").strip()
    consumer_secret = (creds.get("consumer_secret") or "").strip()
    if not consumer_key or not consumer_secret:
        raise ValueError("Missing Splitwise consumer_key/consumer_secret. Run splitwise_save_credentials first.")

    api_key = creds.get("api_key") or None
    s = Splitwise(consumer_key, consumer_secret, api_key=api_key) if api_key else Splitwise(consumer_key, consumer_secret)

    oauth_token = creds.get("oauth_token") or None
    oauth_token_secret = creds.get("oauth_token_secret") or None
    if oauth_token and oauth_token_secret:
        s.setAccessToken({"oauth_token": oauth_token, "oauth_token_secret": oauth_token_secret})

    return s


async def _client_for_request() -> Splitwise:
    sub = _auth0_sub()
    creds = await _get_creds(sub)
    if not creds:
        raise ValueError("Splitwise not connected for this user. Run splitwise_save_credentials first.")
    return _client_from_creds(creds)


# =============================================================================
# Helpers: normalization / lookup
# =============================================================================

def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _user_to_dict(u: Any) -> Dict[str, Any]:
    return {
        "id": getattr(u, "getId", lambda: None)(),
        "first_name": getattr(u, "getFirstName", lambda: None)(),
        "last_name": getattr(u, "getLastName", lambda: None)(),
        "email": getattr(u, "getEmail", lambda: None)(),
    }


def _find_user_id_by_name(users: List[Any], name: str) -> Optional[int]:
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
    target = _norm(group_name)
    if not target:
        return None
    for g in groups:
        if _norm(getattr(g, "getName", lambda: "")() or "") == target:
            return g
    return None


def _d2(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


async def _get_me_friends_groups(s: Splitwise) -> Tuple[Any, List[Any], List[Any]]:
    me = await asyncio.to_thread(s.getCurrentUser)
    friends = await asyncio.to_thread(s.getFriends)
    groups = await asyncio.to_thread(s.getGroups)
    return me, friends, groups


async def _get_group_members(s: Splitwise, group_id: int) -> List[Any]:
    g = await asyncio.to_thread(s.getGroup, int(group_id))
    return getattr(g, "getMembers", lambda: [])() or []


# =============================================================================
# Debug / sanity tool (helps you test connector end-to-end)
# =============================================================================

@mcp.tool()
async def ping() -> Dict[str, Any]:
    return {"ok": True, "msg": "pong"}


@mcp.tool()
async def auth_debug_token_info() -> Dict[str, Any]:
    """
    Debug tool to confirm ChatGPT is sending a valid token.
    Safe: does not reveal secrets.
    """
    token = get_access_token()
    claims = token.claims or {}
    return {
        "iss": claims.get("iss"),
        "aud": claims.get("aud"),
        "scope": claims.get("scope"),
        "sub": claims.get("sub"),
    }


# =============================================================================
# BYO credential management tools
# =============================================================================

@mcp.tool()
async def splitwise_save_credentials(
    consumer_key: str,
    consumer_secret: str,
    api_key: Optional[str] = None,
    oauth_token: Optional[str] = None,
    oauth_token_secret: Optional[str] = None,
) -> Dict[str, Any]:
    """
    User runs this ONCE to save their Splitwise credentials.
    Stored per Auth0 user (sub), encrypted in Redis.
    """
    sub = _auth0_sub()
    creds = {
        "consumer_key": consumer_key.strip(),
        "consumer_secret": consumer_secret.strip(),
        "api_key": api_key.strip() if api_key else None,
        "oauth_token": oauth_token.strip() if oauth_token else None,
        "oauth_token_secret": oauth_token_secret.strip() if oauth_token_secret else None,
    }

    # Validate quickly (so user knows they pasted correct keys)
    s = _client_from_creds(creds)
    me = await asyncio.to_thread(s.getCurrentUser)

    await _set_creds(sub, creds)
    return {"ok": True, "connected_as": _user_to_dict(me)}


@mcp.tool()
async def splitwise_credentials_status() -> Dict[str, Any]:
    """
    Check whether Splitwise creds are saved (does not reveal actual secrets).
    """
    sub = _auth0_sub()
    creds = await _get_creds(sub)
    if not creds:
        return {"ok": True, "saved": False}

    return {
        "ok": True,
        "saved": True,
        "has_api_key": bool(creds.get("api_key")),
        "has_oauth_token": bool(creds.get("oauth_token") and creds.get("oauth_token_secret")),
    }


@mcp.tool()
async def splitwise_clear_credentials() -> Dict[str, Any]:
    sub = _auth0_sub()
    await _delete_creds(sub)
    return {"ok": True}


# =============================================================================
# READ TOOLS
# =============================================================================

@mcp.tool()
async def splitwise_current_user() -> Dict[str, Any]:
    s = await _client_for_request()
    u = await asyncio.to_thread(s.getCurrentUser)
    return _user_to_dict(u)


@mcp.tool()
async def splitwise_friends() -> List[Dict[str, Any]]:
    s = await _client_for_request()
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

    return out


@mcp.tool()
async def splitwise_groups() -> List[Dict[str, Any]]:
    s = await _client_for_request()
    groups = await asyncio.to_thread(s.getGroups)
    return [{"id": g.getId(), "name": g.getName()} for g in groups]


@mcp.tool()
async def splitwise_expenses(
    limit: int = 20,
    offset: int = 0,
    group_id: Optional[int] = None,
    dated_after: Optional[str] = None,
    dated_before: Optional[str] = None,
) -> List[Dict[str, Any]]:
    s = await _client_for_request()

    def _fetch():
        return s.getExpenses(
            offset=offset,
            limit=limit,
            group_id=group_id,
            dated_after=dated_after,
            dated_before=dated_before,
        )

    expenses = await asyncio.to_thread(_fetch)
    return [
        {
            "id": e.getId(),
            "group_id": e.getGroupId(),
            "description": e.getDescription(),
            "cost": e.getCost(),
            "currency_code": e.getCurrencyCode(),
            "date": e.getDate(),
        }
        for e in expenses
    ]


# =============================================================================
# SINGLE CREATE TOOL (shares-based) — equal/unequal/no-split/percent split
# =============================================================================

@mcp.tool()
async def splitwise_create_expense_shares(
    description: str,
    cost: float,
    currency_code: Optional[str] = None,

    group_id: Optional[int] = None,
    group_name: Optional[str] = None,

    paid_by: str = "me",
    participants: Optional[List[str]] = None,
    splits: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if cost <= 0:
        raise ValueError("cost must be > 0")

    total_cost = _d2(Decimal(str(cost)))

    s = await _client_for_request()
    me, friends, groups = await _get_me_friends_groups(s)

    my_id = getattr(me, "getId", lambda: None)()
    if not my_id:
        raise ValueError("Could not determine current user id.")

    resolved_group_id = group_id
    if resolved_group_id is None and group_name:
        g = _find_group_by_name(groups, group_name)
        if not g:
            available = [getattr(x, "getName", lambda: "")() for x in groups]
            return {"ok": False, "errors": [f"Group not found: {group_name}", "Available: " + ", ".join(available)]}
        resolved_group_id = getattr(g, "getId", lambda: None)()

    members = friends
    if resolved_group_id is not None:
        members = await _get_group_members(s, int(resolved_group_id))

    def resolve_id(name: str) -> Optional[int]:
        nn = _norm(name)
        if nn in ("me", "myself", "i"):
            return int(my_id)
        return _find_user_id_by_name(members, name) or _find_user_id_by_name(friends, name)

    split_entries: List[Dict[str, Any]] = []

    # Case A: splits provided (unequal/no-split/percent)
    if splits:
        unresolved: List[str] = []
        resolved: List[Dict[str, Any]] = []

        for item in splits:
            name = item.get("name")
            if not name:
                raise ValueError("Each split must include 'name'")
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
            raise ValueError("Use either owed_percent OR owed_share, not both mixed.")

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
            raise ValueError("Each split must include owed_share or owed_percent.")

    # Case B: no splits -> equal split among participants
    else:
        if not participants:
            raise ValueError("Provide either `splits` or `participants`.")

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


    # Fix tiny drifts
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

    return {
        "ok": True,
        "expense_id": created.getId(),
        "group_id": resolved_group_id,
        "currency_code": currency_code,
        "splits": [
            {"id": int(e["id"]), "paid_share": f"{e['paid_share']:.2f}", "owed_share": f"{e['owed_share']:.2f}"}
            for e in split_entries
        ],
    }


# =============================================================================
# Other write tools
# =============================================================================

@mcp.tool()
async def splitwise_update_expense(
    expense_id: int,
    description: Optional[str] = None,
    cost: Optional[float] = None,
) -> Dict[str, Any]:
    s = await _client_for_request()

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
async def splitwise_delete_expense(expense_id: int) -> Dict[str, Any]:
    s = await _client_for_request()
    success, errors = await asyncio.to_thread(s.deleteExpense, int(expense_id))
    return {"ok": bool(success), "errors": errors}


@mcp.tool()
async def splitwise_add_comment(expense_id: int, content: str) -> Dict[str, Any]:
    s = await _client_for_request()
    comment, errors = await asyncio.to_thread(s.createComment, int(expense_id), content)
    if errors:
        return {"ok": False, "errors": errors}
    return {"ok": True, "comment_id": comment.getId(), "content": comment.getContent()}


# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    # FastMCP Cloud often sets PORT automatically
    port = int(os.environ.get("PORT", "8000"))
    mcp.run(transport="http", host="0.0.0.0", port=port)
