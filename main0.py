"""
Splitwise MCP (Public BYO) — Auth0 + Redis (encrypted creds)

Goal:
- Public MCP server anyone can use
- Each user authenticates with Auth0
- Each user stores THEIR Splitwise keys (BYO)
- Tool calls use ONLY that user's stored creds (never yours)

Key HTTP paths (important for ChatGPT):
- MCP endpoint is mounted at: /mcp
- OAuth discovery endpoints MUST be at root:
  /.well-known/oauth-protected-resource/mcp
  /.well-known/oauth-authorization-server/mcp
FastMCP's auth provider can generate these routes, but they must be mounted at root
when MCP is mounted under /mcp.

Env vars required:
Auth0:
- AUTH0_CONFIG_URL
- AUTH0_CLIENT_ID
- AUTH0_CLIENT_SECRET
- AUTH0_AUDIENCE
- PUBLIC_BASE_URL (e.g. https://visioner.fastmcp.app)

Redis:
- REDIS_HOST
- REDIS_PORT (optional)
- REDIS_PASSWORD (optional)
- REDIS_SSL (optional, default "true")

Encryption:
- FERNET_KEY
"""

import os
import json
import base64
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from cryptography.fernet import Fernet

from fastapi import FastAPI
import uvicorn

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_access_token

from fastmcp.server.auth.providers.auth0 import Auth0Provider

from splitwise import Splitwise
from splitwise.expense import Expense
from splitwise.user import ExpenseUser

from redis.asyncio import Redis

load_dotenv()

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

ROOT_URL = os.environ["PUBLIC_BASE_URL"].rstrip("/")  # e.g. https://visioner.fastmcp.app
MOUNT_PREFIX = "/mcp"  # externally, ChatGPT connects to https://.../mcp
MCP_PATH_INSIDE_MOUNT = "/"  # inside mounted app, MCP endpoint is at "/"

FERNET_KEY = os.environ["FERNET_KEY"]
fernet = Fernet(FERNET_KEY.encode() if isinstance(FERNET_KEY, str) else FERNET_KEY)

AUTH0_CONFIG_URL = os.environ["AUTH0_CONFIG_URL"]
AUTH0_CLIENT_ID = os.environ["AUTH0_CLIENT_ID"]
AUTH0_CLIENT_SECRET = os.environ["AUTH0_CLIENT_SECRET"]
AUTH0_AUDIENCE = os.environ["AUTH0_AUDIENCE"]

# IMPORTANT:
# base_url should include the mount prefix because FastMCP will create operational routes
# like /authorize, /token, /auth/callback under that base.
AUTH_BASE_URL = f"{ROOT_URL}{MOUNT_PREFIX}"

# -----------------------------------------------------------------------------
# Auth provider + MCP server
# -----------------------------------------------------------------------------

auth_provider = Auth0Provider(
    config_url=AUTH0_CONFIG_URL,
    client_id=AUTH0_CLIENT_ID,
    client_secret=AUTH0_CLIENT_SECRET,
    audience=AUTH0_AUDIENCE,
    base_url=AUTH_BASE_URL,
)

mcp = FastMCP("Splitwise MCP (Public BYO)", auth=auth_provider)  # type: ignore


# -----------------------------------------------------------------------------
# Redis (async) + encryption helpers
# -----------------------------------------------------------------------------

_redis: Optional[Redis] = None
_redis_lock = asyncio.Lock()


def _redis_url_from_env() -> str:
    """
    Build a redis:// or rediss:// URL from env vars.
    Supports REDIS_HOST with or without ':port'.
    """
    host_raw = os.environ["REDIS_HOST"].strip()
    port_raw = os.environ.get("REDIS_PORT", "").strip() or None
    password = os.environ.get("REDIS_PASSWORD")
    use_ssl = os.environ.get("REDIS_SSL", "true").strip().lower() in ("1", "true", "yes", "y")

    host = host_raw
    port = port_raw

    # If REDIS_HOST includes a port (host:port) and REDIS_PORT is not set, split it
    if ":" in host_raw and port_raw is None:
        maybe_host, maybe_port = host_raw.rsplit(":", 1)
        if maybe_port.isdigit():
            host = maybe_host
            port = maybe_port

    if port is None:
        port = "6379"

    scheme = "rediss" if use_ssl else "redis"

    if password:
        # Redis Cloud typically uses password auth; username not needed
        return f"{scheme}://:{password}@{host}:{port}"
    return f"{scheme}://{host}:{port}"


async def _get_redis() -> Redis:
    global _redis
    if _redis is not None:
        return _redis

    async with _redis_lock:
        if _redis is not None:
            return _redis

        url = _redis_url_from_env()
        client = Redis.from_url(url, decode_responses=True)

        # quick connectivity check
        await client.ping()

        _redis = client
        return _redis


def _enc_json(data: Dict[str, Any]) -> str:
    raw = json.dumps(data).encode("utf-8")
    return fernet.encrypt(raw).decode("utf-8")


def _dec_json(token: str) -> Dict[str, Any]:
    raw = fernet.decrypt(token.encode("utf-8"))
    return json.loads(raw.decode("utf-8"))


def _auth0_subject() -> str:
    """
    Get the current authenticated user's subject (sub).
    FastMCP verifies the token; we just read identity.
    """
    token = get_access_token()
    if token is None:
        raise RuntimeError("Not authenticated.")

    # Try common attributes
    sub = getattr(token, "subject", None) or getattr(token, "sub", None)

    # Try claims dict if present (depends on provider/version)
    if not sub:
        claims = getattr(token, "claims", None)
        if isinstance(claims, dict):
            sub = claims.get("sub")

    # As a fallback, decode JWT payload WITHOUT verifying (already verified by provider)
    if not sub:
        raw = getattr(token, "token", None) or getattr(token, "raw", None)
        if isinstance(raw, str) and raw.count(".") >= 2:
            payload_b64 = raw.split(".")[1]
            payload_b64 += "=" * (-len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("utf-8")))
            sub = payload.get("sub")

    if not sub:
        raise RuntimeError("Could not determine user identity (sub) from access token.")

    return str(sub)


async def _redis_key_for_user() -> str:
    sub = _auth0_subject()
    return f"splitwise:{sub}:creds"


async def _save_creds(creds: Dict[str, Any]) -> None:
    r = await _get_redis()
    key = await _redis_key_for_user()
    await r.set(key, _enc_json(creds))


async def _load_creds() -> Optional[Dict[str, Any]]:
    r = await _get_redis()
    key = await _redis_key_for_user()
    raw = await r.get(key)
    if not raw:
        return None
    return _dec_json(raw)


async def _delete_creds() -> None:
    r = await _get_redis()
    key = await _redis_key_for_user()
    await r.delete(key)


# -----------------------------------------------------------------------------
# Splitwise client (BYO creds)
# -----------------------------------------------------------------------------

def _client_from_creds(creds: Dict[str, Any]) -> Splitwise:
    consumer_key = creds["SPLITWISE_CONSUMER_KEY"]
    consumer_secret = creds["SPLITWISE_CONSUMER_SECRET"]

    api_key = creds.get("SPLITWISE_API_KEY")
    s = Splitwise(consumer_key, consumer_secret, api_key=api_key) if api_key else Splitwise(consumer_key, consumer_secret)

    oauth_token = creds.get("SPLITWISE_OAUTH_TOKEN")
    oauth_token_secret = creds.get("SPLITWISE_OAUTH_TOKEN_SECRET")
    if oauth_token and oauth_token_secret:
        s.setAccessToken({"oauth_token": oauth_token, "oauth_token_secret": oauth_token_secret})

    return s


async def _client_for_current_user() -> Splitwise:
    creds = await _load_creds()
    if not creds:
        raise RuntimeError(
            "No Splitwise credentials saved for your user yet. "
            "Call splitwise_set_credentials first."
        )
    return _client_from_creds(creds)


# -----------------------------------------------------------------------------
# Helpers: normalization / lookup
# -----------------------------------------------------------------------------

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


async def _get_me_friends_groups(s: Splitwise) -> Tuple[Any, List[Any], List[Any]]:
    me = await asyncio.to_thread(s.getCurrentUser)
    friends = await asyncio.to_thread(s.getFriends)
    groups = await asyncio.to_thread(s.getGroups)
    return me, friends, groups


async def _get_group_members(s: Splitwise, group_id: int) -> List[Any]:
    g = await asyncio.to_thread(s.getGroup, int(group_id))
    return getattr(g, "getMembers", lambda: [])() or []


def _d2(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# -----------------------------------------------------------------------------
# Public BYO tools: credentials management
# -----------------------------------------------------------------------------

@mcp.tool
async def splitwise_credentials_status() -> Dict[str, Any]:
    """Check if you have saved Splitwise credentials (values are never returned)."""
    creds = await _load_creds()
    if not creds:
        return {"ok": True, "has_credentials": False}

    present = sorted([k for k, v in creds.items() if v])
    return {"ok": True, "has_credentials": True, "present_fields": present}


@mcp.tool
async def splitwise_set_credentials(
    SPLITWISE_CONSUMER_KEY: str,
    SPLITWISE_CONSUMER_SECRET: str,
    SPLITWISE_API_KEY: Optional[str] = None,
    SPLITWISE_OAUTH_TOKEN: Optional[str] = None,
    SPLITWISE_OAUTH_TOKEN_SECRET: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Save YOUR Splitwise credentials for future calls (encrypted in Redis).

    Recommended:
    - SPLITWISE_CONSUMER_KEY
    - SPLITWISE_CONSUMER_SECRET
    - SPLITWISE_API_KEY (personal token)

    Optional:
    - SPLITWISE_OAUTH_TOKEN + SPLITWISE_OAUTH_TOKEN_SECRET
    """
    data = {
        "SPLITWISE_CONSUMER_KEY": SPLITWISE_CONSUMER_KEY.strip(),
        "SPLITWISE_CONSUMER_SECRET": SPLITWISE_CONSUMER_SECRET.strip(),
        "SPLITWISE_API_KEY": (SPLITWISE_API_KEY or "").strip() or None,
        "SPLITWISE_OAUTH_TOKEN": (SPLITWISE_OAUTH_TOKEN or "").strip() or None,
        "SPLITWISE_OAUTH_TOKEN_SECRET": (SPLITWISE_OAUTH_TOKEN_SECRET or "").strip() or None,
    }
    if not data["SPLITWISE_CONSUMER_KEY"] or not data["SPLITWISE_CONSUMER_SECRET"]:
        return {"ok": False, "errors": ["consumer key/secret are required"]}

    await _save_creds(data)
    return {"ok": True, "saved_fields": sorted([k for k, v in data.items() if v])}


@mcp.tool
async def splitwise_clear_credentials() -> Dict[str, Any]:
    """Delete your saved Splitwise credentials from Redis."""
    await _delete_creds()
    return {"ok": True, "deleted": True}


# -----------------------------------------------------------------------------
# READ TOOLS
# -----------------------------------------------------------------------------

@mcp.tool
async def splitwise_current_user() -> Dict[str, Any]:
    """Return YOUR Splitwise profile (based on your saved creds)."""
    s = await _client_for_current_user()
    u = await asyncio.to_thread(s.getCurrentUser)
    return _user_to_dict(u)


@mcp.tool
async def splitwise_friends() -> List[Dict[str, Any]]:
    """List YOUR Splitwise friends with balances."""
    s = await _client_for_current_user()
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


@mcp.tool
async def splitwise_groups() -> List[Dict[str, Any]]:
    """List YOUR Splitwise groups."""
    s = await _client_for_current_user()
    groups = await asyncio.to_thread(s.getGroups)
    return [{"id": g.getId(), "name": g.getName()} for g in groups]


@mcp.tool
async def splitwise_expenses(
    limit: int = 20,
    offset: int = 0,
    group_id: Optional[int] = None,
    dated_after: Optional[str] = None,
    dated_before: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch YOUR Splitwise expenses (light projection)."""
    s = await _client_for_current_user()

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


# -----------------------------------------------------------------------------
# SINGLE CREATE TOOL (shares-based)
# -----------------------------------------------------------------------------

@mcp.tool
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
    """
    Create ONE Splitwise expense using shares.
    Supports equal/unequal/no-split/percent split.

    Names:
    - Use "me" to refer to the current Splitwise user.
    """
    if cost <= 0:
        raise ValueError("cost must be > 0")

    total_cost = _d2(Decimal(str(cost)))

    s = await _client_for_current_user()
    me, friends, groups = await _get_me_friends_groups(s)

    my_id = getattr(me, "getId", lambda: None)()
    if not my_id:
        raise ValueError("Could not determine current Splitwise user id.")

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


    owed_total = sum((e["owed_share"] for e in split_entries), Decimal("0"))
    paid_total = sum((e["paid_share"] for e in split_entries), Decimal("0"))

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


# -----------------------------------------------------------------------------
# Other write tools
# -----------------------------------------------------------------------------

@mcp.tool
async def splitwise_update_expense(
    expense_id: int,
    description: Optional[str] = None,
    cost: Optional[float] = None,
) -> Dict[str, Any]:
    """Update an existing expense."""
    s = await _client_for_current_user()

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


@mcp.tool
async def splitwise_delete_expense(expense_id: int) -> Dict[str, Any]:
    """Delete an expense."""
    s = await _client_for_current_user()
    success, errors = await asyncio.to_thread(s.deleteExpense, int(expense_id))
    return {"ok": bool(success), "errors": errors}


@mcp.tool
async def splitwise_add_comment(expense_id: int, content: str) -> Dict[str, Any]:
    """Add a comment to an expense."""
    s = await _client_for_current_user()
    comment, errors = await asyncio.to_thread(s.createComment, int(expense_id), content)
    if errors:
        return {"ok": False, "errors": errors}
    return {"ok": True, "comment_id": comment.getId(), "content": comment.getContent()}


# -----------------------------------------------------------------------------
# Build an ASGI app that:
# - mounts well-known auth discovery at root (RFC requirement)
# - mounts MCP server under /mcp
# -----------------------------------------------------------------------------

well_known_routes = auth_provider.get_well_known_routes(mcp_path=MOUNT_PREFIX)

mcp_app = mcp.http_app(path=MCP_PATH_INSIDE_MOUNT)

app = FastAPI(
    lifespan=mcp_app.lifespan,
    routes=[*well_known_routes],
)

app.mount(MOUNT_PREFIX, mcp_app)


# -----------------------------------------------------------------------------
# Entrypoint (local)
# NOTE: FastMCP CLI doesn't run __main__ when using "fastmcp run".
# FastMCP Cloud typically runs via CLI, but exporting `app` is still useful.
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
