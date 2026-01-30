"""
Splitwise MCP Server (Async) — Public Multi-User (BYO Splitwise creds) + Auth0

Goal:
- Each caller uses THEIR OWN Splitwise credentials.
- Nothing ever uses your Splitwise account unless YOU (your Auth0 user) saved your creds.

How it works:
1) Auth0 authenticates the caller -> we get a stable user id ("sub").
2) We store that user's Splitwise creds (encrypted) in Redis under splitwise:{sub}:creds
3) Every tool loads creds for the current caller and creates a Splitwise client from them.

Requirements (env vars):
Auth0:
- AUTH0_CONFIG_URL
- AUTH0_CLIENT_ID
- AUTH0_CLIENT_SECRET
- AUTH0_AUDIENCE
- PUBLIC_BASE_URL   (e.g. https://visioner.fastmcp.app)

Redis:
- REDIS_HOST
- REDIS_PORT (optional, default 6379)
- REDIS_PASSWORD (optional)
- REDIS_SSL (optional, default "true")

Encryption:
- FERNET_KEY   (generate once; keep secret; do not rotate casually)

Notes:
- Splitwise Python SDK is synchronous; we wrap calls with asyncio.to_thread(...)
- BYO only: users paste their own Splitwise app keys and OAuth token/secret (or API key)
"""

import os
import json
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server import Context
from fastmcp.server.auth.providers.auth0 import Auth0Provider

from cryptography.fernet import Fernet
from key_value.aio.stores.redis import RedisStore
from key_value.aio.wrappers.encryption import FernetEncryptionWrapper

from splitwise import Splitwise
from splitwise.expense import Expense
from splitwise.user import ExpenseUser

load_dotenv()


# =============================================================================
# Auth0 setup (server-level auth)
# =============================================================================

auth = Auth0Provider(
    config_url=os.environ["AUTH0_CONFIG_URL"],
    client_id=os.environ["AUTH0_CLIENT_ID"],
    client_secret=os.environ["AUTH0_CLIENT_SECRET"],
    audience=os.environ["AUTH0_AUDIENCE"],
    base_url=os.environ["PUBLIC_BASE_URL"],
)

mcp = FastMCP("Splitwise MCP (Public BYO)", auth=auth)


# =============================================================================
# Storage (Redis + encryption at rest)
# =============================================================================

def _store():
    base = RedisStore(
        host=os.environ["REDIS_HOST"],
        port=int(os.environ.get("REDIS_PORT", "6379")),
        password=os.environ.get("REDIS_PASSWORD"),
        ssl=os.environ.get("REDIS_SSL", "true").lower() == "true",
    )
    fernet = Fernet(os.environ["FERNET_KEY"])
    return FernetEncryptionWrapper(key_value=base, fernet=fernet)

store = _store()

async def _get_creds(auth0_sub: str) -> Optional[Dict[str, Any]]:
    raw = await store.get(f"splitwise:{auth0_sub}:creds")
    return json.loads(raw) if raw else None

async def _set_creds(auth0_sub: str, creds: Dict[str, Any]) -> None:
    await store.set(f"splitwise:{auth0_sub}:creds", json.dumps(creds))

async def _delete_creds(auth0_sub: str) -> None:
    await store.delete(f"splitwise:{auth0_sub}:creds")


# =============================================================================
# Helpers: auth0 identity, normalization / lookup
# =============================================================================

def _auth0_sub(ctx: Context) -> str:
    """
    Extract user id from Auth0-verified request.
    FastMCP stores auth info on request state; we handle a couple common shapes.
    """
    if ctx is None:
        raise ValueError("Missing request context (ctx).")

    # Most common place FastMCP auth providers attach info
    auth_state = getattr(getattr(ctx, "request_context", None), "request", None)
    auth_state = getattr(getattr(auth_state, "state", None), "auth", None)

    if isinstance(auth_state, dict):
        sub = (auth_state.get("extra", {}) or {}).get("sub") or auth_state.get("sub")
        if sub:
            return str(sub)

    # Fallbacks (just in case a different structure is used)
    # Try ctx.access_token claims style if present
    access_token = getattr(ctx, "access_token", None)
    claims = getattr(access_token, "claims", None)
    if isinstance(claims, dict) and claims.get("sub"):
        return str(claims["sub"])

    raise ValueError("Missing user identity (sub) in Auth0 token.")


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


def _d2(x: Decimal) -> Decimal:
    """Quantize to 2 decimals (currency cents) using HALF_UP rounding."""
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# =============================================================================
# Splitwise client per-user (BYO creds)
# =============================================================================

def _client_from_creds(creds: Dict[str, Any]) -> Splitwise:
    """
    Build a Splitwise SDK client using the caller's own credentials.
    BYO means consumer_key/secret are required. api_key is optional.
    oauth_token/secret are optional but needed for user-scoped actions.
    """
    consumer_key = (creds.get("consumer_key") or "").strip()
    consumer_secret = (creds.get("consumer_secret") or "").strip()
    if not consumer_key or not consumer_secret:
        raise ValueError("Missing Splitwise consumer_key/consumer_secret. Run splitwise_connect_byo.")

    api_key = creds.get("api_key")
    api_key = api_key.strip() if isinstance(api_key, str) else api_key

    s = Splitwise(consumer_key, consumer_secret, api_key=api_key) if api_key else Splitwise(consumer_key, consumer_secret)

    oauth_token = creds.get("oauth_token")
    oauth_token_secret = creds.get("oauth_token_secret")
    oauth_token = oauth_token.strip() if isinstance(oauth_token, str) else oauth_token
    oauth_token_secret = oauth_token_secret.strip() if isinstance(oauth_token_secret, str) else oauth_token_secret

    if oauth_token and oauth_token_secret:
        s.setAccessToken({"oauth_token": oauth_token, "oauth_token_secret": oauth_token_secret})

    return s


async def _get_client_for_ctx(ctx: Context) -> Splitwise:
    """Load the current caller's creds and build their Splitwise client."""
    sub = _auth0_sub(ctx)
    creds = await _get_creds(sub)
    if not creds:
        raise ValueError("Splitwise not connected. Run splitwise_connect_byo first.")
    return _client_from_creds(creds)


# =============================================================================
# Fetch helpers
# =============================================================================

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


# =============================================================================
# CONNECT / DISCONNECT (BYO)
# =============================================================================

@mcp.tool()
async def splitwise_connect_byo(
    consumer_key: str,
    consumer_secret: str,
    api_key: Optional[str] = None,
    oauth_token: Optional[str] = None,
    oauth_token_secret: Optional[str] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Save your own Splitwise credentials for your Auth0 user.
    Run this ONCE per user. Credentials are stored encrypted in Redis.
    """
    sub = _auth0_sub(ctx)

    creds = {
        "consumer_key": consumer_key.strip(),
        "consumer_secret": consumer_secret.strip(),
        "api_key": api_key.strip() if api_key else None,
        "oauth_token": oauth_token.strip() if oauth_token else None,
        "oauth_token_secret": oauth_token_secret.strip() if oauth_token_secret else None,
    }

    # Validate immediately so user knows creds work
    s = _client_from_creds(creds)
    me = await asyncio.to_thread(s.getCurrentUser)

    await _set_creds(sub, creds)
    return {
        "ok": True,
        "connected_as": {
            "id": me.getId(),
            "first_name": me.getFirstName(),
            "last_name": me.getLastName(),
        },
    }


@mcp.tool()
async def splitwise_disconnect(ctx: Context) -> Dict[str, Any]:
    """Forget the current caller's saved Splitwise creds."""
    sub = _auth0_sub(ctx)
    await _delete_creds(sub)
    return {"ok": True}


@mcp.tool()
async def splitwise_connection_status(ctx: Context) -> Dict[str, Any]:
    """Check if the caller has connected Splitwise creds."""
    sub = _auth0_sub(ctx)
    creds = await _get_creds(sub)
    return {"ok": True, "connected": bool(creds)}


# =============================================================================
# READ TOOLS
# =============================================================================

@mcp.tool()
async def splitwise_current_user(ctx: Context) -> Dict[str, Any]:
    """Return the authenticated caller's Splitwise profile."""
    s = await _get_client_for_ctx(ctx)
    u = await asyncio.to_thread(s.getCurrentUser)
    return _user_to_dict(u)


@mcp.tool()
async def splitwise_friends(ctx: Context) -> List[Dict[str, Any]]:
    """List friends with balances (for the caller)."""
    s = await _get_client_for_ctx(ctx)
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
async def splitwise_groups(ctx: Context) -> List[Dict[str, Any]]:
    """List groups (for the caller)."""
    s = await _get_client_for_ctx(ctx)
    groups = await asyncio.to_thread(s.getGroups)
    return [{"id": g.getId(), "name": g.getName()} for g in groups]


@mcp.tool()
async def splitwise_expenses(
    limit: int = 20,
    offset: int = 0,
    group_id: Optional[int] = None,
    dated_after: Optional[str] = None,
    dated_before: Optional[str] = None,
    ctx: Context = None,
) -> List[Dict[str, Any]]:
    """Fetch expenses (light projection) for the caller."""
    s = await _get_client_for_ctx(ctx)

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

    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Create ONE Splitwise expense using shares.
    Supports equal/unequal/no-split/percent split.

    Rules:
    - If `splits` is provided:
        - owed_share OR owed_percent must be present for each participant (owed_percent sums to 100).
        - paid_share is optional; if missing for everyone, defaults to "paid_by pays 100%".
    - If `splits` is NOT provided:
        - `participants` must be provided, and the tool does equal split automatically.
        - paid_by pays 100% by default.

    Names:
    - Use "me" to refer to the current Splitwise user.
    """
    if cost <= 0:
        raise ValueError("cost must be > 0")

    total_cost = _d2(Decimal(str(cost)))

    s = await _get_client_for_ctx(ctx)
    me, friends, groups = await _get_me_friends_groups(s)

    my_id = getattr(me, "getId", lambda: None)()
    if not my_id:
        raise ValueError("Could not determine current user id.")

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

    # Resolve a name to an id, supporting "me"
    def resolve_id(name: str) -> Optional[int]:
        nn = _norm(name)
        if nn in ("me", "myself", "i"):
            return int(my_id)
        return _find_user_id_by_name(members, name) or _find_user_id_by_name(friends, name)

    # -------------------------------------------------------------------------
    # Build a normalized list of split entries:
    # Each entry becomes: {id, owed_share (Decimal), paid_share (Decimal|None)}
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Decide paid shares:
    # - If any paid_share provided explicitly, use those
    # - Else paid_by pays 100%
    # -------------------------------------------------------------------------
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
            "errors": [
                f"Shares invalid after rounding: owed_total={owed_total}, paid_total={paid_total}, cost={total_cost}"
            ],
        }

    # -------------------------------------------------------------------------
    # Create the Splitwise expense (single entry)
    # -------------------------------------------------------------------------
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
# Other write tools (per-user)
# =============================================================================

@mcp.tool()
async def splitwise_update_expense(
    expense_id: int,
    description: Optional[str] = None,
    cost: Optional[float] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """Update an existing expense (for the caller)."""
    s = await _get_client_for_ctx(ctx)

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
async def splitwise_delete_expense(expense_id: int, ctx: Context = None) -> Dict[str, Any]:
    """Delete an expense (for the caller)."""
    s = await _get_client_for_ctx(ctx)
    success, errors = await asyncio.to_thread(s.deleteExpense, int(expense_id))
    return {"ok": bool(success), "errors": errors}


@mcp.tool()
async def splitwise_add_comment(expense_id: int, content: str, ctx: Context = None) -> Dict[str, Any]:
    """Add a comment to an expense (for the caller)."""
    s = await _get_client_for_ctx(ctx)
    comment, errors = await asyncio.to_thread(s.createComment, int(expense_id), content)
    if errors:
        return {"ok": False, "errors": errors}
    return {"ok": True, "comment_id": comment.getId(), "content": comment.getContent()}


# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
