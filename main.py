"""
Splitwise MCP Server (Async) — Per-request User API Key (STRICT)

What changed vs your original:
- Server NO LONGER supports SPLITWISE_API_KEY in env (privacy).
- Every tool requires the caller to provide their own Splitwise API key
  via HTTP headers:
    - Authorization: Bearer <SPLITWISE_API_KEY>   (recommended)
    - OR X-Splitwise-Api-Key: <SPLITWISE_API_KEY>

- Splitwise Python SDK is synchronous, so all SDK calls are wrapped in
  asyncio.to_thread(...) to keep MCP tools async.

Server still requires YOUR app credentials as env vars:
- SPLITWISE_CONSUMER_KEY
- SPLITWISE_CONSUMER_SECRET
"""

import os
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.dependencies import CurrentHeaders  # FastMCP DI for HTTP headers

from splitwise import Splitwise
from splitwise.expense import Expense
from splitwise.user import ExpenseUser

load_dotenv()
mcp = FastMCP("Splitwise MCP")


# =============================================================================
# Strict per-request Splitwise API key extraction
# =============================================================================

def _norm(s: str) -> str:
    """Normalize strings for robust matching."""
    return " ".join((s or "").strip().lower().split())


def _require_splitwise_api_key(headers: Dict[str, str]) -> str:
    """
    Strictly require a Splitwise API key from request headers.

    Accepted formats:
      1) Authorization: Bearer <api_key>
      2) X-Splitwise-Api-Key: <api_key>

    Raises:
      PermissionError if missing.
    """
    headers = headers or {}
    # normalize header keys
    h = {str(k).lower(): (v if isinstance(v, str) else str(v)) for k, v in headers.items()}

    auth = h.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        if token:
            return token

    api_key = (h.get("x-splitwise-api-key") or "").strip()
    if api_key:
        return api_key

    raise PermissionError(
        "Missing Splitwise API key. Provide either "
        "'Authorization: Bearer <SPLITWISE_API_KEY>' or "
        "'X-Splitwise-Api-Key: <SPLITWISE_API_KEY>'."
    )


# =============================================================================
# Splitwise client (per request)
# =============================================================================

def _client(user_api_key: str) -> Splitwise:
    """Create a configured Splitwise client using server app creds + per-request user api key."""
    consumer_key = os.environ["SPLITWISE_CONSUMER_KEY"]
    consumer_secret = os.environ["SPLITWISE_CONSUMER_SECRET"]

    # STRICT: never read SPLITWISE_API_KEY from server env
    if not user_api_key or not str(user_api_key).strip():
        raise PermissionError("User Splitwise API key is required (strict mode).")

    return Splitwise(consumer_key, consumer_secret, api_key=str(user_api_key).strip())


# =============================================================================
# Helpers: normalization / lookup
# =============================================================================

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
# READ TOOLS
# =============================================================================

@mcp.tool()
async def splitwise_current_user(
    headers: Dict[str, str] = CurrentHeaders(),
) -> Dict[str, Any]:
    """Return the authenticated Splitwise user's profile (strict per-request API key)."""
    api_key = _require_splitwise_api_key(headers)
    s = _client(api_key)
    u = await asyncio.to_thread(s.getCurrentUser)
    return _user_to_dict(u)


@mcp.tool()
async def splitwise_friends(
    headers: Dict[str, str] = CurrentHeaders(),
) -> List[Dict[str, Any]]:
    """List Splitwise friends with balances (strict per-request API key)."""
    api_key = _require_splitwise_api_key(headers)
    s = _client(api_key)

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
async def splitwise_groups(
    headers: Dict[str, str] = CurrentHeaders(),
) -> List[Dict[str, Any]]:
    """List Splitwise groups (strict per-request API key)."""
    api_key = _require_splitwise_api_key(headers)
    s = _client(api_key)

    groups = await asyncio.to_thread(s.getGroups)
    return [{"id": g.getId(), "name": g.getName()} for g in groups]


@mcp.tool()
async def splitwise_expenses(
    limit: int = 20,
    offset: int = 0,
    group_id: Optional[int] = None,
    dated_after: Optional[str] = None,
    dated_before: Optional[str] = None,
    headers: Dict[str, str] = CurrentHeaders(),
) -> List[Dict[str, Any]]:
    """Fetch expenses (light projection) (strict per-request API key)."""
    api_key = _require_splitwise_api_key(headers)
    s = _client(api_key)

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
# SINGLE CREATE TOOL (shares-based)
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

    # Injected request headers (strict API key)
    headers: Dict[str, str] = CurrentHeaders(),
) -> Dict[str, Any]:
    """
    Create ONE Splitwise expense using shares.
    This tool supports equal/unequal/no-split/percent split.

    STRICT AUTH:
    - Requires user's Splitwise API key via request headers:
      Authorization: Bearer <key>  OR  X-Splitwise-Api-Key: <key>

    Rules:
    - If `splits` is provided:
        - owed_share OR owed_percent must be present for each participant (owed_percent sums to 100).
        - paid_share is optional; if missing for everyone, defaults to "paid_by pays 100%".
    - If `splits` is NOT provided:
        - `participants` must be provided, and the tool does equal split automatically.
        - paid_by pays 100% by default.

    Names:
    - Use "me" to refer to the current user.
    """
    api_key = _require_splitwise_api_key(headers)
    s = _client(api_key)

    if cost <= 0:
        raise ValueError("cost must be > 0")

    total_cost = _d2(Decimal(str(cost)))

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
    # Build normalized split entries:
    # Each entry: {id, owed_share (Decimal), paid_share (Decimal|None)}
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

        # Compute owed shares
        if uses_percent:
            pct_total = Decimal("0")
            owed_list: List[Decimal] = []
            for x in resolved:
                pct = Decimal(str(x.get("owed_percent", x.get("percent", 0))))
                pct_total += pct
                owed_list.append(_d2(total_cost * pct / Decimal("100")))

            if abs(pct_total - Decimal("100")) > Decimal("0.01"):
                return {"ok": False, "errors": [f"owed_percent must sum to 100. Got {pct_total}."]}

            # Fix rounding drift by adjusting the last owed_share
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

    # Ensure all paid_share are Decimal and 2dp
    for e in split_entries:
        e["paid_share"] = _d2(Decimal(str(e["paid_share"])))

    # Validate sums (and gently fix 1-cent drift for paid/owed)
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
    # Create the Splitwise expense
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
# Other write tools
# =============================================================================

@mcp.tool()
async def splitwise_update_expense(
    expense_id: int,
    description: Optional[str] = None,
    cost: Optional[float] = None,
    headers: Dict[str, str] = CurrentHeaders(),
) -> Dict[str, Any]:
    """Update an existing expense (strict per-request API key)."""
    api_key = _require_splitwise_api_key(headers)
    s = _client(api_key)

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
    headers: Dict[str, str] = CurrentHeaders(),
) -> Dict[str, Any]:
    """Delete an expense (strict per-request API key)."""
    api_key = _require_splitwise_api_key(headers)
    s = _client(api_key)

    success, errors = await asyncio.to_thread(s.deleteExpense, int(expense_id))
    return {"ok": bool(success), "errors": errors}


@mcp.tool()
async def splitwise_add_comment(
    expense_id: int,
    content: str,
    headers: Dict[str, str] = CurrentHeaders(),
) -> Dict[str, Any]:
    """Add a comment to an expense (strict per-request API key)."""
    api_key = _require_splitwise_api_key(headers)
    s = _client(api_key)

    comment, errors = await asyncio.to_thread(s.createComment, int(expense_id), content)
    if errors:
        return {"ok": False, "errors": errors}
    return {"ok": True, "comment_id": comment.getId(), "content": comment.getContent()}


# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
