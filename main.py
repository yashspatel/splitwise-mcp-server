"""
main.py — Splitwise MCP Server (Async) with a Reliable Create-Expense Router

Problem you hit
---------------
ChatGPT sometimes *chooses the wrong tool* (e.g., equal split) and even claims
"the tool only supports equal splits" when it didn't reliably pick/see the
unequal/no-split tools.

Fix in this file
----------------
1) Expose ONE preferred create tool: splitwise_create_expense(...)
   - It can do:
     - equal split
     - no split (100% owed by one person)
     - unequal split by explicit shares
     - unequal split by percentages (e.g., "my share is 35%")
2) Do NOT expose separate create tools (equal/no_split/unequal) as MCP tools.
   - They remain as internal functions.
   - This prevents ChatGPT from mistakenly calling the equal-split tool.
3) Keep other tools: current_user, friends, groups, expenses, update, delete, add_comment

Notes
-----
- Splitwise Python SDK is synchronous, so we wrap calls with asyncio.to_thread(...)
- For percentage split: we create ONE expense using unequal shares (no “jugaad”).
"""

import os
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastmcp import FastMCP

from splitwise import Splitwise
from splitwise.expense import Expense
from splitwise.user import ExpenseUser

load_dotenv()
mcp = FastMCP("Splitwise MCP")


# =============================================================================
# Splitwise client
# =============================================================================

def _client() -> Splitwise:
    """
    Build a Splitwise client from environment variables.

    Required:
      - SPLITWISE_CONSUMER_KEY
      - SPLITWISE_CONSUMER_SECRET

    Optional (personal single-user):
      - SPLITWISE_API_KEY
      - SPLITWISE_OAUTH_TOKEN + SPLITWISE_OAUTH_TOKEN_SECRET
    """
    consumer_key = os.environ["SPLITWISE_CONSUMER_KEY"]
    consumer_secret = os.environ["SPLITWISE_CONSUMER_SECRET"]

    api_key = os.getenv("SPLITWISE_API_KEY")
    s = Splitwise(consumer_key, consumer_secret, api_key=api_key) if api_key else Splitwise(consumer_key, consumer_secret)

    oauth_token = os.getenv("SPLITWISE_OAUTH_TOKEN")
    oauth_token_secret = os.getenv("SPLITWISE_OAUTH_TOKEN_SECRET")
    if oauth_token and oauth_token_secret:
        s.setAccessToken({"oauth_token": oauth_token, "oauth_token_secret": oauth_token_secret})

    return s


# =============================================================================
# Name normalization + ID resolution helpers
# =============================================================================

def _norm(s: str) -> str:
    """Normalize a name for matching (lowercase, collapse spaces)."""
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
    """
    Match by first name or full name.
    Example: "honey" matches first_name="Honey"
             "honey patel" matches full name.
    """
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
    """Find a group by name (case-insensitive)."""
    target = _norm(group_name)
    if not target:
        return None

    for g in groups:
        if _norm(getattr(g, "getName", lambda: "")() or "") == target:
            return g

    return None


async def _get_me_friends_groups(s: Splitwise) -> Tuple[Any, List[Any], List[Any]]:
    """Fetch current user, friends, and groups."""
    me = await asyncio.to_thread(s.getCurrentUser)
    friends = await asyncio.to_thread(s.getFriends)
    groups = await asyncio.to_thread(s.getGroups)
    return me, friends, groups


async def _get_group_members(s: Splitwise, group_id: int) -> List[Any]:
    """Fetch group details and return members list."""
    g = await asyncio.to_thread(s.getGroup, int(group_id))
    return getattr(g, "getMembers", lambda: [])() or []


# =============================================================================
# Internal create-expense builders (NOT exposed as MCP tools)
# =============================================================================

async def _create_expense_equal_split(
    s: Splitwise,
    description: str,
    cost: float,
    payer_id: int,
    participant_ids: List[int],
    group_id: Optional[int],
    currency_code: Optional[str],
) -> Dict[str, Any]:
    """Create equal split expense (internal)."""
    if cost <= 0:
        raise ValueError("cost must be > 0")
    if not participant_ids:
        raise ValueError("participant_ids must not be empty")
    if payer_id not in participant_ids:
        raise ValueError("payer_id must be included in participant_ids")

    expense = Expense()
    expense.setDescription(description)
    expense.setCost(f"{cost:.2f}")
    if group_id is not None:
        expense.setGroupId(int(group_id))
    if currency_code:
        expense.setCurrencyCode(currency_code)

    owed_each = cost / len(participant_ids)

    users: List[ExpenseUser] = []
    for uid in participant_ids:
        eu = ExpenseUser()
        eu.setId(int(uid))
        eu.setOwedShare(f"{owed_each:.2f}")
        eu.setPaidShare(f"{cost:.2f}" if int(uid) == int(payer_id) else "0.00")
        users.append(eu)

    expense.setUsers(users)

    created, errors = await asyncio.to_thread(s.createExpense, expense)
    if errors:
        return {"ok": False, "errors": errors}
    return {"ok": True, "expense_id": created.getId(), "split_type_used": "equal"}


async def _create_expense_no_split(
    s: Splitwise,
    description: str,
    cost: float,
    payer_id: int,
    ower_id: int,
    group_id: Optional[int],
    currency_code: Optional[str],
) -> Dict[str, Any]:
    """Create a 2-person expense where one person owes 100% (internal)."""
    if cost <= 0:
        raise ValueError("cost must be > 0")
    if int(payer_id) == int(ower_id):
        raise ValueError("payer_id and ower_id must be different")

    expense = Expense()
    expense.setDescription(description)
    expense.setCost(f"{cost:.2f}")
    if group_id is not None:
        expense.setGroupId(int(group_id))
    if currency_code:
        expense.setCurrencyCode(currency_code)

    payer = ExpenseUser()
    payer.setId(int(payer_id))
    payer.setPaidShare(f"{cost:.2f}")
    payer.setOwedShare("0.00")

    ower = ExpenseUser()
    ower.setId(int(ower_id))
    ower.setPaidShare("0.00")
    ower.setOwedShare(f"{cost:.2f}")

    expense.setUsers([payer, ower])

    created, errors = await asyncio.to_thread(s.createExpense, expense)
    if errors:
        return {"ok": False, "errors": errors}
    return {"ok": True, "expense_id": created.getId(), "split_type_used": "no_split"}


async def _create_expense_unequal_split_by_ids(
    s: Splitwise,
    description: str,
    cost: float,
    users: List[Dict[str, Any]],
    group_id: Optional[int],
    currency_code: Optional[str],
) -> Dict[str, Any]:
    """
    Create an expense with unequal split (internal).
    users = [{"id": 1, "paid_share": 10, "owed_share": 2}, ...]
    """
    if cost <= 0:
        raise ValueError("cost must be > 0")
    if not users:
        raise ValueError("users must not be empty")

    paid_total = sum(float(u.get("paid_share", 0.0)) for u in users)
    owed_total = sum(float(u.get("owed_share", 0.0)) for u in users)

    # Tolerance for rounding issues
    if abs(paid_total - cost) > 0.01 or abs(owed_total - cost) > 0.01:
        return {
            "ok": False,
            "errors": [
                f"Invalid shares: paid_total={paid_total:.2f}, owed_total={owed_total:.2f}, expected cost={cost:.2f}"
            ],
        }

    expense = Expense()
    expense.setDescription(description)
    expense.setCost(f"{cost:.2f}")
    if group_id is not None:
        expense.setGroupId(int(group_id))
    if currency_code:
        expense.setCurrencyCode(currency_code)

    eu_list: List[ExpenseUser] = []
    for u in users:
        if "id" not in u:
            raise ValueError("Each users entry must include 'id'")
        uid = int(u["id"])
        paid = float(u.get("paid_share", 0.0))
        owed = float(u.get("owed_share", 0.0))

        eu = ExpenseUser()
        eu.setId(uid)
        eu.setPaidShare(f"{paid:.2f}")
        eu.setOwedShare(f"{owed:.2f}")
        eu_list.append(eu)

    expense.setUsers(eu_list)

    created, errors = await asyncio.to_thread(s.createExpense, expense)
    if errors:
        return {"ok": False, "errors": errors}
    return {"ok": True, "expense_id": created.getId(), "split_type_used": "unequal"}


# =============================================================================
# READ TOOLS
# =============================================================================

@mcp.tool()
async def splitwise_current_user() -> Dict[str, Any]:
    """Return the authenticated user's profile."""
    s = _client()
    u = await asyncio.to_thread(s.getCurrentUser)
    return _user_to_dict(u)


@mcp.tool()
async def splitwise_friends() -> List[Dict[str, Any]]:
    """List friends with balances."""
    s = _client()
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
    """List groups."""
    s = _client()
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
    """Fetch expenses (lightweight projection)."""
    s = _client()

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
# WRITE TOOLS (update/delete/comment)
# =============================================================================

@mcp.tool()
async def splitwise_update_expense(
    expense_id: int,
    description: Optional[str] = None,
    cost: Optional[float] = None,
) -> Dict[str, Any]:
    """Update an existing expense (description and/or cost)."""
    s = _client()

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
    """Delete an expense."""
    s = _client()
    success, errors = await asyncio.to_thread(s.deleteExpense, int(expense_id))
    return {"ok": bool(success), "errors": errors}


@mcp.tool()
async def splitwise_add_comment(expense_id: int, content: str) -> Dict[str, Any]:
    """Add a comment to an expense."""
    s = _client()
    comment, errors = await asyncio.to_thread(s.createComment, int(expense_id), content)
    if errors:
        return {"ok": False, "errors": errors}
    return {"ok": True, "comment_id": comment.getId(), "content": comment.getContent()}


# =============================================================================
# ROUTER CREATE TOOL (the only "create expense" tool exposed)
# =============================================================================

@mcp.tool()
async def splitwise_create_expense(
    split_type: str,
    description: str,
    cost: float,
    currency_code: Optional[str] = None,

    # Group resolution (provide either)
    group_id: Optional[int] = None,
    group_name: Optional[str] = None,

    # EQUAL split inputs (names)
    paid_by: Optional[str] = None,                 # e.g. "me" or "Rutik"
    participants: Optional[List[str]] = None,      # e.g. ["me", "Rutik", "Honey"]

    # NO SPLIT inputs (names)
    counterparty: Optional[str] = None,            # e.g. "Honey"
    direction: Optional[str] = None,               # "they_owe_me" or "i_owe_them"

    # UNEQUAL split inputs (explicit shares by name)
    # Example:
    # shares = [{"name":"me","paid_share":100,"owed_share":35},{"name":"Hisaab","paid_share":0,"owed_share":65}]
    shares: Optional[List[Dict[str, Any]]] = None,

    # UNEQUAL split by percentages (simple & common):
    # Example: I paid total 100, my share 35%, Hisaab share 65%
    # percent_split = [{"name":"me","owed_percent":35},{"name":"Hisaab","owed_percent":65}]
    percent_split: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Create a Splitwise expense in ONE entry (no workaround).

    split_type:
      - "equal"     : equal split among participants
      - "no_split"  : one person owes 100% (direction + counterparty)
      - "unequal"   : custom split
          - either provide `shares` with paid_share/owed_share
          - OR provide `percent_split` with owed_percent values summing to 100

    Names:
      Use "me" to refer to your current Splitwise user.

    Group:
      Provide group_id OR group_name. If provided, member name resolution uses group members first.
    """
    if float(cost) <= 0:
        raise ValueError("cost must be > 0")

    split_type_n = _norm(split_type)

    s = _client()
    me, friends, groups = await _get_me_friends_groups(s)

    my_id = getattr(me, "getId", lambda: None)()
    if not my_id:
        raise ValueError("Could not determine current user id.")

    # Resolve group_id from group_name if needed
    resolved_group_id = group_id
    if resolved_group_id is None and group_name:
        g = _find_group_by_name(groups, group_name)
        if not g:
            available = [getattr(x, "getName", lambda: "")() for x in groups]
            return {"ok": False, "errors": [f"Group not found: {group_name}", "Available groups: " + ", ".join(available)]}
        resolved_group_id = getattr(g, "getId", lambda: None)()

    # Choose the best list for name -> id resolution
    members = friends
    if resolved_group_id is not None:
        members = await _get_group_members(s, int(resolved_group_id))

    # Resolve a name to an id, supporting "me"
    def resolve_id(name: str) -> Optional[int]:
        nn = _norm(name)
        if nn in ("me", "myself", "i"):
            return int(my_id)
        # Prefer group members; fallback to friends
        return _find_user_id_by_name(members, name) or _find_user_id_by_name(friends, name)

    # -----------------------
    # NO SPLIT (100% owed)
    # -----------------------
    if split_type_n in ("no_split", "nosplit", "full"):
        if not counterparty:
            raise ValueError("counterparty is required for split_type='no_split'")
        if direction not in ("they_owe_me", "i_owe_them"):
            raise ValueError("direction must be 'they_owe_me' or 'i_owe_them'")

        other_id = resolve_id(counterparty)
        if not other_id:
            return {"ok": False, "errors": [f"Could not find user by name: {counterparty}"]}

        if direction == "they_owe_me":
            payer_id = int(my_id)
            ower_id = int(other_id)
        else:
            payer_id = int(other_id)
            ower_id = int(my_id)

        return await _create_expense_no_split(
            s=s,
            description=description,
            cost=float(cost),
            payer_id=payer_id,
            ower_id=ower_id,
            group_id=resolved_group_id,
            currency_code=currency_code,
        )

    # -----------------------
    # EQUAL SPLIT
    # -----------------------
    if split_type_n in ("equal", "equal_split", "split_equally"):
        if not paid_by:
            raise ValueError("paid_by is required for split_type='equal'")
        if not participants:
            raise ValueError("participants is required for split_type='equal'")

        payer_id = resolve_id(paid_by)
        if not payer_id:
            return {"ok": False, "errors": [f"Could not resolve paid_by name: {paid_by}"]}

        participant_ids: List[int] = []
        unresolved: List[str] = []
        for p in participants:
            pid = resolve_id(p)
            if pid is None:
                unresolved.append(p)
            else:
                participant_ids.append(int(pid))

        if unresolved:
            # helpful: list known group members
            if resolved_group_id is not None:
                names = [
                    f"{getattr(m, 'getFirstName', lambda: '')() or ''} {getattr(m, 'getLastName', lambda: '')() or ''}".strip()
                    for m in members
                ]
                return {
                    "ok": False,
                    "errors": [
                        "Could not resolve these participant names: " + ", ".join(unresolved),
                        "Group members I can see: " + ", ".join([n for n in names if n]),
                    ],
                }
            return {"ok": False, "errors": ["Could not resolve these participant names: " + ", ".join(unresolved)]}

        # Ensure payer included
        if int(payer_id) not in participant_ids:
            participant_ids.append(int(payer_id))

        return await _create_expense_equal_split(
            s=s,
            description=description,
            cost=float(cost),
            payer_id=int(payer_id),
            participant_ids=participant_ids,
            group_id=resolved_group_id,
            currency_code=currency_code,
        )

    # -----------------------
    # UNEQUAL SPLIT
    # -----------------------
    if split_type_n in ("unequal", "unequal_split", "custom"):
        # Case A: percent-based (common: "my share is 35%")
        if percent_split and len(percent_split) > 0:
            if not paid_by:
                raise ValueError("paid_by is required for percent_split (who paid the full amount)")

            payer_id = resolve_id(paid_by)
            if not payer_id:
                return {"ok": False, "errors": [f"Could not resolve paid_by name: {paid_by}"]}

            # Resolve all users + validate percent totals
            resolved: List[Tuple[int, float]] = []
            unresolved: List[str] = []
            pct_total = 0.0

            for item in percent_split:
                nm = item.get("name")
                if not nm:
                    raise ValueError("Each percent_split entry must include 'name'")
                pct = float(item.get("owed_percent", item.get("percent", 0.0)))
                uid = resolve_id(str(nm))
                if uid is None:
                    unresolved.append(str(nm))
                else:
                    resolved.append((int(uid), pct))
                    pct_total += pct

            if unresolved:
                if resolved_group_id is not None:
                    names = [
                        f"{getattr(m, 'getFirstName', lambda: '')() or ''} {getattr(m, 'getLastName', lambda: '')() or ''}".strip()
                        for m in members
                    ]
                    return {
                        "ok": False,
                        "errors": [
                            "Could not resolve these names: " + ", ".join(unresolved),
                            "Group members I can see: " + ", ".join([n for n in names if n]),
                        ],
                    }
                return {"ok": False, "errors": ["Could not resolve these names: " + ", ".join(unresolved)]}

            if abs(pct_total - 100.0) > 0.01:
                return {"ok": False, "errors": [f"percent_split must sum to 100. Got {pct_total:.2f}."]}

            # Build unequal shares payload:
            # - payer pays 100% (paid_share = cost)
            # - others pay 0
            # - owed_share determined by percent
            users_payload: List[Dict[str, Any]] = []
            for uid, pct in resolved:
                owed = float(cost) * (pct / 100.0)
                paid = float(cost) if int(uid) == int(payer_id) else 0.0
                users_payload.append({"id": int(uid), "paid_share": paid, "owed_share": owed})

            # Fix rounding drift by adjusting the last owed_share so totals match exactly (within cents)
            owed_total = sum(u["owed_share"] for u in users_payload)
            drift = float(cost) - owed_total
            if abs(drift) >= 0.005:
                users_payload[-1]["owed_share"] += drift

            return await _create_expense_unequal_split_by_ids(
                s=s,
                description=description,
                cost=float(cost),
                users=users_payload,
                group_id=resolved_group_id,
                currency_code=currency_code,
            )

        # Case B: explicit shares provided
        if shares and len(shares) > 0:
            users_payload: List[Dict[str, Any]] = []
            unresolved: List[str] = []

            for sh in shares:
                nm = sh.get("name")
                if not nm:
                    raise ValueError("Each shares entry must include 'name'")
                uid = resolve_id(str(nm))
                if uid is None:
                    unresolved.append(str(nm))
                    continue

                users_payload.append(
                    {
                        "id": int(uid),
                        "paid_share": float(sh.get("paid_share", sh.get("paidShare", 0.0))),
                        "owed_share": float(sh.get("owed_share", sh.get("owedShare", 0.0))),
                    }
                )

            if unresolved:
                if resolved_group_id is not None:
                    names = [
                        f"{getattr(m, 'getFirstName', lambda: '')() or ''} {getattr(m, 'getLastName', lambda: '')() or ''}".strip()
                        for m in members
                    ]
                    return {
                        "ok": False,
                        "errors": [
                            "Could not resolve these names: " + ", ".join(unresolved),
                            "Group members I can see: " + ", ".join([n for n in names if n]),
                        ],
                    }
                return {"ok": False, "errors": ["Could not resolve these names: " + ", ".join(unresolved)]}

            return await _create_expense_unequal_split_by_ids(
                s=s,
                description=description,
                cost=float(cost),
                users=users_payload,
                group_id=resolved_group_id,
                currency_code=currency_code,
            )

        raise ValueError("For split_type='unequal', provide either 'percent_split' or 'shares'.")

    raise ValueError("split_type must be one of: 'equal', 'no_split', 'unequal'")


# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
