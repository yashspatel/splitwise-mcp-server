"""
main.py — Splitwise MCP Server (Async, with a Create-Expense Router)

Goal
----
Make Splitwise actions available as MCP tools, and make "create expense" reliable:
- Users can say: "Honey owes me 10 CAD" -> NO SPLIT (100% owed by Honey)
- Users can say: "Add 174 CAD in Trip group, I paid, split equally with Rutik" -> EQUAL SPLIT
- Users can specify UNEQUAL SPLITS with explicit shares

Why a router?
-------------
LLM clients sometimes pick the wrong tool (e.g., always calling equal split).
So we provide ONE preferred tool: splitwise_create_expense_router(...)
It resolves:
- group by name
- members by name
- current user id ("me")
- and routes internally to the correct create method
"""

import os  # Environment variables
import asyncio  # Run blocking SDK calls in threads
from typing import Any, Dict, List, Optional, Tuple  # Type hints

from dotenv import load_dotenv  # Local dev env loading
from fastmcp import FastMCP  # MCP server framework

from splitwise import Splitwise  # Splitwise SDK client
from splitwise.expense import Expense  # Splitwise Expense model
from splitwise.user import ExpenseUser  # Splitwise ExpenseUser model

# Load local .env if present (safe in cloud; does nothing if missing)
load_dotenv()

# Create MCP server
mcp = FastMCP("Splitwise MCP")


# =============================================================================
# Splitwise client construction
# =============================================================================

def _client() -> Splitwise:
    """
    Create and return a configured Splitwise client.
    Auth strategy:
      - Prefer API key mode (single-user automation)
      - Otherwise use OAuth1 token+secret if provided
    """
    consumer_key = os.environ["SPLITWISE_CONSUMER_KEY"]  # Splitwise app key
    consumer_secret = os.environ["SPLITWISE_CONSUMER_SECRET"]  # Splitwise app secret

    api_key = os.getenv("SPLITWISE_API_KEY")  # Optional: API key mode
    s = (
        Splitwise(consumer_key, consumer_secret, api_key=api_key)
        if api_key
        else Splitwise(consumer_key, consumer_secret)
    )

    oauth_token = os.getenv("SPLITWISE_OAUTH_TOKEN")  # Optional OAuth token
    oauth_token_secret = os.getenv("SPLITWISE_OAUTH_TOKEN_SECRET")  # Optional OAuth secret

    # If OAuth creds exist, attach them so requests are performed as that user
    if oauth_token and oauth_token_secret:
        s.setAccessToken({"oauth_token": oauth_token, "oauth_token_secret": oauth_token_secret})

    return s


# =============================================================================
# Simple normalization + lookup helpers (name -> id)
# =============================================================================

def _norm(s: str) -> str:
    """Normalize strings for robust comparisons (case/extra spaces)."""
    return " ".join((s or "").strip().lower().split())


def _full_name(first: str, last: str) -> str:
    """Build 'first last' normalized name string."""
    return _norm(f"{first} {last}".strip())


def _user_to_dict(u: Any) -> Dict[str, Any]:
    """Convert a Splitwise user-like object to a JSON-friendly dict."""
    return {
        "id": getattr(u, "getId", lambda: None)(),
        "first_name": getattr(u, "getFirstName", lambda: None)(),
        "last_name": getattr(u, "getLastName", lambda: None)(),
        "email": getattr(u, "getEmail", lambda: None)(),
    }


def _find_user_id_by_name(users: List[Any], name: str) -> Optional[int]:
    """
    Find a user id by matching first name OR full name.
    Works for friends list or group members list.

    Examples:
      name="honey" matches first_name "Honey"
      name="honey patel" matches full name "Honey Patel"
    """
    target = _norm(name)
    if not target:
        return None

    for u in users:
        first = _norm(getattr(u, "getFirstName", lambda: "")() or "")
        last = _norm(getattr(u, "getLastName", lambda: "")() or "")
        full = _full_name(first, last)

        if target == first or target == full:
            return getattr(u, "getId", lambda: None)()

    return None


def _find_group_by_name(groups: List[Any], group_name: str) -> Optional[Any]:
    """
    Find a Splitwise Group object by name (case-insensitive).
    """
    target = _norm(group_name)
    if not target:
        return None

    for g in groups:
        name = _norm(getattr(g, "getName", lambda: "")() or "")
        if name == target:
            return g

    return None


async def _get_me_friends_groups(s: Splitwise) -> Tuple[Any, List[Any], List[Any]]:
    """
    Fetch current user + friends + groups (async wrapper).
    """
    me = await asyncio.to_thread(s.getCurrentUser)
    friends = await asyncio.to_thread(s.getFriends)
    groups = await asyncio.to_thread(s.getGroups)
    return me, friends, groups


async def _get_group_members(s: Splitwise, group_id: int) -> List[Any]:
    """
    Fetch group details (including members) and return member list.
    """
    group = await asyncio.to_thread(s.getGroup, group_id)
    members = getattr(group, "getMembers", lambda: [])() or []
    return members


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
    return [
        {
            "id": g.getId(),
            "name": g.getName(),
            "simplified_by_default": getattr(g, "isSimplifiedByDefault", lambda: None)(),
        }
        for g in groups
    ]


@mcp.tool()
async def splitwise_expenses(
    limit: int = 20,
    offset: int = 0,
    group_id: Optional[int] = None,
    dated_after: Optional[str] = None,   # "YYYY-MM-DD"
    dated_before: Optional[str] = None,  # "YYYY-MM-DD"
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

    out: List[Dict[str, Any]] = []
    for e in expenses:
        out.append(
            {
                "id": e.getId(),
                "group_id": e.getGroupId(),
                "description": e.getDescription(),
                "cost": e.getCost(),
                "currency_code": e.getCurrencyCode(),
                "date": e.getDate(),
            }
        )
    return out


# =============================================================================
# CORE WRITE TOOLS (still exposed, but router is preferred)
# =============================================================================

@mcp.tool()
async def splitwise_create_expense_equal_split(
    description: str,
    cost: float,
    payer_id: int,
    participant_ids: List[int],
    group_id: Optional[int] = None,
    currency_code: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create an expense split equally.
    Note: Prefer splitwise_create_expense_router for natural-language flows.
    """
    if not participant_ids:
        raise ValueError("participant_ids must not be empty")
    if cost <= 0:
        raise ValueError("cost must be > 0")
    if payer_id not in participant_ids:
        raise ValueError("payer_id must be included in participant_ids")

    s = _client()

    expense = Expense()
    expense.setDescription(description)
    expense.setCost(f"{cost:.2f}")
    if group_id is not None:
        expense.setGroupId(group_id)
    if currency_code:
        expense.setCurrencyCode(currency_code)

    owed_each = cost / len(participant_ids)

    users: List[ExpenseUser] = []
    for uid in participant_ids:
        u = ExpenseUser()
        u.setId(uid)
        u.setOwedShare(f"{owed_each:.2f}")
        u.setPaidShare(f"{cost:.2f}" if uid == payer_id else "0.00")
        users.append(u)

    expense.setUsers(users)

    created, errors = await asyncio.to_thread(s.createExpense, expense)
    if errors:
        return {"ok": False, "errors": errors}

    return {"ok": True, "expense_id": created.getId()}


@mcp.tool()
async def splitwise_create_expense_unequal_split(
    description: str,
    cost: float,
    users: List[Dict[str, Any]],
    group_id: Optional[int] = None,
    currency_code: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create an expense with an unequal split.
    users = [{"id": 1, "paid_share": 10, "owed_share": 2}, ...]
    """
    if cost <= 0:
        raise ValueError("cost must be > 0")
    if not users:
        raise ValueError("users must not be empty")

    paid_total = sum(float(u.get("paid_share", u.get("paidShare", 0.0))) for u in users)
    owed_total = sum(float(u.get("owed_share", u.get("owedShare", 0.0))) for u in users)

    if abs(paid_total - cost) > 0.01 or abs(owed_total - cost) > 0.01:
        return {
            "ok": False,
            "errors": [
                f"Invalid shares: sum(paid_share)={paid_total:.2f}, sum(owed_share)={owed_total:.2f}, expected cost={cost:.2f}"
            ],
        }

    s = _client()

    expense = Expense()
    expense.setDescription(description)
    expense.setCost(f"{cost:.2f}")
    if group_id is not None:
        expense.setGroupId(group_id)
    if currency_code:
        expense.setCurrencyCode(currency_code)

    eu_list: List[ExpenseUser] = []
    for u in users:
        if "id" not in u:
            raise ValueError("Each user entry must include 'id'")
        uid = int(u["id"])
        paid = float(u.get("paid_share", u.get("paidShare", 0.0)))
        owed = float(u.get("owed_share", u.get("owedShare", 0.0)))

        eu = ExpenseUser()
        eu.setId(uid)
        eu.setPaidShare(f"{paid:.2f}")
        eu.setOwedShare(f"{owed:.2f}")
        eu_list.append(eu)

    expense.setUsers(eu_list)

    created, errors = await asyncio.to_thread(s.createExpense, expense)
    if errors:
        return {"ok": False, "errors": errors}

    return {"ok": True, "expense_id": created.getId()}


@mcp.tool()
async def splitwise_create_expense_no_split(
    description: str,
    cost: float,
    payer_id: int,
    ower_id: int,
    group_id: Optional[int] = None,
    currency_code: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a 2-person expense where one person owes 100% (no split).
    """
    if cost <= 0:
        raise ValueError("cost must be > 0")
    if payer_id == ower_id:
        raise ValueError("payer_id and ower_id must be different")

    s = _client()

    expense = Expense()
    expense.setDescription(description)
    expense.setCost(f"{cost:.2f}")
    if group_id is not None:
        expense.setGroupId(group_id)
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

    return {"ok": True, "expense_id": created.getId()}


@mcp.tool()
async def splitwise_update_expense(
    expense_id: int,
    description: Optional[str] = None,
    cost: Optional[float] = None,
) -> Dict[str, Any]:
    """Update an existing expense."""
    s = _client()

    e = Expense()
    e.id = int(expense_id)
    if description is not None:
        e.setDescription(description)
    if cost is not None:
        e.setCost(f"{cost:.2f}")

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
# ROUTER TOOL (preferred for "natural language" create expense)
# =============================================================================

@mcp.tool()
async def splitwise_create_expense_router(
    # High-level intent
    split_type: str,
    description: str,
    cost: float,

    # Optional group routing
    group_id: Optional[int] = None,
    group_name: Optional[str] = None,

    # Optional currency
    currency_code: Optional[str] = None,

    # EQUAL SPLIT inputs (name-based recommended)
    paid_by: Optional[str] = None,               # e.g. "me" or "Yash" or "Rutik"
    participants: Optional[List[str]] = None,    # e.g. ["me", "Rutik"]

    # NO SPLIT inputs
    counterparty: Optional[str] = None,          # e.g. "Honey"
    direction: Optional[str] = None,             # "they_owe_me" or "i_owe_them"

    # UNEQUAL SPLIT inputs (name-based)
    # Example:
    # shares = [
    #   {"name":"me", "paid_share": 10, "owed_share": 2},
    #   {"name":"Rutik", "paid_share": 0, "owed_share": 8},
    # ]
    shares: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    PREFERRED entrypoint for creating expenses. This is the "router".

    split_type:
      - "equal": split equally among participants
      - "no_split": one person owes 100% (direction + counterparty)
      - "unequal": custom shares (shares list)

    group:
      Provide either group_id OR group_name.
      If provided, this router will resolve group members by name.

    names:
      Use "me" to refer to the current user.
    """
    # Basic validation
    if cost <= 0:
        raise ValueError("cost must be > 0")

    split_type = _norm(split_type)

    # Create Splitwise client and preload common data for lookups
    s = _client()
    me, friends, groups = await _get_me_friends_groups(s)

    my_id = getattr(me, "getId", lambda: None)()
    if not my_id:
        raise ValueError("Could not determine current user id.")

    # Resolve group_id from group_name if needed
    resolved_group_id: Optional[int] = group_id
    if resolved_group_id is None and group_name:
        g = _find_group_by_name(groups, group_name)
        if not g:
            # Provide a helpful error with available group names
            available = [getattr(x, "getName", lambda: "")() for x in groups]
            return {"ok": False, "errors": [f"Group not found: {group_name}", "Available groups: " + ", ".join(available)]}
        resolved_group_id = getattr(g, "getId", lambda: None)()

    # Resolve members list (friends + (optional) group members)
    # If group provided, group members are the most accurate set for name resolution.
    members: List[Any] = friends
    if resolved_group_id is not None:
        members = await _get_group_members(s, int(resolved_group_id))

    # Small helper: map a name to an id, with special handling for "me"
    def resolve_id(name: str) -> Optional[int]:
        n = _norm(name)
        if n in ("me", "myself", "i"):
            return int(my_id)
        return _find_user_id_by_name(members, name) or _find_user_id_by_name(friends, name)

    # -------------------------------------------------------------------------
    # NO SPLIT: "Honey owes me full" or "I owe Honey full"
    # -------------------------------------------------------------------------
    if split_type in ("no_split", "nosplit", "full"):
        if not counterparty:
            raise ValueError("counterparty is required for split_type='no_split'")
        if direction not in ("they_owe_me", "i_owe_them"):
            raise ValueError("direction must be 'they_owe_me' or 'i_owe_them'")

        other_id = resolve_id(counterparty)
        if not other_id:
            return {"ok": False, "errors": [f"Could not find user by name: {counterparty}"]}

        # Determine payer/ower
        if direction == "they_owe_me":
            payer_id = int(my_id)
            ower_id = int(other_id)
        else:
            payer_id = int(other_id)
            ower_id = int(my_id)

        # Route to the existing no-split tool
        return await splitwise_create_expense_no_split(
            description=description,
            cost=cost,
            payer_id=payer_id,
            ower_id=ower_id,
            group_id=resolved_group_id,
            currency_code=currency_code,
        )

    # -------------------------------------------------------------------------
    # EQUAL SPLIT: "I paid, split equally in group with X and Y"
    # -------------------------------------------------------------------------
    if split_type in ("equal", "equal_split", "split_equally"):
        if not paid_by:
            raise ValueError("paid_by is required for split_type='equal'")
        if not participants or len(participants) == 0:
            raise ValueError("participants is required for split_type='equal'")

        payer_id = resolve_id(paid_by)
        if not payer_id:
            return {"ok": False, "errors": [f"Could not resolve payer name: {paid_by}"]}

        participant_ids: List[int] = []
        unresolved: List[str] = []

        for p in participants:
            pid = resolve_id(p)
            if pid:
            # ensure ints
                participant_ids.append(int(pid))
            else:
                unresolved.append(p)

        if unresolved:
            # Give user a helpful error and list of group members (if group provided)
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

        # Ensure payer is included (Splitwise tool expects that)
        if int(payer_id) not in participant_ids:
            participant_ids.append(int(payer_id))

        # Route to existing equal-split tool
        return await splitwise_create_expense_equal_split(
            description=description,
            cost=cost,
            payer_id=int(payer_id),
            participant_ids=participant_ids,
            group_id=resolved_group_id,
            currency_code=currency_code,
        )

    # -------------------------------------------------------------------------
    # UNEQUAL SPLIT: provide shares by name
    # -------------------------------------------------------------------------
    if split_type in ("unequal", "unequal_split", "custom"):
        if not shares or len(shares) == 0:
            raise ValueError("shares is required for split_type='unequal'")

        # Convert share entries from names -> ids
        users_payload: List[Dict[str, Any]] = []
        unresolved: List[str] = []

        for sh in shares:
            name = sh.get("name")
            if not name:
                raise ValueError("Each shares entry must include 'name'")
            uid = resolve_id(str(name))
            if not uid:
                unresolved.append(str(name))
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

        # Route to existing unequal tool
        return await splitwise_create_expense_unequal_split(
            description=description,
            cost=cost,
            users=users_payload,
            group_id=resolved_group_id,
            currency_code=currency_code,
        )

    # If we reach here, split_type didn't match
    raise ValueError("split_type must be one of: 'no_split', 'equal', 'unequal'")


# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    # HTTP transport is suitable for cloud hosting and remote clients
    mcp.run(transport="http", host="0.0.0.0", port=8000)
