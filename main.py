import os
import json
import asyncio
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from splitwise import Splitwise
from splitwise.expense import Expense
from splitwise.user import ExpenseUser


# -------------------------
# Security: personal-use token gate
# -------------------------
MCP_SERVER_TOKEN = os.getenv("MCP_SERVER_TOKEN")
if not MCP_SERVER_TOKEN:
    raise RuntimeError("Missing MCP_SERVER_TOKEN. Set a strong secret token in your environment variables.")

mcp = FastMCP("Splitwise Personal MCP")


def _require_token(token: str) -> None:
    """
    Simple personal-use protection.
    Every tool must receive token == MCP_SERVER_TOKEN.
    """
    if token != MCP_SERVER_TOKEN:
        raise PermissionError("Unauthorized: invalid token.")


# -------------------------
# Splitwise client
# -------------------------
SPLITWISE_CONSUMER_KEY = os.getenv("SPLITWISE_CONSUMER_KEY")
SPLITWISE_CONSUMER_SECRET = os.getenv("SPLITWISE_CONSUMER_SECRET")

if not SPLITWISE_CONSUMER_KEY or not SPLITWISE_CONSUMER_SECRET:
    raise RuntimeError("Missing SPLITWISE_CONSUMER_KEY / SPLITWISE_CONSUMER_SECRET env vars.")

SPLITWISE_API_KEY = os.getenv("SPLITWISE_API_KEY")
SPLITWISE_OAUTH_TOKEN_JSON = os.getenv("SPLITWISE_OAUTH_TOKEN_JSON")  # JSON dict if you prefer OAuth token storage


def _client() -> Splitwise:
    """
    Personal-use Splitwise client.
    Prefer API Key (simplest). Otherwise, use OAuth access token JSON if provided.
    """
    if SPLITWISE_API_KEY:
        return Splitwise(SPLITWISE_CONSUMER_KEY, SPLITWISE_CONSUMER_SECRET, api_key=SPLITWISE_API_KEY)

    s = Splitwise(SPLITWISE_CONSUMER_KEY, SPLITWISE_CONSUMER_SECRET)
    if SPLITWISE_OAUTH_TOKEN_JSON:
        try:
            token_dict = json.loads(SPLITWISE_OAUTH_TOKEN_JSON)
        except json.JSONDecodeError as e:
            raise RuntimeError("SPLITWISE_OAUTH_TOKEN_JSON must be valid JSON.") from e
        s.setAccessToken(token_dict)

    return s


def _balance_to_dict(b: Any) -> Dict[str, Any]:
    return {
        "currency": getattr(b, "getCurrencyCode", lambda: None)(),
        "amount": getattr(b, "getAmount", lambda: None)(),
    }


def _expense_to_dict(e: Any) -> Dict[str, Any]:
    return {
        "id": e.getId(),
        "group_id": e.getGroupId(),
        "description": e.getDescription(),
        "cost": e.getCost(),
        "currency_code": e.getCurrencyCode(),
        "date": e.getDate(),
        "details": getattr(e, "getDetails", lambda: None)(),
        "created_at": getattr(e, "getCreatedAt", lambda: None)(),
        "updated_at": getattr(e, "getUpdatedAt", lambda: None)(),
    }


# -------------------------
# Tools (all async)
# -------------------------

@mcp.tool()
async def splitwise_current_user(token: str) -> Dict[str, Any]:
    """Return the authenticated Splitwise user's profile (personal account)."""
    _require_token(token)
    s = _client()
    u = await asyncio.to_thread(s.getCurrentUser)
    return {
        "id": u.getId(),
        "first_name": u.getFirstName(),
        "last_name": u.getLastName(),
        "email": getattr(u, "getEmail", lambda: None)(),
        "default_currency": getattr(u, "getDefaultCurrency", lambda: None)(),
        "locale": getattr(u, "getLocale", lambda: None)(),
    }


@mcp.tool()
async def splitwise_friends(token: str) -> List[Dict[str, Any]]:
    """List friends of the current user (personal account)."""
    _require_token(token)
    s = _client()
    friends = await asyncio.to_thread(s.getFriends)

    out: List[Dict[str, Any]] = []
    for f in friends:
        balances = []
        for b in (getattr(f, "getBalances", lambda: [])() or []):
            balances.append(_balance_to_dict(b))

        out.append(
            {
                "id": f.getId(),
                "first_name": f.getFirstName(),
                "last_name": f.getLastName(),
                "email": getattr(f, "getEmail", lambda: None)(),
                "balances": balances,
            }
        )
    return out


@mcp.tool()
async def splitwise_groups(token: str) -> List[Dict[str, Any]]:
    """List groups for the current user."""
    _require_token(token)
    s = _client()
    groups = await asyncio.to_thread(s.getGroups)
    return [
        {
            "id": g.getId(),
            "name": g.getName(),
            "simplified_by_default": getattr(g, "isSimplifiedByDefault", lambda: None)(),
            "invite_link": getattr(g, "getInviteLink", lambda: None)(),
        }
        for g in groups
    ]


@mcp.tool()
async def splitwise_expenses(
    token: str,
    limit: int = 20,
    offset: int = 0,
    group_id: Optional[int] = None,
    friend_id: Optional[int] = None,
    dated_after: Optional[str] = None,   # YYYY-MM-DD
    dated_before: Optional[str] = None,  # YYYY-MM-DD
) -> List[Dict[str, Any]]:
    """
    Fetch expenses with optional filters.
    """
    _require_token(token)
    s = _client()

    def _fetch():
        return s.getExpenses(
            offset=offset,
            limit=limit,
            group_id=group_id,
            friend_id=friend_id,
            dated_after=dated_after,
            dated_before=dated_before,
        )

    expenses = await asyncio.to_thread(_fetch)
    return [_expense_to_dict(e) for e in expenses]


@mcp.tool()
async def splitwise_create_expense_equal_split(
    token: str,
    description: str,
    cost: float,
    payer_id: int,
    participant_ids: List[int],
    group_id: Optional[int] = None,
    currency_code: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create an expense where participants split equally; payer pays 100% (by default).
    """
    _require_token(token)
    if cost <= 0:
        raise ValueError("cost must be > 0")
    if not participant_ids:
        raise ValueError("participant_ids must not be empty")
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
    token: str,
    description: str,
    cost: float,
    users: List[Dict[str, Any]],
    group_id: Optional[int] = None,
    currency_code: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create an expense with an unequal split.
    Provide `users` list as:
    [
      {"id": 123, "paid_share": 10.0, "owed_share": 2.5},
      {"id": 456, "paid_share": 0.0,  "owed_share": 7.5}
    ]

    Must satisfy:
    - sum(paid_share) == cost
    - sum(owed_share) == cost
    """
    _require_token(token)
    if cost <= 0:
        raise ValueError("cost must be > 0")
    if not users:
        raise ValueError("users must not be empty")

    # validate totals (tolerant)
    paid_total = sum(float(u.get("paid_share", 0)) for u in users)
    owed_total = sum(float(u.get("owed_share", 0)) for u in users)

    def _close(a: float, b: float, eps: float = 0.01) -> bool:
        return abs(a - b) <= eps

    if not _close(paid_total, cost) or not _close(owed_total, cost):
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
        uid = u.get("id")
        if uid is None:
            raise ValueError("Each user must include an 'id'")
        paid = float(u.get("paid_share", 0))
        owed = float(u.get("owed_share", 0))

        eu = ExpenseUser()
        eu.setId(int(uid))
        eu.setPaidShare(f"{paid:.2f}")
        eu.setOwedShare(f"{owed:.2f}")
        eu_list.append(eu)

    expense.setUsers(eu_list)

    created, errors = await asyncio.to_thread(s.createExpense, expense)
    if errors:
        return {"ok": False, "errors": errors}
    return {"ok": True, "expense_id": created.getId()}


@mcp.tool()
async def splitwise_update_expense(
    token: str,
    expense_id: int,
    description: Optional[str] = None,
    cost: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Update an expense (description and/or cost).
    """
    _require_token(token)
    if description is None and cost is None:
        raise ValueError("Provide at least one field to update (description or cost).")

    s = _client()

    expense = Expense()
    expense.id = int(expense_id)
    if description is not None:
        expense.setDescription(description)
    if cost is not None:
        if cost <= 0:
            raise ValueError("cost must be > 0")
        expense.setCost(f"{cost:.2f}")

    updated, errors = await asyncio.to_thread(s.updateExpense, expense)
    if errors:
        return {"ok": False, "errors": errors}

    return {"ok": True, "expense_id": updated.getId()}


@mcp.tool()
async def splitwise_delete_expense(token: str, expense_id: int) -> Dict[str, Any]:
    """Delete an expense."""
    _require_token(token)
    s = _client()
    success, errors = await asyncio.to_thread(s.deleteExpense, int(expense_id))
    return {"ok": bool(success), "errors": errors}


@mcp.tool()
async def splitwise_add_comment(token: str, expense_id: int, content: str) -> Dict[str, Any]:
    """Add a comment to an expense."""
    _require_token(token)
    if not content.strip():
        raise ValueError("content must not be empty")

    s = _client()
    comment, errors = await asyncio.to_thread(s.createComment, int(expense_id), content)
    if errors:
        return {"ok": False, "errors": errors}

    return {"ok": True, "comment_id": comment.getId(), "content": comment.getContent()}


if __name__ == "__main__":
    mcp.run()
