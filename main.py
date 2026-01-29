import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastmcp import FastMCP

from splitwise import Splitwise
from splitwise.expense import Expense
from splitwise.user import ExpenseUser

load_dotenv()

mcp = FastMCP("Splitwise MCP")

def _client() -> Splitwise:
    """
    Auth strategy:
    - Prefer API key (single-user automation)
    - Fallback to OAuth1 token+secret if provided
    """
    consumer_key = os.environ["SPLITWISE_CONSUMER_KEY"]
    consumer_secret = os.environ["SPLITWISE_CONSUMER_SECRET"]

    api_key = os.getenv("SPLITWISE_API_KEY")
    s = Splitwise(consumer_key, consumer_secret, api_key=api_key) if api_key else Splitwise(consumer_key, consumer_secret)

    # OAuth1 (optional)
    oauth_token = os.getenv("SPLITWISE_OAUTH_TOKEN")
    oauth_token_secret = os.getenv("SPLITWISE_OAUTH_TOKEN_SECRET")
    if oauth_token and oauth_token_secret:
        # SDK expects dict with oauth_token + oauth_token_secret.
        s.setAccessToken({"oauth_token": oauth_token, "oauth_token_secret": oauth_token_secret})

    return s

def _user_to_dict(u: Any) -> Dict[str, Any]:
    # Works for User/Friend/CurrentUser-like objects
    d = {
        "id": getattr(u, "getId", lambda: None)(),
        "first_name": getattr(u, "getFirstName", lambda: None)(),
        "last_name": getattr(u, "getLastName", lambda: None)(),
        "email": getattr(u, "getEmail", lambda: None)(),
    }
    return d

@mcp.tool()
def splitwise_current_user() -> Dict[str, Any]:
    """Return the authenticated user's profile."""
    s = _client()
    u = s.getCurrentUser()
    return _user_to_dict(u)

@mcp.tool()
def splitwise_friends() -> List[Dict[str, Any]]:
    """List friends with balances (basic fields)."""
    s = _client()
    friends = s.getFriends()
    out: List[Dict[str, Any]] = []
    for f in friends:
        item = _user_to_dict(f)
        balances = []
        for b in (getattr(f, "getBalances", lambda: [])() or []):
            balances.append({
                "currency": getattr(b, "getCurrencyCode", lambda: None)(),
                "amount": getattr(b, "getAmount", lambda: None)(),
            })
        item["balances"] = balances
        out.append(item)
    return out

@mcp.tool()
def splitwise_groups() -> List[Dict[str, Any]]:
    """List groups."""
    s = _client()
    groups = s.getGroups()
    return [
        {
            "id": g.getId(),
            "name": g.getName(),
            "simplified_by_default": getattr(g, "isSimplifiedByDefault", lambda: None)(),
        }
        for g in groups
    ]

@mcp.tool()
def splitwise_expenses(
    limit: int = 20,
    offset: int = 0,
    group_id: Optional[int] = None,
    dated_after: Optional[str] = None,   # "YYYY-MM-DD"
    dated_before: Optional[str] = None,  # "YYYY-MM-DD"
) -> List[Dict[str, Any]]:
    """
    Fetch expenses (lightweight projection).
    """
    s = _client()
    expenses = s.getExpenses(
        offset=offset,
        limit=limit,
        group_id=group_id,
        dated_after=dated_after,
        dated_before=dated_before,
    )
    out: List[Dict[str, Any]] = []
    for e in expenses:
        out.append({
            "id": e.getId(),
            "group_id": e.getGroupId(),
            "description": e.getDescription(),
            "cost": e.getCost(),
            "currency_code": e.getCurrencyCode(),
            "date": e.getDate(),
        })
    return out

@mcp.tool()
def splitwise_create_expense_equal_split(
    description: str,
    cost: float,
    payer_id: int,
    participant_ids: List[int],
    group_id: Optional[int] = None,
    currency_code: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create an expense where participants split equally, payer pays 100%.
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

    created, errors = s.createExpense(expense)
    if errors:
        return {"ok": False, "errors": errors}

    return {"ok": True, "expense_id": created.getId()}

@mcp.tool()
def splitwise_create_expense_unequal_split(
    description: str,
    cost: float,
    users: List[Dict[str, Any]],
    group_id: Optional[int] = None,
    currency_code: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create an expense with an unequal split.

    `users` must be a list of dicts like:
      {"id": 123, "paid_share": 10.0, "owed_share": 2.5}

    Constraints:
    - sum(paid_share) == cost (within 0.01)
    - sum(owed_share) == cost (within 0.01)
    """
    if cost <= 0:
        raise ValueError("cost must be > 0")
    if not users:
        raise ValueError("users must not be empty")

    paid_total = 0.0
    owed_total = 0.0

    for u in users:
        if "id" not in u:
            raise ValueError("Each user entry must include 'id'")
        # allow either snake_case or camelCase just in case
        paid = float(u.get("paid_share", u.get("paidShare", 0.0)))
        owed = float(u.get("owed_share", u.get("owedShare", 0.0)))
        paid_total += paid
        owed_total += owed

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
        uid = int(u["id"])
        paid = float(u.get("paid_share", u.get("paidShare", 0.0)))
        owed = float(u.get("owed_share", u.get("owedShare", 0.0)))

        eu = ExpenseUser()
        eu.setId(uid)
        eu.setPaidShare(f"{paid:.2f}")
        eu.setOwedShare(f"{owed:.2f}")
        eu_list.append(eu)

    expense.setUsers(eu_list)

    created, errors = s.createExpense(expense)
    if errors:
        return {"ok": False, "errors": errors}

    return {"ok": True, "expense_id": created.getId()}

@mcp.tool()
def splitwise_update_expense(
    expense_id: int,
    description: Optional[str] = None,
    cost: Optional[float] = None,
) -> Dict[str, Any]:
    """Update an existing expense."""
    s = _client()

    e = Expense()
    e.id = expense_id
    if description is not None:
        e.setDescription(description)
    if cost is not None:
        e.setCost(f"{cost:.2f}")

    updated, errors = s.updateExpense(e)
    if errors:
        return {"ok": False, "errors": errors}
    return {"ok": True, "expense_id": updated.getId()}

@mcp.tool()
def splitwise_delete_expense(expense_id: int) -> Dict[str, Any]:
    """Delete an expense."""
    s = _client()
    success, errors = s.deleteExpense(expense_id)
    return {"ok": bool(success), "errors": errors}

@mcp.tool()
def splitwise_add_comment(expense_id: int, content: str) -> Dict[str, Any]:
    """Add a comment to an expense."""
    s = _client()
    comment, errors = s.createComment(expense_id, content)
    if errors:
        return {"ok": False, "errors": errors}
    return {"ok": True, "comment_id": comment.getId(), "content": comment.getContent()}

if __name__ == "__main__":
    # Stdio transport (most common local-dev setup)
    mcp.run(transport="http", host="0.0.0.0", port=8000)
