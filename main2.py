import os
import json
import asyncio
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from fastmcp.server.auth import require_auth
from fastmcp.server.middleware import AuthMiddleware
from fastmcp.server.dependencies import get_access_token

# JWT verification provider (OAuth)
# Depending on your FastMCP version, this may be located under:
# fastmcp.server.auth.providers.jwt
from fastmcp.server.auth.providers.jwt import JWKSVerifier

from splitwise import Splitwise
from splitwise.expense import Expense
from splitwise.user import ExpenseUser


# -------------------------
# OAuth / JWT verification config
# -------------------------
# Your OAuth provider's issuer (OIDC issuer URL)
# Examples:
# Auth0: https://YOUR_DOMAIN/
# Clerk: https://clerk.YOUR_DOMAIN (varies)
# Okta: https://YOUR_OKTA_DOMAIN/oauth2/default
OAUTH_ISSUER = os.getenv("OAUTH_ISSUER")
OAUTH_AUDIENCE = os.getenv("OAUTH_AUDIENCE")  # the API identifier / audience configured in your provider

# Optional personal restriction: allow only your subject (sub) or email
ALLOWED_SUB = os.getenv("ALLOWED_SUB")        # e.g. "auth0|123..."
ALLOWED_EMAIL = os.getenv("ALLOWED_EMAIL")    # e.g. "you@example.com"

if not OAUTH_ISSUER or not OAUTH_AUDIENCE:
    raise RuntimeError("Missing OAUTH_ISSUER / OAUTH_AUDIENCE env vars for OAuth validation.")

auth = JWKSVerifier(
    issuer=OAUTH_ISSUER,
    audience=OAUTH_AUDIENCE,
)

mcp = FastMCP(
    "Splitwise Personal MCP (OAuth)",
    auth=auth,
    middleware=[AuthMiddleware(auth=require_auth)],
)


def _assert_allowed_caller() -> None:
    """
    Optional extra lock: even if someone gets a valid token from the same issuer,
    only allow calls from your identity.
    """
    token = get_access_token()
    if token is None:
        raise PermissionError("Not authenticated")

    claims = token.claims or {}
    sub = claims.get("sub")
    email = claims.get("email") or claims.get("preferred_username")

    if ALLOWED_SUB and sub != ALLOWED_SUB:
        raise PermissionError("Forbidden: caller not allowed (sub mismatch).")
    if ALLOWED_EMAIL and email != ALLOWED_EMAIL:
        raise PermissionError("Forbidden: caller not allowed (email mismatch).")


# -------------------------
# Splitwise client (personal)
# -------------------------
SPLITWISE_CONSUMER_KEY = os.getenv("SPLITWISE_CONSUMER_KEY")
SPLITWISE_CONSUMER_SECRET = os.getenv("SPLITWISE_CONSUMER_SECRET")
if not SPLITWISE_CONSUMER_KEY or not SPLITWISE_CONSUMER_SECRET:
    raise RuntimeError("Missing SPLITWISE_CONSUMER_KEY / SPLITWISE_CONSUMER_SECRET env vars.")

SPLITWISE_API_KEY = os.getenv("SPLITWISE_API_KEY")
SPLITWISE_OAUTH_TOKEN_JSON = os.getenv("SPLITWISE_OAUTH_TOKEN_JSON")  # optional


def _client() -> Splitwise:
    if SPLITWISE_API_KEY:
        return Splitwise(SPLITWISE_CONSUMER_KEY, SPLITWISE_CONSUMER_SECRET, api_key=SPLITWISE_API_KEY)

    s = Splitwise(SPLITWISE_CONSUMER_KEY, SPLITWISE_CONSUMER_SECRET)
    if SPLITWISE_OAUTH_TOKEN_JSON:
        s.setAccessToken(json.loads(SPLITWISE_OAUTH_TOKEN_JSON))
    return s


def _balance_to_dict(b: Any) -> Dict[str, Any]:
    return {"currency": getattr(b, "getCurrencyCode", lambda: None)(), "amount": getattr(b, "getAmount", lambda: None)()}


def _expense_to_dict(e: Any) -> Dict[str, Any]:
    return {
        "id": e.getId(),
        "group_id": e.getGroupId(),
        "description": e.getDescription(),
        "cost": e.getCost(),
        "currency_code": e.getCurrencyCode(),
        "date": e.getDate(),
        "created_at": getattr(e, "getCreatedAt", lambda: None)(),
        "updated_at": getattr(e, "getUpdatedAt", lambda: None)(),
    }


# -------------------------
# Tools
# -------------------------

@mcp.tool()
async def splitwise_current_user() -> Dict[str, Any]:
    _assert_allowed_caller()
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
async def splitwise_friends() -> List[Dict[str, Any]]:
    _assert_allowed_caller()
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
async def splitwise_groups() -> List[Dict[str, Any]]:
    _assert_allowed_caller()
    s = _client()
    groups = await asyncio.to_thread(s.getGroups)
    return [{"id": g.getId(), "name": g.getName()} for g in groups]


@mcp.tool()
async def splitwise_expenses(
    limit: int = 20,
    offset: int = 0,
    group_id: Optional[int] = None,
    friend_id: Optional[int] = None,
    dated_after: Optional[str] = None,
    dated_before: Optional[str] = None,
) -> List[Dict[str, Any]]:
    _assert_allowed_caller()
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
    description: str,
    cost: float,
    payer_id: int,
    participant_ids: List[int],
    group_id: Optional[int] = None,
    currency_code: Optional[str] = None,
) -> Dict[str, Any]:
    _assert_allowed_caller()
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
        eu = ExpenseUser()
        eu.setId(uid)
        eu.setOwedShare(f"{owed_each:.2f}")
        eu.setPaidShare(f"{cost:.2f}" if uid == payer_id else "0.00")
        users.append(eu)

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
    _assert_allowed_caller()
    if cost <= 0:
        raise ValueError("cost must be > 0")
    if not users:
        raise ValueError("users must not be empty")

    paid_total = sum(float(u.get("paid_share", 0)) for u in users)
    owed_total = sum(float(u.get("owed_share", 0)) for u in users)

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
    expense_id: int,
    description: Optional[str] = None,
    cost: Optional[float] = None,
) -> Dict[str, Any]:
    _assert_allowed_caller()
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
async def splitwise_delete_expense(expense_id: int) -> Dict[str, Any]:
    _assert_allowed_caller()
    s = _client()
    success, errors = await asyncio.to_thread(s.deleteExpense, int(expense_id))
    return {"ok": bool(success), "errors": errors}


@mcp.tool()
async def splitwise_add_comment(expense_id: int, content: str) -> Dict[str, Any]:
    _assert_allowed_caller()
    if not content.strip():
        raise ValueError("content must not be empty")

    s = _client()
    comment, errors = await asyncio.to_thread(s.createComment, int(expense_id), content)
    if errors:
        return {"ok": False, "errors": errors}
    return {"ok": True, "comment_id": comment.getId(), "content": comment.getContent()}


if __name__ == "__main__":
    mcp.run()
