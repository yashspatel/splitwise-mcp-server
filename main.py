# -----------------------------
# main.py — Splitwise MCP Server (Async)
# -----------------------------
# This file exposes Splitwise actions as MCP tools using FastMCP.
# It uses the Splitwise Python SDK (which is synchronous), so we run SDK calls
# inside asyncio.to_thread(...) to keep the MCP tools async.

import os  # Read environment variables (keys/tokens)
import asyncio  # Run blocking SDK calls in a background thread
from typing import Any, Dict, List, Optional  # Type hints for clean tool interfaces

from dotenv import load_dotenv  # Load local .env (useful for local dev)
from fastmcp import FastMCP  # MCP server framework

from splitwise import Splitwise  # Splitwise SDK main client
from splitwise.expense import Expense  # Splitwise Expense object
from splitwise.user import ExpenseUser  # Splitwise ExpenseUser object

load_dotenv()  # Loads .env into environment when running locally (safe in cloud too)

mcp = FastMCP("Splitwise MCP")  # Create the MCP server instance


def _client() -> Splitwise:
    """
    Create a Splitwise SDK client using env vars.

    Auth strategy:
    - Prefer API key mode (single-user, simplest)
    - Otherwise use OAuth1 token + secret if provided (also single-user)
    """
    consumer_key = os.environ["SPLITWISE_CONSUMER_KEY"]  # Splitwise app consumer key
    consumer_secret = os.environ["SPLITWISE_CONSUMER_SECRET"]  # Splitwise app consumer secret

    api_key = os.getenv("SPLITWISE_API_KEY")  # Optional: personal API key mode

    # If api_key exists, use API key auth; otherwise just create client with app keys
    s = (
        Splitwise(consumer_key, consumer_secret, api_key=api_key)
        if api_key
        else Splitwise(consumer_key, consumer_secret)
    )

    oauth_token = os.getenv("SPLITWISE_OAUTH_TOKEN")  # Optional OAuth token
    oauth_token_secret = os.getenv("SPLITWISE_OAUTH_TOKEN_SECRET")  # Optional OAuth token secret

    # If OAuth creds exist, attach them so requests are performed as that user
    if oauth_token and oauth_token_secret:
        s.setAccessToken({"oauth_token": oauth_token, "oauth_token_secret": oauth_token_secret})

    return s  # Return the configured client


def _user_to_dict(u: Any) -> Dict[str, Any]:
    """
    Convert Splitwise user-like objects (CurrentUser/Friend/User) to a simple dict.
    """
    return {
        "id": getattr(u, "getId", lambda: None)(),  # User ID
        "first_name": getattr(u, "getFirstName", lambda: None)(),  # First name
        "last_name": getattr(u, "getLastName", lambda: None)(),  # Last name
        "email": getattr(u, "getEmail", lambda: None)(),  # Email (may be None)
    }


# -----------------------------
# READ TOOLS
# -----------------------------

@mcp.tool()
async def splitwise_current_user() -> Dict[str, Any]:
    """Return the authenticated user's profile."""
    s = _client()  # Build the Splitwise client
    u = await asyncio.to_thread(s.getCurrentUser)  # Run blocking SDK call in a thread
    return _user_to_dict(u)  # Return as a JSON-friendly dict


@mcp.tool()
async def splitwise_friends() -> List[Dict[str, Any]]:
    """List friends with balances (basic fields)."""
    s = _client()  # Build the Splitwise client
    friends = await asyncio.to_thread(s.getFriends)  # Fetch friends in a thread

    out: List[Dict[str, Any]] = []  # Output list

    for f in friends:  # Loop through each friend
        item = _user_to_dict(f)  # Convert friend object to dict

        balances = []  # Collect balances (currency + amount)
        for b in (getattr(f, "getBalances", lambda: [])() or []):  # Defensive: balances may be missing
            balances.append(
                {
                    "currency": getattr(b, "getCurrencyCode", lambda: None)(),  # Currency code (e.g. USD)
                    "amount": getattr(b, "getAmount", lambda: None)(),  # Balance amount string
                }
            )

        item["balances"] = balances  # Attach balances to friend record
        out.append(item)  # Add to output list

    return out  # Return list of friends


@mcp.tool()
async def splitwise_groups() -> List[Dict[str, Any]]:
    """List groups."""
    s = _client()  # Build the Splitwise client
    groups = await asyncio.to_thread(s.getGroups)  # Fetch groups in a thread

    return [
        {
            "id": g.getId(),  # Group ID
            "name": g.getName(),  # Group name
            "simplified_by_default": getattr(g, "isSimplifiedByDefault", lambda: None)(),  # Simplification flag
        }
        for g in groups  # Convert each group to a dict
    ]


@mcp.tool()
async def splitwise_expenses(
    limit: int = 20,  # Max number of expenses to return
    offset: int = 0,  # Pagination offset
    group_id: Optional[int] = None,  # Filter by group
    dated_after: Optional[str] = None,  # Filter: after YYYY-MM-DD
    dated_before: Optional[str] = None,  # Filter: before YYYY-MM-DD
) -> List[Dict[str, Any]]:
    """
    Fetch expenses (lightweight projection).
    """
    s = _client()  # Build the Splitwise client

    # Wrap the SDK call so we can run it in a background thread
    def _fetch():
        return s.getExpenses(
            offset=offset,  # Offset for pagination
            limit=limit,  # Limit for pagination
            group_id=group_id,  # Optional group filter
            dated_after=dated_after,  # Optional date filter
            dated_before=dated_before,  # Optional date filter
        )

    expenses = await asyncio.to_thread(_fetch)  # Execute the fetch in a thread

    out: List[Dict[str, Any]] = []  # Output list
    for e in expenses:  # Loop through each expense
        out.append(
            {
                "id": e.getId(),  # Expense ID
                "group_id": e.getGroupId(),  # Group ID (or None)
                "description": e.getDescription(),  # Description
                "cost": e.getCost(),  # Cost (string)
                "currency_code": e.getCurrencyCode(),  # Currency code
                "date": e.getDate(),  # Date string
            }
        )

    return out  # Return list of expenses


# -----------------------------
# WRITE TOOLS
# -----------------------------

@mcp.tool()
async def splitwise_create_expense_equal_split(
    description: str,  # Expense description
    cost: float,  # Total cost
    payer_id: int,  # User ID who paid
    participant_ids: List[int],  # User IDs who owe (including payer)
    group_id: Optional[int] = None,  # Optional group ID
    currency_code: Optional[str] = None,  # Optional currency override
) -> Dict[str, Any]:
    """
    Create an expense where participants split equally and payer pays 100%.
    """
    if not participant_ids:  # Must have at least one participant
        raise ValueError("participant_ids must not be empty")
    if cost <= 0:  # Cost must be positive
        raise ValueError("cost must be > 0")
    if payer_id not in participant_ids:  # Payer must be included
        raise ValueError("payer_id must be included in participant_ids")

    s = _client()  # Build the Splitwise client

    expense = Expense()  # Create a new Expense object
    expense.setDescription(description)  # Set description
    expense.setCost(f"{cost:.2f}")  # Set total cost
    if group_id is not None:  # If group specified
        expense.setGroupId(group_id)  # Set group
    if currency_code:  # If currency specified
        expense.setCurrencyCode(currency_code)  # Set currency

    owed_each = cost / len(participant_ids)  # Equal split amount per participant

    users: List[ExpenseUser] = []  # Build list of ExpenseUser splits
    for uid in participant_ids:  # For each participant
        u = ExpenseUser()  # Create split record
        u.setId(uid)  # Set Splitwise user ID
        u.setOwedShare(f"{owed_each:.2f}")  # What they owe
        u.setPaidShare(f"{cost:.2f}" if uid == payer_id else "0.00")  # Payer paid all; others paid 0
        users.append(u)  # Add to list

    expense.setUsers(users)  # Attach users splits to the expense

    created, errors = await asyncio.to_thread(s.createExpense, expense)  # Create expense in a thread
    if errors:  # If API returned validation errors
        return {"ok": False, "errors": errors}  # Return errors

    return {"ok": True, "expense_id": created.getId()}  # Return created expense ID


@mcp.tool()
async def splitwise_create_expense_unequal_split(
    description: str,  # Expense description
    cost: float,  # Total cost
    users: List[Dict[str, Any]],  # Per-user split dicts
    group_id: Optional[int] = None,  # Optional group ID
    currency_code: Optional[str] = None,  # Optional currency override
) -> Dict[str, Any]:
    """
    Create an expense with an unequal split.

    Input format:
      users = [
        {"id": 123, "paid_share": 10.0, "owed_share": 2.5},
        {"id": 456, "paid_share": 0.0,  "owed_share": 7.5},
      ]

    Rules:
    - sum(paid_share) == cost (within 0.01)
    - sum(owed_share) == cost (within 0.01)
    """
    if cost <= 0:  # Cost must be positive
        raise ValueError("cost must be > 0")
    if not users:  # Must provide at least one user split
        raise ValueError("users must not be empty")

    paid_total = 0.0  # Track total paid shares
    owed_total = 0.0  # Track total owed shares

    for u in users:  # Validate and total up shares
        if "id" not in u:  # Must include user id
            raise ValueError("Each user entry must include 'id'")
        paid = float(u.get("paid_share", u.get("paidShare", 0.0)))  # Accept snake_case or camelCase
        owed = float(u.get("owed_share", u.get("owedShare", 0.0)))  # Accept snake_case or camelCase
        paid_total += paid  # Add to paid total
        owed_total += owed  # Add to owed total

    # Validate totals with a small tolerance for floating point rounding
    if abs(paid_total - cost) > 0.01 or abs(owed_total - cost) > 0.01:
        return {
            "ok": False,
            "errors": [
                f"Invalid shares: sum(paid_share)={paid_total:.2f}, sum(owed_share)={owed_total:.2f}, expected cost={cost:.2f}"
            ],
        }

    s = _client()  # Build the Splitwise client

    expense = Expense()  # Create a new Expense object
    expense.setDescription(description)  # Set description
    expense.setCost(f"{cost:.2f}")  # Set total cost
    if group_id is not None:  # If group specified
        expense.setGroupId(group_id)  # Set group
    if currency_code:  # If currency specified
        expense.setCurrencyCode(currency_code)  # Set currency

    eu_list: List[ExpenseUser] = []  # Build list of ExpenseUser splits
    for u in users:  # For each user split input
        uid = int(u["id"])  # User ID
        paid = float(u.get("paid_share", u.get("paidShare", 0.0)))  # Paid share
        owed = float(u.get("owed_share", u.get("owedShare", 0.0)))  # Owed share

        eu = ExpenseUser()  # Create split record
        eu.setId(uid)  # Set user ID
        eu.setPaidShare(f"{paid:.2f}")  # Set paid share
        eu.setOwedShare(f"{owed:.2f}")  # Set owed share
        eu_list.append(eu)  # Add to list

    expense.setUsers(eu_list)  # Attach splits to expense

    created, errors = await asyncio.to_thread(s.createExpense, expense)  # Create expense in a thread
    if errors:  # If API returned validation errors
        return {"ok": False, "errors": errors}  # Return errors

    return {"ok": True, "expense_id": created.getId()}  # Return created expense ID


@mcp.tool()
async def splitwise_create_expense_no_split(
    description: str,  # Expense description
    cost: float,  # Total cost
    payer_id: int,  # Person who paid 100%
    ower_id: int,  # Person who owes 100%
    group_id: Optional[int] = None,  # Optional group ID
    currency_code: Optional[str] = None,  # Optional currency override
) -> Dict[str, Any]:
    """
    Create a 2-person expense with NO split:
    - payer_id paid 100%
    - ower_id owes 100%

    Examples:
    - "They owe me" -> payer_id = me,   ower_id = them
    - "I owe them"  -> payer_id = them, ower_id = me
    """
    if cost <= 0:  # Cost must be positive
        raise ValueError("cost must be > 0")
    if payer_id == ower_id:  # Must be two distinct people
        raise ValueError("payer_id and ower_id must be different")

    s = _client()  # Build the Splitwise client

    expense = Expense()  # Create a new Expense object
    expense.setDescription(description)  # Set description
    expense.setCost(f"{cost:.2f}")  # Set total cost
    if group_id is not None:  # If group specified
        expense.setGroupId(group_id)  # Set group
    if currency_code:  # If currency specified
        expense.setCurrencyCode(currency_code)  # Set currency

    payer = ExpenseUser()  # Split record for payer
    payer.setId(payer_id)  # Set payer user ID
    payer.setPaidShare(f"{cost:.2f}")  # Payer paid everything
    payer.setOwedShare("0.00")  # Payer owes nothing

    ower = ExpenseUser()  # Split record for ower
    ower.setId(ower_id)  # Set ower user ID
    ower.setPaidShare("0.00")  # Ower paid nothing
    ower.setOwedShare(f"{cost:.2f}")  # Ower owes everything

    expense.setUsers([payer, ower])  # Attach both users

    created, errors = await asyncio.to_thread(s.createExpense, expense)  # Create expense in a thread
    if errors:  # If API returned validation errors
        return {"ok": False, "errors": errors}  # Return errors

    return {"ok": True, "expense_id": created.getId()}  # Return created expense ID


@mcp.tool()
async def splitwise_update_expense(
    expense_id: int,  # Expense to update
    description: Optional[str] = None,  # Optional new description
    cost: Optional[float] = None,  # Optional new cost
) -> Dict[str, Any]:
    """Update an existing expense (description and/or cost)."""
    s = _client()  # Build the Splitwise client

    e = Expense()  # Create partial Expense object for update
    e.id = expense_id  # Set required ID
    if description is not None:  # If description is provided
        e.setDescription(description)  # Update description
    if cost is not None:  # If cost is provided
        e.setCost(f"{cost:.2f}")  # Update cost

    updated, errors = await asyncio.to_thread(s.updateExpense, e)  # Update expense in a thread
    if errors:  # If API returned validation errors
        return {"ok": False, "errors": errors}  # Return errors

    return {"ok": True, "expense_id": updated.getId()}  # Return updated expense ID


@mcp.tool()
async def splitwise_delete_expense(expense_id: int) -> Dict[str, Any]:
    """Delete an expense."""
    s = _client()  # Build the Splitwise client
    success, errors = await asyncio.to_thread(s.deleteExpense, expense_id)  # Delete in a thread
    return {"ok": bool(success), "errors": errors}  # Return status + errors (if any)


@mcp.tool()
async def splitwise_add_comment(expense_id: int, content: str) -> Dict[str, Any]:
    """Add a comment to an expense."""
    s = _client()  # Build the Splitwise client
    comment, errors = await asyncio.to_thread(s.createComment, expense_id, content)  # Create comment in a thread
    if errors:  # If API returned validation errors
        return {"ok": False, "errors": errors}  # Return errors
    return {"ok": True, "comment_id": comment.getId(), "content": comment.getContent()}  # Return comment details


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    # Start the MCP server over HTTP (useful for cloud hosting and remote clients)
    mcp.run(transport="http", host="0.0.0.0", port=8000)  # Listen on all interfaces on port 8000
