from copy import deepcopy

import pandas as pd


PIVOT_FIELDS = (
    "pivot_direction",
    "candidate_swing_low",
    "candidate_swing_low_date",
    "confirmed_swing_low",
    "confirmed_swing_low_date",
    "candidate_swing_high",
    "candidate_swing_high_date",
    "confirmed_swing_high",
    "confirmed_swing_high_date",
    "current_structural_stop",
)


def new_pivot_state():
    return {
        "pivot_direction": "searching_low",
        "candidate_swing_low": None,
        "candidate_swing_low_date": None,
        "confirmed_swing_low": None,
        "confirmed_swing_low_date": None,
        "candidate_swing_high": None,
        "candidate_swing_high_date": None,
        "confirmed_swing_high": None,
        "confirmed_swing_high_date": None,
        "current_structural_stop": None,
        "last_processed_week": None,
        "current_weekly_close": None,
        "current_reversal_percent": 0.0,
        "confirmed_pivot_count": 0,
        "total_confirmation_weeks": 0.0,
    }


def completed_weekly_closes(data, as_of=None, timeframe="1wk", price_source="close"):
    """Create Friday-labelled weekly closes without including an incomplete week."""
    column = str(price_source).title()
    if data is None or data.empty or column not in data.columns:
        return pd.Series(dtype=float)
    close = pd.to_numeric(data[column], errors="coerce").dropna().copy()
    index = pd.DatetimeIndex(close.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    close.index = index.normalize()
    rule = {"1wk": "W-FRI", "1w": "W-FRI"}.get(timeframe, timeframe)
    weekly = close.resample(rule).last().dropna()
    cutoff = pd.Timestamp(as_of).tz_localize(None).normalize() if as_of is not None else close.index.max()
    return weekly[weekly.index <= cutoff]


def update_pivot_state(state, weekly_close, week_date, reversal_percent, lookback_weeks, min_weeks):
    """Advance one weekly ZigZag observation and return a descriptive event."""
    state = state if state is not None else new_pivot_state()
    date = _timestamp(week_date)
    date_text = str(date.date())
    price = float(weekly_close)
    event = {
        "new_pivot_confirmed": False,
        "pivot_type": None,
        "pivot_price": None,
        "reversal_percent": 0.0,
        "message": None,
    }
    if state.get("last_processed_week") == date_text:
        return event

    direction = state.get("pivot_direction") or "searching_low"
    if direction == "searching_low":
        _update_low_candidate(state, price, date, lookback_weeks, event)
        low = float(state["candidate_swing_low"])
        rebound = (price - low) / low
        event["reversal_percent"] = rebound
        if rebound >= reversal_percent and _pivot_spacing_ok(
            state["candidate_swing_low_date"],
            state.get("confirmed_swing_high_date"),
            min_weeks,
        ):
            state["confirmed_swing_low"] = low
            state["confirmed_swing_low_date"] = state["candidate_swing_low_date"]
            state["pivot_direction"] = "searching_high"
            state["candidate_swing_high"] = price
            state["candidate_swing_high_date"] = date_text
            _record_confirmation(state, state["confirmed_swing_low_date"], date)
            event.update(
                new_pivot_confirmed=True,
                pivot_type="low",
                pivot_price=low,
                message=f"Meaningful low at {low:.2f} confirmed after a {rebound:.1%} weekly rebound.",
            )
    else:
        _update_high_candidate(state, price, date, lookback_weeks, event)
        high = float(state["candidate_swing_high"])
        pullback = (high - price) / high
        event["reversal_percent"] = pullback
        if pullback >= reversal_percent and _pivot_spacing_ok(
            state["candidate_swing_high_date"],
            state.get("confirmed_swing_low_date"),
            min_weeks,
        ):
            state["confirmed_swing_high"] = high
            state["confirmed_swing_high_date"] = state["candidate_swing_high_date"]
            state["pivot_direction"] = "searching_low"
            state["candidate_swing_low"] = price
            state["candidate_swing_low_date"] = date_text
            _record_confirmation(state, state["confirmed_swing_high_date"], date)
            event.update(
                new_pivot_confirmed=True,
                pivot_type="high",
                pivot_price=high,
                message=f"Meaningful high at {high:.2f} confirmed after a {pullback:.1%} weekly reversal.",
            )

    state["last_processed_week"] = date_text
    state["current_weekly_close"] = price
    state["current_reversal_percent"] = event["reversal_percent"]
    return event


def build_pivot_history(weekly_closes, reversal_percent, lookback_weeks, min_weeks):
    state = new_pivot_state()
    events = []
    snapshots = {}
    for date, price in weekly_closes.items():
        event = update_pivot_state(
            state, price, date, reversal_percent, lookback_weeks, min_weeks
        )
        if event["new_pivot_confirmed"]:
            events.append({"confirmation_date": _timestamp(date), **event})
        snapshots[_timestamp(date)] = deepcopy(state)
    return state, events, snapshots


def update_structural_stop(position, pivot_state, use_tentative_high=False):
    active_low = position.get("active_structural_low")
    confirmed_low = pivot_state.get("confirmed_swing_low")
    if confirmed_low is not None and active_low is not None and confirmed_low > active_low:
        active_low = float(confirmed_low)
        position["active_structural_low"] = active_low
    high = (
        pivot_state.get("candidate_swing_high")
        if use_tentative_high
        else pivot_state.get("confirmed_swing_high")
    )
    if active_low is None or high is None or float(high) <= float(active_low):
        return False
    old_stop = position.get("current_structural_stop")
    midpoint = (float(active_low) + float(high)) / 2
    new_stop = midpoint if old_stop is None else max(float(old_stop), midpoint)
    position["current_structural_stop"] = new_stop
    return old_stop is None or new_stop > float(old_stop)


def _update_low_candidate(state, price, date, lookback_weeks, event):
    candidate_date = state.get("candidate_swing_low_date")
    expired = candidate_date and _weeks_between(candidate_date, date) > lookback_weeks
    old = state.get("candidate_swing_low")
    if old is None or expired or price < float(old):
        state["candidate_swing_low"] = price
        state["candidate_swing_low_date"] = str(date.date())
        if old is not None and not expired:
            event["message"] = f"Candidate low updated from {float(old):.2f} to {price:.2f}; not yet confirmed."


def _update_high_candidate(state, price, date, lookback_weeks, event):
    candidate_date = state.get("candidate_swing_high_date")
    expired = candidate_date and _weeks_between(candidate_date, date) > lookback_weeks
    old = state.get("candidate_swing_high")
    if old is None or expired or price > float(old):
        state["candidate_swing_high"] = price
        state["candidate_swing_high_date"] = str(date.date())
        if old is not None and not expired:
            event["message"] = f"Candidate high updated from {float(old):.2f} to {price:.2f}; not yet confirmed."


def _pivot_spacing_ok(candidate_date, prior_date, minimum_weeks):
    return prior_date is None or _weeks_between(prior_date, _timestamp(candidate_date)) >= minimum_weeks


def _record_confirmation(state, candidate_date, confirmation_date):
    state["confirmed_pivot_count"] = int(state.get("confirmed_pivot_count", 0)) + 1
    state["total_confirmation_weeks"] = float(state.get("total_confirmation_weeks", 0.0)) + _weeks_between(candidate_date, confirmation_date)


def _weeks_between(start, end):
    return max(0.0, (_timestamp(end) - _timestamp(start)).days / 7)


def _timestamp(value):
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
