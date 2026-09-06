"""Section 10's alerting: the refresh telling you when to open the app.

Split three ways so each part can be tested without the other two:

* `rules` decides what is worth saying, as pure functions over frames.
* `discord` sends it, and is the only part that touches the network.
* `scripts/refresh_data.py` reads the database and wires the two together.

Nothing here runs unless a webhook URL is configured, and the URL is never a
literal in this repository -- see `Settings.alert_discord_webhook_url`.
"""

from quantpulse.alerting.rules import Alert, build_digest, pattern_alerts, rating_change_alerts

__all__ = ["Alert", "build_digest", "pattern_alerts", "rating_change_alerts"]
