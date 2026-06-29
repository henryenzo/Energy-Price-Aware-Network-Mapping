"""
    This module was supposed to fetch the energy prices from the british market (after the Brexit they left the ENTSO-E transparency platform).
    My source was the Elexon API, but it seems that they don't provide the day-ahead prices anymore, only the market index prices (which are not useful in any way for our purposes).
    There are other sources, but they are all paid :-(
    So let's admit that Southern Ireland conquered the UK and thus London and Glasgow now have the same price as Dublin ^_^
"""

from datetime import datetime
import requests
import pandas as pd
import numpy as np


BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"


def fetch_gb_market_index(start: datetime, end: datetime) -> pd.DataFrame:
    if start or end is None:
        start = pd.Timestamp.now(tz="Europe/Brussels").floor("D")
        end = start + pd.Timedelta(days=1)

    BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
    provider = "APXMIDP"

    url = f"{BASE_URL}/datasets/MID"

    params = {
        "from": start.strftime("%Y-%m-%dT%H:%MZ"),
        "to": end.strftime("%Y-%m-%dT%H:%MZ"),
        "format": "json",
        "provider": provider,
    }

    r = requests.get(
        url,
        params=params,
        headers={},
        timeout=30,
    )
    r.raise_for_status()
    js = r.json()
    data = js.get("data", [])

    if not data:
        raise Exception("No data returned from bmrs API")

    df = pd.DataFrame(data)

    df = df.rename(columns={"startTime": "datetime_utc", "marketIndexPrice": "price"})

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)

    df["price"] = df["price"].replace(0, np.nan)
    df["price"] = df["price"].ffill()


    df = df.reindex(index=df.index[::-1])


    df.to_csv(f"data/gb_market_index_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv", sep=",", decimal=".", encoding="utf-8-sig", index=False)
    return df[
        [
            "datetime_utc",
            "price"
        ]
    ].sort_values("datetime_utc").reset_index(drop=True)


if __name__ == "__main__":
    start = pd.Timestamp("2024-06-01", tz="UTC")
    end = pd.Timestamp("2024-06-02", tz="UTC")

    df = fetch_gb_market_index(None, None)
    print(df)