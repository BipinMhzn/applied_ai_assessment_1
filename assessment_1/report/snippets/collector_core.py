            params = {
                "site": "stackoverflow",
                "sort": "creation",
                "order": "asc",
                "pagesize": 100,
                "page": page,
                "filter": filter_param,
                "fromdate": int(from_date.timestamp()),
                "todate": int((from_date + timedelta(days=1)).timestamp()) - 1
            }
            if api_key:
                params["key"] = api_key
            try:
                response = requests.get(f"{API}/questions", params=params, timeout=30)
                if response.status_code == 429:
                    # gateway rate limit: stop cleanly instead of hammering; re-run later to resume
                    print(f"Rate limited (HTTP 429) on {from_date.date()} page {page} - stopping. "
                          f"Wait a few minutes, then re-run this cell to resume.")
                    return
                response.raise_for_status()  # other HTTP errors -> retry/skip path below, with status visible
                try:
                    data = response.json()
                except ValueError:
                    raise RuntimeError(f"non-JSON body (HTTP {response.status_code}): {response.text[:80]!r}")
                if "error_id" in data:
                    raise RuntimeError(f"API error {data.get('error_id')}: {data.get('error_message')}")
                if data.get("backoff"):
                    time.sleep(data["backoff"])  # API asks us to slow down
