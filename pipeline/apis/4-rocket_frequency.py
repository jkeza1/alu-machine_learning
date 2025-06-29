#!/usr/bin/env python3
"""Pipeline Api"""
import requests

if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches"
    r = requests.get(url)

    rocket_dict = {}

    for launch in r.json():
        rocket_id = launch["rocket"]
        if rocket_id in rocket_dict:
            rocket_dict[rocket_id] += 1
        else:
            rocket_dict[rocket_id] = 1

    for rocket_id, count in sorted(rocket_dict.items(), key=lambda kv: kv[1], reverse=True):
        rurl = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
        req = requests.get(rurl)
        rocket_name = req.json().get("name", "Unknown")
        print(f"{rocket_name}: {count}")
