from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import json
import numpy as np
from fastapi import FastAPI
import os

AI_API_KEY = os.getenv("AiApiKey")

# =========================================================
# 1. TIME UTILITIES
# =========================================================

def time_to_minutes(time_str):
    try:
        if "AM" in time_str or "PM" in time_str:
            dt = datetime.strptime(time_str, "%I:%M %p")
        else:
            dt = datetime.strptime(time_str, "%H:%M")
        return dt.hour * 60 + dt.minute
    except:
        raise ValueError(f"Invalid time format: {time_str}")


def minutes_to_time(m):
    return f"{m//60:02d}:{m%60:02d}"


def is_overlap(s1, e1, s2, e2, buffer_minutes=0):
    return not (e1 + buffer_minutes <= s2 or e2 + buffer_minutes <= s1)

# =========================================================
# 3. SOLVER ENGINE (مع خيارين)
# =========================================================

def solve_schedule_option_1_balance(course_data, prefs):
    """
    Option 1: تحسين التوازن (Balance) - تقليل ال load اليومي
    """
    model = cp_model.CpModel()
    events = []
    section_vars = {}

    buffer_minutes = prefs.get("buffer_minutes", 0)

    for course_name, info in course_data.items():
        priority = info["priority"]
        difficulty = info.get("difficulty", 1)

        # Lecture
        lec = info["lecture"]
        lec_start = time_to_minutes(lec["start"])
        lec_end = lec_start + lec["duration"]

        if lec["available_seats"] <= 0:
            raise Exception(f"Lecture full for {course_name}")

        events.append({
            "course": course_name,
            "type": "LECTURE",
            "day": lec["day"],
            "start": lec_start,
            "end": lec_end,
            "capacity": lec["capacity"],
            "available_seats": lec["available_seats"],
            "priority": priority,
            "difficulty": difficulty,
            "fixed": True,
            "var": None
        })

        # Sections
        for i, sec in enumerate(info["sections"]):
            var = model.NewBoolVar(f"{course_name}_section_{i}")
            section_vars[(course_name, i)] = var

            start = time_to_minutes(sec["start"])
            end = start + sec["duration"]

            if sec["available_seats"] <= 0:
                model.Add(var == 0)

            events.append({
                "course": course_name,
                "type": "SECTION",
                "section_id": sec["id"],
                "day": sec["day"],
                "start": start,
                "end": end,
                "capacity": sec["capacity"],
                "available_seats": sec["available_seats"],
                "priority": priority,
                "difficulty": difficulty,
                "fixed": False,
                "var": var
            })

    # Exactly One Section
    for course_name, info in course_data.items():
        section_list = [section_vars[(course_name, i)] for i in range(len(info["sections"]))]
        model.AddExactlyOne(section_list)

    # No Overlap
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            e1 = events[i]
            e2 = events[j]

            if e1["day"] != e2["day"]:
                continue

            if is_overlap(e1["start"], e1["end"], e2["start"], e2["end"], buffer_minutes):
                if e1["fixed"] and e2["fixed"]:
                    raise Exception(f"Fixed lectures conflict: {e1['course']} & {e2['course']}")
                elif e1["fixed"]:
                    model.Add(e2["var"] == 0)
                elif e2["fixed"]:
                    model.Add(e1["var"] == 0)
                else:
                    model.Add(e1["var"] + e2["var"] <= 1)

    # Daily Load
    daily_loads = []
    day_used = []

    for d in range(7):
        load = model.NewIntVar(0, 100, f"load_day_{d}")
        terms = []

        for ev in events:
            if ev["day"] == d:
                if ev["fixed"]:
                    terms.append(ev["difficulty"])
                else:
                    terms.append(ev["difficulty"] * ev["var"])

        model.Add(load == sum(terms))
        daily_loads.append(load)

        used = model.NewBoolVar(f"day_used_{d}")
        model.Add(load >= 1).OnlyEnforceIf(used)
        model.Add(load == 0).OnlyEnforceIf(used.Not())
        day_used.append(used)

    max_load = model.NewIntVar(0, 100, "max_daily_load")
    model.AddMaxEquality(max_load, daily_loads)
    total_days = sum(day_used)

    # Priority Score
    priority_score = []
    for ev in events:
        if not ev["fixed"]:
            score = ev["priority"] * 10
            priority_score.append(score * ev["var"])

    # ✅ OPTION 1 WEIGHTS: Balance أهم
    balance_weight = 100
    days_weight = 1
    priority_weight = 10

    model.Minimize(
        balance_weight * max_load +
        days_weight * total_days -
        priority_weight * sum(priority_score)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = prefs.get("max_time", 10)
    status = solver.Solve(model)

    return solver, events, status, {
        "max_load": solver.Value(max_load) if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else 0,
        "total_days": solver.Value(total_days) if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else 0,
        "option_name": "🎯 OPTION 1: Best Balance (Low Daily Load)"
    }



def solve_schedule_option_2_compact(course_data, prefs):
    """
    Option 2: جدول مدمج (Compact) - تقليل عدد الأيام
    """
    model = cp_model.CpModel()
    events = []
    section_vars = {}

    buffer_minutes = prefs.get("buffer_minutes", 0)

    for course_name, info in course_data.items():
        priority = info["priority"]
        difficulty = info.get("difficulty", 1)

        # Lecture
        lec = info["lecture"]
        lec_start = time_to_minutes(lec["start"])
        lec_end = lec_start + lec["duration"]

        if lec["available_seats"] <= 0:
            raise Exception(f"Lecture full for {course_name}")

        events.append({
            "course": course_name,
            "type": "LECTURE",
            "day": lec["day"],
            "start": lec_start,
            "end": lec_end,
            "capacity": lec["capacity"],
            "available_seats": lec["available_seats"],
            "priority": priority,
            "difficulty": difficulty,
            "fixed": True,
            "var": None
        })

        # Sections
        for i, sec in enumerate(info["sections"]):
            var = model.NewBoolVar(f"{course_name}_section_{i}")
            section_vars[(course_name, i)] = var

            start = time_to_minutes(sec["start"])
            end = start + sec["duration"]

            if sec["available_seats"] <= 0:
                model.Add(var == 0)

            events.append({
                "course": course_name,
                "type": "SECTION",
                "section_id": sec["id"],
                "day": sec["day"],
                "start": start,
                "end": end,
                "capacity": sec["capacity"],
                "available_seats": sec["available_seats"],
                "priority": priority,
                "difficulty": difficulty,
                "fixed": False,
                "var": var
            })

    # Exactly One Section
    for course_name, info in course_data.items():
        section_list = [section_vars[(course_name, i)] for i in range(len(info["sections"]))]
        model.AddExactlyOne(section_list)

    # No Overlap
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            e1 = events[i]
            e2 = events[j]

            if e1["day"] != e2["day"]:
                continue

            if is_overlap(e1["start"], e1["end"], e2["start"], e2["end"], buffer_minutes):
                if e1["fixed"] and e2["fixed"]:
                    raise Exception(f"Fixed lectures conflict: {e1['course']} & {e2['course']}")
                elif e1["fixed"]:
                    model.Add(e2["var"] == 0)
                elif e2["fixed"]:
                    model.Add(e1["var"] == 0)
                else:
                    model.Add(e1["var"] + e2["var"] <= 1)

    # Daily Load
    daily_loads = []
    day_used = []

    for d in range(7):
        load = model.NewIntVar(0, 100, f"load_day_{d}")
        terms = []

        for ev in events:
            if ev["day"] == d:
                if ev["fixed"]:
                    terms.append(ev["difficulty"])
                else:
                    terms.append(ev["difficulty"] * ev["var"])

        model.Add(load == sum(terms))
        daily_loads.append(load)

        used = model.NewBoolVar(f"day_used_{d}")
        model.Add(load >= 1).OnlyEnforceIf(used)
        model.Add(load == 0).OnlyEnforceIf(used.Not())
        day_used.append(used)

    max_load = model.NewIntVar(0, 100, "max_daily_load")
    model.AddMaxEquality(max_load, daily_loads)
    total_days = sum(day_used)

    # Priority Score
    priority_score = []
    for ev in events:
        if not ev["fixed"]:
            score = ev["priority"] * 10
            priority_score.append(score * ev["var"])

    # ✅ OPTION 2 WEIGHTS: Days أهم
    balance_weight = 1
    days_weight = 100
    priority_weight = 10

    model.Minimize(
        balance_weight * max_load +
        days_weight * total_days -
        priority_weight * sum(priority_score)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = prefs.get("max_time", 10)
    status = solver.Solve(model)

    return solver, events, status, {
        "max_load": solver.Value(max_load) if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else 0,
        "total_days": solver.Value(total_days) if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else 0,
        "option_name": "📅 OPTION 2: Compact Schedule (Fewer Days)"
    }


# =========================================================
# 4. JSON EXPORT
# =========================================================

def export_schedule_json(solver, events, option_name="Default"):
    """
    تصدير الجدول إلى JSON مع اسم Option
    """
    result = {
        "option_name": option_name,
        "schedule": []
    }

    for ev in events:
        selected = False

        if ev["fixed"]:
            selected = True
        elif solver.Value(ev["var"]) == 1:
            selected = True

        if selected:
            result["schedule"].append({
                "course": ev["course"],
                "type": ev["type"],
                "section_id": ev.get("section_id"),
                "day": ev["day"],
                "start_time": minutes_to_time(ev["start"]),
                "end_time": minutes_to_time(ev["end"]),
                "difficulty": ev["difficulty"],
                "priority": ev["priority"],
                "capacity": ev["capacity"],
                "available_seats": ev["available_seats"]
            })

    return result

def save_schedule_to_file(solver, events, filename, option_name):
    """حفظ جدول في ملف JSON"""
    schedule_data = export_schedule_json(solver, events, option_name)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(schedule_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved: {filename}")
    return schedule_data


app = FastAPI()

def run_solver(data, prefs, option):
    if option == "compact":
        return solve_schedule_option_2_compact(data, prefs)
    return solve_schedule_option_1_balance(data, prefs)

@app.post("/generate-schedule")
def generate_schedule(request: dict):
    if not AI_API_KEY:
        return {"error": "API Key is missing on the server settings!"}
    try:
        data = request["courses"]
        prefs = request.get("preferences", {})
        option = request.get("option", "balance")

        solver, events, status, stats = run_solver(data, prefs, option)

        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return {"error": "No solution found"}

        result = export_schedule_json(solver, events, option)

        return {
            "stats": stats,
            "schedule": result["schedule"]
        }

    except Exception as e:
        return {"error": str(e)}
