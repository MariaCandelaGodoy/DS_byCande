from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from statistics import mean, pstdev


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

WINDOW_DAYS = 14

STUDENT_NAME_ALIASES = {
    "nina": "ninu",
}


def normalize_student_name(name: str) -> str:
    clean = str(name or "").strip()
    return STUDENT_NAME_ALIASES.get(clean.lower(), clean)


def write_output_text(output_name: str, content: str) -> None:
    try:
        (OUTPUT_DIR / output_name).write_text(content, encoding="utf-8")
    except PermissionError as exc:
        print(f"No se pudo escribir outputs/{output_name}: {exc}")


def euclidean_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class KMeans:
    def __init__(self, n_clusters: int, random_state: int | None = None, n_init: int = 10, max_iter: int = 100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.inertia_ = 0.0

    def _initial_centroids(self, vectors: list[list[float]]) -> list[list[float]]:
        centroids = [vectors[0][:]]
        while len(centroids) < self.n_clusters:
            next_vector = max(
                vectors,
                key=lambda vector: min(euclidean_distance(vector, centroid)
                                       for centroid in centroids),
            )
            centroids.append(next_vector[:])
        return centroids

    def fit_predict(self, vectors: list[list[float]]) -> list[int]:
        if not vectors:
            self.inertia_ = 0.0
            return []

        centroids = self._initial_centroids(vectors)
        labels = [0] * len(vectors)

        for _ in range(self.max_iter):
            new_labels = [
                min(range(len(centroids)),
                    key=lambda label: euclidean_distance(vector, centroids[label]))
                for vector in vectors
            ]
            new_centroids = []
            for label in range(len(centroids)):
                members = [vector for vector, assigned in zip(
                    vectors, new_labels) if assigned == label]
                if members:
                    new_centroids.append(
                        [sum(values) / len(values) for values in zip(*members)])
                else:
                    new_centroids.append(centroids[label])

            if new_labels == labels:
                centroids = new_centroids
                break

            labels = new_labels
            centroids = new_centroids

        self.inertia_ = sum(
            euclidean_distance(vector, centroids[label]) ** 2
            for vector, label in zip(vectors, labels)
        )
        return labels


def silhouette_score_local(vectors: list[list[float]], labels: list[int]) -> float:
    scores: list[float] = []
    unique_labels = sorted(set(labels))
    for index, vector in enumerate(vectors):
        own_label = labels[index]
        own_cluster = [
            other
            for other_index, other in enumerate(vectors)
            if labels[other_index] == own_label and other_index != index
        ]
        a = mean([euclidean_distance(vector, other)
                 for other in own_cluster]) if own_cluster else 0.0
        other_distances = []
        for label in unique_labels:
            if label == own_label:
                continue
            members = [other for other, assigned in zip(
                vectors, labels) if assigned == label]
            if members:
                other_distances.append(
                    mean(euclidean_distance(vector, other) for other in members))
        b = min(other_distances) if other_distances else 0.0
        denominator = max(a, b)
        scores.append((b - a) / denominator if denominator else 0.0)
    return mean(scores) if scores else 0.0


def cluster_inertia(vectors: list[list[float]], labels: list[int]) -> float:
    total = 0.0
    for label in sorted(set(labels)):
        members = [vector for vector, assigned in zip(
            vectors, labels) if assigned == label]
        if not members:
            continue
        centroid = [sum(values) / len(values) for values in zip(*members)]
        total += sum(euclidean_distance(vector, centroid) **
                     2 for vector in members)
    return total


def agglomerative_labels(vectors: list[list[float]], n_clusters: int) -> list[int]:
    if not vectors:
        return []

    clusters = [[index] for index in range(len(vectors))]
    target_clusters = max(1, min(n_clusters, len(vectors)))

    def average_linkage(left: list[int], right: list[int]) -> float:
        distances = [
            euclidean_distance(vectors[left_index], vectors[right_index])
            for left_index in left
            for right_index in right
        ]
        return mean(distances) if distances else 0.0

    while len(clusters) > target_clusters:
        best_pair = min(
            (
                (average_linkage(clusters[i], clusters[j]), i, j)
                for i in range(len(clusters))
                for j in range(i + 1, len(clusters))
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )
        _, left_index, right_index = best_pair
        merged = sorted(clusters[left_index] + clusters[right_index])
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {left_index, right_index}
        ]
        clusters.append(merged)
        clusters.sort(key=lambda cluster: cluster[0])

    labels = [0] * len(vectors)
    for label, cluster in enumerate(clusters):
        for index in cluster:
            labels[index] = label
    return labels


def parse_date_text(value: str) -> date | None:
    if not value:
        return None
    text = str(value).strip()
    patterns = [
        "%m/%d/%y",
        "%m/%d/%Y",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d/%m/%y",
        "%d/%m/%Y",
    ]
    for pattern in patterns:
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    return None


def parse_datetime_text(value: str) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    patterns = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]
    for pattern in patterns:
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    return None


def parse_any_date(value: str) -> date | None:
    parsed = parse_date_text(value)
    if parsed is not None:
        return parsed
    if value is None:
        return None
    text = str(value).strip()
    if re.fullmatch(r"\d{13}", text):
        return datetime.fromtimestamp(int(text) / 1000).date()
    if re.fullmatch(r"\d{10}", text):
        return datetime.fromtimestamp(int(text)).date()
    return None


def extract_student_from_path(path: Path, prefix: str) -> str:
    if path.name.startswith(prefix):
        return normalize_student_name(path.name.replace(prefix, "").strip())
    if path.parent.name.startswith(prefix):
        return normalize_student_name(path.parent.name.replace(prefix, "").strip())
    for parent in path.parents:
        if parent.name.startswith(prefix):
            return normalize_student_name(parent.name.replace(prefix, "").strip())
    return normalize_student_name(path.stem)


def read_samsung_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        first_row = next(reader, None)
        second_row = next(reader, None)

        if first_row and first_row[0].startswith("com.samsung") and second_row:
            header = second_row
            data_rows = list(reader)
            return header, data_rows

        header = first_row or []
        data_rows = []
        if second_row is not None:
            data_rows.append(second_row)
        data_rows.extend(list(reader))
        return header, data_rows


def parse_exam_calendar(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            raw_value = (row.get("FECHA_RENDIR") or "").strip()
            match = re.search(r"(\d{2}/\d{2}/\d{2,4})", raw_value)
            if not match:
                continue
            exam_date = parse_date_text(match.group(1))
            if exam_date is None:
                continue
            label_match = re.search(r"\((.*)\)", raw_value)
            rows.append(
                {
                    "subject": (row.get("MATERIA") or "").strip(),
                    "exam_date": exam_date,
                    "exam_label": label_match.group(1).strip() if label_match else "EXAM",
                }
            )
    return rows


def build_relevant_dates(exam_calendar: list[dict], window_days: int) -> set[date]:
    relevant: set[date] = set()
    for exam in exam_calendar:
        exam_date = exam["exam_date"]
        for delta in range(-window_days, window_days + 1):
            relevant.add(exam_date + timedelta(days=delta))
    return relevant


def load_netflix_history(path: Path, relevant_dates: set[date] | None = None) -> list[dict]:
    student = path.stem.replace("NetflixViewingHistory", "").strip()
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            day = parse_date_text(row.get("Date", ""))
            if day is None:
                continue
            if relevant_dates is not None and day not in relevant_dates:
                continue
            rows.append(
                {
                    "student": student,
                    "date": day,
                    "views": 1.0,
                    "title": row.get("Title", ""),
                }
            )
    return rows


def load_spotify_history(path: Path, relevant_dates: set[date] | None = None) -> list[dict]:
    student = extract_student_from_path(path, "Spotify Account Data ")
    source = "music" if "music" in path.stem.lower() else "podcast"
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    rows: list[dict] = []
    for item in data:
        dt = parse_datetime_text(item.get("endTime", ""))
        if dt is None:
            continue
        day = dt.date()
        if relevant_dates is not None and day not in relevant_dates:
            continue
        rows.append(
            {
                "student": student,
                "date": day,
                "minutes": float(item.get("msPlayed", 0)) / 60000.0,
                "source": source,
            }
        )
    return rows


def load_samsung_steps(folder: Path, relevant_dates: set[date] | None = None) -> list[dict]:
    candidates = list(folder.rglob(
        "com.samsung.shealth.activity.day_summary*.csv"))
    if not candidates:
        candidates = list(folder.rglob(
            "com.samsung.shealth.step_daily_trend*.csv"))
    if not candidates:
        return []

    path = candidates[0]
    student = extract_student_from_path(folder, "Samsung Health ")
    rows: list[dict] = []
    header, data_rows = read_samsung_csv_rows(path)
    header_index = {name: index for index, name in enumerate(header)}
    for raw_row in data_rows:
        row = {name: raw_row[index] if index < len(
            raw_row) else "" for name, index in header_index.items()}
        day = parse_date_text((row.get("day_time", "") or "")[
                              :19]) or parse_any_date(row.get("day_time", ""))
        if day is None:
            continue
        if relevant_dates is not None and day not in relevant_dates:
            continue
        step_value = row.get("step_count") or row.get("count") or 0
        distance_value = row.get("distance") or 0
        calorie_value = row.get("calorie") or 0
        rows.append(
            {
                "student": student,
                "date": day,
                "steps": float(step_value or 0),
                "distance": float(distance_value or 0),
                "calories": float(calorie_value or 0),
            }
        )
    return rows


def load_samsung_sleep(folder: Path, relevant_dates: set[date] | None = None) -> list[dict]:
    candidates = list(folder.rglob("com.samsung.shealth.sleep*.csv"))
    if not candidates:
        return []

    path = candidates[0]
    student = extract_student_from_path(folder, "Samsung Health ")
    rows: list[dict] = []
    header, data_rows = read_samsung_csv_rows(path)
    header_index = {name: index for index, name in enumerate(header)}
    for raw_row in data_rows:
        row = {name: raw_row[index] if index < len(
            raw_row) else "" for name, index in header_index.items()}
        day = parse_date_text((row.get("com.samsung.health.sleep.start_time", "") or "")[:19]) or parse_any_date(
            row.get("com.samsung.health.sleep.start_time", "")
        )
        if day is None:
            continue
        if relevant_dates is not None and day not in relevant_dates:
            continue
        raw_sleep = row.get("sleep_duration") or row.get(
            "total_sleep_time_weight") or ""
        try:
            sleep_minutes = float(raw_sleep) / 60000.0
        except (TypeError, ValueError):
            sleep_minutes = math.nan
        rows.append(
            {
                "student": student,
                "date": day,
                "sleep_minutes": sleep_minutes,
            }
        )
    return rows


def load_takeout_daily_steps(folder: Path, relevant_dates: set[date] | None = None) -> list[dict]:
    metrics_dir = folder / "Fit" / "Daily activity metrics"
    if not metrics_dir.exists():
        return []

    student = normalize_student_name(folder.name.replace("Takeout", "").strip())
    rows: list[dict] = []
    for day_file in metrics_dir.glob("*.csv"):
        if day_file.name.lower() == "daily activity metrics.csv":
            continue
        day = parse_date_text(day_file.stem)
        if day is None:
            continue
        if relevant_dates is not None and day not in relevant_dates:
            continue

        step_sum = 0.0
        distance_sum = 0.0
        calories_sum = 0.0
        with day_file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for entry in reader:
                step_sum += float(entry.get("Step count", 0) or 0)
                distance_sum += float(entry.get("Distance (m)", 0) or 0)
                calories_sum += float(entry.get("Calories (kcal)", 0) or 0)
        rows.append(
            {
                "student": student,
                "date": day,
                "steps": step_sum,
                "distance": distance_sum,
                "calories": calories_sum,
            }
        )
    return rows


def _parse_apple_datetime(raw_value: str) -> datetime | None:
    if not raw_value:
        return None
    clean = raw_value.strip()

    # Apple exports often include timezone suffixes like "-0300".
    # Try timezone-aware parsing first.
    try:
        return datetime.strptime(clean, "%Y-%m-%d %H:%M:%S %z")
    except ValueError:
        pass

    # Fallback: strip trailing timezone and parse as naive datetime.
    clean_no_tz = re.sub(r"\s[+-]\d{4}$", "", clean)
    return parse_datetime_text(clean_no_tz)


def load_apple_health_records(folder: Path, relevant_dates: set[date] | None = None) -> tuple[list[dict], list[dict]]:
    candidates = [folder / "export.xml", folder / "exportar.xml"]
    xml_path = next((path for path in candidates if path.exists()), None)
    if xml_path is None:
        return [], []

    student = folder.name.replace("apple_health_export", "").strip()
    steps_acc: dict[date, float] = defaultdict(float)
    sleep_acc: dict[date, float] = defaultdict(float)

    for _, elem in ET.iterparse(str(xml_path), events=("end",)):
        if elem.tag != "Record":
            continue
        record_type = elem.attrib.get("type", "")
        start_dt = _parse_apple_datetime(elem.attrib.get("startDate", ""))
        end_dt = _parse_apple_datetime(elem.attrib.get("endDate", ""))
        if start_dt is None:
            elem.clear()
            continue

        day = start_dt.date()
        if relevant_dates is not None and day not in relevant_dates:
            elem.clear()
            continue

        if record_type == "HKQuantityTypeIdentifierStepCount":
            try:
                steps_acc[day] += float(elem.attrib.get("value", 0) or 0)
            except ValueError:
                pass
        elif record_type == "HKCategoryTypeIdentifierSleepAnalysis" and end_dt is not None:
            duration_minutes = max(
                0.0, (end_dt - start_dt).total_seconds() / 60.0)
            sleep_acc[day] += duration_minutes
        elem.clear()

    step_rows = [
        {"student": student, "date": day, "steps": value,
            "distance": 0.0, "calories": 0.0}
        for day, value in steps_acc.items()
    ]
    sleep_rows = [
        {"student": student, "date": day, "sleep_minutes": value}
        for day, value in sleep_acc.items()
    ]
    return step_rows, sleep_rows


def summarize_daily_windows(records: list[dict], value_key: str, source_name: str, exam_calendar: list[dict]) -> list[dict]:
    if not records:
        return []

    by_student: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_student[row["student"]].append(row)

    result: list[dict] = []
    for student, student_rows in by_student.items():
        dates = [row["date"] for row in student_rows]
        min_date = min(dates)
        max_date = max(dates)
        relevant_exams = [
            exam
            for exam in exam_calendar
            if exam["exam_date"] >= min_date - timedelta(days=WINDOW_DAYS)
            and exam["exam_date"] <= max_date + timedelta(days=WINDOW_DAYS)
        ]
        if not relevant_exams:
            continue

        daily_values: dict[date, list[float]] = defaultdict(list)
        for row in student_rows:
            value = row.get(value_key)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            daily_values[row["date"]].append(float(value))

        # Use daily sums to retain total volume (views, minutes, steps, sleep).
        daily_total = {day: sum(values)
                       for day, values in daily_values.items() if values}
        for exam in relevant_exams:
            for day, value in daily_total.items():
                relative_day = (day - exam["exam_date"]).days
                if abs(relative_day) > WINDOW_DAYS:
                    continue
                result.append(
                    {
                        "student": student,
                        "date": day,
                        "relative_day": relative_day,
                        "exam_date": exam["exam_date"],
                        "exam_label": exam["exam_label"],
                        "source": source_name,
                        value_key: value,
                    }
                )
    return result


def mean_or_none(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None and not (
        isinstance(value, float) and math.isnan(value))]
    return mean(clean) if clean else None


def std_or_none(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None and not (
        isinstance(value, float) and math.isnan(value))]
    if len(clean) < 2:
        return None
    return pstdev(clean)


def add_before_after_summary(windowed: list[dict], value_key: str) -> list[dict]:
    if not windowed:
        return []

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in windowed:
        grouped[(row["student"], row["source"])].append(row)

    summaries: list[dict] = []
    for (student, source), rows in grouped.items():
        pre_values = [row[value_key]
                      for row in rows if -7 <= row["relative_day"] <= -1]
        post_values = [row[value_key]
                       for row in rows if 1 <= row["relative_day"] <= 7]
        event_values = [row[value_key]
                        for row in rows if row["relative_day"] == 0]
        baseline_values = [row[value_key]
                           for row in rows if -14 <= row["relative_day"] <= -8]

        pre = mean_or_none(pre_values)
        post = mean_or_none(post_values)
        event = mean_or_none(event_values)
        baseline = mean_or_none(baseline_values)
        delta_abs = None if pre is None or post is None else post - pre
        delta_pct = None if pre in [
            None, 0] or delta_abs is None else (delta_abs / pre) * 100
        pre_std = std_or_none(pre_values)
        post_std = std_or_none(post_values)
        if delta_abs is None:
            effect_size = None
        else:
            pooled_std = mean_or_none(
                [value for value in [pre_std, post_std] if value is not None])
            effect_size = delta_abs if pooled_std in [
                None, 0] else delta_abs / pooled_std

        pre_n = len(pre_values)
        post_n = len(post_values)
        support_score = None
        if delta_abs is not None:
            support_score = abs(delta_abs) * math.log1p(pre_n + post_n)

        summaries.append(
            {
                "student": student,
                "source": source,
                "baseline_mean": baseline,
                "pre_mean": pre,
                "pre_n": pre_n,
                "pre_std": pre_std,
                "event_day": event,
                "post_mean": post,
                "post_n": post_n,
                "post_std": post_std,
                "delta_abs": delta_abs,
                "delta_pct": delta_pct,
                "effect_size": effect_size,
                "support_score": support_score,
            }
        )
    return summaries


def to_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(converted):
        return None
    return converted


def write_rankings(summary_rows: list[dict], min_days: int = 3) -> None:
    valid_rows = []
    for row in summary_rows:
        pre_n = int(row.get("pre_n", 0) or 0)
        post_n = int(row.get("post_n", 0) or 0)
        delta_abs = to_float_or_none(row.get("delta_abs"))
        if pre_n < min_days or post_n < min_days or delta_abs is None:
            continue
        valid_rows.append(row)

    if not valid_rows:
        return

    by_abs = sorted(
        valid_rows,
        key=lambda item: abs(to_float_or_none(item.get("delta_abs")) or 0.0),
        reverse=True,
    )
    by_pct = sorted(
        [row for row in valid_rows if to_float_or_none(
            row.get("delta_pct")) is not None],
        key=lambda item: abs(to_float_or_none(item.get("delta_pct")) or 0.0),
        reverse=True,
    )
    by_support = sorted(
        [row for row in valid_rows if to_float_or_none(
            row.get("support_score")) is not None],
        key=lambda item: to_float_or_none(item.get("support_score")) or 0.0,
        reverse=True,
    )

    fieldnames = [
        "student",
        "source",
        "pre_n",
        "post_n",
        "pre_mean",
        "post_mean",
        "delta_abs",
        "delta_pct",
        "effect_size",
        "support_score",
    ]

    for filename, rows in [
        ("ranking_delta_abs.csv", by_abs),
        ("ranking_delta_pct.csv", by_pct),
        ("ranking_support_score.csv", by_support),
    ]:
        with (OUTPUT_DIR / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name) for name in fieldnames})


def safe_min_max(values: list[float]) -> tuple[float, float]:
    clean = [value for value in values if value is not None and not (
        isinstance(value, float) and math.isnan(value))]
    if not clean:
        return 0.0, 1.0
    low = min(clean)
    high = max(clean)
    if low == high:
        return low - 1.0, high + 1.0
    return low, high


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0f172a"/>',
        f'<text x="24" y="34" fill="#e2e8f0" font-size="22" font-family="Arial, sans-serif">{title}</text>',
    ]


def svg_footer() -> list[str]:
    return ["</svg>"]


def write_line_chart(windowed: list[dict], value_key: str, title: str, output_name: str) -> None:
    if not windowed:
        return

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in windowed:
        grouped[(row["source"], row["relative_day"])
                ].append(float(row[value_key]))

    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    all_values: list[float] = []
    all_days: list[int] = []
    for (source, relative_day), values in grouped.items():
        value = mean(values)
        series[source].append((relative_day, value))
        all_values.append(value)
        all_days.append(relative_day)

    min_value, max_value = safe_min_max(all_values)
    min_day = min(all_days)
    max_day = max(all_days)
    width, height = 1200, 650
    left, right, top, bottom = 80, 30, 60, 80
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_pos(day: int) -> float:
        if max_day == min_day:
            return left + plot_width / 2
        return left + ((day - min_day) / (max_day - min_day)) * plot_width

    def y_pos(value: float) -> float:
        if max_value == min_value:
            return top + plot_height / 2
        return top + plot_height - ((value - min_value) / (max_value - min_value)) * plot_height

    palette = ["#38bdf8", "#f97316", "#22c55e", "#e879f9", "#facc15"]
    lines = svg_header(width, height, title)

    lines.append(
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#94a3b8" stroke-width="1"/>')
    lines.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#94a3b8" stroke-width="1"/>')
    lines.append(
        f'<text x="{left}" y="{height - 28}" fill="#cbd5e1" font-size="14" font-family="Arial, sans-serif">Días relativos al parcial</text>')
    lines.append(
        f'<text x="18" y="{top + 20}" fill="#cbd5e1" font-size="14" font-family="Arial, sans-serif" transform="rotate(-90 18,{top + 20})">{value_key}</text>')

    for day in range(min_day, max_day + 1):
        x = x_pos(day)
        if day == 0:
            lines.append(
                f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + plot_height}" stroke="#e2e8f0" stroke-width="2" stroke-dasharray="6,4"/>')
        if day % 7 == 0:
            lines.append(
                f'<line x1="{x}" y1="{top + plot_height}" x2="{x}" y2="{top + plot_height + 6}" stroke="#94a3b8" stroke-width="1"/>')
            lines.append(
                f'<text x="{x - 10}" y="{top + plot_height + 24}" fill="#94a3b8" font-size="12" font-family="Arial, sans-serif">{day}</text>')

    tick_count = 5
    for i in range(tick_count + 1):
        value = min_value + (max_value - min_value) * i / tick_count
        y = y_pos(value)
        lines.append(
            f'<line x1="{left - 6}" y1="{y}" x2="{left}" y2="{y}" stroke="#94a3b8" stroke-width="1"/>')
        lines.append(
            f'<text x="18" y="{y + 4}" fill="#94a3b8" font-size="12" font-family="Arial, sans-serif">{value:.2f}</text>')

    for index, (source, points) in enumerate(sorted(series.items())):
        points = sorted(points)
        color = palette[index % len(palette)]
        path_data = " ".join(
            [
                ("M" if pos == 0 else "L") +
                f" {x_pos(day):.2f} {y_pos(value):.2f}"
                for pos, (day, value) in enumerate(points)
            ]
        )
        lines.append(
            f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="3"/>')
        if points:
            day, value = points[-1]
            lines.append(
                f'<circle cx="{x_pos(day):.2f}" cy="{y_pos(value):.2f}" r="4" fill="{color}"/>')
            lines.append(
                f'<text x="{width - right - 140}" y="{80 + index * 22}" fill="{color}" font-size="14" font-family="Arial, sans-serif">{source}</text>')

    lines.extend(svg_footer())
    write_output_text(output_name, "\n".join(lines))


def write_heatmap_svg(windowed: list[dict], value_key: str, title: str, output_name: str) -> None:
    if not windowed:
        return

    matrix: dict[str, dict[int, list[float]]
                 ] = defaultdict(lambda: defaultdict(list))
    for row in windowed:
        matrix[row["student"]][row["relative_day"]].append(
            float(row[value_key]))

    students = sorted(matrix)
    days = sorted({row["relative_day"] for row in windowed})
    if not students or not days:
        return

    all_values = [mean(values) for student_data in matrix.values()
                  for values in student_data.values() if values]
    min_value, max_value = safe_min_max(all_values)
    width, height = 1200, max(420, 40 + 34 * len(students))
    left, top = 150, 60
    cell_w = max(18, (width - left - 40) / max(1, len(days)))
    cell_h = 24

    def color_for(value: float) -> str:
        ratio = 0 if max_value == min_value else (
            value - min_value) / (max_value - min_value)
        ratio = max(0.0, min(1.0, ratio))
        start = (15, 23, 42)
        end = (56, 189, 248)
        r = int(start[0] + (end[0] - start[0]) * ratio)
        g = int(start[1] + (end[1] - start[1]) * ratio)
        b = int(start[2] + (end[2] - start[2]) * ratio)
        return f"rgb({r},{g},{b})"

    lines = svg_header(width, height, title)
    for col_index, day in enumerate(days):
        x = left + col_index * cell_w
        if day % 7 == 0:
            lines.append(
                f'<text x="{x}" y="{top - 10}" fill="#94a3b8" font-size="11" font-family="Arial, sans-serif">{day}</text>')
    lines.append(
        f'<text x="{left}" y="{height - 20}" fill="#cbd5e1" font-size="14" font-family="Arial, sans-serif">Días relativos al parcial</text>')

    for row_index, student in enumerate(students):
        y = top + row_index * cell_h
        lines.append(
            f'<text x="20" y="{y + 16}" fill="#cbd5e1" font-size="12" font-family="Arial, sans-serif">{student}</text>')
        for col_index, day in enumerate(days):
            x = left + col_index * cell_w
            values = matrix[student].get(day)
            if values:
                value = mean(values)
                fill = color_for(value)
            else:
                fill = "#1e293b"
            lines.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w - 1:.2f}" height="{cell_h - 1}" fill="{fill}"/>')

    lines.extend(svg_footer())
    write_output_text(output_name, "\n".join(lines))


def write_top_changes_bar_chart(summary_rows: list[dict], output_name: str, title: str, top_n: int = 10) -> None:
    ranked = [row for row in summary_rows if to_float_or_none(
        row.get("support_score")) is not None]
    ranked = sorted(ranked, key=lambda row: to_float_or_none(
        row.get("support_score")) or 0.0, reverse=True)[:top_n]
    if not ranked:
        return

    width = 1280
    height = max(420, 90 + 42 * len(ranked))
    left = 280
    top = 70
    right = 70
    bottom = 50
    plot_width = width - left - right
    row_h = 34

    max_score = max(abs(to_float_or_none(row.get("support_score")) or 0.0)
                    for row in ranked) or 1.0
    lines = svg_header(width, height, title)
    lines.append(f'<text x="24" y="58" fill="#94a3b8" font-size="13" font-family="Arial, sans-serif">Top cambios ordenados por support_score; rojo = caida, verde = aumento</text>')

    lines.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#334155" stroke-width="1"/>')
    lines.append(
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#334155" stroke-width="1"/>')
    lines.append(
        f'<text x="{left}" y="{height - 18}" fill="#cbd5e1" font-size="12" font-family="Arial, sans-serif">0</text>')

    for idx, row in enumerate(ranked):
        y = top + idx * row_h
        student = str(row.get("student", ""))
        source = str(row.get("source", ""))
        delta = to_float_or_none(row.get("delta_abs")) or 0.0
        support = to_float_or_none(row.get("support_score")) or 0.0
        pct = to_float_or_none(row.get("delta_pct"))
        bar_len = (abs(support) / max_score) * plot_width
        color = "#22c55e" if delta >= 0 else "#ef4444"
        bar_x = left

        lines.append(
            f'<text x="24" y="{y + 22}" fill="#e2e8f0" font-size="16" font-family="Arial, sans-serif">{student}</text>')
        lines.append(
            f'<text x="160" y="{y + 22}" fill="#94a3b8" font-size="13" font-family="Arial, sans-serif">{source}</text>')
        lines.append(
            f'<rect x="{bar_x}" y="{y + 8}" width="{bar_len:.2f}" height="18" rx="8" fill="{color}" opacity="0.85"/>')
        lines.append(f'<text x="{left + min(bar_len + 12, plot_width - 120):.2f}" y="{y + 22}" fill="#e2e8f0" font-size="13" font-family="Arial, sans-serif">{delta:+.1f}' + (
            f' ({pct:+.1f}%)' if pct is not None else '') + f' | score {support:.1f}</text>')

    lines.extend(svg_footer())
    write_output_text(output_name, "\n".join(lines))


def write_student_source_matrix(summary_rows: list[dict], value_key: str, output_name: str, title: str) -> None:
    if not summary_rows:
        return

    students = sorted({str(row.get("student", "")) for row in summary_rows})
    sources = sorted({str(row.get("source", "")) for row in summary_rows})
    lookup = {(str(row.get("student", "")), str(
        row.get("source", ""))): row for row in summary_rows}
    values = [to_float_or_none(row.get(value_key)) for row in summary_rows if to_float_or_none(
        row.get(value_key)) is not None]
    if not values:
        return

    min_value, max_value = min(values), max(values)
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0

    width = max(980, 170 + 130 * len(students))
    height = max(320, 130 + 52 * len(sources))
    left = 170
    top = 70
    cell_w = max(90, (width - left - 40) / max(1, len(students)))
    cell_h = 38

    def color_for(value: float) -> str:
        ratio = 0 if max_value == min_value else (
            value - min_value) / (max_value - min_value)
        ratio = max(0.0, min(1.0, ratio))
        start = (30, 41, 59)
        end = (56, 189, 248)
        r = int(start[0] + (end[0] - start[0]) * ratio)
        g = int(start[1] + (end[1] - start[1]) * ratio)
        b = int(start[2] + (end[2] - start[2]) * ratio)
        return f"rgb({r},{g},{b})"

    lines = svg_header(width, height, title)
    lines.append(f'<text x="24" y="58" fill="#94a3b8" font-size="13" font-family="Arial, sans-serif">Color mas intenso = cambio mayor; fuente por fila y estudiante por columna</text>')

    for col_idx, student in enumerate(students):
        x = left + col_idx * cell_w
        lines.append(
            f'<text x="{x + 8:.2f}" y="{top - 8}" fill="#cbd5e1" font-size="11" font-family="Arial, sans-serif" transform="rotate(-35 {x + 8:.2f},{top - 8})">{student}</text>')

    for row_idx, source in enumerate(sources):
        y = top + row_idx * cell_h
        lines.append(
            f'<text x="24" y="{y + 24}" fill="#cbd5e1" font-size="15" font-family="Arial, sans-serif">{source}</text>')
        for col_idx, student in enumerate(students):
            x = left + col_idx * cell_w
            row = lookup.get((student, source))
            if row is None:
                fill = "#1e293b"
                label = ""
            else:
                value = to_float_or_none(row.get(value_key))
                fill = color_for(value) if value is not None else "#1e293b"
                label = f"{value:+.1f}" if value is not None else ""
            lines.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w - 4:.2f}" height="{cell_h - 4}" rx="6" fill="{fill}" opacity="0.92"/>')
            if label:
                lines.append(
                    f'<text x="{x + cell_w / 2:.2f}" y="{y + 24}" text-anchor="middle" fill="#f8fafc" font-size="12" font-family="Arial, sans-serif">{label}</text>')

    lines.extend(svg_footer())
    write_output_text(output_name, "\n".join(lines))


def write_interactive_dashboard(summary_rows: list[dict], output_name: str) -> None:
    if not summary_rows:
        return

    payload = json.dumps(summary_rows, ensure_ascii=False)
    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>DS_TPO - Dashboard interactivo</title>
    <style>
        :root {{
            color-scheme: dark;
            --bg: #07111f;
            --panel: rgba(15, 23, 42, 0.82);
            --panel-border: rgba(148, 163, 184, 0.25);
            --text: #e2e8f0;
            --muted: #94a3b8;
            --accent: #38bdf8;
            --accent-2: #22c55e;
            --accent-3: #f97316;
            --accent-4: #e879f9;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            min-height: 100vh;
            font-family: Inter, Segoe UI, Arial, sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 30%),
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.12), transparent 28%),
                linear-gradient(180deg, #020617 0%, var(--bg) 100%);
        }}
        .wrap {{ padding: 24px; max-width: 1600px; margin: 0 auto; }}
        .hero {{
            display: grid;
            grid-template-columns: 1.8fr 1fr;
            gap: 18px;
            align-items: stretch;
            margin-bottom: 18px;
        }}
        .card {{
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 20px;
            box-shadow: 0 24px 60px rgba(2, 6, 23, 0.45);
            backdrop-filter: blur(12px);
        }}
        .card.pad {{ padding: 18px; }}
        h1 {{ margin: 0 0 8px; font-size: 34px; line-height: 1.05; }}
        .subtitle {{ color: var(--muted); max-width: 70ch; }}
        .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
        .metric {{ padding: 16px; border-radius: 16px; background: rgba(15, 23, 42, 0.55); border: 1px solid rgba(148, 163, 184, 0.16); }}
        .metric .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
        .metric .value {{ margin-top: 8px; font-size: 28px; font-weight: 700; }}
        .controls {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 18px 0; }}
        .control {{ display: flex; flex-direction: column; gap: 6px; }}
        .control label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }}
        select, input[type="range"] {{ width: 100%; }}
        select {{
            background: #0f172a;
            color: var(--text);
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 12px;
            padding: 10px 12px;
        }}
        .grid {{ display: grid; grid-template-columns: 1.35fr 1fr; gap: 18px; align-items: start; }}
        .cluster-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
        .panel-title {{ margin: 0 0 10px; font-size: 18px; }}
        .panel-desc {{ margin: 0 0 12px; color: var(--muted); font-size: 14px; }}
        canvas {{ width: 100%; height: 560px; display: block; border-radius: 16px; background: linear-gradient(180deg, rgba(15, 23, 42, 0.9), rgba(2, 6, 23, 0.9)); }}
        .small-canvas {{ width: 100%; height: 360px; }}
        .stack {{ display: grid; gap: 18px; }}
        .table-wrap {{ overflow: auto; max-height: 430px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        th, td {{ padding: 10px 8px; border-bottom: 1px solid rgba(148, 163, 184, 0.14); text-align: left; }}
        th {{ position: sticky; top: 0; background: rgba(15, 23, 42, 0.95); color: var(--muted); text-transform: uppercase; font-size: 11px; letter-spacing: 0.08em; }}
        .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
        .chip {{ padding: 6px 10px; border-radius: 999px; background: rgba(56, 189, 248, 0.12); color: #bfdbfe; font-size: 12px; border: 1px solid rgba(56, 189, 248, 0.18); }}
        .small {{ font-size: 12px; color: var(--muted); }}
        @media (max-width: 1100px) {{
            .hero, .grid, .controls, .metrics {{ grid-template-columns: 1fr; }}
            canvas {{ height: 460px; }}
        }}
    </style>
</head>
<body>
<div class="wrap">
    <div class="hero">
        <div class="card pad">
            <h1>Dashboard interactivo de comportamiento alrededor de parciales</h1>
            <p class="subtitle">Filtra por fuente, estudiante y métrica. La vista 3D usa <strong>delta_pct</strong>, <strong>support_score</strong> y <strong>effect_size</strong> para ubicar cada caso en el espacio. Todo se genera sin librerías externas.</p>
            <div class="chips" id="chips"></div>
        </div>
        <div class="card pad">
            <h3 class="panel-title">Lectura rápida</h3>
            <p class="panel-desc">La barra superior cambia con el filtro activo y te muestra el resumen del subconjunto actual.</p>
            <div class="metrics">
                <div class="metric"><div class="label">casos</div><div class="value" id="metric-count">0</div></div>
                <div class="metric"><div class="label">delta pct medio</div><div class="value" id="metric-delta">0</div></div>
                <div class="metric"><div class="label">support medio</div><div class="value" id="metric-support">0</div></div>
                <div class="metric"><div class="label">señal dominante</div><div class="value" id="metric-source">-</div></div>
            </div>
        </div>
    </div>

    <div class="controls card pad">
        <div class="control">
            <label for="sourceFilter">Fuente</label>
            <select id="sourceFilter"></select>
        </div>
        <div class="control">
            <label for="studentFilter">Estudiante</label>
            <select id="studentFilter"></select>
        </div>
        <div class="control">
            <label for="metricFilter">Métrica principal</label>
            <select id="metricFilter">
                <option value="support_score">support_score</option>
                <option value="delta_pct">delta_pct</option>
                <option value="delta_abs">delta_abs</option>
                <option value="effect_size">effect_size</option>
            </select>
        </div>
        <div class="control">
            <label>Rotación 3D</label>
            <div class="small">X <span id="rotXVal">0.75</span> | Y <span id="rotYVal">-0.45</span></div>
            <input id="rotX" type="range" min="-1.5" max="1.5" step="0.01" value="0.75" />
            <input id="rotY" type="range" min="-1.5" max="1.5" step="0.01" value="-0.45" />
        </div>
    </div>

    <div class="grid">
        <div class="card pad">
            <h3 class="panel-title">Vista 3D interactiva</h3>
            <p class="panel-desc">Puntos más altos y hacia adelante indican mayor soporte y efecto; usa los sliders para cambiar la perspectiva.</p>
            <canvas id="scatter3d" width="1200" height="560"></canvas>
            <div class="small" id="scatterHint" style="margin-top:10px;">-</div>
            <div style="margin-top:18px;">
                <h3 class="panel-title">Comparación de clustering</h3>
                <p class="panel-desc">K-Means y jerárquico aglomerativo evaluados con silueta sobre el mismo subconjunto activo.</p>
                <div class="cluster-grid">
                    <div>
                        <div class="small" style="margin-bottom:8px;">K-Means</div>
                        <canvas class="small-canvas" id="clusterKMeans" width="560" height="360"></canvas>
                    </div>
                    <div>
                        <div class="small" style="margin-bottom:8px;">Jerárquico</div>
                        <canvas class="small-canvas" id="clusterAgglomerative" width="560" height="360"></canvas>
                    </div>
                </div>
            </div>
        </div>
        <div class="stack">
            <div class="card pad">
                <h3 class="panel-title">Top cambios del filtro</h3>
                <div class="table-wrap">
                    <table>
                        <thead>
                            <tr><th>Estudiante</th><th>Fuente</th><th>Delta</th><th>%</th><th>Support</th></tr>
                        </thead>
                        <tbody id="topRows"></tbody>
                    </table>
                </div>
            </div>
            <div class="card pad">
                <h3 class="panel-title">Distribución por métrica</h3>
                <p class="panel-desc">Histograma simple del subconjunto activo para ver rápidamente dónde se concentra el cambio.</p>
                <canvas id="histogram" width="1200" height="360"></canvas>
            </div>
        </div>
    </div>
</div>

<script>
const ROWS = {payload};
const COLORS = {{ netflix: '#38bdf8', spotify: '#f97316', steps: '#22c55e', sleep: '#e879f9' }};
const METRIC_LABELS = {{ support_score: 'Support score', delta_pct: 'Delta pct', delta_abs: 'Delta abs', effect_size: 'Effect size' }};

const sourceFilter = document.getElementById('sourceFilter');
const studentFilter = document.getElementById('studentFilter');
const metricFilter = document.getElementById('metricFilter');
const rotX = document.getElementById('rotX');
const rotY = document.getElementById('rotY');
const rotXVal = document.getElementById('rotXVal');
const rotYVal = document.getElementById('rotYVal');
const scatter = document.getElementById('scatter3d');
const clusterKMeans = document.getElementById('clusterKMeans');
const clusterAgglomerative = document.getElementById('clusterAgglomerative');
const histogram = document.getElementById('histogram');
const sctx = scatter.getContext('2d');
const kctx = clusterKMeans.getContext('2d');
const actx = clusterAgglomerative.getContext('2d');
const hctx = histogram.getContext('2d');

const sources = ['all', ...new Set(ROWS.map(r => r.source))];
const students = ['all', ...new Set(ROWS.map(r => r.student))];

function fillSelect(select, values) {{
    select.innerHTML = values.map(v => `<option value="${{v}}">${{v === 'all' ? 'Todos' : v}}</option>`).join('');
}}

function numeric(v) {{ return typeof v === 'number' && Number.isFinite(v); }}

fillSelect(sourceFilter, sources);
fillSelect(studentFilter, students);

function filteredRows() {{
    return ROWS.filter(r =>
        (sourceFilter.value === 'all' || r.source === sourceFilter.value) &&
        (studentFilter.value === 'all' || r.student === studentFilter.value)
    );
}}

function scale(value, min, max) {{
    if (max === min) return 0.5;
    return (value - min) / (max - min);
}}

function project(point, rotX, rotY, dims) {{
    let x = point.x, y = point.y, z = point.z;
    const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
    let x1 = x * cosY + z * sinY;
    let z1 = -x * sinY + z * cosY;
    const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
    let y1 = y * cosX - z1 * sinX;
    let z2 = y * sinX + z1 * cosX;
    const persp = 1 / (1 + z2 * 0.55);
    return {{
        x: dims.cx + x1 * dims.scale * persp,
        y: dims.cy - y1 * dims.scale * persp,
        z: z2
    }};
}}

function drawScatter() {{
    const rows = filteredRows().filter(r => numeric(r.delta_pct) && numeric(r.support_score) && numeric(r.effect_size));
    const w = scatter.width, h = scatter.height;
    sctx.clearRect(0, 0, w, h);
    sctx.fillStyle = '#020617';
    sctx.fillRect(0, 0, w, h);

    if (!rows.length) {{
        sctx.fillStyle = '#e2e8f0';
        sctx.font = '20px Inter, Arial';
        sctx.fillText('No hay datos para el filtro seleccionado', 40, 60);
        document.getElementById('scatterHint').textContent = 'Ajusta los filtros para ver puntos.';
        return;
    }}

    const xs = rows.map(r => r.delta_pct);
    const ys = rows.map(r => r.support_score);
    const zs = rows.map(r => r.effect_size);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const minZ = Math.min(...zs), maxZ = Math.max(...zs);

    const dims = {{ cx: w / 2, cy: h / 2 + 10, scale: Math.min(w, h) * 0.24 }};
    const rX = parseFloat(rotX.value);
    const rY = parseFloat(rotY.value);

    const baseAxes = [
        {{ name: 'delta_pct', from: {{x: -0.5, y: 0, z: 0}}, to: {{x: 0.5, y: 0, z: 0}}, color: '#38bdf8' }},
        {{ name: 'support_score', from: {{x: 0, y: -0.5, z: 0}}, to: {{x: 0, y: 0.5, z: 0}}, color: '#22c55e' }},
        {{ name: 'effect_size', from: {{x: 0, y: 0, z: -0.5}}, to: {{x: 0, y: 0, z: 0.5}}, color: '#f97316' }}
    ];

    sctx.strokeStyle = '#334155';
    sctx.lineWidth = 1;
    sctx.beginPath();
    sctx.moveTo(40, h - 60);
    sctx.lineTo(w - 40, h - 60);
    sctx.stroke();

    baseAxes.forEach(axis => {{
        const a = project(axis.from, rX, rY, dims);
        const b = project(axis.to, rX, rY, dims);
        sctx.strokeStyle = axis.color;
        sctx.lineWidth = 2;
        sctx.beginPath();
        sctx.moveTo(a.x, a.y);
        sctx.lineTo(b.x, b.y);
        sctx.stroke();
    }});

    const points = rows.map(row => {{
        const point = {{
            x: scale(row.delta_pct, minX, maxX) - 0.5,
            y: scale(row.support_score, minY, maxY) - 0.5,
            z: scale(row.effect_size, minZ, maxZ) - 0.5
        }};
        const projected = project(point, rX, rY, dims);
        return {{ row, projected }};
    }}).sort((a, b) => a.projected.z - b.projected.z);

    points.forEach(({{ row, projected }}) => {{
        const color = COLORS[row.source] || '#e2e8f0';
        const radius = 3.5 + Math.min(6, Math.abs(row.delta_pct || 0) / 6);
        sctx.beginPath();
        sctx.fillStyle = color;
        sctx.globalAlpha = 0.9;
        sctx.arc(projected.x, projected.y, radius, 0, Math.PI * 2);
        sctx.fill();
    }});
    sctx.globalAlpha = 1;

    const top = [...rows].sort((a, b) => (b.support_score || 0) - (a.support_score || 0)).slice(0, 8);
    document.getElementById('scatterHint').textContent = `Puntos: ${{rows.length}} | Top: ${{top.map(r => `${{r.student}}/${{r.source}}`).join(', ')}}`;

    const legend = Object.entries(COLORS).map(([k, v]) => `<span class="chip" style="background:${{v}}22;color:${{v}};border-color:${{v}}44">${{k}}</span>`).join('');
    document.getElementById('chips').innerHTML = legend;
}}

function clusterColor(label) {{
    const palette = ['#38bdf8', '#22c55e', '#f97316', '#e879f9', '#facc15', '#fb7185'];
    const index = Math.abs(label || 0) % palette.length;
    return palette[index];
}}

function drawClusterPanel(canvas, ctx, labelKey, title) {{
    const rows = filteredRows().filter(r => numeric(r.delta_pct) && numeric(r.support_score) && r[labelKey] !== undefined && r[labelKey] !== null);
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#020617';
    ctx.fillRect(0, 0, w, h);

    if (!rows.length) {{
        ctx.fillStyle = '#e2e8f0';
        ctx.font = '18px Inter, Arial';
        ctx.fillText('Sin datos para este filtro', 30, 40);
        return;
    }}

    const xs = rows.map(r => r.delta_pct);
    const ys = rows.map(r => r.support_score);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const left = 58, right = 16, top = 28, bottom = 46;
    const plotW = w - left - right;
    const plotH = h - top - bottom;

    function xPos(value) {{ return left + scale(value, minX, maxX) * plotW; }}
    function yPos(value) {{ return top + plotH - scale(value, minY, maxY) * plotH; }}

    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, top + plotH);
    ctx.lineTo(left + plotW, top + plotH);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(left, top);
    ctx.lineTo(left, top + plotH);
    ctx.stroke();

    ctx.fillStyle = '#94a3b8';
    ctx.font = '12px Inter, Arial';
    ctx.fillText('delta_pct', w - 68, top + plotH + 28);
    ctx.save();
    ctx.translate(16, 40);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('support_score', 0, 0);
    ctx.restore();
    ctx.fillStyle = '#e2e8f0';
    ctx.font = '14px Inter, Arial';
    ctx.fillText(title, left, 18);

    rows.sort((a, b) => (a[labelKey] || 0) - (b[labelKey] || 0)).forEach((row) => {{
        const cluster = row[labelKey];
        const color = clusterColor(cluster);
        const x = xPos(row.delta_pct);
        const y = yPos(row.support_score);
        ctx.beginPath();
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.9;
        ctx.arc(x, y, 5.5, 0, Math.PI * 2);
        ctx.fill();
    }});
    ctx.globalAlpha = 1;

    const counts = rows.reduce((acc, row) => (acc[row[labelKey]] = (acc[row[labelKey]] || 0) + 1, acc), {{}});
    const summary = Object.entries(counts).sort((a, b) => Number(a[0]) - Number(b[0])).map(([label, count]) => `${{label}}:${{count}}`).join(' | ');
    ctx.fillStyle = '#94a3b8';
    ctx.font = '12px Inter, Arial';
    ctx.fillText(summary, left, h - 14);
}}

function drawHistogram() {{
    const metric = metricFilter.value;
    const rows = filteredRows().filter(r => numeric(r[metric]));
    const w = histogram.width, h = histogram.height;
    hctx.clearRect(0, 0, w, h);
    hctx.fillStyle = '#020617';
    hctx.fillRect(0, 0, w, h);

    if (!rows.length) return;
    const values = rows.map(r => r[metric]);
    const min = Math.min(...values), max = Math.max(...values);
    const bins = 10;
    const counts = Array(bins).fill(0);
    values.forEach(v => {{
        const idx = Math.min(bins - 1, Math.floor(scale(v, min, max) * bins));
        counts[idx] += 1;
    }});
    const maxCount = Math.max(...counts) || 1;
    const barW = (w - 80) / bins;
    counts.forEach((count, i) => {{
        const bh = (count / maxCount) * (h - 90);
        const x = 40 + i * barW;
        const y = h - 40 - bh;
        hctx.fillStyle = '#38bdf8';
        hctx.fillRect(x + 4, y, barW - 8, bh);
    }});
    hctx.fillStyle = '#94a3b8';
    hctx.font = '13px Inter, Arial';
    hctx.fillText(`Métrica: ${{METRIC_LABELS[metric] || metric}}`, 42, 28);
}}

function updateTable() {{
    const rows = filteredRows().filter(r => numeric(r.support_score));
    const top = rows.sort((a, b) => (b.support_score || 0) - (a.support_score || 0)).slice(0, 10);
    document.getElementById('topRows').innerHTML = top.map(r => `
        <tr>
            <td>${{r.student}}</td>
            <td>${{r.source}}</td>
            <td>${{(r.delta_abs ?? 0).toFixed(1)}}</td>
            <td>${{(r.delta_pct ?? 0).toFixed(1)}}</td>
            <td>${{(r.support_score ?? 0).toFixed(1)}}</td>
        </tr>
    `).join('');
}}

function updateMetrics() {{
    const rows = filteredRows();
    const numericRows = rows.filter(r => numeric(r.delta_pct) && numeric(r.support_score));
    const count = rows.length;
    const avgDelta = numericRows.length ? numericRows.reduce((acc, r) => acc + r.delta_pct, 0) / numericRows.length : 0;
    const avgSupport = numericRows.length ? numericRows.reduce((acc, r) => acc + r.support_score, 0) / numericRows.length : 0;
    const sourceCounts = rows.reduce((acc, r) => (acc[r.source] = (acc[r.source] || 0) + 1, acc), {{}});
    const dominant = Object.entries(sourceCounts).sort((a, b) => b[1] - a[1])[0];
    document.getElementById('metric-count').textContent = count;
    document.getElementById('metric-delta').textContent = avgDelta.toFixed(1) + '%';
    document.getElementById('metric-support').textContent = avgSupport.toFixed(1);
    document.getElementById('metric-source').textContent = dominant ? dominant[0] : '-';
}}

function render() {{
    rotXVal.textContent = parseFloat(rotX.value).toFixed(2);
    rotYVal.textContent = parseFloat(rotY.value).toFixed(2);
    updateMetrics();
    updateTable();
    drawScatter();
    drawClusterPanel(clusterKMeans, kctx, 'kmeans_cluster', 'K-Means');
    drawClusterPanel(clusterAgglomerative, actx, 'agglomerative_cluster', 'Jerárquico');
    drawHistogram();
}}

sourceFilter.addEventListener('change', render);
studentFilter.addEventListener('change', render);
metricFilter.addEventListener('change', render);
rotX.addEventListener('input', render);
rotY.addEventListener('input', render);

render();
</script>
</body>
</html>"""

    write_output_text(output_name, html_content)

def build_normalized_student_vectors(windowed: list[dict], value_key: str) -> tuple[list[str], list[list[float]]]:
    if not windowed:
        return [], []

    by_student: dict[str, dict[int, list[float]]
                     ] = defaultdict(lambda: defaultdict(list))
    all_days = sorted({row["relative_day"] for row in windowed})
    if not all_days:
        return [], []

    for row in windowed:
        by_student[row["student"]][row["relative_day"]].append(
            float(row[value_key]))

    students = sorted(by_student)
    vectors: list[list[float]] = []
    for student in students:
        series = by_student[student]
        vector = [mean(series[day])
                  if day in series else 0.0 for day in all_days]
        vectors.append(vector)

    means = [mean(column) for column in zip(*vectors)]
    stds = []
    for column in zip(*vectors):
        mu = mean(column)
        variance = sum((value - mu) ** 2 for value in column) / \
            max(1, len(column) - 1)
        stds.append(math.sqrt(variance) if variance > 0 else 1.0)

    normalized = [
        [(value - mu) / sigma for value, mu, sigma in zip(vector, means, stds)]
        for vector in vectors
    ]
    return students, normalized


def evaluate_kmeans(normalized: list[list[float]], max_k: int = 6) -> tuple[list[dict], int]:
    n_samples = len(normalized)
    if n_samples < 2:
        return [], 1

    upper_k = min(max_k, n_samples)
    rows: list[dict] = []
    best_k = 1
    best_silhouette: float | None = None

    for k in range(1, upper_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(normalized)
        silhouette: float | None = None
        distinct_labels = len(set(labels))
        if 1 < distinct_labels < n_samples:
            silhouette = float(silhouette_score_local(normalized, labels))
            if best_silhouette is None or silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k
        rows.append(
            {
                "k": k,
                "inertia": float(kmeans.inertia_),
                "silhouette": silhouette,
                "selected": False,
            }
        )

    for row in rows:
        row["selected"] = row["k"] == best_k

    return rows, best_k


def evaluate_agglomerative(normalized: list[list[float]], max_k: int = 6) -> tuple[list[dict], int]:
    n_samples = len(normalized)
    if n_samples < 2:
        return [], 1

    upper_k = min(max_k, n_samples)
    rows: list[dict] = []
    best_k = 1
    best_silhouette: float | None = None

    for k in range(1, upper_k + 1):
        labels = agglomerative_labels(normalized, k)
        silhouette: float | None = None
        distinct_labels = len(set(labels))
        if 1 < distinct_labels < n_samples:
            silhouette = float(silhouette_score_local(normalized, labels))
            if best_silhouette is None or silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k
        rows.append(
            {
                "k": k,
                "inertia": float(cluster_inertia(normalized, labels)),
                "silhouette": silhouette,
                "selected": False,
                "linkage": "average",
            }
        )

    for row in rows:
        row["selected"] = row["k"] == best_k

    return rows, best_k


def open_output_csv(output_name: str):
    output_path = OUTPUT_DIR / output_name
    try:
        return output_path.open("w", encoding="utf-8", newline="")
    except PermissionError:
        output_path = BASE_DIR / output_name
        print(f"No se pudo escribir outputs/{output_name}; se guarda {output_name} en la raiz del proyecto.")
        return output_path.open("w", encoding="utf-8", newline="")


def write_kmeans_evaluation_csv(windowed: list[dict], value_key: str, output_name: str) -> int:
    _, normalized = build_normalized_student_vectors(windowed, value_key)
    rows, best_k = evaluate_kmeans(normalized)

    with open_output_csv(output_name) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["k", "inertia", "silhouette", "selected"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return best_k


def write_agglomerative_evaluation_csv(windowed: list[dict], value_key: str, output_name: str) -> int:
    _, normalized = build_normalized_student_vectors(windowed, value_key)
    rows, best_k = evaluate_agglomerative(normalized)

    with open_output_csv(output_name) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["k", "inertia", "silhouette", "selected", "linkage"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return best_k


def build_cluster_csv(windowed: list[dict], value_key: str, output_name: str, k: int | None = None) -> None:
    students, normalized = build_normalized_student_vectors(windowed, value_key)
    if not students:
        return

    if len(students) < 2:
        labels = [0] * len(students)
    else:
        n_clusters = k or evaluate_kmeans(normalized)[1]
        n_clusters = max(1, min(n_clusters, len(normalized)))
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(normalized)

    with (OUTPUT_DIR / output_name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["student", "cluster"])
        writer.writeheader()
        for student, label in zip(students, labels):
            writer.writerow({"student": student, "cluster": label})


def build_agglomerative_cluster_csv(windowed: list[dict], value_key: str, output_name: str, k: int | None = None) -> None:
    students, normalized = build_normalized_student_vectors(windowed, value_key)
    if not students:
        return

    if len(students) < 2:
        labels = [0] * len(students)
    else:
        n_clusters = k or evaluate_agglomerative(normalized)[1]
        labels = agglomerative_labels(normalized, n_clusters)

    with open_output_csv(output_name) as handle:
        writer = csv.DictWriter(handle, fieldnames=["student", "cluster"])
        writer.writeheader()
        for student, label in zip(students, labels):
            writer.writerow({"student": student, "cluster": label})


def build_clustering_comparison_csv(windowed: list[dict], value_key: str, output_name: str) -> None:
    students, normalized = build_normalized_student_vectors(windowed, value_key)
    if len(students) < 2:
        return

    kmeans_k = evaluate_kmeans(normalized)[1]
    agglomerative_k = evaluate_agglomerative(normalized)[1]
    kmeans = KMeans(
        n_clusters=max(1, min(kmeans_k, len(normalized))),
        random_state=42,
        n_init=10
    )

    kmeans_labels = kmeans.fit_predict(normalized)
    agglomerative = agglomerative_labels(normalized, agglomerative_k)

    with (OUTPUT_DIR / output_name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["student", "kmeans_cluster", "agglomerative_cluster"],
        )
        writer.writeheader()
        for student, kmeans_label, agglomerative_label in zip(students, kmeans_labels, agglomerative):
            writer.writerow(
                {
                    "student": student,
                    "kmeans_cluster": kmeans_label,
                    "agglomerative_cluster": agglomerative_label,
                }
            )


def load_cluster_comparison_map(path: Path) -> dict[str, dict[str, int]]:
    if not path.exists():
        return {}

    mapping: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            student = (row.get("student") or "").strip()
            if not student:
                continue
            mapping[student] = {
                "kmeans_cluster": int(row.get("kmeans_cluster", 0) or 0),
                "agglomerative_cluster": int(row.get("agglomerative_cluster", 0) or 0),
            }
    return mapping


def attach_cluster_labels(summary_rows: list[dict]) -> list[dict]:
    source_to_file = {
        "views": OUTPUT_DIR / "clusters_views_comparison.csv",
        "minutes": OUTPUT_DIR / "clusters_minutes_comparison.csv",
        "steps": OUTPUT_DIR / "clusters_steps_comparison.csv",
    }
    source_maps = {
        source: load_cluster_comparison_map(path)
        for source, path in source_to_file.items()
    }

    enriched_rows: list[dict] = []
    for row in summary_rows:
        enriched = dict(row)
        cluster_map = source_maps.get(row.get("source", ""), {})
        cluster_info = cluster_map.get(str(row.get("student", "")).strip())
        if cluster_info:
            enriched.update(cluster_info)
        else:
            enriched["kmeans_cluster"] = None
            enriched["agglomerative_cluster"] = None
        enriched_rows.append(enriched)
    return enriched_rows


def main() -> None:
    exam_calendar = parse_exam_calendar(
        BASE_DIR / "cronograma_cuatrimestre_2026.csv")
    relevant_dates = build_relevant_dates(exam_calendar, WINDOW_DAYS)

    netflix_records: list[dict] = []
    for path in BASE_DIR.glob("NetflixViewingHistory *.csv"):
        netflix_records.extend(load_netflix_history(path, relevant_dates))
    netflix_windowed = summarize_daily_windows(
        netflix_records, "views", "netflix", exam_calendar)
    netflix_summary = add_before_after_summary(netflix_windowed, "views")

    spotify_records: list[dict] = []
    for path in BASE_DIR.rglob("StreamingHistory*.json"):
        if path.parent.name.startswith("Spotify Account Data"):
            spotify_records.extend(load_spotify_history(path, relevant_dates))
    spotify_windowed = summarize_daily_windows(
        spotify_records, "minutes", "spotify", exam_calendar)
    spotify_summary = add_before_after_summary(spotify_windowed, "minutes")

    data_roots = [BASE_DIR, BASE_DIR / "fit_data"]
    samsung_folders = [
        p
        for root in data_roots
        for p in root.glob("Samsung Health *")
        if p.is_dir()
    ]
    step_records: list[dict] = []
    sleep_records: list[dict] = []
    for folder in samsung_folders:
        step_records.extend(load_samsung_steps(folder, relevant_dates))
        sleep_records.extend(load_samsung_sleep(folder, relevant_dates))

    for root in data_roots:
        for folder in root.glob("Takeout*"):
            if not folder.is_dir():
                continue
            step_records.extend(
                load_takeout_daily_steps(folder, relevant_dates))

    for root in data_roots:
        for folder in root.glob("apple_health_export *"):
            if not folder.is_dir():
                continue
            apple_steps, apple_sleep = load_apple_health_records(
                folder, relevant_dates)
            step_records.extend(apple_steps)
            sleep_records.extend(apple_sleep)

    steps_windowed = summarize_daily_windows(
        step_records, "steps", "steps", exam_calendar)
    steps_summary = add_before_after_summary(steps_windowed, "steps")
    sleep_windowed = summarize_daily_windows(
        sleep_records, "sleep_minutes", "sleep", exam_calendar)
    sleep_summary = add_before_after_summary(sleep_windowed, "sleep_minutes")

    summaries = netflix_summary + spotify_summary + steps_summary + sleep_summary
    if summaries:
        with (OUTPUT_DIR / "before_after_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "student",
                    "source",
                    "baseline_mean",
                    "pre_mean",
                    "pre_n",
                    "pre_std",
                    "event_day",
                    "post_mean",
                    "post_n",
                    "post_std",
                    "delta_abs",
                    "delta_pct",
                    "effect_size",
                    "support_score",
                ],
            )
            writer.writeheader()
            for row in summaries:
                writer.writerow(row)

    write_rankings(summaries)
    write_top_changes_bar_chart(
        summaries, "top_changes_support.svg", "Top cambios con mayor soporte")
    write_student_source_matrix(summaries, "delta_pct", "delta_pct_matrix.svg",
                                "Matriz de cambio porcentual por estudiante y fuente")
    write_student_source_matrix(
        summaries, "support_score", "support_score_matrix.svg", "Matriz de soporte por estudiante y fuente")

    write_line_chart(netflix_windowed, "views",
                     "Netflix: visualizaciones alrededor de parciales", "netflix_relative_day.svg")
    write_line_chart(spotify_windowed, "minutes",
                     "Spotify: minutos alrededor de parciales", "spotify_relative_day.svg")
    write_line_chart(steps_windowed, "steps",
                     "Pasos diarios alrededor de parciales", "steps_relative_day.svg")
    write_line_chart(sleep_windowed, "sleep_minutes",
                     "Sueño alrededor de parciales", "sleep_relative_day.svg")

    write_heatmap_svg(netflix_windowed, "views",
                      "Netflix: mapa por estudiante y día relativo", "netflix_heatmap.svg")
    write_heatmap_svg(spotify_windowed, "minutes",
                      "Spotify: mapa por estudiante y día relativo", "spotify_heatmap.svg")
    write_heatmap_svg(steps_windowed, "steps",
                      "Pasos: mapa por estudiante y día relativo", "steps_heatmap.svg")
    write_heatmap_svg(sleep_windowed, "sleep_minutes",
                      "Sueño: mapa por estudiante y día relativo", "sleep_heatmap.svg")

    netflix_k = write_kmeans_evaluation_csv(
        netflix_windowed, "views", "kmeans_evaluation_views.csv")
    spotify_k = write_kmeans_evaluation_csv(
        spotify_windowed, "minutes", "kmeans_evaluation_minutes.csv")
    steps_k = write_kmeans_evaluation_csv(
        steps_windowed, "steps", "kmeans_evaluation_steps.csv")
    netflix_agglomerative_k = write_agglomerative_evaluation_csv(
        netflix_windowed, "views", "agglomerative_evaluation_views.csv")
    spotify_agglomerative_k = write_agglomerative_evaluation_csv(
        spotify_windowed, "minutes", "agglomerative_evaluation_minutes.csv")
    steps_agglomerative_k = write_agglomerative_evaluation_csv(
        steps_windowed, "steps", "agglomerative_evaluation_steps.csv")

    build_cluster_csv(netflix_windowed, "views",
                      "clusters_views.csv", netflix_k)
    build_cluster_csv(spotify_windowed, "minutes",
                      "clusters_minutes.csv", spotify_k)
    build_cluster_csv(steps_windowed, "steps", "clusters_steps.csv", steps_k)
    build_agglomerative_cluster_csv(
        netflix_windowed, "views", "clusters_views_agglomerative.csv", netflix_agglomerative_k)
    build_agglomerative_cluster_csv(
        spotify_windowed, "minutes", "clusters_minutes_agglomerative.csv", spotify_agglomerative_k)
    build_agglomerative_cluster_csv(
        steps_windowed, "steps", "clusters_steps_agglomerative.csv", steps_agglomerative_k)
    build_clustering_comparison_csv(
        netflix_windowed, "views", "clusters_views_comparison.csv")
    build_clustering_comparison_csv(
        spotify_windowed, "minutes", "clusters_minutes_comparison.csv")
    build_clustering_comparison_csv(
        steps_windowed, "steps", "clusters_steps_comparison.csv")

    enriched_summaries = attach_cluster_labels(summaries)
    write_interactive_dashboard(
        enriched_summaries, "interactive_dashboard.html")

    print("Listo. Revisa outputs/ para CSV y SVG generados.")


if __name__ == "__main__":
    main()
