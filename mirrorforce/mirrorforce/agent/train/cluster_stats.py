"""Append-only terminal observations and bounded per-actor cluster summaries.

No policy imports or RNG use. Zero terminal rewards are deliberately not called
draws: the native API also uses zero for time limits and some failures.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import uuid


def atomic_json(path, value):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".stats-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cluster_name(name):
    return name.split("__", 1)[0] if name.startswith("cluster_") and "__" in name else name


def catalog(manifest, names):
    wanted = set(names)
    decks, families = {}, defaultdict(Counter)
    with Path(manifest).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            name = row["deck_name"]
            if name not in wanted:
                continue
            family = row.get("source_family", "unknown").removeprefix("d-")
            cluster = cluster_name(name)
            decks[name] = {"cluster": cluster, "family": family,
                           "tenpai": bool(re.search(r"(?:^|[-_])tenpai(?:[-_]|$)", family)),
                           "pure_tenpai": family == "tenpai-dragon"}
            families[cluster][family] += 1
    if set(decks) != wanted:
        raise ValueError("Training deck catalog and manifest do not match")
    return {"decks": decks, "clusters": {
        key: {"label": counter.most_common(1)[0][0], "families": dict(counter),
              "deck_count": sum(counter.values())} for key, counter in families.items()},
        "tenpai_definition": "source_family contains the token tenpai",
        "pure_tenpai_definition": "source_family is d-tenpai-dragon",
        "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest()}


def start_run(root, names, manifest, checkpoint, offset, actors):
    root = Path(root)
    folder = root / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
                              + uuid.uuid4().hex[:10])
    folder.mkdir(parents=True, mode=0o700)
    mapping = catalog(manifest, names)
    atomic_json(folder / "catalog.json", mapping)
    atomic_json(folder / "run.json", {
        "schema": 1, "id": folder.name, "started_at": time.time(), "pid": os.getpid(),
        "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip()
                if Path("/proc/sys/kernel/random/boot_id").exists() else None,
        "checkpoint": str(checkpoint), "checkpoint_sha256":
            hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None,
        "start_step": offset, "expected_actors": actors, "deck_count": len(names),
        "cluster_count": len(mapping["clusters"]), "source": "training_selfplay",
        "optimizer_restored": False, "win_rate_definition": "win / (win + loss)",
        "zero_reward_definition": "no_result; draw and truncation cannot always be distinguished"})
    return str(folder)


def classify(acting_seat, reward, terminated, truncated, selfplay=True):
    if not selfplay or acting_seat not in (0, 1) or not math.isfinite(reward):
        return None, "invalid_observation"
    if not terminated or truncated:
        return None, "truncated_or_not_terminal"
    if reward == 0:
        return None, "zero_reward_ambiguous"
    return (acting_seat if reward > 0 else 1 - acting_seat), "decisive"


class Recorder:
    def __init__(self, folder, actor, names):
        self.folder, self.actor, self.names = Path(folder), actor, tuple(names)
        self.mapping = json.loads((self.folder / "catalog.json").read_text(encoding="utf-8"))["decks"]
        self.counts = Counter()
        self.clusters = defaultdict(Counter)
        self.tenpai = {"all": Counter(), "pure": Counter()}
        self.sequence, self.part, self.latest_step = 0, 0, 0
        self.pending = []
        self.last_snapshot = 0
        self.error = None
        self.stream = None
        self.open_part()
        self.flush(force=True)

    def open_part(self):
        self.stream = (self.folder / f"actor-{self.actor:03d}-{self.part:05d}.jsonl").open(
            "x", encoding="utf-8", buffering=1)

    def record(self, *, env, step, deck_indices, acting_seat, reward,
               terminated, truncated, length, selfplay=True, win_reason=0):
        if self.error:
            return
        self.sequence += 1
        self.latest_step = int(step)
        names = [self.names[int(index)] for index in deck_indices
                 if 0 <= int(index) < len(self.names)]
        winner, kind = classify(int(acting_seat), float(reward), bool(terminated),
                                bool(truncated), bool(selfplay))
        if len(names) != 2:
            winner, kind = None, "invalid_deck_indices"
        row = {"seq": self.sequence, "actor": self.actor, "env": int(env), "step": int(step),
               "at": time.time(), "decks": names, "acting_seat": int(acting_seat),
               "reward": float(reward) if math.isfinite(reward) else None,
               "terminated": bool(terminated), "truncated": bool(truncated), "length": int(length),
               "winner": winner, "kind": kind, "win_reason": int(win_reason)}
        self.pending.append(json.dumps(row, separators=(",", ":")) + "\n")
        self.counts["episodes"] += 1
        self.counts[kind] += 1
        if winner is None:
            return
        entries = [self.mapping[name] for name in names]
        for seat, entry in enumerate(entries):
            outcome = "win" if seat == winner else "loss"
            counter = self.clusters[entry["cluster"]]
            counter[outcome] += 1
            counter[f"seat{seat}_{outcome}"] += 1
            if entries[0]["cluster"] == entries[1]["cluster"]:
                counter["mirror_" + outcome] += 1
            for group, key in (("all", "tenpai"), ("pure", "pure_tenpai")):
                if entry[key]:
                    self.tenpai[group][outcome] += 1
                    if not entries[1 - seat][key]:
                        self.tenpai[group]["vs_other_" + outcome] += 1

    def flush(self, force=False):
        if self.error:
            return
        try:
            if self.pending:
                self.stream.writelines(self.pending)
                self.pending.clear()
            now = time.time()
            if not force and now - self.last_snapshot < 10:
                return
            self.stream.flush()
            os.fsync(self.stream.fileno())
            atomic_json(self.folder / f"actor-{self.actor:03d}.json", {
                "schema": 1, "actor": self.actor, "updated_at": now, "step": self.latest_step,
                "sequence": self.sequence, "counts": dict(self.counts),
                "clusters": {key: dict(value) for key, value in self.clusters.items()},
                "tenpai": {key: dict(value) for key, value in self.tenpai.items()}})
            self.last_snapshot = now
            if self.stream.tell() >= 64 * 1024 * 1024:
                self.stream.close()
                self.part += 1
                self.open_part()
        except OSError as error:
            self.error = type(error).__name__
            self.pending.clear()
            print(f"CLUSTER_STATS_ERROR actor={self.actor} error={self.error}", flush=True)
            try:
                atomic_json(self.folder / f"actor-{self.actor:03d}-error.json",
                            {"at": time.time(), "error": self.error})
            except OSError:
                pass

    def close(self):
        self.flush(force=True)
        if self.stream:
            self.stream.close()


def rates(counter):
    row = dict(counter)
    wins, losses = row.get("win", 0), row.get("loss", 0)
    n = wins + losses
    row.update(win=wins, loss=losses, games=n, win_rate=wins / n if n else None)
    if n:
        z = 1.959963984540054
        p = wins / n
        center = p + z * z / (2 * n)
        spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        row["wilson95"] = [(center - spread) / (1 + z * z / n),
                           (center + spread) / (1 + z * z / n)]
    return row


def report(root, minimum=30):
    if minimum < 1:
        raise ValueError("Minimum games must be positive")
    counts, clusters = Counter(), defaultdict(Counter)
    tenpai = {"all": Counter(), "pure": Counter()}
    runs, labels, manifest_sha = [], {}, None
    for folder in sorted((Path(root) / "runs").glob("*")):
        if not (folder / "run.json").exists():
            continue
        metadata = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        mapping = json.loads((folder / "catalog.json").read_text(encoding="utf-8"))
        if manifest_sha is not None and manifest_sha != mapping["manifest_sha256"]:
            raise ValueError("Training manifest changed across runs")
        manifest_sha = mapping["manifest_sha256"]
        for key, info in mapping["clusters"].items():
            if key in labels and labels[key] != info:
                raise ValueError("Cluster catalog changed across runs")
            labels[key] = info
        actors = []
        for actor in sorted(folder.glob("actor-???.json")):
            data = json.loads(actor.read_text(encoding="utf-8"))
            actors.append({key: data[key] for key in ("actor", "updated_at", "step", "sequence")})
            counts.update(data["counts"])
            for key, value in data["clusters"].items():
                clusters[key].update(value)
            for key in tenpai:
                tenpai[key].update(data["tenpai"][key])
        runs.append({**metadata, "actors": actors,
                     "errors": [path.name for path in folder.glob("actor-*-error.json")]})
    rows = [{"cluster": key, **labels[key], **rates(clusters[key])} for key in labels]
    ranked = sorted((row for row in rows if row["games"] >= minimum),
                    key=lambda row: (-row["win_rate"], -row["games"], row["cluster"]))
    return {"captured_at": time.time(), "minimum_games": minimum, "counts": dict(counts),
            "runs": runs, "cluster_count": len(labels),
            "clusters_seen": sum(bool(row["games"]) for row in rows),
            "eligible_clusters": len(ranked), "top10": ranked[:10],
            "tenpai": {key: rates(value) for key, value in tenpai.items()}, "clusters": rows,
            "note": "New training self-play only; no-result episodes excluded; mirrors include both seats."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--minimum-games", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = report(args.root, args.minimum_games)
    if args.output:
        atomic_json(args.output, value)
    print(json.dumps({key: item for key, item in value.items() if key != "clusters"}, ensure_ascii=False))
