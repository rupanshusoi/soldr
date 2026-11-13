from math import ceil
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mocker import MockerLatency
from workload import Trace, earlier_shared, sessions_of

# Per-request worker, times (ms) and reused tokens for each scheduler
Schedules = dict[str, list[dict[str, Any]]]

# Requests that reuse over a block less than the largest hit an earlier request of their session offers: a routing
# miss if no request with that prefix ran on the request's worker, else an eviction miss
def misses(trace: Trace, recs: list[dict[str, Any]]) -> dict[str, list[int]]:
    miss: dict[str, list[int]] = {"routing": [], "eviction": []}
    for j, (shared, q) in enumerate(zip(earlier_shared(trace), recs)):
        best = max(shared.values(), default=0)
        if best - q["reused"] > trace.blk_size:
            local = any(s == best and recs[i]["worker"] == q["worker"] for i, s in shared.items())
            miss["eviction" if local else "routing"].append(j)
    return miss

class Plotter:
    def __init__(self, trace: Trace, schedules: Schedules, prefill_latency: MockerLatency) -> None:
        self.trace = trace
        self.schedules = schedules
        self.latency = prefill_latency
        self.session = sessions_of(trace.requests)
        # Lanes in order of first appearance in the trace, which is by estimated release
        self.lane = {s: k for k, s in enumerate(dict.fromkeys(self.session))}
        self.workers = 1 + max(q["worker"] for recs in schedules.values() for q in recs)
        self.misses = {name: misses(trace, recs) for name, recs in schedules.items()}

    def cache_lanes(self, path: str, dpi: int = 300) -> None:
        reqs, session, lane, workers = self.trace.requests, self.session, self.lane, self.workers
        colors = plt.get_cmap("tab20").colors # type: ignore[attr-defined]
        plt.style.use("dark_background")
        fig, axes = plt.subplots(len(self.schedules), workers, sharex=True, sharey=True, squeeze=False,
                                 figsize=(4.5 * workers, 3.2 * len(self.schedules)))
        for row, (name, recs) in zip(axes, self.schedules.items()):
            miss = self.misses[name]
            peaks = []
            for w, ax in enumerate(row):
                mine = [j for j, q in enumerate(recs) if q["worker"] == w]
                # Breakpoints of the stack: a band changes height at a request's start and ends at the end of its last
                last_end: dict[int, float] = {}
                for j in mine:
                    last_end[session[j]] = max(last_end.get(session[j], 0), recs[j]["end"])
                times = sorted({recs[j]["start"] for j in mine} | set(last_end.values()))
                index = {t: k for k, t in enumerate(times)}
                # held[s][k] is the share of capacity session s holds from times[k] to times[k + 1]
                held = {s: [0.0] * len(times) for s in last_end}
                for j in sorted(mine, key=lambda j: recs[j]["start"]):
                    size = (reqs[j].isl + reqs[j].osl) / self.latency.kv_capacity
                    for k in range(index[recs[j]["start"]], index[last_end[session[j]]]):
                        held[session[j]][k] = size
                order = sorted(held, key=lambda s: lane[s])
                if order:
                    ax.stackplot([t / 1000 for t in times], [[100 * h for h in held[s]] for s in order], step="post",
                                 lw=0, colors=[colors[lane[s] % len(colors)] for s in order])
                peaks.append(max((sum(held[s][k] for s in order) for k in range(len(times))), default=0))

                # A miss is drawn in the middle of the band of its session
                for kind, style in [("routing", dict(marker="x", color="#ff4d4d")),
                                    ("eviction", dict(marker="o", facecolors="none", edgecolors="white"))]:
                    js = [j for j in miss[kind] if recs[j]["worker"] == w]
                    ys = []
                    for j in js:
                        k = index[recs[j]["start"]]
                        below = sum(held[s][k] for s in order if lane[s] < lane[session[j]])
                        ys.append(100 * (below + held[session[j]][k] / 2))
                    ax.scatter([recs[j]["start"] / 1000 for j in js], ys, s=22, lw=1.1, zorder=3, **style)

                ax.axhline(100, color="white", ls="--", lw=0.8)
                ax.set_title(f"worker {w}: {len(order)} sessions, peak {peaks[-1]:.0%}", fontsize=9)
                ax.grid(axis="y", color="0.25", lw=0.5)
                ax.set_axisbelow(True)
            row[0].set_ylabel("session caches")
            label = row[0].annotate(name, xy=(0, 1), xycoords="axes fraction", xytext=(0, 22),
                                    textcoords="offset points", fontsize=11, fontweight="bold")
            row[0].annotate(f"{len(miss['eviction'])} eviction "
                            f"misses, {len(miss['routing'])} routing misses", xy=(1, 0), xycoords=label, xytext=(8, 0),
                            textcoords="offset points", va="bottom", fontsize=10).set_in_layout(False)
            label.set_in_layout(False)
        for ax in axes[-1]:
            ax.set_xlabel("time (s)")
        handles = [Patch(color=colors[k % len(colors)], label=f"session {k}") for k in range(len(lane))] + [
            Line2D([], [], color="white", ls="--", lw=0.8, label="KV capacity"),
            Line2D([], [], color="#ff4d4d", marker="x", ls="", label="routing miss"),
            Line2D([], [], markerfacecolor="none", markeredgecolor="white", marker="o", ls="", label="eviction miss")]
        rows = ceil(len(handles) / 12)
        fig.legend(handles=handles, loc="upper center", ncol=ceil(len(handles) / rows), fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 1 - (0.1 + 0.22 * rows) / fig.get_figheight()), h_pad=3.5)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        print("Wrote", path)

    def swimlane(self, path: str, dpi: int = 300) -> None:
        reqs, lane, workers = self.trace.requests, self.lane, self.workers
        y = [lane[s] for s in self.session]
        achievable = [max(shared.values(), default=0) for shared in earlier_shared(self.trace)]

        colors = plt.get_cmap("tab10" if workers <= 10 else "tab20").colors # type: ignore[attr-defined]
        plt.style.use("dark_background")
        fig, axes = plt.subplots(len(self.schedules), 1, sharex=True, sharey=True, squeeze=False,
                                 figsize=(16, len(self.schedules) * max(3, 0.6 + 0.14 * len(lane))))
        for ax, (name, recs) in zip(axes[:, 0], self.schedules.items()):
            sec = lambda key: [q[key] / 1000 for q in recs]
            release, start, prefill_end, end = sec("release"), sec("start"), sec("prefill_end"), sec("end")
            color = [colors[q["worker"] % len(colors)] for q in recs]
            width = lambda a, b: [t1 - t0 for t0, t1 in zip(a, b)]
            ax.barh(y, width(release, start), left=release, height=0.25, color="0.6", zorder=1)
            ax.barh(y, width(prefill_end, end), left=prefill_end, height=0.8, color=color, alpha=0.35, lw=0, zorder=2)
            ax.barh(y, width(start, prefill_end), left=start, height=0.8, color=color, lw=0, zorder=3)

            miss = self.misses[name]
            switches = sum(u.prev is not None and recs[u.prev]["worker"] != q["worker"] for u, q in zip(reqs, recs))
            ax.scatter([start[j] for j in miss["routing"]], [y[j] for j in miss["routing"]], marker="x", s=30, lw=1.2,
                       color="#ff4d4d", zorder=4)
            ax.scatter([start[j] for j in miss["eviction"]], [y[j] for j in miss["eviction"]], marker="o", s=30, lw=1.2,
                       facecolors="none", edgecolors="white", zorder=4)
            reuse = sum(q["reused"] for q in recs) / max(1, sum(achievable))
            ttft = sum(width(release, prefill_end)) * 1000 / self.latency.decode_ms / len(recs)
            title = ax.set_title(name, fontsize=10, fontweight="bold", loc="left")
            ax.annotate(f"mean TTFT {ttft:.1f} steps, {reuse:.0%} of achievable reuse, {len(miss['routing'])} routing "
                        f"misses, {len(miss['eviction'])} eviction misses, {switches} worker switches", xy=(1, 0),
                        xycoords=title, xytext=(8, 0), textcoords="offset points", va="bottom", fontsize=10)
            ax.set_ylabel("session")
            ax.grid(axis="x", color="0.25", lw=0.5, zorder=0)
            ax.set_axisbelow(True)
        ax = axes[-1, 0]
        ax.set_ylim(len(lane) - 0.4, -0.6)
        # Lanes are numbered from 0, so no ticks in the padding above the first lane
        step = max(1, ceil(len(lane) / 20))
        ax.set_yticks(range(0, len(lane), step))
        ax.set_xlabel("time (s)")
        handles = [Patch(color=colors[w % len(colors)], label=f"worker {w}") for w in range(workers)] + [
            Patch(color="0.8", label="prefill"), Patch(color="0.8", alpha=0.35, lw=0, label="decode"),
            Line2D([], [], color="0.6", lw=3, label="waiting to start"),
            Line2D([], [], color="#ff4d4d", marker="x", ls="", label="routing miss"),
            Line2D([], [], markerfacecolor="none", markeredgecolor="white", marker="o", ls="", label="eviction miss")]
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_figheight()))
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        print("Wrote", path)
