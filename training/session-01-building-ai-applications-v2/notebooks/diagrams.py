"""Architecture diagrams for the session 01 notebooks.

build_notebooks.py draws each diagram with render_svg() and embeds it in the notebook's markdown as an
image, so the notebooks themselves import nothing from here. The usage below is for trying a diagram out.

    import diagrams
    diagrams.show("prompt")                      # notebook 01
    diagrams.show("knowledge")                   # notebook 02
    diagrams.show("retrieval")                   # notebook 03
    diagrams.show("tools")                       # notebook 04
    diagrams.show("agent")                       # notebook 05
    diagrams.show("verification")                # notebook 06
    diagrams.show("architecture", stage=2)       # the six components, one highlighted

Two engines draw the same diagram from the same description:

    "svg"        the default: a small layout written here; needs nothing beyond IPython
    "graphviz"   the Graphviz `dot` program through the `graphviz` package
    "auto"       Graphviz when `dot` is installed, otherwise "svg"

Drawing with "svg" gives the same picture everywhere, including the outputs
saved in the notebook.

Either way the result is an SVG displayed in the notebook, so it is saved with
the notebook's outputs and shows on GitHub without running anything.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from html import escape

from IPython.display import SVG, display

FILL, EDGE, TEXT = "#e8f0fa", "#2f5f9e", "#16315a"
HOT_FILL, HOT_TEXT = "#2f5f9e", "#ffffff"
DIM_FILL, DIM_EDGE, DIM_TEXT = "#f3f4f6", "#8a94a3", "#6b7280"
FONT = "Helvetica, Arial, sans-serif"
DASH = ' stroke-dasharray="5 4"'


ALT = {
    "prompt": "Diagram: the question and the context feed the prompt, which goes to the model and produces the answer.",
    "knowledge": "Diagram: documents, metadata, chunks, embeddings, vector store, then retrieval and the model in notebook 03.",
    "retrieval": "Diagram: the question goes to a retriever, which returns top-k passages for the prompt, then the model and an answer with citations.",
    "tools": "Diagram: the question goes to the model, which calls the get_indicator tool, receives data, and answers; the same tool is served over MCP.",
    "agent": "Diagram: the question goes to the model, which chooses between search_documents and get_indicator, then answers.",
    "verification": "Diagram: the answer with claim tags goes to pcn-core for verified, flagged or unmarked marks, and to the evaluation suite for a table per question."
}


@dataclass
class Flow:
    """Nodes as (id, label, state) with state "normal", "hot", or "dim"; edges as (from, to)."""

    nodes: list[tuple[str, str, str]]
    edges: list[tuple[str, str]]
    note: dict[str, str] = field(default_factory=dict)

    def state(self, node_id: str) -> str:
        return next(s for i, _, s in self.nodes if i == node_id)


def flow(name: str, stage: int | None = None) -> Flow:
    if name == "prompt":
        return Flow(
            [
                ("q", "question", "normal"),
                ("c", "context", "normal"),
                ("p", "prompt\ninstructions + context + question", "hot"),
                ("m", "model", "normal"),
                ("a", "answer", "normal"),
            ],
            [("q", "p"), ("c", "p"), ("p", "m"), ("m", "a")],
        )
    if name == "knowledge":
        return Flow(
            [
                ("d", "documents", "normal"),
                ("md", "metadata", "normal"),
                ("ch", "chunks", "normal"),
                ("e", "embeddings", "normal"),
                ("v", "vector store", "hot"),
                ("r", "retrieval + model\n(notebook 03)", "dim"),
            ],
            [("d", "md"), ("md", "ch"), ("ch", "e"), ("e", "v"), ("v", "r")],
        )
    if name == "retrieval":
        return Flow(
            [
                ("q", "question", "normal"),
                ("r", "retriever", "hot"),
                ("p", "top-k passages", "hot"),
                ("pr", "prompt\ninstructions + passages + question", "normal"),
                ("m", "model", "normal"),
                ("a", "answer\nwith citations", "normal"),
            ],
            [("q", "r"), ("r", "p"), ("p", "pr"), ("pr", "m"), ("m", "a")],
        )
    if name == "tools":
        return Flow(
            [
                ("q", "question", "normal"),
                ("m", "model", "normal"),
                ("t", "tool call\nget_indicator", "hot"),
                ("d", "data", "hot"),
                ("a", "answer, or an object\nthat matches a schema", "normal"),
                ("mcp", "the same tool over MCP\nany MCP client", "hot"),
            ],
            [("q", "m"), ("m", "t"), ("t", "d"), ("d", "a"), ("t", "mcp")],
        )
    if name == "agent":
        return Flow(
            [
                ("q", "question", "normal"),
                ("m1", "model", "hot"),
                ("s", "search_documents(query)", "hot"),
                ("g", "get_indicator(indicator, country)", "hot"),
                ("m2", "model", "normal"),
                ("a", "answer", "normal"),
            ],
            [("q", "m1"), ("m1", "s"), ("m1", "g"), ("s", "m2"), ("g", "m2"), ("m2", "a")],
        )
    if name == "verification":
        return Flow(
            [
                ("q", "question", "normal"),
                ("ag", "agent with tools", "normal"),
                ("an", "answer with\n<claim> tags", "normal"),
                ("pcn", "pcn-core", "hot"),
                ("mk", "verified \u00b7 flagged \u00b7 unmarked", "hot"),
                ("ev", "evaluation suite", "hot"),
                ("tb", "table per question", "hot"),
            ],
            [("q", "ag"), ("ag", "an"), ("an", "pcn"), ("pcn", "mk"), ("an", "ev"), ("ev", "tb")],
        )
    if name == "architecture":
        if stage is None:
            raise ValueError("architecture needs stage=1..6")
        labels = [
            "Instructions\n& context",
            "Prepared\nknowledge",
            "Retrieval",
            "Tools &\nstructured output",
            "Workflow\nor agent",
            "Verification\n& evaluation",
        ]
        nodes = [(f"s{i}", t, "hot" if i == stage else ("dim" if i > stage else "normal")) for i, t in enumerate(labels, 1)]
        return Flow(nodes, [(f"s{i}", f"s{i + 1}") for i in range(1, 6)])
    raise ValueError(f"unknown diagram {name!r}")


# ---- engine: svg -----------------------------------------------------------

def _ranks(f: Flow) -> dict[str, int]:
    rank = {i: 0 for i, _, _ in f.nodes}
    for _ in f.nodes:
        for a, b in f.edges:
            rank[b] = max(rank[b], rank[a] + 1)
    return rank


def _size(label: str) -> tuple[int, int]:
    lines = label.split("\n")
    return max(112, int(max(len(t) for t in lines) * 7.6) + 30), 24 + 19 * len(lines)


def render_svg(f: Flow) -> str:
    rank = _ranks(f)
    size = {i: _size(t) for i, t, _ in f.nodes}
    columns: dict[int, list[str]] = {}
    for i, _, _ in f.nodes:
        columns.setdefault(rank[i], []).append(i)
    gap_x, gap_y, pad = 44, 26, 18
    col_w = {r: max(size[i][0] for i in ids) for r, ids in columns.items()}
    col_h = {r: sum(size[i][1] for i in ids) + gap_y * (len(ids) - 1) for r, ids in columns.items()}
    height = max(col_h.values()) + 2 * pad
    pos, x = {}, pad
    for r in sorted(columns):
        y = (height - col_h[r]) / 2
        for i in columns[r]:
            w, h = size[i]
            pos[i] = (x + (col_w[r] - w) / 2, y, w, h)
            y += h + gap_y
        x += col_w[r] + gap_x
    width = x - gap_x + pad

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="max-width:100%;height:auto" font-family="{FONT}">',
        f'<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="{EDGE}"/></marker>'
        f'<marker id="ahd" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="{DIM_EDGE}"/></marker></defs>',
        f'<rect width="{width}" height="{height}" rx="10" fill="#ffffff"/>',
    ]
    for a, b in f.edges:
        ax, ay, aw, ah = pos[a]
        bx, by, bw, bh = pos[b]
        dim = f.state(b) == "dim"
        out.append(
            f'<line x1="{ax + aw}" y1="{ay + ah / 2}" x2="{bx - 2}" y2="{by + bh / 2}" stroke="{DIM_EDGE if dim else EDGE}" stroke-width="1.8"'
            f'{DASH if dim else ""} marker-end="url(#{"ahd" if dim else "ah"})"/>'
        )
    for i, label, state in f.nodes:
        x, y, w, h = pos[i]
        fill, edge, color = {"hot": (HOT_FILL, HOT_FILL, HOT_TEXT), "dim": (DIM_FILL, DIM_EDGE, DIM_TEXT)}.get(state, (FILL, EDGE, TEXT))
        lines = label.split("\n")
        tspans = "".join(
            f'<tspan x="{x + w / 2}" y="{y + h / 2 + (k - (len(lines) - 1) / 2) * 19 + 5}">{escape(t)}</tspan>' for k, t in enumerate(lines)
        )
        out.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{edge}" stroke-width="1.6"'
            f'{DASH if state == "dim" else ""}/>'
            f'<text text-anchor="middle" font-size="14" fill="{color}" font-weight="{600 if state == "hot" else 500}">{tspans}</text>'
        )
    out.append("</svg>")
    return "".join(out)


# ---- engine: graphviz ------------------------------------------------------

def to_dot(f: Flow) -> str:
    lines = [
        "digraph {",
        '  rankdir=LR; bgcolor="white"; pad=0.2; nodesep=0.35; ranksep=0.45;',
        f'  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=13 fillcolor="{FILL}" color="{EDGE}" fontcolor="{TEXT}" margin="0.2,0.12"];',
        f'  edge [color="{EDGE}" arrowsize=0.8];',
    ]
    for i, label, state in f.nodes:
        extra = {
            "hot": f' fillcolor="{HOT_FILL}" color="{HOT_FILL}" fontcolor="{HOT_TEXT}"',
            "dim": f' style="rounded,dashed,filled" fillcolor="{DIM_FILL}" color="{DIM_EDGE}" fontcolor="{DIM_TEXT}"',
        }.get(state, "")
        text = label.replace("\n", "\\n")
        lines.append(f'  {i} [label="{text}"{extra}];')
    for a, b in f.edges:
        style = f' [style=dashed color="{DIM_EDGE}"]' if f.state(b) == "dim" else ""
        lines.append(f"  {a} -> {b}{style};")
    lines.append("}")
    return "\n".join(lines)


def graphviz_available() -> bool:
    try:
        import graphviz  # noqa: F401
    except ImportError:
        return False
    return shutil.which("dot") is not None


def render_graphviz(f: Flow) -> str:
    import graphviz

    return graphviz.Source(to_dot(f)).pipe(format="svg", encoding="utf-8")


# ---- display ---------------------------------------------------------------

def show(name: str, stage: int | None = None, engine: str = "svg") -> None:
    """Display a diagram in the notebook. `engine` is "svg" (default), "graphviz", or "auto"."""
    f = flow(name, stage)
    if engine not in ("auto", "svg", "graphviz"):
        raise ValueError("engine must be 'auto', 'svg', or 'graphviz'")
    use_graphviz = engine != "svg" and graphviz_available()
    if engine == "graphviz" and not use_graphviz:
        print("Graphviz is not available here (needs the `graphviz` package and the `dot` program); drawing with the built-in SVG engine.")
    display(SVG(data=render_graphviz(f) if use_graphviz else render_svg(f)))
