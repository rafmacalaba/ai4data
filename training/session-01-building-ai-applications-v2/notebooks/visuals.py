"""Helpers that draw tables and views from the notebooks' own data.

workshop.py imports these functions, so notebook 01 calls them as w.show_claim_check and w.show_prompt.

    table_html(headers, rows)  a small table, drawn as a light card
    claim_check_html(rows)   a table of what the model said against what the data says
    show_claim_check(rows)   the same, displayed
    ranking_html(rows, k, legend)  search results as a ranked list of bars, with a line after the first k
    search_comparison_html(question, columns, checks)  one search on two versions of a collection, side by side
    prompt_html(messages)    the messages of one call, colour-coded: instructions, context, question
    show_prompt(messages)    the same, displayed
"""

from html import escape

from IPython.display import HTML, display


CARD = "background:#ffffff;color:#16315a;border:1px solid #e1e6ee;border-radius:10px;padding:10px 14px"  # a light card, whatever the theme


def claim_check_html(rows, source="none"):
    """Each row is (claim, what the model said, what the data says, verdict). Verdict "close" is green."""
    colours = {"close": ("#dff3e7", "#1f6b43"), "none": ("#fbe0dd", "#8e2b21")}
    cell = 'style="padding:8px 10px;border-bottom:1px solid #e5e9f0;vertical-align:top"'

    def pill(text):
        bg, fg = colours.get(text, ("#fdebd0", "#8a4b0a"))
        return f'<span style="display:inline-block;padding:2px 10px;border-radius:999px;background:{bg};color:{fg};font-weight:600;font-size:12.5px">{escape(text)}</span>'

    body = "".join(
        f'<tr><td {cell}><b>{escape(claim)}</b></td><td {cell}>{escape(said)}</td><td {cell}>{escape(data)}</td>'
        f'<td {cell}>{pill(verdict)}</td><td {cell}>{pill(source)}</td></tr>'
        for claim, said, data, verdict in rows
    )
    close = sum(1 for row in rows if row[3] == "close")
    summary = f"{close} of {len(rows)} claims land close to the data."
    if source == "none":
        summary += " The others are marked above. None names a source, so a reader cannot tell which is which."
    head = "".join(f'<th style="padding:8px 10px">{h}</th>' for h in ["Claim", "What the model said", "What the data says", "Verdict", "Source named"])
    return (
        f'<div style="{CARD}">'
        f'<table style="border-collapse:collapse;font:14px Helvetica,Arial,sans-serif;width:100%;color:#16315a">'
        f'<thead><tr style="text-align:left;background:#f3f4f6">{head}</tr></thead><tbody>{body}</tbody></table>'
        f'<p style="font:600 14px Helvetica,Arial,sans-serif;margin:10px 2px 2px;color:#16315a">{escape(summary)}</p></div>'
    )


def table_html(headers, rows, source=""):
    """A small table in a light card. Cells that look like numbers are right-aligned."""
    import re

    cell = "padding:7px 12px;border-bottom:1px solid #e5e9f0;vertical-align:top"

    def align(text):
        return "right" if re.fullmatch(r"[0-9,.-]+", str(text)) else "left"

    def mark(text):
        """Green for a hit or a full count, red for a miss or a short count; nothing for other cells."""
        text = str(text)
        count = re.fullmatch(r"(\d+)/(\d+)", text)
        good = text == "hit" or (count and count.group(1) == count.group(2))
        bad = text == "miss" or (count and count.group(1) != count.group(2))
        return ";font-weight:600;color:" + ("#1f6b43" if good else "#8e2b21") if good or bad else ""

    head = "".join(f'<th style="padding:7px 12px;text-align:{"right" if i >= 2 else "left"}">{escape(str(h))}</th>' for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(f'<td style="{cell};text-align:{align(c)}{mark(c)}">{escape(str(c))}</td>' for c in row) + "</tr>"
        for row in rows
    )
    note = f'<p style="font:12.5px Helvetica,Arial,sans-serif;margin:8px 2px 2px;color:#6b7280">{escape(source)}</p>' if source else ""
    return (
        f'<div style="{CARD};display:inline-block"><table style="border-collapse:collapse;font:14px Helvetica,Arial,sans-serif;color:#16315a">'
        f'<thead><tr style="background:#f3f4f6">{head}</tr></thead><tbody>{body}</tbody></table>{note}</div>'
    )


def show_table(headers, rows, source=""):
    display(HTML(table_html(headers, rows, source)))


def show_claim_check(rows, source="none"):
    display(HTML(claim_check_html(rows, source)))


def ranking_html(rows, k, legend, question=""):
    """Search results as a ranked list of bars, with a line after the first k.

    rows: [(label, score, kind), ...], best first. legend: {kind: (name, colour)}.
    """
    top = max(score for _, score, _ in rows)
    key = "".join(
        f'<span style="display:inline-block;margin-right:16px"><span style="display:inline-block;width:12px;height:12px;border-radius:3px;'
        f'background:{colour};margin-right:5px;vertical-align:-1px"></span>{escape(name)}</span>'
        for name, colour in legend.values()
    )
    lines = ""
    for rank, (label, score, kind) in enumerate(rows, 1):
        colour = legend[kind][1]
        faded = "" if rank <= k else ";opacity:.4"
        lines += (
            f'<div style="display:flex;align-items:center;gap:10px;margin:3px 0">'
            f'<div style="width:22px;text-align:right;font:12px Helvetica,Arial,sans-serif;color:#6b7280">{rank}</div>'
            f'<div style="width:490px;flex:none;font:12.5px Menlo,Consolas,monospace;color:#16315a;white-space:nowrap{faded}">{escape(label)}</div>'
            f'<div style="flex:1;background:#f3f4f6;border-radius:5px;height:18px"><div style="width:{score / top * 100:.1f}%;height:18px;'
            f'background:{colour};border-radius:5px{faded}"></div></div>'
            f'<div style="width:44px;font:600 12.5px Helvetica,Arial,sans-serif;color:#16315a">{score:.3f}</div></div>'
        )
        if rank == k:
            lines += (
                f'<div style="margin:6px 0 6px 32px;border-top:2px dashed #9a3a2f;color:#9a3a2f;font:600 12.5px Helvetica,Arial,sans-serif;padding-top:3px">'
                f'only the top {k} go to the model</div>'
            )
    head = f'<div style="font:600 13.5px Helvetica,Arial,sans-serif;margin-bottom:6px">Question: {escape(question)}</div>' if question else ""
    return f'<div style="{CARD}">{head}<div style="font:12.5px Helvetica,Arial,sans-serif;margin-bottom:8px;color:#374151">{key}</div>{lines}</div>'


def show_ranking(rows, k, legend, question=""):
    display(HTML(ranking_html(rows, k, legend, question)))


def search_comparison_html(question, columns, checks):
    """Side-by-side results of one search on two versions of a collection.

    Each column is a dict: title, tone ("bad" or "good"), hits [(citation, text), ...], and found [one bool
    per check]. `checks` names the figures the question needs.
    """
    colours = {"bad": "#9a3a2f", "good": "#2f8f5b"}

    def excerpt(text):
        flat = " ".join(text.split())
        if len(flat) <= 170:
            return flat
        return flat[:100].rsplit(" ", 1)[0] + " … " + flat[-60:].split(" ", 1)[1]

    blocks = ""
    for column in columns:
        colour = colours.get(column.get("tone"), "#2f5f9e")
        hits = "".join(
            f'<div style="margin:8px 0;padding:8px 10px;background:#f6f7f9;border-radius:8px">'
            f'<div style="font:700 12.5px Menlo,Consolas,monospace;color:{colour}">{escape(label)}</div>'
            f'<div style="font:12.5px Helvetica,Arial,sans-serif;color:#374151;margin-top:3px">{escape(excerpt(text))}</div></div>'
            for label, text in column["hits"]
        )
        ticks = "".join(
            f'<div style="font:600 13px Helvetica,Arial,sans-serif;color:{"#1f6b43" if ok else "#8e2b21"}">'
            f'{"✓ found" if ok else "✗ missing"} {escape(check)}</div>'
            for check, ok in zip(checks, column["found"])
        )
        blocks += (
            f'<div style="flex:1;min-width:300px;border:1.5px solid {colour};border-radius:10px;padding:10px 12px">'
            f'<div style="font:700 14px Helvetica,Arial,sans-serif;color:{colour}">{escape(column["title"])}</div>{hits}'
            f'<div style="margin-top:8px">{ticks}</div></div>'
        )
    return (
        f'<div style="{CARD}"><div style="font:600 13.5px Helvetica,Arial,sans-serif;margin-bottom:10px">Question: {escape(question)}</div>'
        f'<div style="display:flex;gap:14px;flex-wrap:wrap">{blocks}</div></div>'
    )


def show_search_comparison(question, columns, checks):
    display(HTML(search_comparison_html(question, columns, checks)))


def prompt_html(messages):
    """Messages as produced by prompt.format_messages: a system message, then a human message of the form
    "CONTEXT:\\n...\\n\\nQUESTION: ..."."""
    colours = {"instructions": "#2f5f9e", "context": "#2f8f5b", "question": "#d9822b"}

    def block(kind, meta, text):
        c = colours[kind]
        return (f'<div style="border-left:5px solid {c};background:{c}1a;padding:8px 12px;margin:8px 0;border-radius:0 8px 8px 0">'
                f'<div style="font:700 12.5px Helvetica,Arial,sans-serif;color:{c};margin-bottom:4px">{kind.capitalize()} '
                f'<span style="font-weight:400;color:#6b7280">· {escape(meta)}</span></div>'
                f'<pre style="margin:0;white-space:pre-wrap;font:12.5px Menlo,Consolas,monospace;color:#16315a">{escape(text)}</pre></div>')

    system = next(m for m in messages if m.type == "system").content
    human = next(m for m in messages if m.type == "human").content
    head, _, question = human.rpartition("\n\nQUESTION: ")
    context = head[len("CONTEXT:\n"):] if head.startswith("CONTEXT:\n") else head
    shown = context if len(context) <= 330 else context[:330].rsplit(" ", 1)[0] + " …"
    return (f'<div style="{CARD}">' + block("instructions", f"the system message, {len(system):,} characters", system)
            + block("context", f"{len(context):,} characters" + ("" if shown == context else ", the first lines shown"), shown)
            + block("question", "from the user", question) + "</div>")


def show_prompt(messages):
    display(HTML(prompt_html(messages)))
