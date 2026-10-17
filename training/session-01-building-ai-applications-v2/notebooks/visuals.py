"""Helpers that draw tables and views from the notebooks' own data.

build_notebooks.py puts the source of these functions into a hidden helper cell of notebook 01, so the
notebook runs without importing this module (the Colab setup clones a repository that does not contain it).
Each function therefore uses only `escape`, `HTML` and `display`, which the helper cell imports.

    claim_check_html(rows)   a table of what the model said against what the data says
    show_claim_check(rows)   the same, displayed
    prompt_html(messages)    the messages of one call, colour-coded: instructions, context, question
    show_prompt(messages)    the same, displayed
"""

from html import escape

from IPython.display import HTML, display


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
        f'<table style="border-collapse:collapse;font:14px Helvetica,Arial,sans-serif;width:100%">'
        f'<thead><tr style="text-align:left;background:#f3f4f6">{head}</tr></thead><tbody>{body}</tbody></table>'
        f'<p style="font:600 14px Helvetica,Arial,sans-serif;margin:10px 2px">{escape(summary)}</p>'
    )


def show_claim_check(rows, source="none"):
    display(HTML(claim_check_html(rows, source)))


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
    return (block("instructions", f"the system message, {len(system):,} characters", system)
            + block("context", f"{len(context):,} characters" + ("" if shown == context else ", the first lines shown"), shown)
            + block("question", "from the user", question))


def show_prompt(messages):
    display(HTML(prompt_html(messages)))
