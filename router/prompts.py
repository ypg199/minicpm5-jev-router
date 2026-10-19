"""Prompt builders. These must stay byte-for-byte identical to the ones used in training and evaluation
(notebooks/02_build_dataset.ipynb and notebooks/01_compare_models.ipynb), so no torch here and no changes
without retraining.

`render` and `option_text` are taken from Kev (https://github.com/jaredpalmer/kev, kev/api.py, Apache-2.0).
"""
import json

LETTERS = [chr(ord("A") + i) for i in range(26)]
PREFILL = "Answer:"
DESC_CHARS = 300

DECISION_SYSTEM = ("You are a decision model. Read the text, then answer the question by choosing exactly one of the "
                   "lettered options. Reply with the option letter only.")
ROUTE_SYSTEM = ("You are a tool-routing decision model. Read the conversation and the lettered options, "
                "then choose the ONE option that should happen next: the single tool to call, or NONE if "
                "no listed tool can fulfil the request. Reply with the option letter only.")

# The four next steps the model was trained on (same wording as When2Call / Kev's devtools suite).
ACTIONS = {"tool_call": "Call one of the available tools",
           "request_for_info": "Ask the user for information that a tool call needs",
           "cannot_answer": "Say it cannot help with the tools available",
           "direct": "Answer directly without calling a tool"}


def render(v, indent=0):
    """Flatten str | object | array into text the model sees. Field names are kept as labels."""
    pad = "  " * indent
    if v is None: return ""
    if isinstance(v, (str, int, float, bool)): return str(v)
    if isinstance(v, list): return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}" for k, x in v.items())


def option_text(name, desc):
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def decision_options(q):
    """[(answer key, text shown)] in display order for a Jev-style question."""
    if q["type"] == "choice":
        crit = q["criteria"]
        if isinstance(crit, list):
            crit = {str(c): None for c in crit}
        return [(k, option_text(k, d)) for k, d in crit.items()]
    if q["type"] == "noul":
        crit = q.get("criteria") or {}
        return [("true", option_text("Yes", crit.get("true"))), ("false", option_text("No", crit.get("false")))]
    if q["type"] == "score":
        return [(str(i), render(c)) for i, c in enumerate(q["criteria"])]   # lowest -> highest
    raise ValueError(f"unknown question type {q['type']!r}")


def decision_user(state, q, opts):
    instructions = render(q.get("instructions")) or "Choose the best option."
    lines = [f"({LETTERS[i]}) {text}" for i, (_, text) in enumerate(opts)]
    if q["type"] == "score":
        lines.append("(The options are ordered from lowest to highest.)")
    return (f"Text:\n{render(state)}\n\nQuestion: {instructions}\n\nOptions:\n" + "\n".join(lines)
            + "\n\nReply with the letter of the best option only.")


def action_state(tools, user_message):
    """State for the four-way "what should the assistant do next?" question. Tools are JSON-schema dicts."""
    available = [t if isinstance(t, str) else json.dumps(t) for t in tools] if tools else list("none")
    return {"available_tools": available, "user_message": user_message}


def action_question():
    return {"type": "choice", "instructions": "What should the assistant do next?", "criteria": dict(ACTIONS)}


def render_tool(fn):
    params = fn.get("parameters") or {}
    props = params.get("properties") or {}
    required = set(params.get("required") or [])
    args = []
    for name, spec in props.items():
        typ = spec.get("type", "any") if isinstance(spec, dict) else "any"
        args.append(f"{name}{'' if name in required else '?'}: {typ}")
    desc = " ".join(str(fn.get("description", "")).split())
    if len(desc) > DESC_CHARS:
        desc = desc[:DESC_CHARS].rstrip() + "..."
    return f"{fn['name']}({', '.join(args)}) - {desc}"


def route_user(messages, tools):
    """Pick-a-tool-or-NONE question. messages: [{"role", "content"}]; tools: function schemas (name, description, parameters)."""
    conversation = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    lines = [f"({LETTERS[i]}) {render_tool(t)}" for i, t in enumerate(tools)]
    lines.append(f"({LETTERS[len(tools)]}) NONE - none of the tools above fits this request")
    return ("Conversation:\n" + conversation + "\n\nOptions:\n" + "\n".join(lines)
            + "\n\nWhich option should happen next? Reply with the letter only.")


def chat_prompt(tok, system, user):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
    try:
        text = tok.apply_chat_template(msgs, **kwargs)
    except Exception:  # template without a system role
        text = tok.apply_chat_template([{"role": "user", "content": system + "\n\n" + user}], **kwargs)
    return text + PREFILL
