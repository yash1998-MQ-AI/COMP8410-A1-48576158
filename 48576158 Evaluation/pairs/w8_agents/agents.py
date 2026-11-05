"""Week 8 agents: one rule-based baseline and two VLM agents built on the same Qwen2.5-VL-7B family.

  rule        reads the DOM, no model (the lecture's "simple rule-based agent", made slightly more general)
  qwen_dom    Qwen2.5-VL-7B-Instruct, text only: task + DOM element list in, JSON action out
  uitars_vis  UI-TARS-1.5-7B (a GUI-trained Qwen2.5-VL-7B derivative), screenshot in, click/type text out

Both model agents are greedy (temperature 0) and see the same task text. The ONLY intended differences are the
observation (DOM text vs pixels) and the GUI training of UI-TARS; the report states this as the controlled contrast.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from w8_common import HF_ID, SHOT_H, SHOT_W


@dataclass
class Act:
    kind: str                      # click_element | click_xy | type | focus_type | finish | invalid
    ref: int | None = None
    x: float | None = None         # page coordinates (task area, pixels)
    y: float | None = None
    text: str = ""
    note: str = ""

    def brief(self) -> str:
        if self.kind == "click_element":
            return f"click_element({self.ref})"
        if self.kind == "click_xy":
            return f"click({self.x:.0f},{self.y:.0f})"
        if self.kind == "focus_type":
            return f"type({self.ref},{self.text!r})"
        if self.kind == "type":
            return f"type({self.text!r})"
        return self.kind


# ---------------------------------------------------------------- observations
def dom_lines(obs: dict, limit: int = 70) -> list[str]:
    lines = []
    for e in obs["dom_elements"]:
        tag = str(e["tag"]).lower()
        text = str(e["text"]).strip()
        value = str(e.get("value", "")).strip()
        interactive = tag in {"button", "input", "textarea", "select", "option", "a", "label"}
        if not (text or value or interactive):
            continue
        if tag in {"body", "html", "div"} and not text:
            continue
        extra = f' value="{value}"' if value else ""
        lines.append(f'[{int(e["ref"])}] <{tag}> "{text[:60]}"{extra}')
        if len(lines) >= limit:
            break
    return lines


# ---------------------------------------------------------------- parsers (unit-tested)
def parse_json_action(raw: str) -> Act:
    """Qwen DOM agent output: {"action": "click"|"type"|"finish", "ref": 12, "text": "abc"}."""
    m = re.search(r"\{.*?\}", raw, re.S)
    if not m:
        return Act("invalid", note="no json")
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return Act("invalid", note="bad json")
    a = str(d.get("action", "")).lower()
    try:
        if a in {"click", "click_element"}:
            return Act("click_element", ref=int(d["ref"]))
        if a in {"type", "type_text", "enter"}:
            return Act("focus_type", ref=int(d["ref"]), text=str(d.get("text", "")))
    except (KeyError, ValueError, TypeError):
        return Act("invalid", note="missing field")
    if a in {"finish", "done", "finished"}:
        return Act("finish")
    return Act("invalid", note=f"unknown action {a!r}")


_NUM = r"(-?\d+(?:\.\d+)?)"


def parse_uitars(raw: str, sx: float = SHOT_W / 160, sy: float = SHOT_H / 210) -> Act:
    """UI-TARS output: 'Thought: ...\\nAction: click(start_box='(x,y)')'. Coordinates are in the pixels of the screenshot we
    sent (we send a multiple-of-28 size so the model's processor does not resize); divide by the upscale factor."""
    m = re.search(r"Action:\s*(.*)", raw, re.S)
    body = (m.group(1) if m else raw).strip()
    if re.match(r"(finished|wait)\b", body):
        return Act("finish")
    c = re.match(r"(?:left_single|click)\(\s*start_box\s*=\s*['\"]?(?:<\|box_start\|>)?\s*\(?\s*" + _NUM + r"\s*,\s*" + _NUM, body)
    if c:
        return Act("click_xy", x=float(c.group(1)) / sx, y=float(c.group(2)) / sy)
    t = re.match(r"type\(\s*content\s*=\s*(['\"])(.*?)\1\s*\)", body, re.S)
    if t:
        return Act("type", text=t.group(2).rstrip("\n"))
    return Act("invalid", note=f"unparsed: {body[:60]}")


def to_env_action(env_u: Any, act: Act, width: float = 160, height: float = 210):
    """Map an Act to a MiniWoB action. env_u is env.unwrapped (has create_action)."""
    import numpy as np

    if act.kind == "click_element":
        return env_u.create_action("CLICK_ELEMENT", ref=np.array(act.ref, dtype=np.int64))
    if act.kind == "click_xy":
        x = float(min(max(act.x, 0), width - 1))
        y = float(min(max(act.y, 0), height - 1))
        return env_u.create_action("CLICK_COORDS", coords=np.array([x, y], dtype=np.float32))
    if act.kind == "focus_type":
        return env_u.create_action("FOCUS_ELEMENT_AND_TYPE_TEXT", ref=np.array(act.ref, dtype=np.int64), text=act.text)
    if act.kind == "type":
        return env_u.create_action("TYPE_TEXT", text=act.text)
    return env_u.create_action("NONE")


# ---------------------------------------------------------------- agents
class RuleAgent:
    """Finds a quoted/UPPER-CASE target in the utterance and acts on the matching DOM element; types quoted text.
    Deliberately narrow: it is the lecture's rule-based baseline, a floor that shows what the tasks need."""

    name = "rule"

    def reset(self, task: str):
        self.done_types = 0

    @staticmethod
    def _targets(utt: str) -> list[str]:
        q = re.findall(r"\"([^\"]+)\"|'([^']+)'", utt)
        out = [a or b for a, b in q]
        out += re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", utt)
        return [t for t in out if t]

    def act(self, obs: dict, step: int):
        utt = str(obs["utterance"])
        els = obs["dom_elements"]
        low = utt.lower()
        typed = [e for e in els if str(e["tag"]).lower() in {"input", "textarea"}]
        if ("type" in low or "enter" in low) and typed and self.done_types < 1:
            tg = self._targets(utt)
            self.done_types += 1
            if tg:
                return Act("focus_type", ref=int(typed[0]["ref"]), text=tg[0]), "rule:type", 0
        for t in self._targets(utt):
            for e in els:
                if str(e["text"]).strip().lower() == t.lower() and str(e["tag"]).lower() in {"button", "a", "option", "label", "input", "span", "div"}:
                    return Act("click_element", ref=int(e["ref"])), f"rule:{t}", 0
        for e in els:
            if str(e["tag"]).lower() == "button":
                return Act("click_element", ref=int(e["ref"])), "rule:first-button", 0
        return Act("finish"), "rule:nothing", 0


QWEN_SYSTEM = (
    "You control a small web page by reading its DOM. Reply with ONE JSON object and nothing else.\n"
    'Actions: {"action":"click","ref":<int>}  {"action":"type","ref":<int>,"text":"<string>"}  {"action":"finish"}\n'
    "Use the [ref] numbers shown in the element list. Finish only when the task is done."
)


class _HFBase:
    def __init__(self, key: str):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(HF_ID[key])
        self.model = AutoModelForImageTextToText.from_pretrained(HF_ID[key], torch_dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()

    def _gen(self, messages, images=None, max_new_tokens=160):
        torch = self.torch
        text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=images, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[0, inputs["input_ids"].shape[1]:]
        return self.proc.decode(new, skip_special_tokens=True), int(new.shape[0])


class QwenDomAgent(_HFBase):
    name = "qwen_dom"

    def __init__(self):
        super().__init__("qwen_dom")

    def reset(self, task: str):
        self.history: list[str] = []

    def act(self, obs: dict, step: int):
        dom = "\n".join(dom_lines(obs))
        hist = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(self.history)) or "(none)"
        user = f"Task: {obs['utterance']}\n\nPage elements:\n{dom}\n\nActions so far:\n{hist}\n\nNext action (JSON):"
        msgs = [{"role": "system", "content": [{"type": "text", "text": QWEN_SYSTEM}]},
                {"role": "user", "content": [{"type": "text", "text": user}]}]
        raw, n = self._gen(msgs)
        act = parse_json_action(raw)
        self.history.append(act.brief())
        return act, raw, n


UITARS_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
Thought: ...
Action: ...

## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.

## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""


class UITarsAgent(_HFBase):
    name = "uitars_vis"

    def __init__(self):
        super().__init__("uitars_vis")

    def reset(self, task: str):
        self.history: list[str] = []

    def act(self, obs: dict, step: int):
        from PIL import Image

        shot = Image.fromarray(obs["screenshot"]).convert("RGB").resize((SHOT_W, SHOT_H), Image.BICUBIC)
        instr = str(obs["utterance"])
        if self.history:
            instr += ("\n\nActions you have already performed, in order (each one took effect): " + "; ".join(self.history)
                      + ". Note: clicking a text field focuses it even though no caret is visible, so do not repeat a click that is "
                      "already listed; carry out the next step of the task (e.g. type the text, then submit).")
        msgs = [{"role": "user", "content": [{"type": "text", "text": UITARS_PROMPT.format(instruction=instr)},
                                             {"type": "image", "image": shot}]}]
        raw, n = self._gen(msgs, images=[shot])
        act = parse_uitars(raw)
        m = re.search(r"Action:\s*(.+)", raw)
        self.history.append((m.group(1).strip() if m else act.brief())[:120])
        return act, raw, n


def make_agent(name: str):
    if name == "rule":
        return RuleAgent()
    if name == "qwen_dom":
        return QwenDomAgent()
    if name == "uitars_vis":
        return UITarsAgent()
    raise ValueError(name)
