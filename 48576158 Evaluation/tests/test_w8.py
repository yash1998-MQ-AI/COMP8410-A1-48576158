"""Week 8 tests: parsers, action mapping, rule agent, the episode loop on a fake environment, report tables. No GPU, no browser."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pairs" / "w8_agents"))
sys.path.insert(0, str(ROOT))

import agents as A  # noqa: E402
import report_w8 as R  # noqa: E402
import run_w8 as RUN  # noqa: E402
from a1eval import Record, append  # noqa: E402
import w8_common as C  # noqa: E402


def test_tasks_exist_in_config():
    assert len(C.TASKS) == 8 and set(C.INJECT_TASKS) <= set(C.TASKS)
    assert (C.SHOT_W % 28, C.SHOT_H % 28) == (0, 0)


def test_parse_json_action():
    assert A.parse_json_action('sure {"action":"click","ref":7}').ref == 7
    a = A.parse_json_action('{"action":"type","ref":3,"text":"hello"}')
    assert (a.kind, a.ref, a.text) == ("focus_type", 3, "hello")
    assert A.parse_json_action('{"action":"finish"}').kind == "finish"
    assert A.parse_json_action("no json here").kind == "invalid"
    assert A.parse_json_action('{"action":"click"}').kind == "invalid"


def test_parse_uitars_click_scales_back():
    a = A.parse_uitars("Thought: click it.\nAction: click(start_box='<|box_start|>(280,364)<|box_end|>')")
    assert a.kind == "click_xy" and a.x == pytest.approx(80.0) and a.y == pytest.approx(105.0)
    b = A.parse_uitars("Action: click(start_box='(140,182)')")
    assert b.x == pytest.approx(40.0)


def test_parse_uitars_type_and_finish():
    assert A.parse_uitars("Action: type(content='abc\\n')").text.startswith("abc")
    assert A.parse_uitars("Action: finished(content='done')").kind == "finish"
    assert A.parse_uitars("Action: scroll(foo)").kind == "invalid"


class FakeUnwrapped:
    def create_action(self, kind, **kw):
        return {"type": kind, **kw}


class FakeEnv:
    """Click the button labelled ONE. Reward 1 for the right ref, terminates after any click."""
    unwrapped = FakeUnwrapped()

    def reset(self, seed=None):
        self.obs = {"utterance": 'Click on the button "ONE".', "screenshot": np.zeros((210, 160, 3), np.uint8),
                    "dom_elements": [{"ref": 1, "tag": "button", "text": "TWO", "value": "", "id": "", "left": np.array([5.]), "top": np.array([60.]), "width": np.array([40.]), "height": np.array([20.])},
                                     {"ref": 2, "tag": "button", "text": "ONE", "value": "", "id": "", "left": np.array([60.]), "top": np.array([60.]), "width": np.array([40.]), "height": np.array([20.])}]}
        return self.obs, {}

    def step(self, action):
        ok = action["type"] == "CLICK_ELEMENT" and int(action["ref"]) == 2
        return self.obs, 1.0 if ok else -1.0, True, False, {}

    def close(self):
        pass


def test_to_env_action_mapping():
    u = FakeUnwrapped()
    assert A.to_env_action(u, A.Act("click_element", ref=5))["type"] == "CLICK_ELEMENT"
    c = A.to_env_action(u, A.Act("click_xy", x=999, y=-4))
    assert c["type"] == "CLICK_COORDS" and c["coords"][0] == 159 and c["coords"][1] == 0   # clipped into the task area
    assert A.to_env_action(u, A.Act("invalid"))["type"] == "NONE"


def test_rule_agent_solves_fake_task():
    res = RUN.run_episode(FakeEnv(), A.RuleAgent(), "fake", 0)
    assert res["success"] == 1.0 and res["n_steps"] == 1 and res["invalid"] == 0


class WrongAgent:
    def reset(self, task): pass
    def act(self, obs, step): return A.Act("click_element", ref=1), "raw", 12


def test_failure_and_token_counts():
    res = RUN.run_episode(FakeEnv(), WrongAgent(), "fake", 0)
    assert res["success"] == 0.0 and res["gen_tokens"] == 12


def test_injection_hit_detection():
    obs = {"dom_elements": [{"ref": 9, "tag": "button", "text": C.INJ_BUTTON_TEXT, "id": C.INJ_ID,
                             "left": np.array([10.]), "top": np.array([190.]), "width": np.array([50.]), "height": np.array([10.])}]}
    assert RUN.hit_injection(A.Act("click_element", ref=9), obs)
    assert RUN.hit_injection(A.Act("click_xy", x=20, y=195), obs)
    assert not RUN.hit_injection(A.Act("click_xy", x=20, y=60), obs)
    assert not RUN.hit_injection(A.Act("click_element", ref=1), {"dom_elements": []})


def test_dom_lines_skip_empty_containers():
    obs = {"dom_elements": [{"ref": 1, "tag": "div", "text": "", "value": ""}, {"ref": 2, "tag": "button", "text": "Go", "value": ""}]}
    assert A.dom_lines(obs) == ['[2] <button> "Go"']


def test_report_tables(tmp_path):
    p = tmp_path / "w8.jsonl"
    for a, wins in (("qwen_dom", [1, 1, 1, 0]), ("uitars_vis", [1, 0, 0, 0])):
        for i, w in enumerate(wins):
            for seed in (0, 1):
                append(p, Record("w8", a, f"clean/t{i}", "success", float(w), seed))
    df = R.frame(p)
    st = R.success_table(df)
    q = st[st.agent == "qwen_dom"].iloc[0]
    assert q.episodes == 8 and q.successes == 6 and q.ci_lo < q.success_rate < q.ci_hi
    assert R.paired(df)["diff_a_minus_b"] == pytest.approx(-0.5)
