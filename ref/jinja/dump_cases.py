#!/usr/bin/env python3
"""Record what Jinja2 makes of the templates golem/jinja interprets.

golem renders a model's chat template with an interpreter of its own, and this
script is what that interpreter is held to: Jinja2 itself, in the environment
transformers builds for apply_chat_template — sandboxed, trim_blocks and
lstrip_blocks on, loop controls, the generation tag, and transformers' own
tojson, raise_exception and strftime_now.

Two corpora. The first is small templates, one construct each, rendered with
the variables beside them. The second is every template in
testdata/jinja/templates, which were read out of the GGUF files golem runs,
rendered over a set of conversations in the shape transformers hands them
over: content as a string or as a list of parts, tool calls with their
arguments as mappings.

    python3 ref/jinja/dump_cases.py testdata/jinja

Run by hand. Never at test time: the fixture is committed.
"""

import json
import os
import sys
from datetime import datetime

import jinja2
from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment


class Generation(Extension):
    """transformers' {% generation %}, which renders its body."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.CallBlock(self.call_method("_body"), [], [], body).set_lineno(lineno)

    def _body(self, caller):
        return caller()


def environment():
    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True,
        extensions=[Generation, jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda f: datetime.now().strftime(f)
    return env


def render(env, source, variables):
    try:
        return {"rendered": env.from_string(source).render(**variables)}
    except Exception as e:  # a template may fail in Python, not only by raising
        return {"error": f"{type(e).__name__}: {e}"}


# One construct each. The variables go through JSON, so what the Go side reads
# is exactly what was rendered here.
SNIPPETS = [
    ("{{ x }}", {"x": "plain"}),
    ("a  {{- x -}}  b", {"x": 1}),
    ("  {% if true %}\n  a\n  {% endif %}\n  {{ 'b' }}\n{# c #}\nd {%+ if 1 %}e{% endif %}", {}),
    ("{% for i in xs %}\n  {{ i }}\n{% endfor %}\n", {"xs": [1, 2]}),
    ("{%- for i in xs -%}\n  {{ i }}\n{%- endfor %}", {"xs": [1, 2]}),
    ("x {%- if 1 %} y {% endif -%} z", {}),
    ("{# a\ncomment #}\nafter", {}),
    ("{% raw %}{{ not }}{% endraw %}", {}),
    ("{{ 1.0 }} {{ 1e20 }} {{ 0.0001 }} {{ 1/3 }} {{ 10/2 }} {{ 2.5e-7 }} {{ 123456789012345678.0 }}", {}),
    ("{{ [1, 'a', None, True, (1,), 'it\\'s', \"q\\\"\"] }} {{ {'a': \"b'c\"} }}", {}),
    ("{{ 'x' ~ none ~ 1.5 ~ true }}", {}),
    ("{{ 7 // 2 }} {{ -7 // 2 }} {{ -7 % 3 }} {{ 7 % -3 }} {{ 2 ** 10 }} {{ 'ab' * 2 }} {{ [1] + [2] }} {{ 1 + 2.5 }}", {}),
    ("{{ -1 | abs }} {{ -x | abs }}", {"x": 3}),
    ("{{ 'a' if false }}|{{ 'a' if true else 'b' }}|{{ 'x' if 0 else 'y' if 1 else 'z' }}", {}),
    ("{{ x | length > 1 }} {{ not x | length }} {{ x is string and x | length == 2 }}", {"x": "ab"}),
    ("{{ 'héllo'[1:3] }} {{ 'héllo' | length }} {{ 'héllo'[::-1] }} {{ 'héllo'[-2:] }}", {}),
    ("{{ xs[1:] }} {{ xs[:1] }} {{ xs[::-1] }} {{ xs[-1:] }} {{ xs[1:-1] }} {{ xs[::2] }} {{ xs[-10:] }} {{ xs[5:] }}", {"xs": [1, 2, 3, 4]}),
    ("{{ xs[-1] }} {{ xs[0] }} {{ d['k'] }} {{ d.k }} {{ d.missing is defined }} {{ xs[9] is defined }}", {"xs": [1, 2], "d": {"k": "v"}}),
    ("{{ {'B': 1, 'a': 2, 'c': 0} | dictsort }} {{ {'B': 1, 'a': 2} | dictsort(true) }} {{ {'a': 2, 'b': 1} | dictsort(by='value') }}", {}),
    ("{{ d | tojson }} {{ d | tojson(indent=2) }} {{ [] | tojson }} {{ {} | tojson(indent=2) }}", {"d": {"z": [1, 2.0, "é\"\n'<>&"], "a": None, "t": True, "n": {"x": {}}}}),
    ("{% for a in [1, 2, 3] if a > 1 %}{{ loop.index }}{{ loop.length }}{{ loop.last }}{% else %}none{% endfor %}", {}),
    ("{% for a in [] %}x{% else %}none{% endfor %}", {}),
    ("{% for a in 'abc' %}{{ loop.index0 }}{{ loop.first }}{{ loop.revindex }}{{ loop.previtem is defined }}{{ loop.nextitem | default('-') }};{% endfor %}", {}),
    ("{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}{% for k in d %}{{ k }}{% endfor %}", {"d": {"b": 1, "a": 2}}),
    ("{% for i in range(5) %}{% if i == 1 %}{% continue %}{% endif %}{% if i == 3 %}{% break %}{% endif %}{{ i }}{% endfor %}", {}),
    ("{{ range(3) | list }} {{ range(1, 7, 2) | list }} {{ range(3, 0, -1) | list }}", {}),
    ("{% set x = 'o' %}{% for i in range(3) %}[{{ x }}]{% set x = i %}{% endfor %}{{ x }}", {}),
    ("{% for i in range(2) %}{% if i == 1 %}{{ y }}{% endif %}{% set y = 5 %}{% endfor %}", {}),
    ("{% if true %}{% set z = 3 %}{% endif %}{{ z }}", {}),
    ("{% set ns = namespace(a=1, b=[]) %}{% for i in range(3) %}{% set ns.a = ns.a + i %}{% endfor %}{{ ns.a }} {{ ns.b }}", {}),
    ("{% set a, b = 1, 2 %}{{ a }}{{ b }}{% set t = (3, 4) %}{% set c, d = t %}{{ c }}{{ d }}", {}),
    ("{% set x %}  inner {{ 1 }}\n{% endset %}[{{ x }}]", {}),
    ("{% macro m(a, b='B', c=none) %}{{ a }}{{ b }}{{ c }}{% endmacro %}{{ m(1) }} {{ m(1, 2) }} {{ m(1, c=3) }}", {}),
    ("{% macro m() %}{{ q }}{% endmacro %}{% set q = 7 %}{{ m() }}", {}),
    ("{% for i in [1] %}{% macro m() %}{{ i }}{% endmacro %}{{ m() }}{% endfor %}", {}),
    ("{% macro fact(n) %}{% if n <= 1 %}1{% else %}{{ n }}*{{ fact(n - 1) }}{% endif %}{% endmacro %}{{ fact(4) }}", {}),
    ("{{ ' a b  '.split() }} {{ 'a,b,,c'.split(',') }} {{ 'a b c'.split(' ', 1) }} {{ 'abc'.rsplit('b') }} {{ 'a b c'.rsplit(' ', 1) }} {{ '  x  y '.split(none, 1) }}", {}),
    ("{{ '  x  y '.rsplit(none, 1) }} {{ ' a '.split(none, 0) }} {{ 'a  b c '.rsplit(none, 5) }} {{ ''.split(none, 1) }} {{ ' x y '.rsplit(none, 0) }} {{ 'a b c'.split(none, 1) }}", {}),
    ("{{ 2.675 | round(2) }} {{ 3.5 | round }} {{ -2.5 | round }} {{ 7 | round(1, 'floor') }}", {}),
    ("{{ '  x '.strip() }}|{{ 'xxaxx'.strip('x') }}|{{ '  x '.lstrip() }}|{{ '  x '.rstrip() }}|{{ 'abc'.startswith('a') }}{{ 'abc'.endswith(('x', 'c')) }}", {}),
    ("{{ 'a-b'.replace('-', '+') }} {{ 'hello world'.title() }} {{ 'hELLO'.capitalize() }} {{ 'abc'.upper() }} {{ 'ABC'.lower() }} {{ 'abcb'.find('b') }} {{ 'abcb'.rfind('b') }} {{ 'abcb'.count('b') }}", {}),
    ("{{ none | string }} {{ none | trim }} {{ u | length }}|{{ u | trim }}|{{ u | list }} {{ u | default('d') }} {{ '' | default('d') }} {{ '' | default('d', true) }}", {}),
    ("{{ 'x' in u }} {{ 'a' in 'cat' }} {{ 'k' in d }} {{ 2 in [1, 2] }} {{ 3 not in [1, 2] }}", {"d": {"k": 1}}),
    ("{{ x is none }} {{ x is not none }} {{ y is mapping }} {{ y is iterable }} {{ y is sequence }} {{ 's' is sequence }} {{ 1 is number }} {{ 1.5 is float }} {{ 1 is integer }} {{ true is boolean }} {{ true is true }} {{ 0 is false }}", {"x": None, "y": {"a": 1}}),
    ("{{ 4 is divisibleby 2 }} {{ 3 is odd }} {{ 'a' is eq 'a' }} {{ 'A' is upper }} {{ x is callable }} {{ namespace is callable }}", {"x": 1}),
    ("{{ [3, 1, 2] | sort }} {{ ['b', 'A', 'c'] | sort }} {{ [3, 1, 2] | sort(reverse=true) }} {{ xs | sort(attribute='n') | map(attribute='n') | list }}", {"xs": [{"n": 2}, {"n": 1}]}),
    ("{{ [1, 2, 3] | first }} {{ [1, 2, 3] | last }} {{ [1, 2] | join(', ') }} {{ xs | map(attribute='n') | join }} {{ ['a', 'b'] | map('upper') | list }}", {"xs": [{"n": 2}, {"n": 1}]}),
    ("{{ xs | selectattr('role', 'equalto', 'user') | list | length }} {{ xs | rejectattr('role', 'eq', 'user') | map(attribute='role') | list }} {{ [0, 1, 2] | select | list }} {{ [1, 2, 3, 4] | select('odd') | list }}", {"xs": [{"role": "user"}, {"role": "assistant"}, {"role": "user"}]}),
    ("{{ [1, 2, 1] | unique | list }} {{ [1, 2] | reverse | list }} {{ 'ab' | reverse }} {{ [1, 2, 3] | sum }} {{ [1, 5, 3] | max }} {{ [1, 5, 3] | min }}", {}),
    ("{{ '3' | int + 1 }} {{ 'x' | int }} {{ '2.5' | float }} {{ 2.567 | round(2) }} {{ 2.5 | round }} {{ 'a b' | wordcount }} {{ 'x' | upper }} {{ 'Hi There' | lower }}", {}),
    ("{{ 'a\nb\n\nc' | indent(2) }}|{{ 'a\nb' | indent(2, true) }}", {}),
    ("{{ {'a': 1}.items() | list }} {{ {'a': 1}.keys() | list }} {{ {'a': 1}.values() | list }} {{ {'a': 1}.get('a') }} {{ {'a': 1}.get('b') }} {{ {'a': 1}.get('b', 2) }} {{ {'a': 1} | items | list }}", {}),
    ("{{ 'a' 'b' }} {{ \"\\u00e9\\n\\t|\" }} {{ '\\x41' }}", {}),
    ("{{ 1 == 1.0 }} {{ 1 < 2 < 3 }} {{ 'a' < 'b' }} {{ [1, 2] == [1, 2] }} {{ (1, 2) == [1, 2] }} {{ none == none }} {{ 1 != '1' }}", {}),
    ("{{ x and y }} {{ x or y }} {{ not x }} {{ 0 or '' or 'z' }} {{ 1 and 2 }}", {"x": 0, "y": "y"}),
    ("{{ raise_exception('refused: ' ~ x) }}", {"x": "why"}),
    ("{{ u.x }}", {}),
    ("{{ d.x.y }}", {"d": {}}),
    ("{{ ns.x }}|", {"ns": None}),
    ("{% generation %}gen {{ 1 }}{% endgeneration %}", {}),
    ("{{ ({'a': 1}) }} {{ () }} {{ (1, 2) }} {{ [] }} {{ {} }} {{ [[1], {'k': [2]}] }}", {}),
    ("{{ [1, 2] | tojson }} {{ 'x' | tojson }} {{ 3 | tojson }} {{ 1.5 | tojson }} {{ none | tojson }} {{ true | tojson }} {{ ('a', 1) | tojson }}", {}),
    ("{{ d | tojson(indent=4) }}", {"d": {"a": [1, {"b": []}], "c": "x"}}),
    ("{{ d | dictsort | first }} {{ (d | dictsort)[0][1] }}", {"d": {"b": 1, "a": 2}}),
    ("{{ dict(a=1, b='x') }} {{ namespace(k=1).k }}", {}),
    ("{% for m in msgs %}{% set role = 'model' if m.role == 'assistant' else m.role %}{{ role }};{% endfor %}", {"msgs": [{"role": "user"}, {"role": "assistant"}]}),
    ("{%- if x is defined and x is not none and x | length > 0 -%}yes{%- else -%}no{%- endif -%}", {"x": [1]}),
    ("{{ '%s' }} {{ '{}' }} {{ '}' }}{{ '{' }}", {}),
    ("{{ \"a\" ~ 'b' ~ 3 ~ none ~ [1] }}", {}),
    ("{{ xs | map('string') | join(',') }} {{ xs | map('tojson') | join(' ') }}", {"xs": [1, "a", None]}),
    ("{%- set ns = namespace(found=false) -%}{%- for m in msgs[::-1] -%}{%- if not ns.found and m.role == 'user' -%}{%- set ns.found = true -%}{{ loop.index0 }}{%- endif -%}{%- endfor -%}", {"msgs": [{"role": "user"}, {"role": "assistant"}, {"role": "user"}, {"role": "assistant"}]}),
    ("{{ x.content is string }} {{ x.get('content') is none }} {{ x['tool_calls'] is defined }}", {"x": {"content": None}}),
    ("line1\n    {%- if true %}\n    kept\n    {% endif %}\nend", {}),
    ("{%- for i in [1, 2] %}\n    {{- i }}\n{%- endfor %}", {}),
    ("{%+ if true %}  keep{% endif +%}\nnext", {}),
]


def image(kind="image"):
    return {"type": kind}


# Keys in sorted order, which is the order golem hands them over in: a Go map
# keeps none, so chat.FileTemplate sorts them, and chat's test replays these
# conversations through it.
CALL = {"type": "function", "function": {"name": "get_weather",
        "arguments": {"city": "Paris", "days": 3, "detail": {"hourly": True, "note": "it's \"fine\""}, "units": "metric"}}}
CALL_ID = {"id": "call_1", **CALL}
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the weather for a city, in the user's units.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "days": {"type": "integer", "description": "How many days", "nullable": True},
            "detail": {"type": "object", "properties": {"hourly": {"type": "boolean"}}},
            "tags": {"type": "array", "items": {"type": "string"}},
            "units": {"type": "string", "enum": ["metric", "imperial"]},
        }, "required": ["city"]}}},
    {"type": "function", "function": {"name": "now", "description": ""}},
]

# The conversations every template is rendered over.
CONVERSATIONS = [
    ("user_only", [{"role": "user", "content": "hi"}]),
    ("system_user", [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "hi"}]),
    ("developer_user", [{"role": "developer", "content": "Be brief."}, {"role": "user", "content": "hi"}]),
    ("multi_turn", [
        {"role": "user", "content": "  one  "},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three\n"},
    ]),
    ("trailing_assistant", [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}]),
    ("gemma_thinking_history", [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "<|channel>thought\nhmm<channel|>a1"},
        {"role": "user", "content": "q2"},
    ]),
    ("qwen_thinking_history", [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "<think>\nhmm\n</think>\n\na1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "<think>\nmore\n</think>\n\na2"},
    ]),
    ("reasoning_field", [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "because"},
        {"role": "user", "content": "q2"},
    ]),
    ("picture", [{"role": "user", "content": [image(), {"type": "text", "text": "What is this?"}]}]),
    ("two_pictures", [
        {"role": "system", "content": "Look closely."},
        {"role": "user", "content": [image(), image(), {"type": "text", "text": " Compare. "}]},
        {"role": "assistant", "content": "Same."},
        {"role": "user", "content": [image(), {"type": "text", "text": "And this one?"}]},
    ]),
    ("recording", [{"role": "user", "content": [image("audio"), {"type": "text", "text": "Transcribe."}]}]),
    ("tool_round", [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [CALL_ID]},
        {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "{\"temp\": 21}"},
        {"role": "assistant", "content": "It is 21 degrees."},
        {"role": "user", "content": "Thanks"},
    ]),
    ("tool_pending", [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "Let me check.", "tool_calls": [CALL]},
        {"role": "tool", "name": "get_weather", "content": "sunny"},
    ]),
    ("two_calls", [
        {"role": "user", "content": "Weather and time?"},
        {"role": "assistant", "content": None, "tool_calls": [
            CALL, {"type": "function", "function": {"name": "now", "arguments": {}}}]},
        {"role": "tool", "name": "get_weather", "content": "sunny"},
        {"role": "tool", "name": "now", "content": "noon"},
    ]),
    ("unicode", [{"role": "user", "content": "Ça va ? 日本語 — «ok» 'q' \"dq\" \\ {{ x }}"}]),
]

FLAGS = [
    # (tools, enable_thinking, add_generation_prompt)
    (None, False, True),
    (None, True, True),
    (None, False, False),
    (TOOLS, False, True),
]


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = sys.argv[1]
    env = environment()

    snippets = []
    for source, variables in SNIPPETS:
        case = {"template": source, "vars": variables}
        case.update(render(env, source, variables))
        snippets.append(case)

    templates = []
    folder = os.path.join(out, "templates")
    for name in sorted(os.listdir(folder)):
        source = open(os.path.join(folder, name)).read()
        cases = []
        for title, messages in CONVERSATIONS:
            for tools, thinking, generation in FLAGS:
                variables = {
                    "messages": messages, "tools": tools,
                    "enable_thinking": thinking, "add_generation_prompt": generation,
                    "bos_token": "<bos>", "eos_token": "<eos>",
                }
                case = {"name": title, "vars": variables}
                case.update(render(env, source, json.loads(json.dumps(variables))))
                cases.append(case)
        templates.append({"file": name, "cases": cases})

    with open(os.path.join(out, "cases.json"), "w") as f:
        json.dump({"jinja2": jinja2.__version__, "snippets": snippets, "templates": templates},
                  f, ensure_ascii=False, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
