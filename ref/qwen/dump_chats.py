#!/usr/bin/env python3
"""Record what Qwen3's own chat template renders.

The template sits in the GGUF metadata under tokenizer.chat_template. golem
does not interpret Jinja; it reimplements the subset a conversation with tools
exercises, and this script produces the fixture that pins the reimplementation
to the original.

The rendering here is done by Jinja2 itself, on the template read straight out
of the model file, so the fixture says what the template says and not what any
reimplementation of Jinja believes it says.

    python3 ref/qwen/dump_chats.py <model.gguf> testdata/qwen/chat/cases.json

Run by hand. Never at test time: the fixture is committed.

Two things this fixture was checked against llama.cpp's /apply-template for,
and one of them it deliberately disagrees with.

The empty <think></think> block: llama.cpp leaves enable_thinking undefined,
which the template reads as on, and so writes no such block. The cases here
that pass enable_thinking=True match llama.cpp character for character, which
is what says the disagreement is about the variable and not about the template.
golem's chat.Options.EnableThinking is false by default, so its prompts carry
the block; that is the same default cmd/golem-cli's -think flag has.

The tool declarations: Jinja2's tojson sorts an object's keys and escapes an
apostrophe as \u0027; minja, which llama.cpp renders with, keeps insertion
order and writes the apostrophe. Transformers renders with Jinja2, so the
spelling recorded here is the one the checkpoint was trained on, and it is what
qwen/toolrender.go reproduces.
"""

import json
import struct
import sys

from jinja2 import Environment
from jinja2.exceptions import TemplateError

# GGUF value types, as the format numbers them.
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)

_FIXED = {
    U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i",
    F32: "<f", BOOL: "<?", U64: "<Q", I64: "<q", F64: "<d",
}


class Reader:
    """Just enough of the GGUF header to reach one string."""

    def __init__(self, data):
        self.data = data
        self.at = 0

    def take(self, n):
        chunk = self.data[self.at:self.at + n]
        self.at += n
        return chunk

    def fixed(self, kind):
        fmt = _FIXED[kind]
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        return self.take(self.fixed(U64)).decode("utf-8")

    def value(self, kind):
        if kind == STRING:
            return self.string()
        if kind == ARRAY:
            item = self.fixed(U32)
            count = self.fixed(U64)
            if item in _FIXED:
                # Skipped whole: the arrays in this file hold a quarter of a
                # million entries and none of them is wanted here.
                self.at += count * struct.calcsize(_FIXED[item])
                return None
            for _ in range(count):
                self.value(item)
            return None
        return self.fixed(kind)


def metadata(path):
    with open(path, "rb") as f:
        data = f.read(64 << 20)  # the header lives at the front of the file
    r = Reader(data)
    if r.take(4) != b"GGUF":
        raise SystemExit(f"{path} is not a GGUF file")
    r.fixed(U32)  # version
    r.fixed(U64)  # tensor count
    count = r.fixed(U64)
    out = {}
    for _ in range(count):
        key = r.string()
        out[key] = r.value(r.fixed(U32))
    return out


# The corpus. Each case names the messages and the two flags the template reads.
CASES = [
    ("user_only", [{"role": "user", "content": "hi"}], False, True, None),
    ("system_user", [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "hi"},
    ], False, True, None),
    ("multi_turn", [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ], False, True, None),
    ("no_generation_prompt", [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ], False, False, None),
    ("thinking", [{"role": "user", "content": "hi"}], True, True, None),
    ("thinking_with_system", [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "hi"},
    ], True, True, None),
    ("assistant_with_think_block", [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "<think>\nhidden\n</think>\n\nafter"},
        {"role": "user", "content": "three"},
    ], False, True, None),
    ("empty_content", [{"role": "user", "content": ""}], False, True, None),
    ("unicode", [{"role": "user", "content": "héllo — 世界 🙂"}], False, True, None),
    ("whitespace", [
        {"role": "system", "content": "  padded  \n"},
        {"role": "user", "content": "\n  spaced  \n"},
    ], False, True, None),
]

# The tool path of the template: declarations in the system turn, the calls the
# model makes, and the results fed back to it.
WEATHER = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": "Today's weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Which city."},
                "unit": {"type": "string", "description": "Scale.",
                         "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}
NOW = {"type": "function",
       "function": {"name": "now", "description": "The current time."}}
TAGS = {
    "type": "function",
    "function": {
        "name": "tags",
        "description": "Tag a note.",
        "parameters": {
            "type": "object",
            "properties": {
                "labels": {"type": "array", "description": "The labels.",
                           "items": {"type": "string"}},
                "pinned": {"type": "boolean", "description": "Keep it on top."},
            },
        },
    },
}

TOOL_CASES = [
    ("tools_declared", [{"role": "user", "content": "weather in Lyon?"}],
     False, True, [WEATHER]),
    ("tools_no_parameters", [{"role": "user", "content": "what time is it?"}],
     False, True, [NOW]),
    ("tools_array_and_boolean", [{"role": "user", "content": "tag it"}],
     False, True, [TAGS]),
    ("tools_two_declarations", [{"role": "user", "content": "hi"}],
     False, True, [WEATHER, NOW]),
    ("tools_with_system", [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "hi"},
    ], False, True, [WEATHER]),
    ("tool_call_unanswered", [
        {"role": "user", "content": "weather in Lyon?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "weather",
                          "arguments": {"city": "Lyon", "unit": "celsius"}}},
        ]},
    ], False, False, [WEATHER]),
    ("tool_call_answered", [
        {"role": "user", "content": "weather in Lyon?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "weather", "arguments": {"city": "Lyon"}}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "name": "weather",
         "content": "18 degrees, clear"},
    ], False, True, [WEATHER]),
    ("tool_call_two_calls_answered", [
        {"role": "user", "content": "both please"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "weather", "arguments": {"city": "Lyon"}}},
            {"id": "call_2", "type": "function",
             "function": {"name": "now", "arguments": {}}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "name": "weather",
         "content": "18 degrees"},
        {"role": "tool", "tool_call_id": "call_2", "name": "now",
         "content": "14:03"},
    ], False, True, [WEATHER, NOW]),
    ("tool_call_then_answer", [
        {"role": "user", "content": "weather in Lyon?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "weather", "arguments": {"city": "Lyon"}}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "name": "weather",
         "content": "18 degrees"},
        {"role": "assistant", "content": "It is 18 degrees."},
        {"role": "user", "content": "and tomorrow?"},
    ], False, True, [WEATHER]),
]

CASES = CASES + TOOL_CASES


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    model, out_path = sys.argv[1], sys.argv[2]

    meta = metadata(model)
    template_text = meta["tokenizer.chat_template"]

    # The template controls its own whitespace with {%- -%} throughout, so all
    # four combinations of these two flags render identically. They are set
    # here to what transformers uses, not because it matters.
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(template_text)

    cases = []
    for name, messages, thinking, add_generation_prompt, tools in CASES:
        try:
            text = template.render(
                messages=messages,
                enable_thinking=thinking,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
            )
        except TemplateError as e:
            raise SystemExit(f"{name}: {e}")
        cases.append({
            "name": name,
            "messages": messages,
            "tools": tools,
            "enable_thinking": thinking,
            "add_generation_prompt": add_generation_prompt,
            "rendered": text,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"cases": cases}, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{len(cases)} cases written to {out_path}")


if __name__ == "__main__":
    main()
