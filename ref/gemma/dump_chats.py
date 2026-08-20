#!/usr/bin/env python3
"""Record what Gemma's own chat template renders.

The template is 18 KB of Jinja sitting in the GGUF metadata. golem does not
interpret Jinja; it reimplements the subset a text conversation exercises, and
this script produces the fixture that pins the reimplementation to the original.

The rendering here is done by Jinja2 itself, on the template read straight out
of the model file, so the fixture says what the template says and not what any
reimplementation of Jinja believes it says.

    python3 ref/gemma/dump_chats.py <model.gguf> testdata/gemma/chat/cases.json

Run by hand. Never at test time: the fixture is committed.
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
    ("developer_user", [
        {"role": "developer", "content": "You are terse."},
        {"role": "user", "content": "hi"},
    ], False, True, None),
    ("multi_turn", [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ], False, True, None),
    ("assistant_continuation", [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "assistant", "content": "still two"},
    ], False, True, None),
    ("thinking", [{"role": "user", "content": "hi"}], True, True, None),
    ("thinking_with_system", [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "hi"},
    ], True, True, None),
    ("no_generation_prompt", [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ], False, False, None),
    ("whitespace_trimmed", [
        {"role": "system", "content": "  padded  \n"},
        {"role": "user", "content": "\n  spaced  \n"},
    ], False, True, None),
    ("assistant_with_channel", [
        {"role": "user", "content": "one"},
        {"role": "assistant",
         "content": "before<|channel>thought\nhidden<channel|>after"},
        {"role": "user", "content": "three"},
    ], False, True, None),
    ("assistant_channel_unclosed", [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "kept<|channel>dropped"},
    ], False, False, None),
    ("system_only", [{"role": "system", "content": "alone"}], False, True, None),
    ("empty_content", [
        {"role": "user", "content": ""},
    ], False, True, None),
    ("unicode", [
        {"role": "user", "content": "héllo — 世界 🙂"},
    ], False, True, None),
]

# The tool path of the template: declarations in the system turn, the calls
# the model makes, and the results fed back to it.
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
    ("tools_thinking", [{"role": "user", "content": "hi"}], True, True, [WEATHER]),
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
    ("tool_call_argument_types", [
        {"role": "user", "content": "tag it"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "tags",
                          "arguments": {"labels": ["a", "b"], "pinned": True,
                                        "count": 2}}},
        ]},
    ], False, False, [TAGS]),
]

CASES = CASES + TOOL_CASES


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    model, out_path = sys.argv[1], sys.argv[2]

    meta = metadata(model)
    template_text = meta["tokenizer.chat_template"]
    bos_id = meta["tokenizer.ggml.bos_token_id"]

    env = Environment(trim_blocks=False, lstrip_blocks=False)
    template = env.from_string(template_text)

    cases = []
    for name, messages, thinking, add_generation_prompt, tools in CASES:
        try:
            text = template.render(
                messages=messages,
                bos_token="<bos>",
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

    # Which of the two spellings of the generation prompt this template uses.
    # The 12B closes it with an empty thought channel when thinking is off; E2B
    # does not, and the Go renderer is told which by the fixture rather than by
    # guessing from the model's name.
    # The literal as the Jinja spells it: a backslash and an n, not a newline.
    empty_thought = r"<|channel>thought\n<channel|>" in template_text

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"bos_token_id": bos_id, "empty_thought": empty_thought,
                   "cases": cases}, f,
                  ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{len(cases)} cases written to {out_path}")


if __name__ == "__main__":
    main()
