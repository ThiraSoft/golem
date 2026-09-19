// Package jinja renders the Jinja a model file carries as its chat template.
//
// It is the language as transformers runs it for apply_chat_template, and no
// wider: Jinja2's syntax and scoping, trim_blocks and lstrip_blocks on, the
// loop controls and the generation tag, the filters and tests chat templates
// call, the methods of Python's strings and dicts they reach for, and
// transformers' own tojson, raise_exception and strftime_now. There is no
// autoescaping, no inheritance and no include: a chat template is one string.
//
// Values are Python's, spelled in Go (value.go says how), because what a
// template prints is str() of a Python object and a template is written
// against Python's semantics down to how a float is printed.
//
// ref/jinja/dump_cases.py records what Jinja2 renders, and the tests compare
// character for character.
package jinja
