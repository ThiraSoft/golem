# ref/tha3: Talking Head(?) Anime 3

The weights are by Pramook Khungurn, distributed under CC-BY 4.0 from
[talking-head-anime-3-demo](https://github.com/pkhungurn/talking-head-anime-3-demo).
They are not in this repository. Credit him wherever you redistribute them.

## Weights

```sh
uv venv -p 3.12 /tmp/tha3-venv
uv pip install -p /tmp/tha3-venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -p /tmp/tha3-venv/bin/python safetensors matplotlib numpy pillow
sh ref/tha3/fetch.sh /tmp/tha3-pt
/tmp/tha3-venv/bin/python ref/tha3/convert.py /tmp/tha3-pt ~/.cache/golem/tha3/separable_float
```

`GOLEM_THA3` points the Go side elsewhere.

## Fixtures

`dump.py` runs the upstream code itself, from a checkout pinned at
8946939ec7b417f443d7b0e3fcd97384313fcdb8:

```sh
git clone https://github.com/pkhungurn/talking-head-anime-3-demo /tmp/tha3-demo
git -C /tmp/tha3-demo checkout 8946939ec7b417f443d7b0e3fcd97384313fcdb8
/tmp/tha3-venv/bin/python ref/tha3/dump.py /tmp/tha3-demo /tmp/tha3-pt \
    ~/dev/avatar/avatar_tha.png testdata/tha3
```

It writes one directory per pose under `testdata/tha3/`. Only `neutral`
records the blocks inside each network; the others record what goes in and
out of each network, which is what an end-to-end test needs.
