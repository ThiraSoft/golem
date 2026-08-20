"""Tokenizer reference: the expected identifiers for a corpus of sentences.

Usage:
    python ref/dump_tokens.py <tokenizer.model> <out.json> [french|english]

Pocket TTS ships one tokenizer per language, and they are not the same model:
the parity test runs against both, so there is one corpus per language here.
Each covers what a sentence in that language runs into — elisions and accents on
one side, contractions and quotation marks on the other, and on both sides
numbers, repeated spaces, and enough to exercise the byte fallback.
"""

import json
import sys

import sentencepiece

FRENCH = [
    "Bonjour le monde.",
    "Bonjour, je suis Estelle, et je parle français.",
    "L'hôtel était déjà complet, hélas !",
    "Où ça ? Là-bas, près de l'église.",
    "Il a dit : « c'est fini ».",
    "Cent quatre-vingt-dix-neuf euros et vingt-trois centimes.",
    "Le 14 juillet 1789, à 6 h 30.",
    "Ça coûte 3,50 € — pas plus.",
    "Æsculape, cœur, naïve, sûr, français.",
    "Un texte    avec   des espaces    répétés.",
    "   Espaces en tête et en queue.   ",
    "MAJUSCULES et minuscules, MéLaNgÉeS.",
    "cent%2Fun@exemple.fr ; http://exemple.fr/chemin?x=1",
    "Emoji 🙂 et symbole ∑ dans la phrase.",
    "Phrase sans ponctuation finale",
    "a",
    "Deux\nlignes\tséparées.",
    "L'élève dit qu'aujourd'hui, c'est l'anniversaire d'Élodie.",
]

ENGLISH = [
    "Hello world.",
    "Hello, I am Alba, and I speak English.",
    "The hotel wasn't full — she'd booked it weeks ago.",
    "Where? Over there, next to the church.",
    "He said: “it's finished”.",
    "One hundred and ninety-nine pounds and twenty-three pence.",
    "On the 14th of July 1789, at 6:30 a.m.",
    "It costs $3.50 — no more.",
    "Naïve, café, résumé, coöperate, façade.",
    "A text    with   repeated    spaces.",
    "   Spaces at the head and at the tail.   ",
    "CAPITALS and lowercase, MiXeD.",
    "one%2Ftwo@example.com ; http://example.com/path?x=1",
    "Emoji 🙂 and the symbol ∑ in the sentence.",
    "A sentence with no final punctuation",
    "a",
    "Two\nlines\tapart.",
    "The student says that today is Elodie's birthday.",
]

CORPORA = {"french": FRENCH, "english": ENGLISH}


def main() -> None:
    path, out = sys.argv[1], sys.argv[2]
    language = sys.argv[3] if len(sys.argv) > 3 else "french"
    sp = sentencepiece.SentencePieceProcessor(path)
    cases = [
        {"text": s, "tokens": sp.encode(s, out_type=int), "pieces": sp.encode(s, out_type=str)}
        for s in CORPORA[language]
    ]
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"vocab_size": sp.vocab_size(), "cases": cases}, f, ensure_ascii=False, indent=2)
    print(f"{len(cases)} cases written to {out}")


if __name__ == "__main__":
    main()
