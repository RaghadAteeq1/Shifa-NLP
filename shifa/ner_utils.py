"""NER subword alignment (Lab 3B). Pure functions so alignment is unit-tested without a tokenizer."""


def align_labels(word_ids, word_label_ids, ignore=-100):
    """First sub-token of each word gets the word's label; the rest and specials get -100."""
    out, prev = [], None
    for w in word_ids:
        if w is None:
            out.append(ignore)
        elif w != prev:
            out.append(word_label_ids[w])
        else:
            out.append(ignore)
        prev = w
    return out


def words_from_subwords(word_ids, pred_ids, n_words, id2tag, default="O"):
    """Collapse sub-token predictions back to one tag per word (first sub-token wins)."""
    tags = [default] * n_words
    seen = set()
    for pos, w in enumerate(word_ids):
        if w is None or w in seen:
            continue
        seen.add(w)
        tags[w] = id2tag[int(pred_ids[pos])]
    return tags


def fix_bio(tags):
    """Turn an I-X that does not follow B-X/I-X into B-X (keeps spans valid)."""
    out, prev = [], "O"
    for t in tags:
        if t.startswith("I-") and prev[2:] != t[2:]:
            t = "B-" + t[2:]
        out.append(t)
        prev = t
    return out


def spans_to_entities(words, tags):
    ents, cur = [], None
    for w, t in zip(words, tags):
        if t.startswith("B-") or (t.startswith("I-") and (cur is None or cur[0] != t[2:])):
            if cur:
                ents.append(cur)
            cur = [t[2:], [w]]
        elif t.startswith("I-"):
            cur[1].append(w)
        else:
            if cur:
                ents.append(cur)
            cur = None
    if cur:
        ents.append(cur)
    return [{"type": typ, "text": " ".join(ws)} for typ, ws in ents]
