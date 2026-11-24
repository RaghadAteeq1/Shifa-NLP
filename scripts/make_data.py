"""Generate the synthetic splits and freeze the test set (writes data/test.sha256)."""
from shifa import data, synth


def main():
    splits = synth.build()
    data.write_splits(splits)
    for name, rows in splits.items():
        print(f"{name:5s} {len(rows):5d} messages, {len({r['patient'] for r in rows})} patients")
    print("test set frozen: data/test.sha256")


if __name__ == "__main__":
    main()
