"""Train the lightweight dialect/language ID used for slices and the API response."""
from sklearn.metrics import accuracy_score

from shifa import dialect
from shifa.data import load_split


def main():
    tr, va = load_split("train"), load_split("val")
    pipe = dialect.fit(tr.text, tr.dialect)
    acc = accuracy_score(va.dialect, dialect.predict(pipe, va.text))
    print(f"dialect-ID val accuracy {acc:.3f}")
    dialect.save(pipe)


if __name__ == "__main__":
    main()
