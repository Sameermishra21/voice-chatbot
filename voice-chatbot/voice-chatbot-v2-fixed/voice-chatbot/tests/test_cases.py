"""
tests/test_cases.py
-------------------
Runs the 15 demonstration test cases through the REAL trained model and prints
a results table (used in report.md).  None of these sentences are copied from
intents.json – they are new phrasings, as a user would actually speak them.

Run from the project root:
    python tests/test_cases.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

from nlp_utils import IntentClassifier  # noqa: E402

# (spoken input, expected intent)
# "refuse" = an off-topic question: the bot must NOT give a college answer.
#   It passes if the intent is "out_of_scope" (polite refusal) or "unknown"
#   (low confidence -> "please say it again").
REFUSALS = {"out_of_scope", "unknown"}
TEST_CASES = [
    ("Hello", "greeting"),
    ("What is your name?", "bot_name"),
    ("Can you help me?", "help"),
    ("Thank you", "thanks"),
    ("Goodbye", "goodbye"),
    ("Tell me about courses", "courses"),
    ("What are your working hours?", "working_hours"),
    ("How can I contact you?", "contact"),
    ("What can you do?", "capabilities"),
    ("How much is the fee for B.Tech?", "fees"),
    ("Is there a hostel for girls?", "hostel"),
    ("What's the weather like on Mars?", "refuse"),
    ("Tell me about the Taj Mahal", "refuse"),
    ("Who built the Taj Mahal?", "refuse"),
    ("What is the capital of France?", "refuse"),
]


def run(classifier=None):
    clf = classifier or IntentClassifier(BACKEND)
    rows = []
    for text, expected in TEST_CASES:
        r = clf.predict(text)
        rows.append({
            "input": text, "expected": expected, "intent": r["intent"],
            "predicted": r["predicted_intent"], "confidence": r["confidence"],
            "response": r["response"],
            "passed": (r["intent"] in REFUSALS) if expected == "refuse" else r["intent"] == expected,
        })
    return rows


if __name__ == "__main__":
    rows = run()
    print("| # | Input | Expected | Predicted intent | Confidence | Response | Status |")
    print("|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        shown = r["intent"] if r["intent"] not in REFUSALS else f"{r['intent']} (model: {r['predicted']})"
        print(f"| {i} | {r['input']} | {r['expected']} | {shown} | {r['confidence']*100:.1f}% | "
              f"{r['response']} | {'PASS' if r['passed'] else 'FAIL'} |")
    passed = sum(r["passed"] for r in rows)
    print(f"\n{passed}/{len(rows)} test cases passed.")
