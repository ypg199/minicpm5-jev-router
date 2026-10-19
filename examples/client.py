"""Ask a running router what to do with a request.  Start the server first (see README)."""
import json
import urllib.request

URL = "http://127.0.0.1:8009"

TOOLS = [
    {"name": "get_card_statement", "description": "Fetch the statement for a credit card and month.",
     "parameters": {"type": "dict", "required": ["card_id", "month"],
                    "properties": {"card_id": {"type": "string"}, "month": {"type": "string"}}}},
    {"name": "block_card", "description": "Block a lost or stolen card.",
     "parameters": {"type": "dict", "required": ["card_id"], "properties": {"card_id": {"type": "string"}}}},
]


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


for msg in ["Show my September statement for card 4021.",   # tool_call
            "Show my statement.",                            # request_for_info: which card, which month?
            "Transfer 500 dollars to my brother.",           # cannot_answer: no tool for that
            "What does APR mean?"]:                          # direct
    print(msg, "->", post("/v1/action", {"user_message": msg, "tools": TOOLS}))

print(post("/v1/route", {"user_message": "I lost my card 4021, please block it", "tools": TOOLS}))

# Jev / TypeSafe request shape
print(post("/v1/systemone", {
    "state": "Customer: I was charged twice for the same order and I want my money back.",
    "questions": {
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": None, "technical": None, "sales": None}},
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "normal", "high"]},
    }}))
