"""Build the AttachIQ text request dataset.

Produces 7,200 labelled prompts across the 10 request classes
(720 per class) using controlled templates plus paraphrases plus messy
realistic phrasings plus light typo variants, with a fixed seed and an
80/10/10 stratified split.

Outputs:
    data/processed/text_prompts.csv
    data/splits/text_{train,val,test}.csv
    reports/text_data_summary.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from attachiq.config import (
    GLOBAL_SEED,
    PROCESSED_DIR,
    REPORTS_DIR,
    REQUEST_CLASSES,
    SPLITS_DIR,
    ensure_dirs,
)
from attachiq.logging import get_logger

log = get_logger("data.text")

TEMPLATES: dict[str, list[str]] = {
    "summarization": [
        "Summarize this document for me.",
        "Give me a short summary of this.",
        "Can you summarize the attached file?",
        "Provide a brief overview of this document.",
        "TL;DR this please.",
        "What is this document about in one paragraph?",
        "Summarize the main points of the attached.",
        "I need a quick summary.",
        "Boil this down to a few sentences.",
        "Give me the gist of this attachment.",
        "Condense the attached document.",
        "Explain the key takeaways from this.",
        "Could you give a high-level overview?",
        "Wrap this up in a paragraph for me.",
        "Pull the highlights from this file.",
        "What are the most important points in this?",
        "Give me the executive summary.",
        "Brief me on this document.",
        "Provide a summary of the attached attachment.",
        "Walk me through this at a high level.",
    ],
    "information_extraction": [
        "Extract the key information from this document.",
        "Pull out the names and dates mentioned in this.",
        "What are the main entities in the attached file?",
        "List the important details in this document.",
        "Get all the relevant fields from this.",
        "Extract the data points from this attachment.",
        "Find every email address in this document.",
        "Pull out the contact info.",
        "Extract the addresses and phone numbers.",
        "Get the structured information from this.",
        "Pull all dates listed in the document.",
        "Find the names mentioned here.",
        "Capture the main fields from this file.",
        "Tell me which people are referenced in this.",
        "List all numbers and identifiers in the attached.",
        "Get the metadata out of this document.",
        "Identify the key entities in the attachment.",
        "Extract the most relevant fields from this.",
        "Pull anything that looks like an ID from this.",
        "Pick out the structured info in this document.",
    ],
    "financial_extraction": [
        "Extract the total amount from this invoice.",
        "Pull the line items and their prices.",
        "What is the grand total on this document?",
        "Get the tax amount from this invoice.",
        "Tell me the subtotal listed.",
        "Extract the currency and amount.",
        "Get the invoice number and amount due.",
        "Pull the price column from this.",
        "What does this invoice charge in total?",
        "Find the dollar amounts in this document.",
        "Extract the billing total from the attached.",
        "Tell me the amount owed on this invoice.",
        "Pull the financial figures from this file.",
        "What are the costs listed on this?",
        "Get the payment due and due date.",
        "Extract the unit prices from each line.",
        "Sum the amounts on this invoice for me.",
        "Find the total payable in this attachment.",
        "Extract the receipt total.",
        "Pull every currency-formatted number from this document.",
    ],
    "document_classification": [
        "What kind of document is this?",
        "Classify this attachment.",
        "Identify the document type.",
        "Is this an invoice, a form, or something else?",
        "Tell me what type of document I'm looking at.",
        "What category does this document fall into?",
        "Determine the type of this file.",
        "Is this a resume?",
        "What sort of document have I uploaded?",
        "Give me the document class for this file.",
        "Categorize this attachment.",
        "Label this document by type.",
        "What document type is shown here?",
        "Tell me whether this is a letter or a memo.",
        "Identify whether this is an email or a form.",
        "What type of document was just attached?",
        "Classify the uploaded file.",
        "Is this presentation or report?",
        "Recognize the document type for me.",
        "Just tell me the kind of document this is.",
    ],
    "internal_sharing": [
        "Forward this document to my team.",
        "Share this attachment with my manager internally.",
        "Send this to my coworker on the same project.",
        "Email this internally to my department.",
        "Share this within our company only.",
        "Pass this along to my boss.",
        "Send a copy of this to my team lead.",
        "Forward this to internal stakeholders.",
        "Share this with the internal review group.",
        "Distribute this to my colleagues.",
        "Send this to the team in our private channel.",
        "Forward this attachment to internal counsel.",
        "Share this internally with engineering.",
        "Pass this to my supervisor for review.",
        "Send this to our internal mailing list.",
        "Forward this within the org.",
        "Share with internal stakeholders only.",
        "Send this to my project group.",
        "Loop in my internal team on this attachment.",
        "Share this with my direct manager.",
    ],
    "public_sharing": [
        "Post this on our public website.",
        "Share this on social media.",
        "Tweet this attachment.",
        "Publish this document publicly.",
        "Upload this where everyone can see it.",
        "Make this attachment public.",
        "Share this on LinkedIn for the world to see.",
        "Post this to our public blog.",
        "Put this on our public-facing site.",
        "Share this with the general public.",
        "Publish this on the open web.",
        "Push this to our public Twitter account.",
        "Upload this to the public news feed.",
        "Make this available publicly online.",
        "Post this attachment to a public forum.",
        "Share this to the public Slack workspace.",
        "Publish this externally on our blog.",
        "Put this up on the company's public page.",
        "Release this attachment publicly.",
        "Make this attachment world-readable.",
    ],
    "delete_permanent": [
        "Delete this permanently.",
        "Permanently remove this from the system.",
        "Erase this from records.",
        "Wipe this document.",
        "Destroy this file after review.",
        "Delete the original file.",
        "Permanently remove this attachment.",
        "Purge this document from the database.",
        "Erase the uploaded file completely.",
        "Take this file out of the system.",
        "Remove this file permanently.",
        "Wipe the attached document from records.",
        "Clear out this file forever.",
        "Eradicate this attachment.",
        "Delete the file and all of its versions.",
        "Permanently delete the attachment.",
        "Permanently and irreversibly delete this file.",
        "Erase this document for good.",
        "Wipe this file from disk.",
        "Destroy and remove this attachment.",
    ],
    "archive_retain": [
        "Archive this, do not delete it.",
        "Keep this for records.",
        "Store this for audit.",
        "Move this to long-term storage.",
        "Retain this but remove it from active view.",
        "Save this for our audit trail.",
        "Keep the original but remove it from active use.",
        "Hold this for retention.",
        "Archive this for compliance.",
        "Preserve this in cold storage.",
        "Add this to the archive folder.",
        "Place this in long-term retention.",
        "Save this for record-keeping.",
        "Keep this on file for our archival records.",
        "Archive but keep accessible to compliance.",
        "Snapshot this for retention only.",
        "Move to the records archive.",
        "Retain this document for the audit period.",
        "Keep this on file for legal retention.",
        "Archive this attachment indefinitely.",
    ],
    "ambiguous_or_unclear": [
        "Can you handle this?",
        "Do the necessary.",
        "Take care of this.",
        "Use this where needed.",
        "Proceed with this.",
        "Prepare this.",
        "Complete this.",
        "Check this and tell me what to do.",
        "Do whatever is appropriate.",
        "Handle as discussed.",
        "Move forward with this.",
        "Do the next step.",
        "Can you make this ready?",
        "Follow up on this.",
        "Take action on this.",
        "Look into this.",
        "Process this.",
        "Please deal with this.",
        "Review this and proceed.",
        "Use this however you see fit.",
    ],
    "redaction_or_safe_transform": [
        "Summarize this without personal details.",
        "Share after removing sensitive information.",
        "Make a public-safe summary.",
        "Remove names before summarizing.",
        "Extract only non-sensitive fields.",
        "Anonymize this first.",
        "Hide financial figures before sharing.",
        "Prepare a redacted version.",
        "Mask private details first.",
        "Strip personal info and summarize.",
        "Redact PII then provide a summary.",
        "Give me a sanitized version of this.",
        "Anonymize personal information and pass it on.",
        "Remove identifiers, then describe the content.",
        "Censor financial details and summarize the rest.",
        "Provide a safe-to-share excerpt.",
        "Extract content but mask the names.",
        "Prepare a non-confidential summary.",
        "Hide personal info and share a brief.",
        "Make this safe for external view by anonymising.",
    ],
}


MESSY: dict[str, list[str]] = {
    "summarization": [
        "can u summarize this quickly", "tldr", "short summary plz",
        "summarize this real quick", "give me the gist", "what's this say in short",
    ],
    "information_extraction": [
        "pull the important info from this", "need the names + dates from this",
        "extract whats useful", "grab the key fields", "get the names out of this",
        "give me the important data",
    ],
    "financial_extraction": [
        "need the amount/date from this", "whats the total?", "how much is this for",
        "extract the $ amount", "tell me the total due", "amount on this invoice?",
    ],
    "document_classification": [
        "what kind of doc is this", "tell me what this file is", "is this an invoice?",
        "type of doc?", "wat is this file", "what category is this",
    ],
    "internal_sharing": [
        "forward this to my boss", "send to my team", "share w/ coworker",
        "send to teammate plz", "fwd to manager", "share internal only",
    ],
    "public_sharing": [
        "upload this where everyone can see", "post this online", "share publicly",
        "put it on the website", "make this public", "tweet this out",
    ],
    "delete_permanent": [
        "delete it permanently", "wipe this file forever", "nuke this attachment",
        "remove this for good", "erase forever", "permanently get rid of this",
    ],
    "archive_retain": [
        "archive only", "keep for records", "save for audit", "keep on file",
        "move to archive", "store this don't delete",
    ],
    "ambiguous_or_unclear": [
        "handle this", "deal with this", "process this", "do something with this",
        "see what to do", "you decide",
    ],
    "redaction_or_safe_transform": [
        "redact and summarize", "anonymize first", "strip pii then summarize",
        "remove names plz", "safe summary only", "hide $ and summarize",
    ],
}


TYPOS: dict[str, list[str]] = {
    "summarization": ["sumarize this pls", "give summary"],
    "information_extraction": ["extarct the key info", "get teh names"],
    "financial_extraction": ["pull the totl", "extract teh amount"],
    "document_classification": ["clasify this", "wat doc type"],
    "internal_sharing": ["fwd to manger", "shar internaly"],
    "public_sharing": ["post publicaly", "tweet thsi"],
    "delete_permanent": ["delette permanantly", "wipe forevr"],
    "archive_retain": ["archve this", "keep for recrds"],
    "ambiguous_or_unclear": ["handel this", "do necesary"],
    "redaction_or_safe_transform": ["anonymze first", "redct and summarize"],
}


def _expand(label: str, base: list[str], rng: random.Random) -> list[str]:
    pre = ["", "Hey, ", "Hi, ", "Please ", "Could you ", "I need you to ",
           "Quick ask: ", "Heads up — "]
    post = ["", " Thanks!", " Thx.", " ASAP please.", " when you can.",
            " ok?", " — sooner the better.", " kindly."]
    out: list[str] = []
    for tpl in base:
        for _ in range(2):
            out.append((rng.choice(pre) + tpl + rng.choice(post)).strip())
    return out


def build_dataset(target_per_class: int = 720, seed: int = GLOBAL_SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for label in REQUEST_CLASSES:
        pool: list[str] = []
        pool.extend(TEMPLATES[label])
        pool.extend(_expand(label, TEMPLATES[label], rng))
        pool.extend(MESSY[label] * 4)
        pool.extend(TYPOS[label] * 2)
        rng.shuffle(pool)
        if len(pool) >= target_per_class:
            chosen = pool[:target_per_class]
        else:
            chosen = pool + [rng.choice(pool) for _ in range(target_per_class - len(pool))]
        for text in chosen:
            rows.append({"text": text, "label": label})
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def stratified_split(df: pd.DataFrame, seed: int = GLOBAL_SEED) -> dict[str, pd.DataFrame]:
    train, temp = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["label"])
    val, test = train_test_split(temp, test_size=0.5, random_state=seed, stratify=temp["label"])
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def main(target_per_class: int = 720) -> None:
    ensure_dirs()
    log.info(f"Building text dataset, target_per_class={target_per_class}.")
    df = build_dataset(target_per_class=target_per_class)
    log.info(f"Generated {len(df)} prompts across {df['label'].nunique()} classes.")

    out_csv = PROCESSED_DIR / "text_prompts.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"Wrote {out_csv}")

    splits = stratified_split(df)
    for name, part in splits.items():
        path = SPLITS_DIR / f"text_{name}.csv"
        part.to_csv(path, index=False)
        log.info(f"Wrote {path} (n={len(part)})")

    summary = {
        "total": int(len(df)),
        "per_class": df["label"].value_counts().to_dict(),
        "splits": {n: int(len(p)) for n, p in splits.items()},
        "seed": GLOBAL_SEED,
        "caveat": "synthetic templated text dataset; intended as a controlled benchmark.",
    }
    Path(REPORTS_DIR / "text_data_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=720)
    args = parser.parse_args()
    main(target_per_class=args.per_class)


if __name__ == "__main__":
    cli()
