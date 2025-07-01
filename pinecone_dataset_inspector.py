"""
pinecone_inspect.py
-------------------
Quick inspection utility for a Pinecone index (SDK ≥ 1.0).

Required in .env
----------------
PINECONE_API_KEY     your Pinecone API key
PINECONE_INDEX_HOST  full host, e.g.  my-index-abc123.svc.us-east1-aws.pinecone.io
PINECONE_INDEX_NAME  index name, e.g.  my-index
"""

import os
import sys
from collections import defaultdict
from pprint import pformat

from dotenv import load_dotenv
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from pinecone.grpc import PineconeGRPC as Pinecone 

load_dotenv()

API_KEY      = os.getenv("PINECONE_API_KEY")
INDEX_HOST   = os.getenv("PINECONE_INDEX_HOST") 
INDEX_NAME   = os.getenv("PINECONE_INDEX_NAME")

missing = [k for k, v in {
    "PINECONE_API_KEY": API_KEY,
    "PINECONE_INDEX_HOST": INDEX_HOST,
    "PINECONE_INDEX_NAME": INDEX_NAME,
}.items() if not v]

if missing:
    sys.exit(
        f"[bold red]Error:[/bold red] .env must define {', '.join(missing)}"
    )

# ─────────────────── 1. Connect  ───────────────────────
pc     = Pinecone(api_key=API_KEY)   # control-plane
index  = pc.Index(host=INDEX_HOST)   # data-plane

console = Console()
console.rule("[bold cyan]🔍  Pinecone inspection")

# ─────────────────── 2. Describe index ───────────────────────
rprint("[bold yellow]Fetching index description…[/bold yellow]")
desc = pc.describe_index(name=INDEX_NAME)          # ← name comes from .env
console.print(pformat(desc), highlight=False)

# ─────────────────── 3. Stats ────────────────────────────────
rprint("\n[bold yellow]Gathering vector statistics…[/bold yellow]")
stats        = index.describe_index_stats()
dimension    = stats["dimension"]
total_vecs   = stats["total_vector_count"]
ns_meta      = stats.get("namespaces", {})

table = Table(title=f"Vector counts (dim={dimension})")
table.add_column("Namespace")
table.add_column("# Vectors", justify="right")
for ns, meta in ns_meta.items():
    table.add_row(ns, f"{meta['vector_count']:,}")
table.add_row("[bold]TOTAL", f"{total_vecs:,}", style="bold")
console.print(table)

# ─────────────────── 4. Sample IDs ───────────────────────────
SAMPLE_IDS_PER_NS = 3
rprint(f"\n[bold yellow]Listing up to {SAMPLE_IDS_PER_NS} IDs per namespace…[/bold yellow]")

id_samples = defaultdict(list)
for ns in (ns_meta.keys() or ["__default__"]):
    for id_page in index.list(namespace=ns):
        id_samples[ns].extend(id_page)
        if len(id_samples[ns]) >= SAMPLE_IDS_PER_NS:
            break

for ns, ids in id_samples.items():
    console.print(f"\n[bold]Namespace '{ns}':[/bold] {ids[:SAMPLE_IDS_PER_NS]}")

# ─────────────────── 5. Fetch sample records ─────────────────
FETCH_LIMIT = 3
rprint("\n[bold yellow]Fetching sample record objects…[/bold yellow]")
for ns, ids in id_samples.items():
    if not ids:
        continue
    records = index.fetch(ids=ids[:FETCH_LIMIT], namespace=ns)
    console.print(f"\n[bold green]Sample records from '{ns}':[/bold green]")
    for vid, vect in records.vectors.items():
        console.print(f"[bold]{vid}[/bold]")
        console.print(f"  metadata: {pformat(vect['metadata'])}")
        console.print(f"  embedding: <{len(vect['values'])} floats omitted>")

console.rule("[bold cyan]✅ Inspection complete")
