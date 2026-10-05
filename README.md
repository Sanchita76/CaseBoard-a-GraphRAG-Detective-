# Project 1 — CaseBoard: a GraphRAG Detective  🕵️

**Tech stack:** GraphRAG (knowledge graph + community detection + graph-based retrieval)
**Cost:** 100% free. No API keys. No accounts. Runs in Google Colab.

## What does it do?
You give it a pile of text files (here: 7 short fictional police case files about a stolen pearl).
It then:

1. **Builds a knowledge graph** – people, organisations, places, items become *nodes*; "they appear together in a sentence" becomes an *edge* (with the sentence kept as evidence).
2. **Finds communities** (groups of closely linked entities) and writes a short report for each group.
3. **Answers questions by walking the graph** (this is the "GraphRAG" part):
   - *"How is Silas Crane connected to the Marlow Museum?"* → follows a multi-hop path and shows the evidence.
   - *"Tell me about Nils Arden"* → collects his neighbourhood.
   - *"What are the main groups?"* → uses the community reports (big-picture question).
4. **Compares against plain RAG** (TF-IDF sentence search) to show where GraphRAG is better: plain RAG only finds sentences that *look similar* to the question, it cannot chain facts across documents.
5. Optionally uses a **free local language model (FLAN-T5)** to phrase the final answer in natural language.

## Easiest way: Google Colab (no installation on your computer)
1. Go to <https://colab.research.google.com> and sign in with any Google account.
2. Click **File → Upload notebook** and choose `CaseBoard_GraphRAG.ipynb` (in this folder).
3. Click **Runtime → Run all**. (If asked "Run anyway?", click yes.)
4. Wait ~3–6 minutes (the first run downloads a small model). Scroll down to see:
   - the graph picture,
   - the answers to the demo questions,
   - the "plain RAG vs GraphRAG" comparison,
   - a download button for the whole `outputs/` folder.
5. To ask **your own question**, edit the last code cell (`MY_QUESTION = "..."`) and press ▶ on that cell only.

The notebook is self-contained: it creates all folders and files itself, so you only need to upload that one file.

## Running on your own computer instead (optional)
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python run_demo.py --no-llm       # fast, no language model
python run_demo.py                # with the free FLAN-T5 model
python run_demo.py --question "How is Felix Ortega connected to Silas Crane?"
```

## Folder structure
```
1_GraphRAG_CaseBoard/
├── README.md                    <- you are here
├── requirements.txt             <- libraries to install
├── CaseBoard_GraphRAG.ipynb     <- upload this to Google Colab
├── run_demo.py                  <- runs everything from the command line
├── data/
│   ├── entities.json            <- list of names the graph should recognise (+ aliases)
│   └── case_files/*.txt         <- the documents (replace with your own!)
├── graphrag/
│   ├── build_graph.py           <- text -> knowledge graph
│   ├── communities.py           <- Louvain community detection + group reports
│   ├── query.py                 <- GraphRAG retrieval (link / profile / global / keyword)
│   ├── baseline.py              <- plain RAG baseline for comparison
│   ├── llm.py                   <- optional free FLAN-T5 answer writer
│   └── viz.py                   <- draws the graph pictures
├── outputs/                     <- sample results (re-created when you run it)
└── docs/HOW_IT_WORKS.md         <- simple explanation of every step
```

## Use your own documents
1. Replace the `.txt` files in `data/case_files/` with your own text.
2. Edit `data/entities.json`: add every person/organisation/place you care about with optional `aliases` (nicknames, surnames).
3. Run again. (Tip: write names fully in the text; the code resolves simple "He/She" sentences but not complex pronouns.)

## Troubleshooting
| Problem | Fix |
|---|---|
| `No module named spacy` / model not found | Run the install cell first (Runtime → Run all does this) |
| Model download is slow | Set `USE_LLM = False` in the notebook; everything else still works |
| Weird or missing links | Add the missing name/alias to `data/entities.json` |
