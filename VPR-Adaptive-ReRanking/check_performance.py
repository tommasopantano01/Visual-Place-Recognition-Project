"""
check_performance.py — Statistiche finali su un output di adaptive reranking.
[STUB — corpo da implementare/incollare dopo.]

CLI prevista:
    python VPR-adaptive-re-ranking/check_performance.py \
        --preds-dir       <preds folder>            \
        --adaptive-RR-dir <output folder del reranking adattivo> \
        --num-preds       <int, es. 20>             \
        --recall-values   1 5 10 20

LOGICA PREVISTA
1) Distribuzione degli stop: per ogni topK in --adaptive-RR-dir conta i file
   -> quante query si sono fermate al top-1 / top-5 / top-10 (sequenziale) /
   top-20. Il nome cartella (topK) da' direttamente il budget.
2) Ranking finale per ogni query (per calcolare recall@N):
     - carica il suo file dalla cartella topK in cui si trova;
     - .torch (k risultati IM): ordina i k candidati per num_inliers desc,
       poi accoda i restanti (num_preds - k) candidati nell'ordine di
       retrieval letto da --preds-dir (coda non rerankata);
     - .txt (solo su/ sulle skip): ordine di retrieval puro.
3) recall@N: positivo se almeno un candidato corretto entro i primi N del
   ranking finale. Riporta recall per ogni valore in --recall-values e la
   distribuzione di stop del punto 1.
"""
