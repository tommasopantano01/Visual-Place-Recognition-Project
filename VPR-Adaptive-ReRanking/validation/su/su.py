"""
validation/su — Trova la soglia tau ottimale per il metodo SU-only.
[METODO ACCANTONATO dal team — tenuto solo per riferimento.]

Input: model.json (regressore gia' allenato, P(rerank|SU)) + candidate_level
del validation set (per ogni query: SU, e se il rerank aiuta o meno = label).
Per ogni candidato tau: applica il regressore -> probability, partiziona
rerank/skip (probability > tau), calcola recall@1 sul validation set.
Tiene il tau che massimizza recall@1. Scrive threshold.csv (colonna: tau).
[VUOTO — codice da incollare.]
"""
