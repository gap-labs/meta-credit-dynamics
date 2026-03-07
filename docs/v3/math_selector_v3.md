# math_selector_v3.md — Term-Aware CapitalSelector

## 1. Scope

math-v3 erweitert den CapitalSelector aus math-v2 um eine zeitstrukturierte Erwartungsbildung über Kanäle.

Die Weltaxiome aus math-v2 bleiben unverändert, insbesondere:

* Kanaldefinition
* Offer-Semantik
* Settlement
* Wealth / Dead
* Rebirth
* Stacking

math-v3 definiert ausschließlich eine Policyklasse im $P$-Raum: einen Selector, der Erwartungen über zeitliche Cashflow-Horizonte lernt.

Der Selector bleibt strikt lokal und nutzt ausschließlich:

$$
I(\tau), \quad R_P(\tau), \quad V_P(\tau), \quad \sigma_P
$$

## 2. Horizontstruktur

Der Selector führt eine endliche Menge von Zeit-Horizonten

$$
H = {h_0, \dots, h_{H-1}}
$$

Typisch:

$$
|H| \in {2,3,4}
$$

Beispiel:

```
h0 : immediate
h1 : short
h2 : long
```

Horizonte entstehen durch eine Abbildung

$$
\phi : \mathbb{R}_{\ge 0} \rightarrow H
$$

$$
\phi(\Delta \tau) = h
$$

### 2.1 Deterministische Bucketisierung (v1)

Für Implementierungen muss $\phi$ deterministisch sein. Eine minimale v1‑Definition mit drei Buckets:

Seien Schwellen $T_1 < T_2$.

$$
\phi(\Delta \tau) =
\begin{cases}
h_0 & 0 \le \Delta \tau \le T_1 \\
h_1 & T_1 < \Delta \tau \le T_2 \\
h_2 & \Delta \tau > T_2
\end{cases}
$$

Die Intervalle sind disjunkt und nutzen feste Randregeln, um deterministische Tie-Breaker zu garantieren.

## 3. Kanalweise Term-Erwartung

Für jeden Kanal $i$ und Horizont $h$ führt der Selector

$$
\mu_{i,h}
$$

mit

$$
\mu_{i,h} \approx \mathbb{E}[\text{Netto-Cashflow aus Kanal } i \text{ im Horizont } h]
$$

Selector-Zustand:

$$
\sigma_P = (\mu_{i,h}, \rho_i, \dots)
$$

## 4. Attribution von Ereignissen

Realisierte Cashflows entstehen aus

* Returns $R_P(\tau)$
* Verpflichtungen $V_P(\tau)$

Jeder Cashflow wird einem Kanal $i$ und Horizont $h$ zugeordnet:

$$
(i,h,\pi,r) = \psi(\text{event})
$$

Dabei gilt:

* $i$ – Kanal
* $h$ – Horizontbucket
* $\pi$ – signed Cashflow
* $r$ – optionaler Risikoimpuls

$\psi$ basiert ausschließlich auf lokal beobachtbaren Informationen:

* Ursprungsoffer
* Claim-ID
* ursprüngliche Maturity

### 4.0 Eventtypen (v1)

Die folgende Tabelle dokumentiert die minimale Event-Menge, die $\psi$ akzeptiert.
Dabei gilt jeweils $x > 0$.

| `event.kind` | $\pi$ | $r$ |
| --- | --- | --- |
| `RETURN` | $+x$ | $0$ |
| `DUE_CASH` | $-x$ | $0$ |
| `COST` | $-x$ | $0$ |
| `ROLLOVER` | $0$ | $1$ |
| `FAIL` | $0$ | $1$ |

### 4.1 Kanalzuordnung

Wenn ein Event mit einem Claim verknüpft ist:

$$
i = \text{claim.origin\_channel\_id}
$$

Fallback: Falls kein Claim referenziert ist, wird der Kanal des Events verwendet.

### 4.2 Horizonzuordnung

Für einen Claim mit ursprünglicher Maturity $t''$ gilt

$$
\Delta \tau = \max(0, t'' - \tau_{now})
$$

$$
h = \phi(\Delta \tau)
$$

Die **origin maturity** wird verwendet, damit Umschreibungen (Rollovers) neue Claims mit neuer Horizon erzeugen.

### 4.3 Cashflowbeitrag

Cashflows werden signed gebucht:

Return-Inflow $x>0$

$$
\pi = +x
$$

Due-Cash-Zahlung $x>0$

$$
\pi = -x
$$

Kosten werden ebenfalls negativ gebucht.

Falls ein Event mehrere Claims betrifft, wird der Betrag proportional zu den Claim-Nominalen aufgeteilt.

### 4.4 Risikoimpuls

Optional wird ein Risikoimpuls $r$ erzeugt.

Minimaldefinition (v1):

* Rollover / Claim-Rewrite

$$
r = 1
$$

* sonst

$$
r = 0
$$

Dieser Impuls treibt den kanalweisen Risikoindikator.

## 5. Termed-EWMA Update

Für Cashflow $\pi$ aus Kanal $i$ im Horizont $h$:

$$
\mu_{i,h} \leftarrow (1-\beta)\mu_{i,h} + \beta\pi
$$

## 6. Risikoindikator

Optional führt der Selector

$$
\rho_i
$$

Beispiele:

* Settlement-Fail
* Roll-Over
* Liquiditätsstress

Update:

$$
\rho_i \leftarrow (1-\beta_r)\rho_i + \beta_r r_t
$$

## 7. Kanalbewertung

$$
u_i = \sum_h \gamma_h \mu_{i,h} - \lambda \rho_i
$$

## 8. Kapitalallokation

$$
w'_i = w_i \exp(\eta u_i)
$$

$$
w_i = \frac{w'_i}{\sum_j w'_j}
$$

## 9. Selector-Taxonomie

math-v3 definiert eine Familie von Selector-Policies.

Umsetzungsphasen:

* **Phase I (ohne Engine-Change):** Selector-2 bis Selector-3
* **Phase II (mit Engine-Change):** Selector-4 bis Selector-5

Selector-0/1 dienen als Baselines.

### Selector-0 — Random

Keine Erwartungsbildung.

$$
w_i = \frac{1}{K}
$$

Baseline für Experimente.

### Selector-1 — Myopic Bandit

Score basiert auf

$$
\mu_i = \text{EWMA}(\pi_i)
$$

Keine Zeitstruktur.

### Selector-2 — Term-Aware

Erwartungen über

$$
\text{channel} \times \text{horizon}
$$

$$
\mu_{i,h}
$$

Diskontierte Bewertung.

### Selector-3 — Term + Risk

Zusätzliche Risikoindikatoren

$$
\rho_i
$$

Beispiele:

* rollover rate
* settlement failure

### Selector-4 — Liquidity-Aware

Zusätzliche Bewertung der eigenen Verpflichtungsstruktur.

Due-Curve:

$$
D_h
$$

Penalty:

$$
\text{LiquidityMismatch} = \max(0, D_h - \text{buffer}_h)
$$

### Selector-5 — Strategic Credit Actor

Explizite Nutzung von

* Fristentransformation
* Settlement-Strategien
* strukturellen Kreditketten

Selector beginnt aktiv

$$
\text{borrow short / invest long}
$$

zu betreiben.

## 10. Spezialfall

Für

$$
|H| = 1
$$

ergibt sich eine nahe Verwandtschaft zum Phase-G Selector, aber keine exakte Gleichheit.

## 11. Minimalitätsprinzip

Selector nutzt ausschließlich:

* beobachtete Cashflows
* beobachtete Verpflichtungen
* sichtbare Kanalangebote

Keine globale Weltinformation.

---
