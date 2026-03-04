# Meta‑Credit Dynamics – Math v1 (kanonisch)

Dieses Dokument führt `math.md` und `math-delta.md` zusammen. Ziel ist eine **formal umsetzbare** Spezifikation.
Jeder Abschnitt enthält:
- **Status:** implemented | partial | planned
- **Decisions:** offene Entscheidungen, die vor Implementierung zu klären sind

**Inhaltsverzeichnis**
- Teil I – Mathematische Grundlagen
  - 0. Leitidee: Learning as Inhibition
  - 1. Notation und Konventionen
  - 2. Systemprimitiven
  - 3. ETF‑Baseline und Overperformance
  - 4. Gewicht‑Update (Reweighting)
  - 5. Aktivitätsgrad und Mischportfolio
  - 6. Statistik und Stabilität
  - 7. Stacks (Mediation)
  - 8. Sedimented Mediation Paths (Phase E)
  - 9. Rebirth
  - 10. Freeze (Inference Mode)
- Teil II – Kapitalstruktur und Inhibition (Fundamentalinvestor‑Erweiterung)
  - 11. Equity/Debt und Kapitaldynamik
  - 12. Kreditkondition und Austrocknung
  - 13. Sparsity‑Maske
  - 14. Rebirth‑Pool (Kapitalerhaltung)
- Teil III – Implementierungsstatus und offene Entscheidungen
  - 15. Status‑Marker
  - 16. Entscheidungstypen (Config vs. Architektur vs. Modus)
  - 17. Offene Entscheidungen (Liste)
- Teil IV – Betriebsprofile (Config‑Kohärenz)
  - 18. Profil A – Canonical Inhibition Mode (Default / v1)
  - 19. Profil B – Analytical / Research Mode (optional)
- Teil V – Zusammenfassung (Formal)

---

# Teil I – Mathematische Grundlagen

## 0. Leitidee: Learning as Inhibition
**Status:** implemented (konzeptionell)  
**Decisions:** keine

**Kernidee:** Lernen erfolgt primär durch **Inhibition** (Ausschluss schlechter Pfade), nicht durch Optimierung.
Es gibt keine positive Zielfunktion; Stabilität entsteht durch Schwellen und irreversible Ausschlüsse.

---

## 1. Notation und Konventionen
**Status:** planned  
**Decisions:**
- D1: Zeit $t$ diskret mit fester Schrittweite vs. variablem $dt$?

Einheiten:
- Zeit $t \in \mathbb{N}$ (diskret), optional mit $dt>0$.
- **Klarstellung (v1):** $\Delta t$ ist **intern** definiert; die World liefert kein $dt$.
- Indizes: Einheiten/Explorer $i \in \mathcal{U}$, Kanäle $k=1,\dots,n$.

Simplex:
$$
\Delta^n = \{ w \in \mathbb{R}^n_{\ge 0} \mid \sum_k w_k = 1 \}
$$

---

## 2. Systemprimitiven
**Status:** implemented (partial)  
**Decisions:**
- D2: Kanäle als semantikfreie Renditequellen oder eigenständige Einheiten?

**Implemented in:** `capitalmarket/capitalselector/core.py` (Channel, CapitalSelector)  
**Partial notes:** Kostenmodell ist im Code nicht kanalweise aufgeteilt.

### 2.1 Gewichte
Für jede Einheit $i$:
$$
w_i(t) \in \Delta^n
$$

### 2.2 Returns, Kosten, Nettofluss
Brutto-Return je Kanal:
$$
r_k(t) \in \mathbb{R}
$$

Gewichteter Brutto‑Return:
$$
R_i(t) := \sum_k w_{ik}(t)\, r_k(t)
$$

Kosten (skalar):
$$
C_i(t) := C_{info}(t) + C_{activity}(t) + C_{aggregation}(t)
$$

Nettofluss:
$$
\Pi_i(t) := R_i(t) - C_i(t)
$$

Kanal‑spezifischer Netto‑Beitrag (für Reweighting):
$$
\pi_{ik}(t) := w_{ik}(t)\, r_k(t) - \frac{w_{ik}(t)}{\sum_j w_{ij}(t)}\, C_i(t)
$$

**Decisions:**
- D3: Kostenverteilung proportional zu $w_{ik}$ (wie oben) vs. alternative Kostenmodelle.

---

## 3. ETF‑Baseline und Overperformance
**Status:** planned  
**Decisions:**
- D4: ETF‑Baseline verpflichtend (kanonisch) oder optional?

Referenz‑Allokation $q(t)\in\Delta^n$:
$$
q(t) = (q_1(t),\dots,q_n(t))
$$

ETF‑Return:
$$
R^{ETF}(t) := \sum_k q_k(t)\, r_k(t)
$$

Brutto‑Overperformance:
$$
\Delta_i^{gross}(t) := R_i(t) - R^{ETF}(t)
$$

Netto‑Overperformance:
$$
\Delta_i^{net}(t) := \Pi_i(t) - R^{ETF}(t)
$$

---

## 4. Gewicht‑Update (Reweighting)
**Status:** partial  
**Decisions:**
- D5: Score basiert auf $\pi_{ik}$, $\Pi_i$ oder $\Delta_i$?

**Implemented in:** `capitalmarket/capitalselector/reweight.py`, `capitalmarket/capitalselector/core.py`  
**Partial notes:** Score im Code basiert auf $r$‑Vektor minus skalarer $\mu$ (ohne Kosten).

Score pro Kanal:
$$
s_{ik}(t) := \pi_{ik}(t) - \mu_i(t)
$$

Exponentiated‑Gradient‑Update:
$$
w_{ik}(t+1) = \frac{w_{ik}(t)\, e^{\eta s_{ik}(t)}}{\sum_j w_{ij}(t)\, e^{\eta s_{ij}(t)}}
$$

**Hinweis:** Falls ETF‑Baseline aktiv ist, kann $s_{ik}(t)$ auf $\Delta_i^{net}$ bzw. $g_{ik}(t):=r_k(t)-R^{ETF}(t)$ basieren.

---

## 5. Aktivitätsgrad und Mischportfolio
**Status:** planned  
**Decisions:**
- D6: $\alpha_i$ fest vs. dynamisch?

Aktives Portfolio $a_i(t)\in\Delta^n$:
$$
w_i(t) = (1-\alpha_i(t))\, q(t) + \alpha_i(t)\, a_i(t)
$$

Aktive Allokation:
$$
a_{ik}(t+1) \propto a_{ik}(t)\, \exp(\eta\, g_{ik}(t))
$$

Minimaler Score:
$$
g_{ik}(t) := r_k(t) - R^{ETF}(t)
$$

---

## 6. Statistik und Stabilität
**Status:** partial  
**Decisions:**
- D7: Statistik auf $\Pi_i$ (Netto) oder $R_i$ (Brutto)?

**Implemented in:** `capitalmarket/capitalselector/stats.py`, `capitalmarket/capitalselector/stack.py`  
**Partial notes:** EWMA wird im Code auf $r$ (Brutto) angewendet; Stabilitätsgrenzen existieren nur in Stack‑Config.

EWMA‑Mittelwert:
$$
\mu_i(t) = (1-\beta)\mu_i(t-1) + \beta\, \Pi_i(t)
$$

EWMA‑Varianz:
$$
\sigma_i^2(t) = (1-\beta)\sigma_i^2(t-1) + \beta\,(\Pi_i(t)-\mu_i(t))^2
$$

Drawdown auf kumuliertem Nettofluss:
$$
DD_i(t) = \max_{\tau \le t}\left(\sum_{u=\tau}^{t} -\Pi_i(u)\right)
$$

Stabilität:
$$
\text{stable}_i(t) := (\mu_i(t) \ge \tau_\mu) \land (\sigma_i(t) \le \tau_\sigma) \land (DD_i(t) \le \tau_{dd})
$$

---

## 7. Stacks (Mediation)
**Status:** partial  
**Decisions:**
- D8: Stack‑Weights $\alpha_i$ gleichgewichtet oder lernbar?

**Implemented in:** `capitalmarket/capitalselector/stack.py`  
**Partial notes:** interner Stack nutzt Gleichgewichtung; Stack‑Weights $\alpha_i$ sind nicht extern steuerbar.

Stack‑Return:
$$
R_S(t) = \sum_{i\in S} \alpha_i\, R_i(t),\quad \alpha \in \Delta^{|S|}
$$

Stack‑Kosten:
$$
C_S(t) = C_{agg} + \sum_{i\in S} C_i(t)
$$

Dissolution:
$$
\neg \text{stable}_S(t_d)
$$

---

## 8. Sedimented Mediation Paths (Phase E)
**Status:** partial  
**Decisions:**
- D9: Bedeutung von $w$ im Sediment‑Node (behalten vs. entfernen)?

**Implemented in:** `capitalmarket/capitalselector/sediment.py`, `capitalmarket/capitalselector/stack.py`  
**Partial notes:** Sediment‑Node speichert kein $w$ im Code; DAG‑Kanten werden nur geloggt.

Sediment‑Node:
$$
\nu = (M, P, w, \phi, t_d, r)
$$

Interpretation (minimal):
$$
w := (w_i(t_d))_{i\in M}
$$

Sediment‑DAG:
$$
\mathcal{G}=(\mathcal{V},\mathcal{E}),\quad (\nu_i\to\nu_j) \iff t_i < t_j \land r_i=r_j
$$

Exclusion‑Rule:

Sediment wirkt ausschließlich als struktureller Filter bei der Bildung neuer Stacks.
Es hat keinerlei Einfluss auf Reweighting, Statistik-Updates oder Kapitaldynamik.
$$
\exists \nu \in \mathcal{V}: M' = M_\nu \land \phi' = \phi_\nu
$$

---

## 9. Rebirth
**Status:** partial  
**Decisions:**
- D10: welche Zustände resetten? (Weights, Stats, Wealth, Flags)

**Implemented in:** `capitalmarket/capitalselector/core.py`, `capitalmarket/capitalselector/rebirth.py`  
**Partial notes:** Gewichte werden zurückgesetzt, Stats nicht.

**BehavioralState (definiert):**  
Der Verhaltenszustand einer Einheit umfasst ausschließlich:
- Gewichte $w_i$
- zentrale Statistiken $(\mu_i, \sigma_i^2, DD_i)$

Minimaler Reset:
$$
w_i(t+1) \leftarrow \text{uniform simplex}
$$
$$
\mu_i(t+1) \leftarrow 0,\quad \sigma_i^2(t+1) \leftarrow \sigma^2_{seed}
$$
$$
DD_i(t+1) \leftarrow 0
$$

Wealth‑Clamp:
$$
\text{wealth}_i(t+1) \leftarrow \max(\text{wealth}_i(t+1),\tau_{rebirth})
$$

---

## 10. Freeze (Inference Mode)
**Status:** planned  
**Decisions:**
- D11: Stats weiter updaten oder einfrieren?

**Implemented in:** n/a (nicht im Code vorhanden)

Freeze‑Invarianten:
$$
w_i(t+1)=w_i(t),\quad \mathcal{G}_{t+1}=\mathcal{G}_t,\quad S_{t+1}=S_t
$$

---

# Teil II – Kapitalstruktur und Inhibition (Fundamentalinvestor‑Erweiterung)

## 11. Equity/Debt und Kapitaldynamik
**Status:** planned  
**Decisions:**
- D12: Negative Equity zulassen bis Exit‑Trigger?

Zustand:
$$
E_i(t)\ge 0,\quad D_i(t)\ge 0,\quad K_i(t)=E_i(t)+D_i(t)
$$

**Entscheidung (v1 / Prod‑v0):** D12.B – Nicht zulassen (Hard‑Clamp)  
Nach jedem Schritt gilt:
$$
E_i(t+1) \leftarrow \max\big(0,\; E_i(t+1)\big)
$$

Brutto‑Profit:
$$
\Pi_i^{gross}(t) := K_i(t)\, R_i(t)
$$

Kosten:
$$
\text{Cost}_i(t) := D_i(t)c_{D,i}(t) + E_i(t)c_{E,i}(t) + b_i(t)
$$

Netto‑Profit:
$$
\Pi_i^{net}(t) := \Pi_i^{gross}(t) - \text{Cost}_i(t)
$$

Equity‑Update:
$$
E_i(t+1) := E_i(t) + \Pi_i^{net}(t)
$$

---

## 12. Kreditkondition und Austrocknung
**Status:** planned  
**Decisions:**
- D13: Form der Konditionsfunktion $\psi_i$ (Sigmoid vs. Schwelle)?

**Hinweis (v1):**  
Teil II ist in Profil A vollständig inaktiv.  
Die Konditionsfunktion $\psi_i$ wird in v1 weder berechnet noch ausgewertet.


Kreditkondition:
$$
\psi_i(t) := \sigma\big(a_i\mu_i(t) - b_i\sigma_i(t) - d_iDD_i(t)\big)
$$

Debt‑Update:
$$
D_i(t+1) = \operatorname{clip}_{[0,D_{\max,i}]}
\big(\ell_{\max,i}\,E_i(t+1)\,\psi_i(t)\big)
$$

Austrocknung (monoton):
$$
D_i(t+1) \le D_i(t)\quad \text{falls}\quad \psi_i(t+1) < \psi_i(t)
$$

---

## 13. Sparsity‑Maske
**Status:** planned  
**Decisions:**
- D14: Masken‑Update deterministisch vs. stochastisch?

Maske $m_i(t)\in\{0,1\}^n$:
$$
a_{ik}(t)=0\quad \text{falls}\quad m_{ik}(t)=0
$$

Minimal‑Update:
$$
m_{ik}(t+1)=
\begin{cases}
0 & \text{falls } a_{ik}(t)\approx 0 \text{ über } H \text{ Schritte}\\
1 & \text{falls } k \sim q(t)
\end{cases}
$$

---

## 14. Rebirth‑Pool (Kapitalerhaltung)
**Status:** planned  
**Decisions:**
- D15: Seed‑Kapital aus Pool verpflichtend?

Pool‑Update:
$$
P(t+1) \leftarrow P(t) + \max(0, K_i(t+1))
$$

Seed aus Pool:
$$
E_i(t+1) \leftarrow E_{seed},\quad P(t+1) \leftarrow P(t+1) - E_{seed}
$$

---

# Teil III – Implementierungsstatus und offene Entscheidungen

## 15. Status‑Marker
Jeder Abschnitt enthält:
```
Status: implemented | partial | planned
```

## 16. Entscheidungstypen (Config vs. Architektur vs. Modus)

**Config (runtime‑wählbar):** D1, D3, D5, D7, D8, D11, D13, D14, D15

**Architektur (einmalig festlegen):** D2, D4, D10, D12

**Modus (Use‑Case abhängig):** D6, D9

## 17. Offene Entscheidungen (Liste)
- D1 (Config): Zeitdiskretisierung / dt‑Semantik
- D2 (Architektur): Kanal‑Ontologie
- D3 (Config): Kostenverteilung auf Kanäle
- D4 (Architektur): ETF‑Baseline verpflichtend?
- D5 (Config): Score‑Definition für Reweighting
- D6 (Modus): Dynamik von $\alpha_i$
- D7 (Config): Statistik‑Signal (Netto vs. Brutto)
- D8 (Config): Stack‑Weights
- D9 (Modus): Sediment‑Node $w$
- D10 (Architektur): Rebirth‑Reset‑Menge
- D11 (Config): Freeze‑Stats
- D12 (Architektur): Negative Equity zulassen?
- D13 (Config): Konditionsfunktion $\psi_i$
- D14 (Config): Sparsity‑Update‑Regeln
- D15 (Config): Pool‑Seed verpflichtend?

---

# Teil IV – Betriebsprofile (Config‑Kohärenz)

Obwohl mehrere Parameter technisch konfigurierbar sind (vgl. §16/§17),
definiert das System **keinen offenen Konfigurationsraum**.

Stattdessen existieren **wenige, intern konsistente Betriebsprofile**,
in denen alle Konfigurationsentscheidungen aufeinander abgestimmt sind.

Beliebige Kombinationen einzelner Config‑Parameter sind **explizit nicht unterstützt**.

---

## 18. Profil A – Canonical Inhibition Mode (Default / v1)

Dieses Profil definiert den **kanonischen Betrieb** des Systems.
Alle mathematischen Aussagen in `math-v1.md` beziehen sich,
sofern nicht anders angegeben, auf dieses Profil.

**Ziel:** maximale Robustheit, deterministische Dynamik,
keine implizite Optimierung.

| Decision | Festlegung |
|--------|------------|
| D1 Zeitdiskretisierung | diskret, konstantes $\Delta t$ |
| D3 Kostenverteilung | proportional zu $w_{ik}$ |
| D5 Score | kanalweiser Netto‑Beitrag $s_{ik} = \pi_{ik} - \mu_i$ |
| D7 Statistik‑Signal | Nettofluss $\Pi_i$ |
| D8 Stack‑Weights | gleichgewichtet |
| D11 Freeze | vollständiges Einfrieren von Gewichten **und** Stats; Topologie/Sediment konstant |
| D13 Kreditkondition | **inaktiv** (Teil II nicht aktiv in v1) |
| D14 Sparsity | **inaktiv** (nicht Teil von v1) |
| D15 Rebirth‑Pool | deaktiviert |

**Eigenschaften:**
- Lernen erfolgt ausschließlich durch Inhibition
- Keine positive Zielfunktion
- Stabilität ist primäres Selektionskriterium
- Deterministische Semantik (CPU/GPU‑kompatibel)

Dieses Profil ist **verbindlich für v1 / Prod‑v0**.

---

## 19. Profil B – Analytical / Research Mode (optional)

Dieses Profil dient ausschließlich der Analyse, Exploration
und wissenschaftlichen Untersuchung.

Es ist **nicht kanonisch** und **nicht produktiv**.

Abweichungen gegenüber Profil A sind zulässig, z. B.:

- alternative Score‑Definitionen (Brutto vs. Netto)
- parallele Statistik‑Signale
- lernbare Stack‑Weights
- fortlaufende Statistik‑Updates im Freeze
- aktivierter Rebirth‑Pool

**Wichtige Invariante:**  
Auch im Research Mode dürfen keine gespeicherten Sedimentdaten
in die Dynamik rückgespeist werden.

**Warnhinweis:**  
Ergebnisse aus Profil B sind **nicht** direkt mit Profil A vergleichbar.

---

**Normative Festlegung:**  
Implementierungen müssen mindestens **Profil A vollständig unterstützen**.
Profil B ist optional und explizit als nicht‑kanonisch zu kennzeichnen.

# Teil V – Zusammenfassung (Formal)

Das System lernt durch **irreversiblen Ausschluss** instabiler Strukturen.
Die Optimierung ist lokal (Reweighting), jedoch ohne globale Zielfunktion.
Stabilität und Sediment definieren die Topologie des verbleibenden Suchraums.
