# CapitalSelector – Abstract Core v1  
## (Der stackbare, selbstähnliche „Disembodied Mind“)

Datum: 2026-01-28  
Status: **kanonisch**  
Geltungsbereich: Fundamentalinvestor‑Neuron, Neuronen‑Stacks, Stack‑von‑Stacks  
Ziel: minimale, weltagnostische Spezifikation eines selektierenden Geistes

---

## 0. Leitidee

> **Ein CapitalSelector ist kein Modell der Welt.  
> Er ist eine ökonomische Bewegung,  
> die sich auf beliebigen Skalen wiederholen kann.**

Diese Bewegung ist:
- semantikfrei
- selbstähnlich
- rekursiv
- stackbar

Alles Weitere (Welten, Richtungen, Räume, Labyrinthe) sind **Adapter**, nicht Teil des Kerns.

---

## 1. Ontologie

Es existiert **genau eine** Entität:

> **CapitalSelector**

Es gibt **keine** Unterscheidung zwischen:
- Neuron
- Neuronen‑Stack
- Stack‑von‑Stacks

Diese Unterscheidungen sind rein **strukturell**, nicht ontologisch.

---

## 2. Kanonische Schnittstelle

### 2.1 Input: InvestmentOffer

Ein CapitalSelector empfängt ein Investitionsangebot:

$$
I \in \mathbb{R}_{\ge 0}
$$

Optional strukturiert in $K$ **unterscheidbare Kanäle**:

$$
I \rightarrow \{I_1, I_2, \dots, I_K\}
$$

**Wichtig:**
- Kanäle tragen **keine Semantik**
- sie sind nur unterscheidbar, nicht bedeutungsvoll
- „Welt“, „Neuron“ oder „Stack“ erscheinen hier identisch

---

### 2.2 Output: Allocation

Der CapitalSelector antwortet mit einer Allokation:

$$
\mathbf{w} \in \Delta(K)
$$

wobei $\Delta(K)$ der $K$‑dimensionale Simplex ist.

Interpretation:
- $w_k$ ist der Kapitalanteil für Kanal $k$
- keine Aktion, kein Befehl, keine Richtung
- reine Kapitalgewichtung

---

### 2.3 Feedback: Rückfluss & Kosten

Der CapitalSelector erhält Rückmeldung in Form von:

$$
(r, c)
$$

- $r$: Rückfluss / Return
- $c$: Kosten (Informations‑, Betriebs‑, Kapitalkosten)

Dies ist die **einzige** Form von Erfahrung.

---

## 3. Interner Zustand (vollständig)

Der gesamte innere Zustand eines CapitalSelectors ist:

$$
\theta = \{\text{wealth},\; \mathbf{w},\; \text{statistics}(r),\; \text{alive / rebirth flag}\}
$$

Explizit **nicht vorhanden**:
- Weltzustände
- Repräsentationen
- Karten
- Planung
- Ziele

---

## 4. Dynamik (Update‑Regel)

Ein CapitalSelector führt iterativ aus:

$$
\begin{aligned}
\text{wealth} &\leftarrow \text{wealth} + r - c \\
\mathbf{w} &\leftarrow \text{Reweight}(\mathbf{w}, r, \text{statistics}) \\
\text{if } \text{wealth} &< \tau:\ \text{rebirth}
\end{aligned}
$$

Eigenschaften:
- lokal
- rein ökonomisch
- skaleninvariant
- keine explizite Lernregel notwendig

---

## 5. Rebirth (Reset)

Rebirth ist:
- **kein Tod**
- **keine Evolution**
- sondern ein **episodischer Neustart**

Beim Rebirth können Teile von $\theta$ zurückgesetzt werden:
- Weltkopplung
- Allokation
- Statistiken

Der CapitalSelector selbst bleibt identisch.

---

## 6. Rekursion: Stackbarkeit

### 6.1 Zentrale Regel

> **Ein CapitalSelector kann selbst als Kanal  
> in einem anderen CapitalSelector auftreten.**

Formal:
- der Rückfluss eines Selectors ist selbst ein $r$
- seine Kostenstruktur ist selbst ein $c$

Damit ist jeder CapitalSelector:
- Kapitalnehmer
- Kapitalgeber
- Welt
- Kanal

zugleich.

---

### 6.2 Definition eines Stacks

Ein **Stack** ist definiert als:

$$
S := \text{CapitalSelector}(\text{Kanäle} = \{C_1, C_2, \dots, C_n\})
$$

mit $C_i$ selbst CapitalSelectors.

Es entsteht **kein neuer Typ**.

---

## 7. Selbstähnlichkeit (Fixpunkt)

Das System besitzt die Fixpunkt‑Eigenschaft:

$$
\text{CapitalSelector} \equiv \text{Stack} \equiv \text{Stack\text{-}von\text{-}Stacks}
$$

- gleiche Schnittstelle
- gleiche Dynamik
- gleiche Selektionslogik

Diese Selbstähnlichkeit ist die Grundlage für:
- Abstraktion
- Lernpfade
- Skalierung
- Robustheit

---

## 8. Bedeutung für „schwierige Probleme“

Schwierige Probleme zeichnen sich dadurch aus, dass:
- relevante Rückflüsse selten sind
- sie erst nach mehreren Selektionsstufen stabil werden
- frühe Kanäle irreführend sind

Der CapitalSelector‑Stack adressiert dies, indem:
- jede Ebene nur lokal selektiert
- höhere Ebenen nur mit bereits geglätteten Kanälen arbeiten
- Invarianzen unter Verfeinerung entstehen

---

## 9. Grenzen (bewusst)

Der CapitalSelector:
- besitzt kein Weltmodell
- kennt keine Symbole
- plant nicht
- garantiert keinen Erfolg

Er ist stattdessen:
- ehrlich scheiternd
- kostenbewusst
- selektiv
- rekursiv einsetzbar

---

## 10. Essenz

> **Der Disembodied Mind ist kein Gehirn,  
> sondern eine wiederholbare Selektionsbewegung.**

Weil diese Bewegung selbstähnlich ist, kann sie:
- Welten koppeln
- sich selbst koppeln
- und über Stacks hinweg Relevanz herausarbeiten

---

_Ende der kanonischen Spezifikation (v1)._  