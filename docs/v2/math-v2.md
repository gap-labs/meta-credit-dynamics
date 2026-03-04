# Phase H – Kambrische Explosion (hoffentlich)

## 1. Weltrepräsentation (Kanalbasiertes Modell)

### 1.1. Ziel

Wir ersetzen den unscharfen Begriff „Welt“ durch eine strikt operative,
kanalbasierte Beschreibung.

Die Welt ist kein expliziter Zustandsraum.
Sie ist ein System von Angeboten.

Ein System interagiert ausschließlich über veröffentlichte Kanäle.

---

### 1.2. Zeit

Wir betrachten kontinuierliche Zeit:

$$
\tau \in \mathbb{R}_{\ge 0}
$$

Kanäle publizieren zu beliebigen Zeitpunkten.  
Ein Kanal $K_\mu$, $\mu \in I$, $I$ beliebige Indexmenge, ist zu einem Zeitpunkt $\tau$ aktiv, wenn er zu diesem Zeitpunkt ein **aktives** (s.u.) Kreditangebot publiziert hat.

Es wird **keine Existenzfunktion** eingeführt.
Nichtexistenz ist modelliert durch Nicht-Publikation.

Damit ist die Welt als Ereignisprozess über kontinuierlicher Zeit beschrieben.
Diskretisierung gehört zur Implementierung und ist **nicht** Teil von math‑v2.


---

### 1.3. Kanaldefinition

Ein Kanal $K_\mu$ publiziert zu Zeitpunkt $\tau = t_k$ ein Kreditangebot k:
$$
k = (\mu, (t_k, t'_k, t''_k, \delta_k, A_k, C_k)) \in S \subset I \times \mathbb{R}^6
$$

mit:

$A_k ∈ \mathbb{R}_{> 0}$ maximal ausschüttbares Kapital, $C_k \in \mathbb{R}$ Rückzahlungsfaktor, $\delta_k > 0$ Angebotsdauer, $t'_k > t_k$ Zusicherungsbeginn, $t''_k > t'_k + \delta_k$ spätester Ablösezeitpunkt.

Interpretation:

Ein Kreditnehmer kann zu einem Zeitpunkt $\tau$ mit $t_k \le \tau < t'_k$ einen Betrag $0 < x_{\tau} < A^{rem}_k(\tau)$ reservieren, Def. $A^{rem}_k$ s.u. Es ist dem Designer freigestellt, einen Mindestbezug zu definieren.
Die Reservation verpflichtet zum Kapitalbezug zum Zeitpunkt $t'_k$. Gilt $t'_k \le \tau < t'_k + \delta_k$ sind Reservation und Bezug ein und dieselbe Handlung. Ab $\tau \ge t'_k + \delta_k$ ist weder Reservation noch Bezug möglich.

Ein Kreditangebot hat Zustand. Durch Reservation oder Bezug ändert sich das insgesamt verfügbare Kapital für $t'_k + \delta_k > \tau \ge t_k$ gem.
$$
A^{rem}_k(\tau) = A_k - \sum_{x_{t}, t \le \tau} x_t \qquad \text{Bem.:Integral für Kontinuum}
$$
Impl.Notes: Diskret vs. Kontinuum ist eine Implementierungsentscheidung (Summe vs. Integral).

Angebotsschluss kann auch durch $A^{rem}_k(\tau) = 0 ,\; \tau \ge t'_k + \delta_k$ modelliert werden.

Ein Kreditangebot k ist zu einem Zeitpunkt $\tau$ **aktiv** im Kanal $\mu$, wenn $k = (\mu,(...))$ und mindestens eine der folgenden Handlungen zulässig ist:
Reservation ($t_k \le \tau < t'_k$), Bezug ($t'_k \le \tau < t'_k + \delta_k$) oder Rückzahlung ($t'_k \le \tau \le t''_k$) für bereits bezogenes Kapital. Insbesondere gilt: Ohne tatsächlichen Bezug existiert keine Rückzahlungspflicht; ist das Bezugsfenster $[t'_k, t'_k+\delta_k)$ ohne Bezug abgelaufen, ist das Angebot ab $t'_k+\delta_k$ inaktiv.

Wir definieren die Kanalaktivität $\theta$ als Abbildung
$$
\theta: \mathbb{R}_{\ge 0} \times I \rightarrow \mathcal{P}(S)
$$
durch
$$
\theta(\tau,\mu) = 
\{k | k\;\text{ist aktives Angebot von}\;K_\mu\;\text{zum Zeitpunkt}\;\tau\}
$$
Dabei bedeutet „aktiv" explizit: Es ist mindestens eine zulässige Handlung vorhanden (Reservation, Bezug oder Rückzahlung), wobei Rückzahlung nur für zuvor tatsächlich bezogenes Kapital zulässig ist.

Falls mehrere Anfragen zum gleichen Zeitpunkt $\tau$ auftreten,
ist die **Zuteilungsregel Teil der Offer-Definition** (z.B. greedy fill / proportional / priority).

Ein Kanal $K_\mu$ ist zu einem Zeitpunkt $\tau$ vollständig beschrieben durch die Menge seiner Kreditangebote:
$$
K_\mu(\tau) = \{k | k \in \theta(\tau,\mu)\}
$$

Bezugszeiträume verschiedener Angebote **dürfen** sich überschneiden.
In diesem Fall ist eine eindeutige Zuteilungs-/Matching-Regel pro Offer erforderlich,
und jedes Offer führt sein Restvolumen als eigenen Zustand.

---

### 1.4. Spezialfälle (keine neuen Kategorien)

Dieses Modell erlaubt ohne Zusatzsemantik:

| Bedingung | → | Bedeutung |
| --- | --- | --- |
| $C > 1$ | → | verzinster Kredit |
| $C = 1$ | → | zinsfreier Kredit |
| $0 < C < 1$ | → | Subvention |
| $C = 0$ | → | Geschenk |
| $C < 0$ | → | Belohnung für Kapitalaufnahme |

Es existieren keine separaten Kanaltypen.
Alles ist Parametrisierung desselben Objekts.

---

### 1.5. Welt (Systemschnittstelle)

Eine Teilmenge der Indexmenge $I$ wird partitioniert in zwei Mengen $M$ und $N$, so dass gilt:
$$
M \cap N = \emptyset \\
M \cup N \subset I
$$
Der Eingaberaum zum Zeitpunkt $\tau$ wird definiert als:
$$
I(\tau) = \bigcup_{\mu \in M} \theta(\tau,\mu)
$$

Damit ist Welt $W$ zum Zeitpunkt $\tau$ das Tupel
$$
W(\tau) = (I(\tau), N)
$$

$N$ definiert den Aktionsraum als Raum, auf den Kreditangebote ausgegeben werden können.
In diesem Design ist einem Akteur nicht global bekannt, welche Aktionen (=Kreditangebote) im Aktionsraum von ihm ausgelöst wurden.

Es steht dem Weltdesigner frei, eine Abbildung
$$
\phi : N \rightarrow M
$$

zu definieren, welche veröffentlichte Offers (auch vermittelt) wieder in den Inputraum einspeist.

Dadurch ist rekursive Kanalstruktur auf Weltebene zulässig.
Primitive $P$-Abbildungen bleiben jedoch nicht-rekursiv; Rekursion entsteht ausschließlich durch Kanalabbildung über die Zeitfolge $\tau_n$.

---

## 2. $P$-Raum

### 2.1. Definition

Eine $P$-Abbildung ist ein iteratives Konstrukt mit zeitbehaftetem Zustand $\sigma_P$, welches aus einem Zustandsupdate $Z_P$ und einer Angebotserstellung $O_P$ besteht, sowie den nächsten Bewertungszeitpunkt ausgibt.

Sei 
$$
X(\tau) = \{k = (\mu, (t_k, t'_k, t''_k, \delta_k, A_k, C_k))  | k \in S \land \mu \in N \land t_k = \tau \}
$$

Initialisierung: 

Sei $\tau_0$ gegeben mit Startzustand $\sigma_P(\tau_0)$. Die Iteration beginnt bei $\tau_0$ mit $O_P$.
$$
O_P(\tau_0, I(\tau_0), \sigma_P(\tau_0)) = (A_P(\tau_0) \subset X(\tau_0), \tau_1)
$$
Danach gilt rekursiv:
$$
Z_P(\sigma_P(\tau_{n-1}), V_P(\tau_{n}), R_P(\tau_{n})) = \sigma_P(\tau_{n})
$$
und
$$
O_P(\tau_n, I(\tau_n), \sigma_P(\tau_n)) = (A_P(\tau_n) \subset X(\tau_n), \tau_{n+1})
$$


$V_P(\tau)$ resp. $R_P(\tau)$ sind die zum Zeitpunkt $\tau$ einzulösenden Verpflichtungen bzw. erhaltenen Returns. $A_P(\tau)$ ist die neu publizierte Menge an Offers. Es gilt $\tau_{n+1} > \tau_n$. $\tau_{n+1}$ ist der nächste Zeitpunkt, zu welchem die beschriebene Iteration durchgeführt wird.

---

### 2.2. Invarianten

#### 2.2.1. Gliederung

Invarianten sind in **drei Klassen** gliedern:

1. **Typ-/Schnittstelleninvarianten** (damit Algebra/Stackbarkeit möglich ist)
2. **Finanz-/Abwicklungsinvarianten** (damit es kein magisches Geld gibt)
3. **Zeit-/Ablauf-Invarianten** (damit Iteration wohldefiniert ist)

Keine Präferenzen, keine Ziele.

#### 2.2.2. Typ- und Schnittstelleninvarianten

**(I1) Lokale Sichtbarkeit**
$O_P$ darf nur von $(\tau_n, I(\tau_n), \sigma_P(\tau_n))$ abhängen.
$Z_P$ darf nur von $(\sigma_P(\tau_{n-1}), V_P(\tau_n), R_P(\tau_n))$ abhängen.
Kein Zugriff auf $I\setminus(M\cup N)$ oder andere „verdeckte“ Weltzustände.

**(I2) Output-Kanalbindung**
Alle von $P$ publizierten Offers liegen auf Outputkanälen:
$$
k \in A_P(\tau_n)\ \Rightarrow\ \mu \in N
$$
und Publikationszeit passt:
$$
k=(\mu,(t_k,\dots))\in A_P(\tau_n)\ \Rightarrow\ t_k=\tau_n
$$

**(I3) Offer-Wohldefiniertheit**
Für jedes publizierte Offer $(k \in A_P(\tau_n))$ gelten die Offer-Nebenbedingungen aus 1.3:
$$
A_k>0,\ \delta_k>0,\ t'_k>t_k,\ t''_k>t'_k+\delta_k
$$
(und $C_k\in\mathbb{R}$ beliebig).

> Zweck: Damit kann *jedes* $P$ als „Anbieter“ auftreten, ohne neue Semantik.


#### 2.2.3. Wealth und Dead

**(I4) Wealth**

Für jeden Prozess $P$ definieren wir die bilanzielle Eigenkapitalposition

$$
W_P(\tau) = L_P(\tau) - \sum_{\text{zum Zeitpunkt }\tau\text{ fällige Verpflichtungen}} V_i(\tau)
$$
Zeitpunktkonvention: $W_P(\tau)$ wird nach Verbuchung von $R_P(\tau)$ und nach Abschluss der zulässigen bilateralen Settlement-Verarbeitung bei $\tau$ bestimmt. Maßgeblich sind die danach verbleibenden bei $\tau$ fälligen Verpflichtungen.
Impl.Notes: "fällige Verpflichtungen" meint exakt zum Zeitpunkt $\tau$ fällige Zahlungen; nicht-fällige zukünftige Verpflichtungen gehen zu diesem Zeitpunkt nicht in $W_P(\tau)$ ein.

wobei $L_P(\tau)$ die Liquidität bezeichnet.

Wealth ist vollständig aus lokaler Buchhaltung ableitbar.
Es enthält keine Bewertungsannahmen über externe Positionen.
Insbesondere werden zukünftige Returns nicht als Wealth gezählt; sie können nur dann Wealth beeinflussen,
wenn sie als Return $R_P$ realisiert oder durch Abtretung in Liquidität transformiert wurden.

**(I5) Dead**

Definiere den Todeszeitpunkt

$$
\tau_{dead} = \inf \{ \tau \mid W_P(\tau) < 0 \}
$$

Für $\tau \ge \tau_{dead}$ gilt:

$$
\sigma_P(\tau) = \text{dead}
$$

Dead ist irreparabel und absorbierend.


**(I6) Verschleppungscharakterisierung**

Insolvenz kann durch Aufnahme neuer Verpflichtungen zeitlich verschoben werden.

Wird bei $\tau$ Kapital $x$ aufgenommen, so gilt:

$$
W_P(\tau) \mapsto W_P(\tau) + x
$$

und es existiert $\tau' > \tau$ mit

$$
\Delta W_P(\tau') = -x \cdot C
$$

Insolvenz wird nur vermieden, wenn zwischen $\tau$ und $\tau'$ ausreichender Return erfolgt.

Verschleppung ist somit rein cashflow-getrieben und kein Sondermechanismus.

**(I6a) Rückzahlungszulässigkeit**

Rückzahlungsverpflichtungen dürfen nur aus tatsächlich bezogenem Kapital entstehen.
Ist ein Offer ohne Bezug durch das Bezugsfenster gelaufen, erzeugt es keine Rückzahlungs-Events in $V_P$/$R_P$.

**(I6b) Bilaterale Umschreibung fälliger Claims**

Fällige Verpflichtungen dürfen bilateral durch Settlement umgeschrieben werden.
Dabei gilt: Ein Schuldner kann statt voller Cash-Zahlung ein Paket aus Teil-Cash und einem oder mehreren neuen zukünftigen Claims anbieten.
Die Annahme erfolgt ausschließlich bilateral durch den jeweiligen Gläubiger (lokale Policy); es gibt keinen globalen Markt und kein zentrales Preissignal.
Wird kein Settlement akzeptiert und kann die fällige Verpflichtung nicht in Cash bedient werden, folgt Insolvenz gemäß (I5).

#### 2.2.4. Zeit- und Ablauf-Invarianten

**Settlement-Ablauf je Zeitpunkt $\tau$ (operativ)**

1. Ermittele alle bei $\tau$ fälligen Verpflichtungen.
2. Schuldner bietet pro Verpflichtung optional ein bilaterales Settlement (Cash + neue Claims) an.
3. Gläubiger akzeptiert oder lehnt je Settlement lokal gemäß eigener Policy ab.
4. Akzeptierte Settlements werden verbucht (Teiltilgung + Erzeugung neuer Claims mit neuen Fälligkeiten).
5. Nicht akzeptierte Restbeträge müssen sofort in Cash bedient werden; andernfalls tritt Dead gemäß (I5) ein.

Hinweis: Der folgende Pseudocode ist **informativ** und dient nur der operativen Präzisierung; er ist **nicht** Teil der formalen Axiome von math‑v2.

```text
settle_at_tau(P, τ):
  due_claims = obligations_due_at(P, τ)            # V_P(τ)
  returns_in  = realized_returns_at(P, τ)          # R_P(τ)
  credit_liquidity(P, returns_in)
  settlement_failed = false

  for claim in due_claims:
    debtor   = claim.debtor
    creditor = claim.creditor
    D        = claim.amount_due

    proposal = debtor.propose_settlement(claim, τ)
    # proposal := (cash c, new_claims {F_j}, bilateral discounts {q_j})

    if proposal is not None and creditor.accepts(proposal, τ):
      c, {F_j}, {q_j} = proposal
      value = c + Σ_j (q_j * nominal(F_j))
      booked = min(D, value)
      book_cash_transfer(debtor, creditor, c)
      issue_assigned_new_claims(debtor, creditor, {F_j})
      reduce_or_close(claim, booked)
    else:
      if debtor.liquidity >= D:
        book_cash_transfer(debtor, creditor, D)
        close(claim)
      else:
        settlement_failed = true
        break

  # I4-Zeitpunktkonvention: Wealth/Dead erst nach Returns + Settlement bei τ
  W_tau = compute_wealth_at_tau_after_settlement(P, τ)
  if settlement_failed or W_tau < 0:
    mark_dead(P, τ)
```

**(I7) Strikte Fortschreitung**
$$
\tau_{n+1} > \tau_n
$$
> ADR 2.6.2 Optionaler Zeno-Schutz:
$$
\tau_{n+1}-\tau_n \ge \epsilon >0
$$

**(I8) Aktionsfenster**
Zwischen $\tau_n$ und $\tau_{n+1}$ werden die von $P$ publizierten Offers $A_P(\tau_n)$ in der Welt verarbeitet; die Effekte erscheinen nur über $R_P(\tau_{n+1})$ und $V_P(\tau_{n+1})$.
Das ist eine wichtige Schnittstelleninvariante: **keine direkte Rückmeldung außerhalb des Inputs**.

---

### 2.3. Was leisten die Invarianten?

* Sie definieren einen **zulässigen P-Raum** $\mathcal{P}$ ohne Präferenzen.
* **Broker** und **CapitalSelector** sind später einfach Elemente von $\mathcal{P}$, weil sie:

  * Offers publizieren können ($I2/I3$)
  * Settlement korrekt behandeln ($I4$–$I6b$)
  * nur lokal sehen ($I1/I8$)
  * adaptive Zeitwahl haben ($I7$)

Damit erzeugt man einen echten Unterraum, in dem man evolutionär suchen kann.

---

### 2.5. Fitness (neu)

Für jeden Prozess $P$ definieren wir das Erfolgsmaß

$$
\mathcal F(P) = \int_{\tau_0}^{\tau_{dead}} W_P(\tau)\, d\tau
$$
Impl.Notes: Numerische Approximation der Integration wird in der Implementierung festgelegt (Diskretisierung). Falls $\tau_{dead}=\infty$, wird auf endlichem Evaluationshorizont $T$ integriert, d.h. $\mathcal F_T(P)=\int_{\tau_0}^{T} W_P(\tau)\,d\tau$.

Das Fitnessmaß ist rein buchhalterisch definiert.

Dead schneidet die Integration ab.
Es wird keine vollständige Wealth-Historie gespeichert.

---

### 2.6 ADR

#### 2.6.1 Dead als beendete Iteration

Bei $\sigma_P = \text{dead}$ endet die Iteration.
Es erfolgt keine weitere Offer-Erstellung.

#### 2.6.2 Optionaler Zeno-Schutz

Optional:

$$
\tau_{n+1} - \tau_n \ge \epsilon > 0
$$

---

# 3. Stacking-Algebra

## 3.1. Grundidee

Jeder Prozess (P) läuft in einer eigenen Welt

$$
W(\tau) = (I(\tau), N)
$$

mit festem Inputkanalraum $M\subset I$ und festem Aktionsraum $N\subset I$.

**Stacking** bedeutet:

> Die Output-Kanäle einer Welt werden als Input-Kanäle einer anderen Welt verwendet.

Es gibt **keine Parallelausführung innerhalb einer Welt**.
Komposition geschieht ausschließlich durch Kanalabbildung.

---

## 3.2. Kanalabbildung (Wiring)

Seien zwei Welten

$$
W_1(\tau) = (I_1(\tau), N_1), \qquad
W_2(\tau) = (I_2(\tau), N_2)
$$

Eine Stack-Verbindung ist eine Abbildung

$$
\varphi: N_1 \rightarrow M_2
$$

wobei $M_2\subset I$ die Inputkanäle von $W_2$ sind.

Interpretation:

Ein Offer $k$, das in $W_1$ auf Kanal $\mu\in N_1$ publiziert wird,
erscheint in $W_2$ auf Kanal $\varphi(\mu)$ als Input.

Damit ist die **komponierte Welt**

$$
W_2 \circ_\varphi W_1
$$

definiert durch:

* Input:

  $$
  I_{comp}(\tau)
  =
  I_2(\tau)
  \cup
  \varphi\big(A_{P_1}(\tau)\big)
  $$

* Aktionsraum:

  $$
  N_{comp} = N_2
  $$

$W_1$ bleibt als eigene Welt erhalten;
$W_2$ sieht zusätzlich die durch $\varphi$ eingespeisten Offers.

> Zulässig ist auch eine Selbstabbildung
> $$
> \varphi : N \rightarrow M
> $$
> innerhalb derselben Welt.
>
> Dadurch entstehen Rückkopplungsschleifen.
> Diese wirken ausschließlich über die zeitliche Iteration.
>
> Primitive Prozesse bleiben nicht-rekursiv; Rekursion ist eine Eigenschaft der Kanalstruktur.

---

## 3.3. Disjunktheit und Konkurrenz

Zwei Fälle sind zulässig:

### (A) Disjunkte Einspeisung

$$
\varphi(N_1) \cap M_2^{ext} = \emptyset
$$

Dann ist Stacking reine Erweiterung.

### (B) Konkurrenz

$$
\varphi(N_1) \cap M_2^{ext} \neq \emptyset
$$

Dann konkurrieren Offers aus $W_1$ mit externen Offers in $W_2$.

Die Zuteilungsregel bleibt Offer-lokal (vgl. 1.3).

Es wird **keine globale Priorität eingeführt**.

---

## 3.4. Verteilerwelt

Eine Verteilerwelt $V$ ist eine Welt mit

$$
N_V = \bigcup_{i=1}^k M_i
$$

und festen Projektionen

$$
\pi_i: N_V \rightarrow M_i
$$

mit

$$
\pi_i(N_V) \cap \pi_j(N_V) = \emptyset \quad (i\neq j)
$$

Sie repliziert Output disjunkt auf Teilwelten als Input.

Formal ist $V$ ein normales $P$ mit rein struktureller Offer-Erstellung.

---

## 3.5. Koordinatorwelt

Eine Koordinatorwelt $C$ integriert mehrere Outputräume

$$
N_i
$$

über Abbildungen

$$
\psi_i: N_i \rightarrow M_C
$$

und erzeugt daraus einen gemeinsamen Inputraum.

Auch hier:

* keine globale Sicht,
* keine implizite Optimierung,
* nur lokale Offer-Filterung.

---

## 3.6. Algebraische Eigenschaften

### (S1) Typabschluss

Ist $P_1, P_2 \in \mathcal{P}$,
dann ist

$$
W_2 \circ_\varphi W_1
$$

wieder eine gültige Welt mit gültigem $P_2$.

### (S2) Assoziativität (bis auf Isomorphie)

Für kompatible Abbildungen gilt:

$$
(W_3 \circ_{\varphi_2} W_2) \circ_{\varphi_1} W_1
\cong
W_3 \circ_{\varphi_2 \circ \varphi_1} W_1
$$

Die Komposition ist strukturell assoziativ.

### (S3) Keine implizite Parallelität

Parallelausführung zweier Prozesse ist äquivalent zu

* disjunkter Kanalabbildung
* oder expliziter Koordinatorwelt.

Es existiert keine „versteckte“ globale Synchronisierung.

---

## 3.7. Dead-Propagation

Wenn $P_1$ in $W_1$ dead wird:

* Seine Outputs entfallen.
* Die durch $\varphi$ eingespeisten Offers verschwinden.
* $W_2$ erhält nur noch externen Input.

Dead propagiert also **nicht strukturell**, sondern nur über ausbleibenden Input.

---

## 3.8 Ergebnis

Damit gilt:

* Stacking ist reine Kanalabbildung.
* Jede Welt bleibt lokal.
* Konkurrenz entsteht nur über Kanalüberlagerung.
* Verteiler und Koordinator sind normale $P$.


---

# 4. Einwohnerbuch

## 4.1. Ziel

Das Einwohnerbuch ist eine **Meta-Struktur** außerhalb einer einzelnen Welt $W$.

Es speichert Informationen über beendete Prozesse $P$,
um Selektion und Rebirth zu ermöglichen.

Das Einwohnerbuch ist **nicht Teil des $P$-Raums**.
Es wirkt ausschließlich auf Metaebene.

---

## 4.2. Erfassungsereignis

Ein Prozess $P$ wird in das Einwohnerbuch eingetragen, wenn

$$
\sigma_P(\tau_{dead}) = \text{dead}
$$

oder wenn eine externe Meta-Regel eine Beendigung erzwingt.

Eintragungszeitpunkt:

$$
\tau_{entry} := \tau_{dead}
$$

---

## 4.3. Gespeicherte Größen

Für jeden verstorbenen Prozess $P$ wird gespeichert:

### (E1) Fitness

$$
\mathcal F(P) = \int_{\tau_0}^{\tau_{dead}} W_P(\tau)\, d\tau
$$

Falls $\tau_{dead} = \infty$, wird ein Evaluationshorizont $T$ verwendet.

---

### (E2) Lebensdauer

$$
\mathcal L(P) = \tau_{dead} - \tau_0
$$

---

### (E3) Strukturelle Metadaten

* Kanalstruktur $(M_P, N_P)$
* Position im Stack-DAG
* Topologiegrad (z.B. Anzahl eingehender / ausgehender Verbindungen)
* Parametrisierung des $P$ (z.B. Policy-Koeffizienten)

Diese Daten enthalten **keine Zustandsverläufe**.

---

### (E4) Optional: Ursachenmarkierung

Falls der Tod durch

* Settlement-Fehlschlag,
* Liquiditätsmangel,
* exogenen Schock

ausgelöst wurde, kann eine Klassifikation gespeichert werden.

Diese ist rein informativ.

---

## 4.4. Struktur des Einwohnerbuchs

Das Einwohnerbuch ist eine Menge

$$
\mathcal B = { (P_i, \mathcal F(P_i), \mathcal L(P_i), \text{meta}_i) }
$$

mit möglicher Indexierung nach Zeit.

Optional kann eine Fensterung erfolgen:

$$
\mathcal B_T = \{ P_i \in \mathcal B \mid \tau_{entry,i} \ge T_{min} \}
$$

---

## 4.5. Rebirth-Mechanismus

Rebirth ist eine Meta-Operation

$$
\mathcal R : \mathcal B \rightarrow \mathcal P
$$

die aus einem Eintrag im Einwohnerbuch einen neuen Prozess erzeugt.

---

### 4.5.1. Auswahlregel

Ein Prozess $P_i$ wird mit Wahrscheinlichkeit

$$
\pi_i = \frac{g(\mathcal F(P_i))}{\sum_j g(\mathcal F(P_j))}
$$
ausgewählt.

Falls $\sum_j g(\mathcal F(P_j)) = 0$, wird statt dessen eine uniforme Auswahl über die zulässigen Kandidaten verwendet.

Typische Wahl:

$$
g(x) = \max(x, 0)
$$

oder exponentielle Gewichtung:

$$
g(x) = e^{\beta x}
$$

$\beta$ ist Selektionsdruck.

---

### 4.5.2. Mutation

Der neue Prozess $P_{new}$ erhält:

* gleiche Struktur wie $P_i$
* aber mutierte Parameter

Formal:

$$
\theta_{new} = \theta_i + \epsilon
$$

mit

$$
\epsilon \sim \mathcal D
$$

für geeignete Mutationsverteilung $\mathcal D$.

Topologische Mutation ist ebenfalls zulässig:

* Hinzufügen / Entfernen eines Kanals
* Änderung von Wiring
* Änderung der Settlement-Policy

---

### 4.5.3. Kapitalisierung bei Rebirth

Rebirth startet mit initialer Liquidität

$$
L_{init}
$$

Diese kann:

* konstant sein,
* oder aus globalem Meta-Kapitalpool stammen,
* oder proportional zu $\mathcal F(P_i)$.

Wichtig:

Rebirth übernimmt **keine alten Verpflichtungen**.

---

## 4.6. Energie-Interpretation

Das System benötigt exogenen Zufluss, wenn:

* Rebirth Liquidität erzeugt,
* oder Meta-Kapital injiziert wird.

Ohne exogenen Zufluss gilt:

$$
\sum W_{gesamt} \rightarrow 0
$$

Das System konvergiert gegen Aussterben.

---

## 4.7. Stabilitätsimplikation

Rebirth wirkt als:

* strukturelle Entkopplung von Fehler und Bauplan
* zeitliche Arbitrage
* Informationsspeicher ohne Bilanzübernahme

Das System wird dadurch:

* nicht arbitragefrei,
* aber evolutiv stabilisierbar.

---

## 4.8. Extremfälle

### Kein Rebirth

Modellannahme: System kollabiert nach endlicher Zeit.

### Unbegrenzter Rebirth ohne Selektion

Explosion instabiler Varianten.

### Rebirth mit starkem Selektionsdruck

Konvergenz auf lokale Fitnessmaxima.

---

## 4.9. Zusammenhang mit Stack-Welt

Da Dead nur lokal wirkt, gilt:

* lokale Prozesse sterben zuerst,
* Stack verliert Input,
* höhere Ebenen verlieren Signal,
* Kaskade ist rein kausal über Inputverlust.

Das Einwohnerbuch speichert nur lokale Prozesse.

Stack selbst ist kein Lebewesen.

---

## 4.10. Ergebnis

Mit Kapitel 4 wurde eingeführt:

* formale Todeserfassung
* Fitnessmaß
* selektive Wiedergeburt
* strukturelle Mutation
* Energiebedingung
* keine implizite Moral
* keine globale Preisbildung

---

Perfekt.
Dann sauber, minimal, anschlussfähig an Phase H.

---

# 5. CapitalSelector (Minimalform v1)

Der CapitalSelector ist ein spezielles Element
$$
CS \in \mathcal P
$$
mit strikt lokaler Sicht, bilanziellem Zwang und mutierbaren Parametern.

Er ist eine Verallgemeinerung des Phase-G-Selectors unter den neuen Invarianten.

---

## 5.1. Struktur

Ein CapitalSelector ist definiert durch:

$$
CS = ( \sigma, \Theta )
$$

mit

* $\sigma$ = Zustand (Liquidität, offene Claims, interne Parameter)
* $\Theta$ = Parametervektor

Minimal enthält $\sigma$:

* $L(\tau)$ – Liquidität
* $\mathcal O(\tau)$ – aktive Verpflichtungen
* ggf. interne Statistiken

---

## 5.2. Parameterraum

Minimaler Parametervektor:

$$
\Theta = (\beta, \ell_{max}, \rho, \gamma, \lambda)
$$

mit:

* $\beta$ – Selektionssensitivität
* $\ell_{max}$ – maximale Leverage
* $\rho$ – Settlement-Neigung
* $\gamma$ – Zeitpräferenz / Diskontfaktor ($0 < \gamma \le 1$)
* $\lambda$ – Cash-Anteil im Settlement ($0 \le \lambda \le 1$)

Diese Parameter sind evolvierbar.

---

## 5.3. Input-Selektion

Für eingehende Offers
$$
k \in I(\tau)
$$

definiert der Selector eine Bewertungsfunktion:

$$
u(k;\Theta,\sigma) = \beta \cdot f(C_k, t_k'', \tau) - r(\sigma)
$$

Minimalform von $f$:

$$
f(C_k, t_k'', \tau) = \gamma^{(t_k''-\tau)} \cdot C_k
$$

Interpretation:

* höherer $C_k$ → attraktiver
* spätere Fälligkeit → diskontiert
* $r(\sigma)$ → Risikokorrektur aus aktuellem Hebel

---

## 5.4. Allokationsregel

Aus Bewertung entsteht Gewicht:

$$
\alpha_k = \frac{e^{u(k)}}{\sum_j e^{u(j)}}
$$

Investitionsbetrag:

$$
x_k(\tau) = \alpha_k \cdot L_{avail}(\tau)
$$
mit $L_{avail}(\tau)$ als nach Returns- und Settlement-Verbuchung verfügbare Liquidität bei $\tau$.

Optional unter Hebelbegrenzung:

$$
\frac{\text{Gesamtverpflichtungen}}{L(\tau)} \le \ell_{max}
$$
für $L(\tau) > 0$; bei $L(\tau)=0$ werden keine neuen Verpflichtungen eingegangen.

---

## 5.5. Eigene Offer-Erstellung

Der Selector darf selbst Kapital aufnehmen.

Minimalregel:

Wenn erwartete Rendite über Schwelle:

$$
\bar C_{exp} > 1
$$

dann publiziert er Offer:

$$
k^{out} = (\mu,( \tau, t', t'', \delta, A, C_{borrow}))
$$

mit

* $A \le L(\tau) \cdot (\ell_{max}-1)$
* $C_{borrow}$ fix oder parametrisierbar

---

## 5.6. Settlement-Policy

Für fällige Verpflichtung $D$:

Mit Wahrscheinlichkeit $\rho$:

* wird bilaterales Settlement vorgeschlagen

Sonst:

* reine Cash-Zahlung

Settlement-Minimalform:

$$
(c, F) \text{ mit } c = \lambda D
$$

und $F$ zukünftiger Claim über Rest.

Akzeptanz liegt beim Gläubiger.

---

## 5.7. Wealth-Dynamik

Wealth gemäß Kapitel 2:

$$
W(\tau) = L(\tau) - \sum_{\text{fällig}} V_i(\tau)
$$

Fitness:

$$
\mathcal F(CS) = \int_{\tau_0}^{\tau_{dead}} W(\tau)\, d\tau
$$
Falls $\tau_{dead}=\infty$, wird analog zu Kapitel 2 ein endlicher Evaluationshorizont $T$ verwendet.

---

## 5.8. Mutationsraum

Rebirth mutiert:

$$
\Theta \mapsto \Theta + \epsilon
$$

mit

$$
\epsilon \sim \mathcal N(0,\Sigma)
$$

Optional:

* Strukturmutation (neue Bewertungsfunktion)
* neue Hebelregel
* neue Settlement-Strategie

---

## 5.9. Minimalitätsgrad

Diese Version enthält:

* keine globale Marktinformation
* keine explizite Erwartungsbildung über Welt
* keine Portfolio-Historie
* keine Mehrperiodenoptimierung

Er ist rein:

> reaktives, lokal bilanzgetriebenes Kapitalorgan.

---

## 5.10. Beziehung zu Phase G

Phase G Selector war:

* diskret
* ohne echte Insolvenz
* ohne Settlement
* ohne Rekursion

Minimal-v1 ist:

* kontinuierlich
* insolvenzfähig
* settlementfähig
* rekursiv stackbar
* evolvierbar

---

Perfekt.
Minimal, formal, anschlussfähig an Kapitel 4 und 5.

---

## 5.11. Rebirth-Mechanismus (Minimalform)

Der Rebirth-Mechanismus operiert **ausschließlich auf Metaebene**.
Er ist kein Bestandteil eines $P$ und verletzt keine Invariante.


### 5.11.1. Trigger

Ein Rebirth-Event wird ausgelöst, wenn

$$
\sigma_{CS}(\tau) = \text{dead}
$$

zum Zeitpunkt $\tau = \tau_{dead}$.

Der Prozess wird beendet und in das Einwohnerbuch eingetragen:

$$
\mathcal B \ni
\left(
CS_i,\ \mathcal F(CS_i),\ \mathcal L(CS_i),\ \text{meta}_i
\right)
$$

### 5.11.2. Auswahlmechanismus

Aus dem Einwohnerbuch wird ein Eltern-Parametervektor $\Theta^{(parent)}$ gewählt.

Minimalform: Fitness-proportionale Selektion

$$
\mathbb P(\Theta^{(parent)} = \Theta_i) = \frac{\max(\mathcal F_i,0)} {\sum_j \max(\mathcal F_j,0)}
$$

Falls $\sum_j \max(\mathcal F_j,0) = 0$, wird eine uniforme Auswahl über die zulässigen Kandidaten verwendet.

Alternative Minimalformen:

* Turnierselektion
* Top-$k$ Eliten
* Rein zufällige Auswahl

Die Auswahlregel ist ein Meta-Parameter und nicht Teil des $P$-Raums.


### 5.11.3. Mutation

Neuer Parametervektor:

$$
\Theta^{(new)} = \Theta^{(parent)} + \epsilon
$$

mit

$$
\epsilon \sim \mathcal N(0,\Sigma)
$$

Optional:

* Parameter-Clipping
* Strukturmutation (z.B. neue Bewertungsfunktion)


### 5.11.4. Initialisierung des Rebirth-Selectors

Ein neuer CapitalSelector wird erzeugt:

$$
CS^{(new)} = (\sigma_0, \Theta^{(new)})
$$

mit:

* $L(\tau_{rebirth}) = L_0$ (exogenes Startkapital)
* $\mathcal O = \emptyset$
* keine Altlasten

Rebirth ist somit **keine Wiederbelebung**, sondern
eine neue Instanz mit mutiertem Genom.


### 5.11.5. Zeitliche Arbitrage

Rebirth erzeugt eine zeitliche Struktur:

* Erfolgreiche Strategien akkumulieren Fitness.
* Tote Strategien verschwinden lokal.
* Meta-Selektion verschiebt Parameterverteilungen über Generationen.

Wichtig:

Rebirth ist **keine Reparatur eines toten Prozesses**.
Dead bleibt irreversibel auf Prozessebene.


### 5.11.6. Stabilitätsinterpretation

Ein Parametervektor ist evolutionsstabil, wenn gilt:

$$
\mathbb E_{\epsilon\sim\mathcal D}\big[\mathcal F(\Theta)\big]
>
\mathbb E_{\epsilon\sim\mathcal D}\big[\mathcal F(\Theta + \epsilon)\big]
$$

für kleine $\epsilon$.

Dann konzentriert sich die Population in einem Parametergebiet.


### 5.11.7. Kambrische Dynamik

Wenn

* Mutationsvarianz groß
* Fitnesslandschaft rau
* Kapital schnell zirkuliert

entsteht:

> hohe Geburtenrate + hohe Sterberate

### 5.11.8. Zusammenfassung

Rebirth ist:

* meta-evolutionär
* fitnessgetrieben
* lokal bilanzbasiert
* rekursionskompatibel
* nicht reparierend

Er erzeugt:

* Konkurrenz
* Parameterdrift
* Selektionsdruck
* zeitliche Struktur

---

# 6. Normative Semantik-Festlegungen (bindend)

Die folgenden Festlegungen sind **normativ** und gelten für alle Implementierungen von math-v2.

## 6.1. Ordnungsregel je Zeitpunkt $\tau$

Für jeden Prozess $P$ gilt bei gleichem Zeitpunkt $\tau$ die feste Reihenfolge:

1. Ermittlung fälliger Verpflichtungen $V_P(\tau)$ und realisierter Returns $R_P(\tau)$.
2. Verbuchung von $R_P(\tau)$ in die Liquidität.
3. Verarbeitung zulässiger bilateraler Settlements für fällige Verpflichtungen.
4. Berechnung von $W_P(\tau)$ gemäß (I4) auf Basis der danach verbleibenden fälligen Verpflichtungen.
5. Dead-Entscheidung gemäß (I5).
6. Nur falls nicht dead: Offer-Erstellung $O_P(\tau, I(\tau), \sigma_P(\tau))$.

Eine spätere Stufe darf semantisch nicht auf eine frühere Stufe rückwirken.

## 6.2. Gleichzeitigkeit und Kausalität

Ereignisse mit identischem Zeitstempel sind gleichzeitig.
Implementierungen dürfen hierfür deterministische Tie-Breaker verwenden, müssen aber die Ordnungsregel aus 6.1 erhalten.
Eine kausale Umkehr (Nutzung von Informationen aus $\tau'>\tau$ für Entscheidungen bei $\tau$) ist unzulässig.

## 6.3. Diskretisierung kontinuierlicher Zeit

Diskretisierung ist zulässig, sofern die semantische Zeitordnung erhalten bleibt:

* Monotonie: $\tau_a < \tau_b \Rightarrow n(\tau_a) \le n(\tau_b)$.
* Keine Kausalitätsverletzung zwischen Buckets.
* Ereignisse mit gleichem $\tau$ bleiben semantisch gleichzeitig.

Die konkrete Bucket-/Rundungsregel ist Implementierungsdetail und gehört ins impl_spec.

## 6.4. Claim-/Offer-Identität und Umschreibung

Jedes Offer und jeder Claim besitzt eine eindeutige Identität.
Bei Umschreibung (Settlement) entstehen neue Claims mit eigener Identität;
die Eltern-Kind-Beziehung ist semantisch nachvollziehbar zu halten.
Eine Umschreibung ist keine rückwirkende Mutation vergangener Ereignisse.

## 6.5. Rebirth-Timing

Ein Dead-Ereignis bei $\tau_{dead}$ beendet den Prozess irreversibel.
Rebirth ist Meta-Operation und darf denselben Prozess nicht reaktivieren.
Ein Rebirth-Selector darf frühestens als neue Instanz nach Abschluss der Dead-Verarbeitung bei $\tau_{dead}$ erscheinen.

## 6.6. Exogene Sicherheitsgrenzen

Externe Stop-/Safety-Signale sind zulässig und stehen semantisch über internen Selektionsmechanismen.
Erzwingt eine externe Regel die Beendigung bei $\tau$, gilt der Prozess ab diesem Zeitpunkt als beendet und wird wie in Kapitel 4 behandelt.

