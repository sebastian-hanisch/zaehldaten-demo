# Zählen: Poisson, Überdispersion und die Frage, welcher Standardfehler stimmt – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-zaehldaten-demo.streamlit.app/)** -->

Diese Demo gehört zum Themenfeld **Regression und verallgemeinerte lineare Modelle** auf der Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie hebt eine Annahme der Demo [`logit-separation-demo`](https://github.com/sebastian-hanisch/logit-separation-demo) auf:
Bei einem Ja/Nein-Ausgang ist die Varianz durch den Mittelwert festgelegt (p(1 − p)) und lässt sich ohne Wiederholungen nicht prüfen. Bei **Zählwerten** gilt die Annahme „Varianz = Mittel“ (Poisson), und man kann nachsehen, ob sie stimmt. Im Container-Vehikel stimmt sie nicht: der Schätzer existiert,
aber seine **Standardfehler stimmen nicht**. Das Sandwich-Verfahren (HC0 bis zur Hebelkorrektur) kennt die Demo [`streuung-wls-demo`](https://github.com/sebastian-hanisch/streuung-wls-demo) für stetige Ausgänge; hier ersetzt es die Zählmodelle. Familie und Link (Identität, andere Links, Deviance als Gütetest)
behandelt [`glm-familie-demo`](https://github.com/sebastian-hanisch/glm-familie-demo), den Kunden als Cluster [`cluster-gemischt-demo`](https://github.com/sebastian-hanisch/cluster-gemischt-demo), den Standgeld-Binärausgang [`kalibrierung-schwelle-demo`](https://github.com/sebastian-hanisch/kalibrierung-schwelle-demo).
Poisson mit Ridge und Newton für Tore und Mannschaften steht in [`tore-remis-demo`](https://github.com/sebastian-hanisch/tore-remis-demo); dort geht es um die Wahrscheinlichkeit eines Ergebnisses, hier um die Inferenz über Koeffizienten (Überdeckung der Intervalle, Quasi-Poisson, Sandwich, Fehlalarm).
Vehikel ist die Container-Verweilzeit aus [`verweilzeit-demo`](https://github.com/sebastian-hanisch/verweilzeit-demo), die Zählvariable heißt **Standtage** (begonnene Tage = aufgerundete Verweilzeit). Der Generator ist hier **nachgebaut** (Merkmale und Verteilung der Verweilzeit, ohne ereignisdiskrete Simulation), mit eigener Zufallsquelle: die Zahlen dieser README sind die der Demo, nicht die der Nachbar-Demo.

## Warum dieses Problem

Zählt man Standtage und rechnet Poisson, nimmt man Varianz = Mittel an. **Wie viel mehr streuen die Standtage, und was geschieht mit den Intervallen der Koeffizienten, wenn man das ignoriert?**
Im Container-Vehikel ist die Varianz der Standtage insgesamt das 6,8-Fache des Mittels (6,15 und 41,73; eine Historie aus einer Million Containern, B11). Das ist die rohe Streuung über alle Container zusammen und **nicht** die Überdispersion gegenüber dem Modell: schon ein exakt passendes Poisson-Modell mit denselben angepassten Mitteln hätte roh das 2,56-Fache, weil die erwarteten Werte der Container verschieden sind. Gegen das angepasste Modell gemessen ist die Streuung das 3,12-Fache (Pearson-φ̂), und bei Export und Umschlag ist sie kleiner als bei Poisson (B11). Bei n = 3000 Containern (ohne Kundenspalten im Modell) deckt das 95-%-Intervall des Poisson-Modells den Grenzwert des Zoll-Koeffizienten in 68,9 % der Historien (2000 Wiederholungen),
und das Rauschmerkmal Gewicht (wahre Wirkung null) schlägt in 38,2 % an statt in 5 % (B12). Der mittlere Poisson-Standardfehler ist dabei halb so groß wie die wahre Streuung des Schätzers (0,51). Die üblichen Reparaturen stellen die 95 % nicht her:
Quasi-Poisson (z) deckt bei Zoll 91,6 %, die negativ binomiale Anpassung (NB2) 87,2 %; nur das Sandwich-Verfahren (HC0) deckt 95,3 % (bei n = 300 deckt HC0 92,3 %, B12). Die beiden Reparaturen sind außerdem **nicht dasselbe**: sie gewichten Zählwerte verschieden (bei α = 0,5 zählt ein Fall mit Mittel 20 gegenüber einem mit Mittel 1 das 2,727-Fache statt des 20-Fachen),
und bei falschem Mittelwertmodell schätzen sie verschiedene Größen (B7, B10).
**Alles ist eigene Nachrechnung in diesem Modell, bei diesen Parametern** (Zähleinheit überall: eine Wiederholung = ein Datensatz bzw. eine Historie aus n Containern, außer wo „Container“ steht). Die Gleichungen sind Lehrbuchstoff (Quellen unten); eine Neuheit wird nicht behauptet.

## Modell

- **Konvention (gilt für jede Zahl):** Zählwert y ≥ 0, Erwartung μ = exp(x'β) (Log-Link, bei Poisson der kanonische), Exposition t als Offset log t (Rate μ/t). Varianzfunktionen: **Poisson** V = μ; **Quasi-Poisson** V = φ μ mit φ̂ = Σ (y − μ̂)²/μ̂ / (n − p) (Pearson-Dispersion, nicht Deviance);
  **NB2** V = μ + α μ² (Gamma-Mischung der Poisson-Rate, α = 1/Form); **NB1** V = (1 + δ) μ nur als Messwelt in Experiment 4.
- **Sandwich:** Var(β̂) = A⁻¹ B A⁻¹ mit A = Σ μ̂ x x', B = Σ (y − μ̂)² x x'; **HC0** so, **HC1** = n/(n − p) · HC0, **„HC3a“** mit Score (y − μ̂)/(1 − h) und h = μ̂ x'A⁻¹x. Namensfalle: bei MacKinnon/White (1985) heißt HC3 das Jackknife, die heutige Bedeutung kommt von Davidson/MacKinnon (1993).
  Die Fassung hier ist ein **Analogon** zu HC3 der linearen Regression; die App schreibt „Hebelkorrektur (HC3-Analogon)“, nie „HC3“.
- **Intervalle:** z (1,96) für Poisson, Sandwich und NB2, bei Quasi-Poisson z und t mit n − p Freiheitsgraden. Quasi-Poisson hat dieselben Koeffizienten wie Poisson, NB2 nicht. φ̂ unter 1 (Unterdispersion) wird wie in R unverändert angewandt: das Quasi-Intervall wird dann schmaler als das Poisson-Intervall.
- **Messwelt 1** (Experimente 2 und 4): ein Merkmal x ~ N(0, 1), log μ = 1 + 0,4 x (+ c x²), der Zählwert ist NB2 mit α oder NB1 mit δ = α · E[μ] (dieselbe mittlere Dispersion E[V/μ]); Ziel der Intervalle ist der wahre Anstieg 0,4.
- **Messwelt 2, das Container-Vehikel** (Experimente 3 und 5): Art (Import 35 %, Export 30 %, Umschlag 25 %, Leer 10 %), Kunde (gleichverteilt auf zwölf), Zollhalt (nur bei Import, Wahrscheinlichkeit 0,25), Wochenende (Wahrscheinlichkeit 2/7), Vorlauf bei Export (gleichverteilt von 1 bis 10) und Umschlag (gleichverteilt von 0,5 bis 6), Rauschmerkmal Gewicht (N(0, 1), unabhängig, Wirkung null).
  Die Verweilzeit d (in Tagen) ist lognormal: log d = log m + u_k · f + s · ε mit ε ~ N(0, 1); die Standtage sind ⌈d⌉ (mindestens 1). Median m und Streuung s hängen an der Art: **Import** m = 3,5 · exp(0,7 · Zoll + 0,30 · Wochenende), s = 0,55; **Export** m = 0,6 + 0,85 · Vorlauf, s = 0,25; **Umschlag** m = 0,3 + 0,8 · Vorlauf, s = 0,35; **Leer** m = 7 · exp(0,30 · Wochenende), s = 0,90.
  Der Kundeneffekt u_k (k = 0 bis 11) wirkt je Art mit dem Faktor f (Import 1, Export 0,3, Umschlag 0,3, Leer 1); u_k auf drei Stellen gerundet (genau in `zae_constants.py`): −0,427, 0,379, −0,261, −0,078, −0,023, −0,222, −0,410, 0,195, 0,108, −0,586, 0,704, 0,291.
  Modell: Art-Dummies, Zoll, Wochenende, log(1 + Vorlauf), Gewicht, **ohne Kunden** (mit Schalter: elf Kundenspalten). Alle Parameter sind erfunden. **Das Mittelwertmodell ist nicht log-linear** (Standtage sind die Aufrundung einer lognormalen Größe, je Art mit anderer Streuung): es gibt keinen „wahren“ Koeffizienten, nur den **Grenzwert des jeweiligen Schätzers**
  (hier aus einer Historie von einer Million Containern; Poisson und NB2 haben verschiedene Grenzwerte).
- Alle Formeln stehen im 📐-Expander der App.

## Methodik

- **Kern** (`zae_models.py`): IRLS für Poisson und NB2 mit festem α (Gewicht μ/(1 + αμ)), NB2 mit α aus der Profilgleichung (β und α abwechselnd, wie `glm.nb` in R), Pearson-Dispersion, Sandwich HC0, HC1 und die Hebelkorrektur; Kappung des linearen Prädiktors bei |η| ≤ 30. Nur numpy und scipy.
- **Von Hand** (`zae_hand.py`): acht erfundene Zählwerte (keine Standtage: sie enthalten Nullen, die echte Standtage nicht haben) in zwei Gruppen, die Varianzen in Brüchen (`fractions`), Gewichte, Exposition, Gamma-Mischung.
- **Messungen:** Zählwerte per Inversion der Verteilungsfunktion aus einer Gleichverteilten (direkt aus der Wahrscheinlichkeitsfunktion, kein Gamma-Zieher); alle Zufallszahlen aus SplitMix64. Eine Wiederholung r hat den Zufalls-Seed `rep_seed(Basis, r)`; die ersten R Wiederholungen sind also unabhängig von R.
- **Grenzwert der Überdeckung ohne Zufall** (`zae_asymptotics.py`): A = E[μ x x'] und B = E[V(μ) x x'] sind Erwartungen über x ~ N(0, 1) mit der wahren Varianzfunktion V, auf zwei Wegen gerechnet (Gauss-Hermite mit 80 Knoten, `scipy.integrate.quad`). Stimmt das Mittelwertmodell, ist √n (β̂₁ − β₁) für n → ∞ normalverteilt mit der Varianz (A⁻¹BA⁻¹)₁₁; **SD** ist deren Wurzel, die wahre Streuung auf dieser Skala. Ein z-Intervall, dessen geschätzte Varianz auf derselben Skala s² ist, deckt dann mit Wahrscheinlichkeit 2Φ(1,96 s / SD) − 1 (Φ: Verteilungsfunktion der Standardnormalverteilung); bei Poisson ist s² = (A⁻¹)₁₁, bei Quasi-Poisson (z) s² = φ (A⁻¹)₁₁ mit φ = E[V(μ)/μ]. Das ist das Orakel für die Balken von Experiment 2.
- **Vorgerechnet** (`tools/zae_make_frozen.py` → `data/zae_frozen.json`, rund sieben Minuten): das Raster von Experiment 2 (21 Zellen, 4000 bis 20 000 Wiederholungen), das Vehikel (n = 3000 mit 2000 und n = 300 mit 4000 Historien, je ohne und mit Kunden), das falsche Mittelwertmodell, die Nullen. Die App kennzeichnet diese Zahlen als „vorgerechnet“; sie rechnet live nur das Handbeispiel,
  den Grenzwert aus der Formel, **einen** Datensatz bzw. **eine** Historie und auf Knopfdruck eine **Live-Probe** (50 bis 500 Wiederholungen der Messwelt 1; gemessen bei n = 200 etwa 2 ms je Wiederholung, 500 Wiederholungen 0,8 bis 1,0 s, bei n = 30 und α = 1 1,0 s; Windows, unter Last durch andere Prozesse). Der erste Seitenaufruf braucht im AppTest unter 2 s.
- **Experimente:** (1) Zählen von Hand, (2) Wann kippt das Poisson-Intervall?, (3) Standtage der Container, (4) Quasi-Poisson ist nicht NB2 (Gewichte, zwei Welten, falsches Mittelwertmodell), (5) Nullen und fehlende Merkmale.

## Befunde (gemessen, keine Behauptungen)

Zähleinheit: Wiederholung. Monte-Carlo-Standardfehler einer Überdeckung bei 0,95: 0,15 Prozentpunkte (20 000 Wiederholungen), 0,34 (4000), 0,49 (2000). **Quasi-Poisson** hat zwei Intervalle: **Quasi z** (z-Quantil 1,96) und **Quasi t** (t-Quantil mit n − p Freiheitsgraden); wo eine Zahl nur eines davon meint, steht es dabei, und „Quasi“ allein heißt Quasi z. Die Spalte „Test“ nennt den Test in `tests/test_claims.py` (Band, nicht exakt), wo nichts anderes steht.

| Nr | Frage | Befund (Voraussetzungen im Satz) | Test |
|---|---|---|---|
| B1 | Wie oft deckt das Poisson-Intervall bei Überdispersion? | NB2-Welt, n = 200, Anstieg 0,4: α = 0 / 0,25 / 0,5 / 1 gibt **94,8 / 84,0 / 74,1 / 63,6 %** (4000, 4000, 20 000, 4000 Wiederholungen). Für die Hauptzelle (α = 0,5) steht darum das Band **74 bis 76 %**, keine Stelle hinter dem Komma aus 2000 Läufen (siehe „Befunde und Korrekturen“). **In der Simulation** liegt das Poisson-Intervall in allen 18 Zellen mit α > 0 (NB2- und NB1-Welt, n = 30, 100 und 200) unter 90 % (63,6 bis 86,6 %). | `test_b1_*` |
| B2 | Was sagt die Formel für n → ∞? | Poisson **83,4 / 74,2 / 61,9 %**, Quasi-Poisson (z; Skalar φ = E[V/μ]) **93,2 / 92,5 / 91,8 %** für α = 0,25 / 0,5 / 1 (NB2-Welt, Messwelt 1); α = 0 gibt exakt 95 %. Nur das Poisson-Intervall liegt im Grenzwert unter 90 %, Quasi-Poisson nicht. | `test_b2_*`, `test_oracle_*` |
| B3 | Hält Quasi-Poisson die 95 %? | **Nein, nicht für NB2-Varianz:** schon im Grenzwert 92,5 % (α = 0,5), in der Simulation in allen 9 Zellen der NB2-Welt (3 Werte α > 0 mal n = 30, 100, 200) zwischen 91,9 und 93,2 % (**Quasi z**; Quasi t 92,3 bis 94,0 %, ebenfalls in keiner Zelle innerhalb von zwei Standardfehlern von 95 %). **In der NB1-Welt** (Varianz linear im Mittel) ist Quasi z im Grenzwert exakt 95,0 % und bei n = 200 zwischen 93,7 und 94,8 %, bei n = 30 zwischen 92,5 und 93,9 %. Deutung (aus der Formel, nicht eigens gemessen): ein einziges φ passt nicht zu einer Varianz, die quadratisch im Mittel wächst. | `test_b3_*` |
| B4 | Hält irgendein Verfahren die 95 %? | Zählung über die 18 Zellen mit α > 0 (Abstand höchstens 2 Monte-Carlo-Standardfehler von 95 %): Poisson 0, HC0 0, HC1 0, Hebelkorrektur 3, Quasi z 4, Quasi t 7, NB2 7 von 18 (die Einzelzahlen hängen am Rauschen und können bei anderen Zufalls-Seeds um einige Zellen abweichen). **Kein Verfahren hält das Niveau in allen Zellen.** | `test_b3_count_*` |
| B5 | Kleine Stichprobe | n = 30, α > 0 (6 Zellen): die einfachste Sandwich-Form **HC0** deckt **86,3 bis 89,9 %** (das gilt für HC0, nicht für Sandwich-Verfahren allgemein), die Hebelkorrektur 91,4 bis 94,1 % (in 6 von 6 Zellen mehr als 2 Prozentpunkte darüber); bei α = 0,5: 88,0 gegen 92,8 %. | `test_b4_*` |
| B6 | Quasi-Poisson gegen NB2 | α = 0,5, n = 200: **NB2-Welt** NB2 94,5 % gegen Quasi z 92,2 % (NB2 liegt vorn, Abstand über zwei Standardfehlern); **NB1-Welt** NB2 94,7 % gegen Quasi z 94,8 % (kein Abstand über zwei Standardfehlern). Bei n = 3000 im Vehikel (ohne Kunden) liegt NB2 bei Zoll **unter** Quasi z (87,2 gegen 91,6 %), beim Wochenende **darüber** (87,3 gegen 84,2 %): beim Wochenende in 4 von 4, bei Zoll in 3 von 4 Zellen (n = 3000 und 300, ohne und mit Kunden; Ausnahme bei Zoll: n = 300 mit Kunden, 89,5 gegen 87,4 %). Beide Verfahren haben verschiedene Ziele (Grenzwerte des Wochenendes: NB2 0,145, Poisson und Quasi 0,160), die Zahlen sind darum nicht Zahl für Zahl vergleichbar. | `test_b5_*`, `test_b7_ranking_*` |
| B7 | Wie gewichten die Verfahren? | Verhältnis des Gewichts bei Mittel 20 gegen Mittel 1: Poisson und Quasi 20; NB2 **4,167 / 2,727 / 1,905** für α = 0,25 / 0,5 / 1 (Brüche 30/11 bei α = ½ und 40/21 bei α = 1). | `test_b5_weights_*`, `tests/test_hand.py` |
| B8 | Kostet der Punktschätzer etwas? | Bei richtigem Mittelwertmodell (NB2-Welt der Messwelt 1, nicht das Vehikel) ist die Verzerrung beider Schätzer unter 0,01 (n ≥ 100) und das RMSE-Verhältnis Poisson/NB2 **1,000** (α = 0), **1,055** (α = 0,5), **1,078** (α = 1) bei n = 200; 1,011 bei n = 30, α = 0,5. Der Effizienzverlust ist klein. **Negativbefund:** bei α = 1, n = 30 hat NB2 einen entarteten Datensatz (Wiederholung 3480 von 4000: Randlösung α̂ ≈ 403, β̂₁ von rund 5000; die Standardabweichung der Schätzung 79,6 gegen 0,24 bei Poisson gibt im Wesentlichen nur diesen einen Datensatz wieder, ihre Größe hängt an der Abbruchregel für α̂; `test_b5_nb2_degenerate_replicate`), das RMSE-Verhältnis ist dort wertlos. | `test_b5_rmse_*`, `test_b5_nb2_estimate_*` |
| B9 | Randlösungen | α = 0, n = 200: α̂ liegt in 2253 von 4000 Datensätzen am Rand des Suchbereichs (Randlösung, wie bei Varianzkomponenten); bei α = 0,5 in keinem. In keiner der 106 000 Wiederholungen des Rasters scheiterte eine Anpassung. | `test_b5_boundary_*` |
| B10 | Falsches Mittelwertmodell | Wahr log μ = 1 + 0,4 x + c x², angepasst 1 + x, NB2-Zählung α = 0,5. Anstieg β̂₁ bei n = 200 000 (**ein** Datensatz) Poisson / NB2: c = 0: 0,400 / 0,401; c = 0,1: 0,505 / 0,436; c = 0,2: **0,678 / 0,466**; c = 0,3: 0,970 / 0,461 (nicht stabil, siehe unten). Bei n = 200, c = 0,2 (3000 Wiederholungen): Unterschied β̂₁(NB2) − β̂₁(Poisson) im **Median −0,141** (Quartile −0,203 und −0,096) bei einer robusten Streuung von β̂₁(Poisson) von 0,152; in 8 von 3000 Datensätzen ist der Betrag des Unterschieds größer als 1 (c = 0,3: 268 von 3000). Mittel und Standardabweichung des Unterschieds nennt die Demo nicht (einzelne Ausreißer dominieren sie). | `test_b6_*` |
| B11 | Wie viel streuen die Standtage? | Vehikel, eine Historie aus 10⁶ Containern (Zufalls-Seed 8500): Mittel 6,15, Varianz 41,73, **Varianz/Mittel 6,78** (roh, alle Container zusammen), kleinster Wert 1. **Das ist nicht die Überdispersion gegenüber dem Modell:** ein exakt passendes Poisson-Modell mit denselben angepassten Mitteln hätte roh 1 + Var(μ̂)/Mittel = **2,56** (Var(μ̂) = 9,58), weil die erwarteten Werte der Container selbst streuen (Art, Zollhalt, Vorlauf). Überdispersion gegenüber dem Modell ist das Pearson-φ̂ **3,12** (zweiter Zufalls-Seed 3,08); **je Containerart Import 3,05, Export 0,43, Umschlag 0,44, Leer 18,04** (Mittel von (y − μ̂)²/μ̂ über die Container der Art, mit dem μ̂ des Gesamtmodells): ein einziges φ passt nicht, in 2 von 4 Arten (Export, Umschlag) ist die Streuung gegen das angepasste Mittel **kleiner** als Poisson. Die rohe Varianz/Mittel je Art (Import 4,21, Export 1,34, Umschlag 1,00, Leer 18,43) zeigt diese Unterdispersion nicht: sie steckt nur im Vergleich mit dem angepassten Mittel. Mit 11 Kundenspalten im Modell fällt φ̂ auf 2,46. Grenzwerte der Koeffizienten Poisson / NB2: Zoll 0,649 / 0,648, Wochenende 0,160 / 0,145, Gewicht 0,000 / 0,000 (beide unter 0,005 im Betrag). | `test_b7_dispersion_*`, `test_b7_raw_*`, `test_b7_pseudo_*` |
| B12 | Überdeckung im Vehikel (Ziel: Grenzwert des jeweiligen Schätzers) | n = 3000 Container, 2000 Historien, ohne Kunden. **Zoll:** Poisson 68,9 %, Quasi z 91,6 %, HC0 95,3 %, NB2 87,2 %. **Wochenende:** 59,6 / 84,2 / 94,9 / 87,3 %. **Fehlalarm des Rauschmerkmals Gewicht** (nominal 5 %): Poisson **38,2 %**, Quasi z 13,5 %, NB2 11,6 %, HC0 4,7 %. Mittlerer Standardfehler ÷ wahre Streuung (Zoll): Poisson 0,51, Quasi z 0,88, HC0 1,01, NB2 0,77. **Diese Zahlen gelten ohne Kundenspalten und bei n = 3000;** bei n = 300 (ohne Kunden) deckt HC0 beim Zoll 92,3 %, der Fehlalarm liegt bei Quasi z bei 12,7 % und bei NB2 bei 8,8 % (unter 10 %). In allen vier Zellen (n = 3000 und 300, ohne und mit Kunden) schlägt Gewicht bei Poisson in 25 bis 45 % an, bei HC0 in weniger als 8 %, bei der Hebelkorrektur in weniger als 6 %. | `test_b7_coverage_*`, `test_b7_noise_*` |
| B13 | Kunden im Modell | n = 3000: φ̂ im Mittel 3,10 ± 0,90 ohne und 2,38 ± 0,59 mit elf Kundenspalten (n = 300: 2,75 ± 1,38 und 1,93 ± 0,70). Das Poisson-Intervall wird dadurch nicht repariert: Zoll 72,7 %, Wochenende 60,5 %, Gewicht-Fehlalarm 36,4 % (n = 3000, 2000 Historien). **Die Streuung von φ̂ je Historie ist groß** (Standardabweichung 0,90 bei Mittel 3,10); die Zahl hängt am Heavy Tail der Art „Leer“. | `test_b7_customers_*`, `test_b7_phi_hat_*` |
| B14 | Nullen | Standtage haben keine Nullen (begonnene Tage, immer mindestens 1). Bei der abgeleiteten Zählvariablen **Standgeld-Tage** (Annahme dieser Messung: max(0, Standtage − 7), Standgeld ab dem 8. Tag) ist der Nullanteil bei n = 3000 (300 Historien) **0,766 beobachtet, 0,562 von Poisson erwartet, 0,758 von NB2 erwartet**. Je Containerart (10⁶ Container, beobachtet / Poisson / NB2): Import 0,734 / 0,369 / 0,664, Export 0,733 / 0,658 / 0,812, Umschlag 0,963 / 0,931 / 0,949, Leer 0,477 / 0,002 / 0,440: NB2 trifft den Gesamtanteil, liegt je Art aber um bis zu 8 Prozentpunkte daneben. **Ob ein Zero-Inflation- oder Hurdle-Modell das behebt, ist nicht gemessen.** Der Nullanteil gilt für den Generator dieser Demo; andere Demos bauen den Generator eigenständig nach und haben andere Standgeld-Anteile. | `test_b8_*` |
| B15 | Von Hand | Acht erfundene Zählwerte (keine Standtage: sie enthalten Nullen), ohne Zollhalt 0, 0, 1, 5, mit Zollhalt 1, 3, 6, 10: β̂ = (ln 3/2; ln 10/3), Faktor 3,33. Var(β̂₁) naiv **13/60**, Pearson-φ̂ **154/45**, Quasi **1001/1350**, HC0 **1057/1800**, HC1 **1057/1350**. z naiv 2,59 (Intervall [0,29; 2,12] schließt 0 aus), Quasi mit t (6 Freiheitsgrade) 1,40 ([−0,90; 3,31] enthält 0; die Intervalle stehen auf der Skala von β̂₁, nicht des Faktors), HC0 1,57, NB2 1,67 (α̂ = 0,61). Gleicher Punktschätzer, anderer Schluss. Gilt für diese acht Werte, nichts über Häufigkeiten. | `tests/test_hand.py` |
| B16 | Die Gamma-Mischung | Mittel 3, α = 1/2: P(Y = 0) = 4/25 = 0,16 gegen 0,0498 bei Poisson mit Mittel 3; Varianz 15/2. Exposition: 21 Standtage bei 3 Containern und 20 bei 5 geben die Raten 7 und 4 (β = (ln 7; ln 4/7)). | `tests/test_hand.py` |
| B17 | Passt die vorgerechnete Datei zum Code? | Die ersten 40 Wiederholungen der Hauptzelle und 5 Historien des Vehikels werden neu gerechnet und stimmen mit der Datei überein (Band 10⁻⁶); eine Zelle (n = 100, α = 0,5) mit anderem Zufalls-Seed und 800 Wiederholungen liegt innerhalb von 4 Standardfehlern der Differenz. | `test_b17_*`, `tools/zae_make_frozen.py --check` |

**Bei c = 0,3 ist die Schätzung des Poisson-Anstiegs nicht stabil, der Grenzwert existiert aber:** der Grenzwert folgt aus der Schätzgleichung und braucht nur erste Momente; mit x ~ N(0, 1) ist er 0,4/(1 − 2c), also 0,500 bei c = 0,1, 0,667 bei c = 0,2 und 1,000 bei c = 0,3 (`test_b6_poisson_limit_closed_form`). μ² hat dagegen für c ≥ 0,25 keinen endlichen Erwartungswert (das Integral von exp(2c x²) gegen die Normaldichte divergiert; Herleitung, nicht gemessen); darum konvergiert die Schätzung nicht mit der üblichen Rate, und einzelne Datensätze weichen stark ab. Die frühere Messung (ein Datensatz mit n = 10⁶) gab dort 1,038, diese Demo mit n = 200 000: 0,970; beide liegen nahe am Grenzwert 1,000. Der Befund gilt für c = 0,1 und 0,2.

### Rechenregeln und Hinweise zu den Zahlen

- **Alle Überdeckungs-, Fehlalarm- und Nullzahlen sind Bänder**, keine exakten Werte: Monte-Carlo-Standardfehler plus die Abweichung zwischen Zufalls-Seeds und Plattformen. Nur Brüche (B15, B16) sind exakt.
- **Der Zufall:** SplitMix64 ganzzahlig, Normalzahlen nach Box-Muller aus Grundrechenarten und Wurzel (Gleichheit mit der exakten Fassung auf 10⁻¹³ im Test); keine C-Bibliothek, kein `numpy.random`. Ein Grenzfall bleibt: `exp` und `log1p` aus der C-Bibliothek können in der letzten Stelle abweichen und einen Zählwert in etwa 10⁻¹⁶ der Ziehungen umkippen; darum sind alle Tests mit Neu-Rechnung Bänder.
- **Ziel im Vehikel:** die Überdeckung bezieht sich auf den Grenzwert des jeweiligen Schätzers aus **einer** Historie von 10⁶ Containern. Ein zweiter Zufalls-Seed gibt für die drei Koeffizienten Grenzwerte, die sich um höchstens 0,015 unterscheiden (B11); der Standardfehler dort liegt bei etwa 6 % der Streuung bei n = 3000.
- **Frühere Messung:** die Zahlen der früheren Messung (numpy-Zufallsströme) stehen nur im Abschnitt „Befunde und Korrekturen“; jede andere Zahl ist mit dem Strom dieser Demo gerechnet.

### Befunde und Korrekturen gegenüber der früheren Messung

Die frühere Messung (`out_rn_v2.txt`, `out_rn_v17_ueberdeckung.txt`, `out_rn_v17_standtage.txt` im Arbeitsordner der Messungen) hatte andere Zufallsströme und teils weniger Wiederholungen. Abweichungen:

- **Poisson-Überdeckung der Hauptzelle:** 76,1 % (2000 Wiederholungen) und 74,4 % (10 000 Wiederholungen, anderer Strom) früher, hier **74,1 %** (20 000) und der Grenzwert 74,2 %. Die README nennt das Band 74 bis 76 %.
- **NB1-Welt, NB2 gegen Quasi:** früher Quasi z 94,9 % gegen NB2 94,1 % (4000 Wiederholungen, „Quasi liegt in der NB1-Welt vorn“). Hier **94,8 gegen 94,7 %** (10 000 Wiederholungen): kein Abstand über zwei Standardfehlern. Der frühere Vorsprung von Quasi-Poisson in der NB1-Welt bestätigt sich nicht; bestätigt ist, dass NB2 in der NB2-Welt vorn liegt (94,5 gegen 92,2 %).
- **Fehlalarm des Rauschmerkmals bei n = 3000:** früher Poisson 32,3 %, Quasi z 11,7 %, NB2 9,7 %, HC0 4,3 % (300 Wiederholungen, Standardfehler 2,7 Prozentpunkte beim Poisson-Wert), hier **38,2 / 13,5 / 11,6 / 4,7 %** (2000 Wiederholungen). Die Reihenfolge der Verfahren ist gleich, die Zahlen liegen höher; die Streuung von φ̂ je Historie (B13) erklärt, warum 300 Historien nicht reichen.
- **φ̂ je Containerart:** früher Import 3,02, Export 0,44, Umschlag 0,44, Leer 19,16 (ein Zufallsstrom), hier 3,05 / 0,43 / 0,44 / 18,04. Der Befund „in zwei von vier Arten kleiner als Poisson“ bleibt; die Zahl für Leer hängt am Heavy Tail.
- **Grenzwert bei c = 0,3:** früher 1,038, hier 0,970, der Grenzwert selbst ist 1,000 (siehe oben); die Schätzung ist dort nicht stabil, das Experiment nennt es ausdrücklich.
- **Vehikel mit Kunden im Modell:** früher nur φ̂ gemessen (3,23 auf 2,54), hier auch die Überdeckung (B13): die Rangfolge der Verfahren bleibt, das Poisson-Intervall wird nicht repariert.

## Quellen

Die Gleichungen sind Lehrbuchstoff; die Zahlen dieser Demo sind eigene Nachrechnung. Stufen: **W** Wortlaut gelesen, **A** nur Abstract oder Kapitelliste, **S** nur Suchtreffer; alles in eigenen Worten.
- Zeileis, Kleiber, Jackman (2008), *Regression Models for Count Data in R*, Journal of Statistical Software 27(8) (**W**): Quasi-Poisson hat dieselben Koeffizienten wie Poisson, nur die Inferenz ist angepasst; NB hat eine Likelihood.
- Cameron, *Count Data Regression Made Simple* (4 Seiten, Kurzfassung des Buchs von Cameron/Trivedi; **W**): Poisson-ML ist konsistent, wenn der bedingte Mittelwert stimmt; die üblichen Standardfehler gelten nur bei Equidispersion; der Effizienzgewinn von NB2 sei klein (B8 rechnet das in diesem Modell nach). Das Buch selbst ist nicht gelesen (**A**).
- Brooks u. a. (2017), glmmTMB, R Journal 9(2) (**W**): Überdispersion zu ignorieren macht Intervalle zu eng und erhöht die Rate falscher Funde.
- MacKinnon (2011), *Thirty Years of Heteroskedasticity-Robust Inference* (**W**): HC0 bis HC3 und die Namensverschiebung des HC3 (Jackknife bei MacKinnon/White 1985, heutige Fassung seit Davidson/MacKinnon 1993). MacKinnon/White 1985 selbst nur darüber (**S**).
- Ver Hoef, Boveng (2007), Ecology 88 (nur **S**, Volltext nicht gelesen): Quasi-Poisson und NB gewichten Zählwerte verschieden. Die Gewichte in B7 und alle Zahlen dazu sind eigene Nachrechnung.
- Warton (2005) und Harrison (2014) sind nicht gelesen (**S**); sie stehen nicht als Beleg, nur die Frage „viele Nullen sind nicht gleich Zero-Inflation“ in B14 ist offen formuliert.
- Eine Neuheitsaussage gibt es nicht; neu ist die Vorführbarkeit mit Zähleinheiten und Gegenproben.

## Ehrliche Grenzen

- **Ein Generator, eine Messwelt.** Container-Vehikel (erfundene Parameter) und eine zweite Welt mit einem Merkmal x ~ N(0, 1) und Anstieg 0,4. Andere Mittelwerte (μ unter 1, kleine Zählwerte mit vielen Nullen), mehrere Merkmale in der Messwelt 1 und eine lognormale Mischung statt Gamma sind **nicht gemessen**.
- **Ziel ist der Grenzwert des Schätzers, kein wahrer Koeffizient** (Vehikel). Die Überdeckung zeigt, wie oft das Intervall den Grenzwert des jeweiligen Schätzers trifft, nicht den Koeffizienten der Wahrheit; für NB2 ist das ein anderes Ziel als für Poisson.
- **Quasi-Poisson, NB2 und die Rangfolge** gelten für das gemessene Paar und die gemessenen Zellen: „NB2 ist besser“ und „Quasi-Poisson ist besser“ gelten beide nicht allgemein (B6). Kein Verfahren hält das Niveau in allen Zellen (B4).
- **Keine Pearson-gegen-Deviance-Messung** bei Poisson; keine Kenward-Roger-artigen Freiheitsgrade für das Sandwich; keine Zählmodelle mit Zufallseffekt auf Beobachtungsebene. Pearson-φ̂ ist Konvention, nicht Ergebnis eines Vergleichs.
- **Ursachen („weil“) sind Deutungen**, wo es in den Texten so steht: dass ein einziges φ nicht zu einer quadratisch wachsenden Varianz passt (B3), und der Grund für die Instabilität bei c = 0,3.
- **Die Zahl von φ̂ im Vehikel hängt am Zufallsstrom** (Heavy Tail der Art „Leer“): 3,12 gegen 3,08 bei zwei Zufalls-Seeds für den Grenzwert, 3,10 ± 0,90 über Historien; die Zahl der früheren Messung war 3,23.
- **Plattform.** Alle Zahlen sind unter Windows (Python 3.14, numpy 2.5.3, scipy 1.18.1) vorgerechnet; Tests mit Neu-Rechnung prüfen Bänder, damit Linux (Python 3.12) nicht an der letzten Stelle scheitert.
- **Keine Echtdaten.** Das Vehikel ist erfunden. Die App wurde mit AppTest geprüft; Diagramme sind im Browser nur angesehen, nicht auf dem Telefon und nicht in beiden Designs.

## Verifikation

- **Von Hand** (`tests/test_hand.py`, `tests/test_models.py`): acht Zählwerte in Brüchen (13/60, 308/15, 154/45, 1001/1350, 1057/1800, 1057/1350), Matrixform des Sandwichs gleich Gruppenformel in Brüchen, erster IRLS-Schritt (1/2; 7/2), Hebel 1/4 je Beobachtung, Gewichte 30/11 und 40/21, Gamma-Mischung (Brüche, 400 Glieder), Exposition.
- **Orakel** (`tests/test_oracle_zaehldaten.py`): IRLS gegen scikit-learn `PoissonRegressor` (Newton-Cholesky, 10⁻⁸) und gegen BFGS mit analytischem Gradienten; Offset gegen scikit-learn mit Rate und Gewicht; NB2-Log-Likelihood gegen `scipy.stats.nbinom`; α̂ gegen das Gitter-Maximum und gegen die gemeinsame Maximierung mit `jac="3-point"`; Sandwich gegen eine Schleife über die Beobachtungen;
  Grenzwert der Überdeckung auf zwei Wegen (Gauss-Hermite und `quad`, 10⁻⁶); Zählziehung gegen `scipy.stats.nbinom.ppf` (mindestens 99,95 % gleich) und Momente; Generator gegen die geschlossene Verteilung. scikit-learn steht nur in `requirements-dev.txt`.
- **Eigenschaften** (`tests/test_eigenschaften_zaehldaten.py`, je 20 Zufallsinstanzen): Quasi-Poisson hat dieselben Koeffizienten und SE = √φ̂ · SE_Poisson, NB2 nicht (bei falschem Mittelwertmodell); Vertauschen, Verschieben, Verdoppeln der Beobachtungen; Offset gleich Rate mit Gewicht; Skalieren der Exposition; NB2 mit α → 0 gleich Poisson; Gruppenmittel; Poisson-Überdeckung fällt mit α; φ̂ < 1 verkleinert das Quasi-Intervall.
- **Randfälle** (`tests/test_edge_cases.py`): Gruppe ohne Ereignis, alle Nullen, n = p, n < p, konstante Spalte, Unterdispersion (Randlösung), große Zählwerte, Ausfälle werden gezählt statt verschwiegen, Ausreißer-Datensätze ohne Mittelwert.
- **Claims** (`tests/test_claims.py`): jede Zahl der Befunde B1 bis B14 und B17 als Band, dazu die Vehikel-Parameter der README gegen den Generator (`test_b7_vehicle_parameters_*`); die Aussage „kein Verfahren hält das Niveau in allen Zellen“ als Zählung über 18 Zellen; die vorgerechnete Datei wird neu angestoßen (`tools/zae_make_frozen.py --check`) und mit einer kleinen Zelle verglichen (`tests/test_frozen.py` prüft Aufbau und Vollständigkeit).
- **Verdikte** (`tests/test_verdicts.py`): jede Zahl im Text und jede Schwelle (93 und 90 % der Klassen, Abstand über zwei Standardfehler strikt, Fehlalarmwarnung ab über 10 %) an handgesetzten Zeilen; je Preset die erwartete Verdikt-Zeile; das Preset „Keine Überstreuung“ meldet keinen Befund.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an Minimum und Maximum, jede Auswahl, Live-Probe, Gruppe ohne Ereignis, Permalink), Layout (Schnellstart in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, jede Regleränderung hebt die Markierung auf), Footer, Laufzeit-Importe.
- **Stand:** 228 Tests (`pytest --collect-only -q tests`), alle grün im Standardlauf (Windows, Python 3.14). Linux-Check (`demo_linux_check.py`, Docker, Python 3.12) vor den sechs zuletzt ergänzten Tests: 222 grün in 43 s Testzeit, 64 s gesamt (2026-11-16).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Kernfragen, fünf Experimente, Grenzen, Formeln |
| `data/zae_frozen.json` | vorgerechnete Wiederholungs-Messungen (aus `tools/zae_make_frozen.py`) |
| `requirements-dev.txt` | Laufzeit plus pytest und scikit-learn (nur Gegenproben) |
| `requirements.txt` | Laufzeit: streamlit, numpy, scipy, pandas, plotly |
| `tools/zae_make_frozen.py` | erzeugt (`--only <Teil>`) und prüft (`--check`) die vorgerechnete Datei |
| `zae_asymptotics.py` | Grenzwert der Überdeckung aus der Sandwich-Formel (Gauss-Hermite und `quad`) |
| `zae_constants.py` | Regler-Grenzen, Parameter der Messwelten, Presets |
| `zae_evaluation.py` | Auswertung, Verdikte als reine Funktionen, Zugriff auf die vorgerechnete Datei |
| `zae_generator.py` | Container-Vehikel: Merkmale, Verweilzeit, Standtage, Standgeld-Tage, Entwurfsmatrix |
| `zae_hand.py` | Handbeispiel in Brüchen, Gewichte, Exposition, Gamma-Mischung |
| `zae_layout.py` | Fester Kopf (Stylesheet) |
| `zae_models.py` | Kern: IRLS, Pearson-Dispersion, Sandwich, NB2, sieben Verfahren |
| `zae_presets.py` | Permalink (Begrenzen und Einrasten), Presets, Zufalls-Seed-Button |
| `zae_rng.py` | SplitMix64, vektorisierte Gleich- und Normalzahlen ohne C-Bibliothek, `rep_seed` |
| `zae_sample.py` | Zählwerte per Inversion, Datensätze der Messwelt 1 |
| `zae_simulation.py` | Wiederholungs-Messungen (Überdeckung, falsches Mittelwertmodell, Vehikel, Nullen) |
| `zae_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Randfälle, Claims, Verdikte, Presets, vorgerechnete Datei, Zufallsquelle, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Zero-Inflation und Hurdle:** nur der Nullanteil ist gemessen (B14), nicht, ob diese Modelle ihn beheben.
- **Pearson- gegen Deviance-Dispersion bei Poisson**, Freiheitsgrade für das Sandwich, Zufallseffekt auf Beobachtungsebene: nicht gemessen.
- **Ein Regler für φ̂ < 1** (Begrenzen auf 1): die App wendet φ̂ wie R unverändert an. Sie zeigt φ̂ < 1 nur als Pearson-φ̂ je Containerart (B11), nicht als schmaleres Intervall und nicht als Schalter.
- **Familie und Link** (Identität, andere Links, Deviance-Gütetest): [`glm-familie-demo`](https://github.com/sebastian-hanisch/glm-familie-demo). **Kunde als Cluster** (Zufallseffekt): [`cluster-gemischt-demo`](https://github.com/sebastian-hanisch/cluster-gemischt-demo). **Standgeld als Ja/Nein** (Kalibrierung, Schwelle): [`kalibrierung-schwelle-demo`](https://github.com/sebastian-hanisch/kalibrierung-schwelle-demo).
- **PDF-Export:** die Diagramme sind interaktiv, einen Bericht erzeugt die Demo nicht.
- **Statsmodels und andere Bibliotheken zur Laufzeit:** alles Rechnen mit numpy und scipy selbst; scikit-learn nur als Orakel in den Tests.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die vorgerechnete Datei neu erzeugen (rund sieben Minuten): `python tools/zae_make_frozen.py`; nur prüfen: `python tools/zae_make_frozen.py --check`.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, pandas, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Regression und GLM: wann die Gleichung trügt](https://sebastianhanisch.net/konzepte-regression.html).
