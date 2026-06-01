# NILM et le déséquilibre de classe — le cœur du problème

Ce document explique le défi central d'Ignis : entraîner un modèle qui **détecte
vraiment** les appareils, et pas un modèle qui obtient un bon score en ne
prédisant rien. C'est le piège classique du NILM (Non-Intrusive Load
Monitoring) et le vrai travail du projet, au-delà de la plomberie.

> **TL;DR** — Nos appareils sont allumés ~2-3 % du temps. Un modèle naïf
> minimise l'erreur moyenne (MAE) en prédisant « toujours 0 W » : il a raison
> 97 % du temps mais ne détecte **rien** (F1 = 0). On corrige à trois niveaux :
> les **données** (équilibrage), la **loss** (pénaliser les ratés, pas les
> fausses alertes), et la **normalisation** (échelle par appareil).

---

## 1. Le problème : un signal rare noyé dans le bruit

Le NILM part d'**un seul chiffre** — la puissance agrégée du compteur Linky
(ex. 447 W en moyenne, pics à 3570 W) — et doit **désagréger** : combien le
four ? le lave-vaisselle ? la télé ?

La vérité terrain (ce qu'on cherche à prédire) vient des prises Meross qui
mesurent chaque appareil. Problème mesuré sur 14 jours de données réelles :

| Appareil        | % du temps ON | Puissance ON typique |
|-----------------|---------------|----------------------|
| four            | 2.2 %         | jusqu'à 2914 W       |
| lave_vaisselle  | 3.3 %         | jusqu'à 2186 W       |
| pc              | 2.4 %         | jusqu'à 245 W        |
| television      | 35 %          | 50-135 W             |
| smart_plug      | 30 %          | 55-120 W             |
| lave_linge      | < 1 %         | (rare)               |

Deux difficultés cumulées :
1. **Rareté** — la plupart des appareils sont OFF la quasi-totalité du temps.
2. **Faible amplitude** — la télé à 130 W représente 3,6 % de l'agrégat (3570 W
   max). Son signal est noyé.

---

## 2. Le piège : « MAE basse mais F1 nul »

### Les deux métriques (vulgaire)

- **MAE** (Mean Absolute Error) = erreur moyenne en watts. *Une métrique de
  moyenne.*
- **F1** = qualité de détection ON/OFF. Combine :
  - **précision** : quand le modèle dit « ON », a-t-il raison ? (évite les
    fausses alertes),
  - **rappel** : quand c'est vraiment ON, le voit-il ? (évite les ratés).
  - F1 = moyenne harmonique des deux, de 0 (nul) à 1 (parfait). **Gate = 0.8.**
- **Energy error** = écart relatif d'énergie cumulée. Gate = ≤ 15 %.

### Pourquoi la MAE ment

Un appareil OFF 97 % du temps : le modèle « prédis toujours 0 W » se trompe
seulement 3 % du temps → **MAE minuscule** (~0.01) → *a l'air excellent*. Mais
il ne détecte **aucun** appareil → **F1 = 0**. Inutile.

> **Leçon** : sur des événements rares, une métrique de moyenne (MAE) est
> trompeuse. F1 ne ment pas, il regarde précisément les moments ON.

C'est exactement ce qu'on a observé sur les premiers modèles (3 epochs, puis 50
epochs) : `val_mae ≈ 0.01`, `F1 = 0.000` partout, `energy_err ≈ 0.97` (le modèle
sous-estime de ~97 %, c.-à-d. ne prédit presque rien).

---

## 3. Le diagnostic : ne pas spéculer, mesurer

Plutôt que deviner, on a sondé le modèle réel
(`scripts/diag_predict.py`) — ce qu'il prédit vraiment vs la vérité :

```
appliance        pred[min/max/mean]      truth[min/max/mean]
four                0.0/    1.2/  1.16      0.0/ 2914.2/ 25.48
lave_vaisselle      0.0/    0.0/  0.00      0.0/ 2185.9/  9.60
television          0.0/    1.0/  0.90      0.0/  134.9/ 45.42
```

Verdict sans appel : **le modèle sort une quasi-constante proche de 0** (pred max
1,2 W là où la vérité monte à 2914 W). Effondrement vers zéro — *mode collapse*.

> **Méthode** : toujours inspecter les prédictions brutes avant de changer le
> modèle. Un script de diagnostic de 40 lignes a tranché ce qu'une heure de
> spéculation n'aurait pas tranché.

---

## 4. Les trois leviers

On les attaque **un à la fois**, en mesurant entre chaque, pour savoir lequel
agit (méthode scientifique, pas tout changer d'un coup).

### Levier 1 — Données : équilibrer (`training/windows.py::balance_windows`)

On jette une partie des fenêtres « mortes » (aucun appareil ON au centre).
À ~2-3 % d'activité, 97 % des fenêtres sont du vide. On garde **toutes** les
fenêtres actives + un nombre égal de mortes (`dead_ratio=1.0` → ~50/50).
Le modèle ne peut plus gagner en prédisant zéro.

*Résultat seul : insuffisant.* Le F1 est resté à 0 — l'équilibrage aide mais ne
suffit pas, car la **loss** poussait encore vers zéro (levier 2).

### Levier 2 — Loss : pénaliser les ratés, pas les fausses alertes

Le moteur hérité (Linkya) utilisait `asymmetric_loss` qui pénalise **plus** les
**faux positifs** (prédire de la puissance quand la vérité est ~0). Conçu pour
des signatures propres, c'est **l'inverse de ce qu'il faut** ici : avec 97 % de
OFF, cette loss apprend au modèle « ne prédis jamais haut, reste à zéro, c'est
plus sûr ». Elle **cause** l'effondrement.

Correctif (`training/trainer.py::_recompile_under_penalty`) : on recompile avec
une MAE qui pénalise la **sous-prédiction** (rater un ON) `fn_penalty=3×` plus
que la sur-prédiction. L'asymétrie opposée, adaptée à des cibles rares.

```python
def under_penalty_loss(y_true, y_pred):
    err = y_true - y_pred
    under = tf.cast(err > 0.0, tf.float32)   # vérité au-dessus de la prédiction = un raté
    weight = 1.0 + (fn_penalty - 1.0) * under
    return tf.reduce_mean(weight * tf.abs(err))
```

### Levier 3 — Normalisation : échelle par appareil *(suspect, pas encore appliqué)*

Le `target_scaler` est un **MinMax global** sur tous les appareils poolés
(max ~2179 W, dominé par le four). Conséquence : un cycle télé à 130 W se
normalise à `130/2179 ≈ 0.06`. Le modèle a juste à sortir ~0.01 pour les petits
appareils → ils sont **écrasés** par l'échelle du four.

Piste si les leviers 1+2 ne suffisent pas : **un scaler par appareil**, chacun
normalisé sur sa propre plage, pour que la télé et le four pèsent autant dans la
loss.

---

## 5. Ce qu'on surveille à l'entraînement (apprentissage ML)

- **loss (train) qui descend** : le modèle apprend.
- **val_loss** (15 % de données jamais vues) : la vraie généralisation. Si
  `train` descend mais `val` stagne/monte → **overfitting** (le modèle
  mémorise au lieu d'apprendre). `EarlyStopping(patience=10)` coupe avant.
- **F1 et energy_error à l'eval** : le seul juge réel (gate F1 ≥ 0.8,
  err ≤ 15 %). Une MAE basse ne suffit jamais.
- **diag_predict** : `pred.max` doit être du même ordre que `truth.max`. S'il
  reste ridiculement bas → le modèle s'effondre encore.

---

## 6. Champion / challenger (pourquoi)

Chaque entraînement produit un **challenger** daté. On le compare au
**champion** (le modèle de prod, `champion.keras`). Promotion seulement si
meilleur (`Score = (#appareils passant le gate, F1 moyen)`). Ainsi **on ne
régresse jamais** en prod : un mauvais réentraînement n'écrase pas un bon
modèle. La promotion est **locale** ; l'envoi au Raspberry Pi (`make ship`)
reste **manuel** — on valide avant de déployer.

---

## 7. Journal des résultats

| Date       | Changement                                | F1 par appareil | Note |
|------------|-------------------------------------------|-----------------|------|
| 2026-06-01 | baseline 3 epochs                         | tous 0.000      | collapse zéro |
| 2026-06-01 | baseline 50 epochs                        | tous 0.000      | idem, MAE basse trompeuse |
| 2026-06-01 | + balance_windows (levier 1)              | tous 0.000      | insuffisant seul |
| 2026-06-01 | + under_penalty_loss (levier 2)           | tous 0.000      | insuffisant |
| 2026-06-01 | + sample_weight par appareil (levier "3") | **tv 0.547**, rares 0.000 | 1er signe de vie ! mais seuls les appareils à forte activité percent |
| 2026-06-01 | + sample_weight + seq_len 599 (5h window)  | tv **0.547**, reste 0.000 | seuls les appareils à forte activité percent |
| 2026-06-01 | → seq_len 599→99 (5h→50min)                | tv **0.701**, pc **0.416**, reste 0.000 | net progrès + train 6× plus rapide (3s/epoch). pc apparait. four/lave_vaisselle/smart_plug encore 0. |

### Constat après seq_len=99

Fenêtre courte = gros gain : tv 0.55→0.70, pc 0→0.42, et le train passe de 30s à
3s par epoch. Mais une **séparation nette** demeure :
- **Apprennent** : tv (232 h actives sur 30 j), pc (24 h) — assez d'exemples.
- **Restent à 0** : four (11 h), lave_vaisselle (11 h), smart_plug (signal faible
  ~55 W). Leur `val_loss` reste figée (four 0.204), preuve que le modèle ne voit
  pas assez de leurs cycles. C'est désormais une **limite de données** (peu
  d'activations sur 30 j), pas seulement de méthode.

Pistes restantes : (a) accumuler plus de jours (le live MQTT alimente en continu) ;
(b) split train/val **par cycle** plutôt qu'aléatoire (un cycle rare ne doit pas
tomber uniquement en val) ; (c) seq_len adapté au pic court du four.

### Le vrai coupable : la longueur de fenêtre

Levier découvert en lisant les résultats : `sequence_length=599` venait de Linkya
où les données étaient à **1 Hz** (599 points = 10 min). Sur notre grille **30 s**,
599 points = **5 heures**. Le modèle regarde 5 h d'agrégat pour deviner un point
central. Un four qui tourne 20 min ne représente que 0,3 % de la fenêtre — noyé.
La télé (allumée 35 % du temps) perçait malgré tout (F1 0.55) car son signal est
présent quelle que soit la fenêtre ; les appareils rares à cycles courts, non.

Fix : `seq_len` calibré sur la **durée d'un cycle**, pas sur un héritage 1 Hz.
~99 points × 30 s ≈ 50 min. Fenêtre 6× plus courte, le cycle d'un appareil
redevient visible. C'est le levier qui manquait.

*(À compléter à chaque itération. Le but : per-appliance F1 ≥ 0.8 et energy
error ≤ 15 % vs HA.)*
