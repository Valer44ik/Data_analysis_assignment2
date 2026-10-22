# 5. Task 4 — Explainable fuzzy classifier (10 %)

Answers: how human-understandable rules about facial expressions perform; are the rules consistent
with expectations; do they make sense. Source: `04_fuzzy_kdef.ipynb` (fuzzy environment).

```latex
\section{Task 4: Explainable fuzzy classifier}
\textbf{Features.} Six geometric measures from the dlib landmarks of the KDEF faces, all divided by
the inter-ocular distance: \emph{mouth opening} (inner lips), \emph{mouth width} (lip corners),
\emph{eye opening} (eye aspect ratio), \emph{brow height} (eye centre to brow), \emph{smile ratio}
(vertical position of the lip corners relative to the lip centre) and \emph{brow distance} (inner
brows, frowning). They are min-max scaled to $[0,1]$ so that linguistic terms are comparable.
All six differ significantly between emotions (ANOVA, \texttt{results/task4\_feature\_means.csv}).

\textbf{Model.} For each emotion a first-order Takagi--Sugeno model is identified with
pyFUME~\cite{fuchs2020pyfume}: fuzzy c-means clustering defines the rules (one per cluster, Gaussian
membership functions), and the linear consequents are fitted by weighted least squares on the
one-vs-rest target. The emotion with the highest output wins. The number of rules (\todo{k}) and
the consequent order were selected with subject-wise 5-fold CV. We verified that the membership
functions did not degenerate into "universal" sets (median width $\sigma=\todo{}$ on a $[0,1]$
scale), a failure mode in which the model reduces to a linear regression.

\textbf{Performance.} Table~\ref{tab:t4}: the fuzzy model reaches \todo{} macro-F1 on the KDEF
test subjects, close to \todo{/below} a logistic regression (\todo{}) and an RBF-SVM (\todo{}) on
the same six features, and \todo{} points below the best Task~2 model that uses
$\sim$1{,}000 HOG+landmark values. Interpretability thus costs little relative to black boxes on the
same information; the gap to Task~2 is the information that six distances cannot capture
(texture: wrinkles, nasolabial folds, nose wrinkling). Applied to FER2013 the rules reach only
\todo{}, mostly because dlib's landmarks are inaccurate on FER faces.

\begin{table}[h]\centering\small
\caption{Fuzzy classifier vs crisp models (KDEF test; last row cross-dataset).}
\label{tab:t4}\input{results/task4_comparison.tex}
\end{table}

\textbf{Do the rules make sense?} The strongest rule per emotion (\texttt{results/task4\_rules.csv})
reads e.g.\ \todo{"IF smile ratio is HIGH AND mouth width is HIGH THEN Happy"}, \todo{"IF brow height
is HIGH AND eye opening is HIGH AND mouth opening is HIGH THEN Surprise"}, \todo{"IF brow distance
is LOW AND brow height is LOW THEN Angry"}. These agree with the Facial Action Coding System:
lip-corner puller (AU12) for happiness; brow raisers (AU1+2), upper-lid raiser (AU5) and jaw drop
(AU26) for surprise; brow lowerer (AU4) for anger. \todo{Inconsistent/unexpected rules, e.g.\ Fear
sharing the Surprise antecedents, Disgust/Sad rules close to Neutral} explain the confusions of
the confusion matrix: geometry alone cannot separate expressions that differ mainly in texture.

\begin{figure}[h]\centering
\includegraphics[width=\linewidth]{figures/task4_feature_boxplots.png}
\caption{Distribution of the six geometric features per emotion (KDEF train).}
\end{figure}
```
