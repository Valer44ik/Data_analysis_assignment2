# 4. Task 3 — Cross-dataset generalization (20 %)

Answers: what adaptations were needed to run the model, what domain shift is, why performance
degrades, which features transfer best (+ the start.md point 7 options: crop/alignment,
resolution, intensity, feature-space alignment). Source: `03_cross_dataset.ipynb`.

```latex
\section{Task 3: Cross-dataset generalization}
\textbf{Protocol.} The best Task~2 configuration (\todo{model + features}) is retrained on the
preprocessed KDEF images and tested on FER2013 PrivateTest (KDEF$\to$FER); symmetrically, the same
configuration trained on FER2013 Training is tested on the KDEF test subjects (FER$\to$KDEF). Both
directions use the identical FER-format pipeline. Hyper-parameters always come from the
\emph{source} domain; no target label is ever used.

\begin{table}[h]\centering\small
\caption{Train (rows) / test (columns) macro-F1 and accuracy. Diagonal: in-domain.}
\label{tab:t3main}
\begin{tabular}{cc}
Macro-F1 & Accuracy\\
\input{results/task3_transfer_macro_f1.tex} & \input{results/task3_transfer_acc.tex}
\end{tabular}
\end{table}

\textbf{Results.} Performance collapses in both directions (Table~\ref{tab:t3main}): KDEF$\to$FER
reaches \todo{} macro-F1 (vs \todo{} in-domain on FER), FER$\to$KDEF \todo{} (vs \todo{} in-domain
on KDEF). Training on FER transfers better than training on KDEF, as also reported with deep
networks (48\,\% vs 33\,\% accuracy~\cite{kirbiz2026}): FER's variability covers part of KDEF's
clean distribution, but not vice versa. Training on all 980 KDEF images instead of 784 changes
KDEF$\to$FER by only \todo{}, so the problem is the \emph{kind} of data, not its amount.

\textbf{Adaptations needed to run the model.} KDEF images had to be face-detected, cropped with a
margin, converted to gray, resized to $48\times48$ and passed through exactly the same feature
extractors and normalisation; KDEF labels were mapped to FER indices (\emph{afraid}$\to$Fear);
there is no target validation set, so hyper-parameters were taken from source-domain tuning.

\textbf{What is domain shift?} The joint distribution $P(x,y)$ of the training (source) data differs
from the one of the deployment (target) data. Here we see (i) \emph{covariate shift}, $P(x)$
changes: framing and alignment (mean faces, Fig.~\ref{fig:intensity}), resolution and image quality,
illumination and contrast, head pose, occlusions; (ii) \emph{prior shift}, $P(y)$ changes: balanced
KDEF vs imbalanced FER; (iii) \emph{concept shift}, $P(y|x)$ changes: posed, exaggerated KDEF
expressions vs spontaneous, subtle and noisily labelled FER expressions. A logistic regression can
tell the two datasets apart with \todo{}--\todo{}\,\% accuracy depending on the feature
(\texttt{results/task3\_domain\_classifier.csv}; 50\,\% would mean no shift).

\textbf{Why performance degrades.} A KDEF model sees one lighting condition, one pose and 56
identities; its decision boundaries lie in a region of feature space that most FER images do
not occupy, so they fall on arbitrary sides of the boundaries. The prediction histogram shows
\todo{collapse onto a few classes, e.g.\ Surprise/Fear for open-mouth FER images}
(Fig.~\ref{fig:t3dist}). Conversely, subtle FER expressions are learned as "weak" versions, which
partly carries over to KDEF's strong versions.

\textbf{Which features transfer best?} Table~\ref{tab:t3feat}: \todo{landmark/geometric features
keep the largest fraction of their in-domain performance} since they are normalised for position
and scale and do not depend on illumination, while PCA on pixels transfers worst --- its components
encode the acquisition conditions. HOG lies in between: invariant to global brightness, but
sensitive to framing and blur.

\begin{table}[h]\centering\small
\caption{Feature transfer (macro-F1); same classifier, source-domain hyper-parameters.}
\label{tab:t3feat}\input{results/task3_features_macro_f1.tex}
\end{table}

\textbf{Reducing the shift.} Table~\ref{tab:t3adapt} removes one source of shift at a time:
(i) cropping --- the KDEF margin that best matches FER's landmark geometry (chosen without labels:
\todo{m}) and eye-line alignment of both datasets; (ii) resolution/quality --- KDEF degraded with
blur, noise, contrast jitter and JPEG artefacts; (iii) intensity --- per-image z-scoring plus CLAHE;
feature-space alignment with per-domain standardisation and CORAL~\cite{sun2016coral}; and
(iv) \emph{prior shift}: the target class priors are estimated without labels with the EM algorithm
of Saerens et al.~\cite{saerens2002} on the source model's posteriors for unlabelled target images,
and the posteriors are re-weighted accordingly (\texttt{results/task3\_em\_priors.csv} compares the
estimated and true priors). All steps use only unlabelled target \emph{training} images. The crop
margin is chosen without labels as the one whose mean landmark configuration is closest to FER's;
since FER crops are tighter than the Haar box we also tried negative margins (cropping inside the
box): \todo{chosen margin}. \todo{Summarise which helped and how much; the
combined row.}

\begin{table}[h]\centering\small
\caption{Domain-adaptation steps (macro-F1).}
\label{tab:t3adapt}\input{results/task3_adapt_macro_f1.tex}
\end{table}

\begin{figure}[h]\centering
\includegraphics[width=0.7\linewidth]{figures/task3_prediction_distribution.png}
\caption{True vs predicted class distribution after transfer.}
\label{fig:t3dist}
\end{figure}
```
