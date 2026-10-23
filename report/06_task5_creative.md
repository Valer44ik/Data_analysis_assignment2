# 6. Task 5 — Creative challenge (25 %)

Covers: description of the features and processing in the creative challenge, at least 2
strategies (we use: hyper-parameter tuning, feature engineering, feature selection, class
balancing, data augmentation, data curation/cleaning, label verification and KDEF-derived features,
prototypes and dataset merging). Source: `05_creative_challenge.ipynb`.

```latex
\section{Task 5: Creative challenge --- improving FER2013 with KDEF}
\textbf{Approach.} Starting from the best Task~1 configuration (\todo{model + features}), groups of
improvements are tested in sequence. Within a group the options are compared on PublicTest and the
best is adopted only if it improves validation macro-F1 (greedy forward selection); PrivateTest is
only reported. Every option is trained on the \emph{full} training set: in a first run, ablations
on a 12k subset under-estimated the full-data score by 8 points and hid real gains in noise.
The GPU kernel computation (Task~1) makes one full-data SVM fit take about a minute.

\textbf{Strategies.}
\begin{itemize}\itemsep0pt
\item \emph{Feature engineering}: CLAHE input; HOG cell size ($6/8$ px) and orientations ($9/12$).
\item \emph{Class balancing}: none vs inverse-frequency class weights vs SMOTE (inside an
  \texttt{imblearn} pipeline so that synthetic samples never leak into validation).
\item \emph{Data augmentation}: horizontal flips, and flips + random rotations ($\pm10^\circ$) of
  the training images.
\item \emph{KDEF-derived geometric features}: 20 scale-invariant geometric measures ranked on KDEF
  by mutual information with the emotion \emph{and} by transfer consistency (Spearman correlation
  between per-emotion means on KDEF and on FER). Only features that are informative on clean data
  \emph{and} behave the same in the wild are added (\todo{n} selected,
  \texttt{results/task5\_geo\_ranking.csv}).
\item \emph{KDEF emotion prototypes}: Procrustes-aligned mean landmark shape of each emotion on
  KDEF; the distances of a face to the 7 prototypes are added as identity-normalised features (the
  nearest prototype alone classifies FER with \todo{} accuracy).
\item \emph{Data curation and label verification}: dHash duplicates and blank images removed (all
  copies when they disagree on the label); training images flagged when a cross-validated model
  and the nearest KDEF prototype \emph{both} contradict the label with high confidence
  (\todo{n} images, Fig.~\ref{fig:t5flag}). Only the training set is curated.
\item \emph{Dataset merging} (following~\cite{kirbiz2026}): the 56 KDEF training subjects, clean
  or degraded to FER-like quality, are added to the FER training set with inverse-frequency class
  weights.
\item \emph{Texture}: uniform LBP histograms~\cite{shan2009} on a $6\times6$ grid (360 dims),
  capturing wrinkles, furrows and teeth that HOG cells and geometry only partly see.
\item \emph{Feature selection} (SelectKBest/ANOVA vs PCA 95\,\%) and a finer
  \emph{hyper-parameter search} of $C$ and the kernel width.
\item \emph{Test-time augmentation}: the class scores of a face and of its mirror image are averaged.
\item \emph{Score-level ensemble}: SVMs on complementary representations (selected configuration,
  LBP+landmarks, eigenfaces), z-normalised decision scores averaged with weights chosen on PublicTest.
\item \emph{More data}: after all choices are frozen, the selected model is refitted on Training +
  PublicTest; both the Training-only and the refitted model are reported on PrivateTest.
\end{itemize}

\begin{table}[h]\centering\small
\caption{Ablation (macro-F1; adopted steps in bold). PrivateTest is reported, never used for selection.}
\label{tab:t5abl}\input{results/task5_ablation_macro_f1.tex}
\end{table}

\textbf{Results.} \todo{Describe adopted steps and their gains.} The final model reaches
\todo{acc}/\todo{bal-acc}/\todo{F1} on PrivateTest, compared with \todo{} for the Task~1 best model
and \todo{} for the provided baseline (Table~\ref{tab:t5cmp}). \todo{Per-class: which classes
improved (Disgust with balancing/merging, ...).} Merging KDEF \todo{helps/does not help} FER but
makes the model work on both domains (KDEF test \todo{}).

\begin{table}[h]\centering\small
\caption{Final comparison on FER2013 PrivateTest. Last year's number is on a landmark-filtered test set.}
\label{tab:t5cmp}\input{results/task5_comparison.tex}
\end{table}

\begin{figure}[h]\centering
\includegraphics[width=\linewidth]{figures/task5_flagged_examples.png}
\caption{FER training images whose label is contradicted by both the model and the KDEF prototypes (label$\to$suggested).}
\label{fig:t5flag}
\end{figure}
```
