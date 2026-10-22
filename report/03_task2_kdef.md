# 3. Task 2 — KDEF dataset study (15 %)

Answers: why KDEF is easier/harder than FER2013, which models change ranking, which emotions
transfer well between datasets. Source: `02_kdef_study.ipynb`.

```latex
\section{Task 2: KDEF study}
\textbf{Setup.} The same features and classifiers are retrained on the 784 frontal KDEF training
images and tested on the 196 images of the 14 held-out subjects (no person in both splits, sessions
A and B of a person in the same split). Without a validation split, hyper-parameters are chosen
by 5-fold GroupKFold over the training subjects (macro-F1). The full grid is reported in the same
layout as Task~1 (Table~\ref{tab:t2grid}); the column \todo{HOG+LM} is the best representation
identified on FER2013.

\begin{table}[h]\centering\small
\caption{KDEF test results (14 unseen subjects). Rows: classifier, columns: features.}
\label{tab:t2grid}
\begin{tabular}{cc}
Macro-F1 & Accuracy \\
\input{results/task2_grid_test_macro_f1.tex} & \input{results/task2_grid_test_acc.tex}
\end{tabular}
\end{table}

\textbf{Results.} The best configuration (selected on CV) is \todo{model + features} with
\todo{} macro-F1 (bootstrap 95\,\% CI \todo{}--\todo{}); on average KDEF scores are \todo{x} points
higher than on FER2013 (\texttt{results/task2\_minus\_task1\_macro\_f1.csv}).

\textbf{Why KDEF is easier.} The expressions are posed and exaggerated by trained actors, all faces
are frontal with identical framing and studio lighting, there are no occlusions, no label noise or
duplicates, and the classes are balanced. Every image therefore varies essentially only in
expression and identity, so even 784 images suffice. \textbf{Why it is also harder.} With so few
images, high-capacity models overfit and estimates are noisy (one test image $=0.5$\,\% accuracy;
CV standard deviations of \todo{}); 56 identities make person-specific face shape a strong
confounder, which the subject-independent split exposes.

\textbf{Ranking changes.} \todo{Describe from results/task2\_rank\_models.csv and
task2\_rank\_features.csv, e.g.: LR moves up because with few samples and clean data a linear
model is less prone to overfit than an RBF-SVM; landmarks gain the most because dlib is accurate on
clean frontal faces, while PCA improves because pixel variance in KDEF is mostly expression and
identity rather than illumination.}

\textbf{Gray vs RGB.} Colour adds \todo{} macro-F1 for HOG and \todo{} for PCA
(\texttt{results/task2\_color\_macro\_f1.csv}): expressions are encoded by shape and shading, and
colour mainly reflects skin and lighting --- and a colour model could not be applied to
grayscale FER anyway. Binarised and edge inputs behave as on FER (\todo{}).

\textbf{Which emotions transfer well.} Table~\ref{tab:t2emo} compares per-class F1 of the same
configuration within each dataset. Happy and Surprise are easy in both (large, unambiguous
geometric changes). The largest gains on KDEF are for \todo{Fear, Disgust and Sad}: in FER these
are rare, subtle or noisily labelled, whereas the actors' posed versions are prototypical. The
actual cross-dataset transfer is measured in Task~3.

\begin{table}[h]\centering\small
\caption{Per-class F1, same configuration trained/tested within each dataset.}
\label{tab:t2emo}\input{results/task2_per_class_fer_vs_kdef.tex}
\end{table}
```
