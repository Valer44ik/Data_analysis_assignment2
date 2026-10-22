# 1. Introduction and datasets

Covers: assignment goal, FER2013 vs KDEF characteristics, dataset-specific and shared challenges
(grading criterion 2 "Understanding of the datasets"). Figures/tables come from
`00_data_preparation.ipynb` (`figures/data_*.png`, `results/data_*.csv|tex`).
The numbers below are final (they do not depend on model training).

```latex
\section{Introduction}
Facial expression recognition (FER) assigns one of seven basic emotions (angry, disgust, fear,
happy, sad, surprise, neutral) to a face image. We study how much the \emph{dataset} determines
the difficulty of this task by training classical machine-learning models (logistic regression,
RBF-SVM, random forest) on hand-crafted features (HOG, facial landmarks, PCA of the pixels) on two
datasets built for the same purpose: FER2013~\cite{goodfellow2013} (in-the-wild) and
KDEF~\cite{goeleven2008} (studio). We (i) build FER2013 and KDEF baselines, (ii) measure
cross-dataset generalization in both directions and analyse the domain shift, (iii) build an
interpretable fuzzy classifier from geometric features, (iv) use KDEF to improve FER2013, and (v)
run the best model in real time on a webcam.

\section{Datasets}
\textbf{FER2013} contains 35{,}887 grayscale $48\times48$ faces collected automatically with a web
image search and labelled into seven emotions. We use the official splits: Training (28{,}709) for
fitting, PublicTest (3{,}589) for every model-selection decision and PrivateTest (3{,}589) only for
the reported results. \textbf{KDEF} contains photographs of 70 actors (35 female, 35 male; two
sessions A/B) posing the same seven expressions in five head poses, in RGB at $562\times762$ pixels
under controlled studio conditions. We use the 980 frontal images and the official
subject-independent split (56 training / 14 test subjects; 784 / 196 images). KDEF images are
converted to the FER format with the provided procedure: Haar-cascade face detection, box
expanded by 20\,\% on each side, grayscale, resize to $48\times48$ (Fig.~\ref{fig:samples});
our implementation reproduces the kit output exactly (maximum pixel difference 0).

\begin{table}[h]\centering\small
\caption{Characteristics of the two datasets (after conversion of KDEF to the FER format).}
\label{tab:datasets}
\begin{tabular}{lll}\toprule
 & FER2013 & KDEF (frontal) \\\midrule
Images (train/val/test) & 28{,}709 / 3{,}589 / 3{,}589 & 784 / -- / 196 (56 / 14 subjects)\\
Acquisition & web, spontaneous \& posed, in the wild & studio, posed by trained actors \\
Original format & $48\times48$ gray & $562\times762$ RGB \\
Class balance & Happy 25.1\,\%, Disgust 1.5\,\% (ratio 16.5) & 14.3\,\% each \\
Mean intensity / lighting variability$^a$ & 129.5 / 33.5 & 83.8 / 26.3 \\
Mean per-image contrast (pixel std) & 54.0 & 40.5 \\
Pose, framing & variable pose, tight crops & identical frontal pose and framing \\
Label noise, duplicates & 12 blank images, 1{,}236 exact duplicates in train, & none \\
 & 59 duplicate groups with conflicting labels, & \\
 & 288 test images identical to a training image & \\
Identities & thousands, unknown & 70 known $\Rightarrow$ subject-independent split \\\bottomrule
\end{tabular}\\[2pt]
{\footnotesize $^a$ standard deviation of the per-image mean intensity.}
\end{table}

\begin{figure}[h]\centering
\includegraphics[width=\linewidth]{figures/data_samples.png}
\caption{Random training samples per emotion: FER2013 (5 per row) and KDEF after conversion (3 per row).}
\label{fig:samples}
\end{figure}

\paragraph{Dataset-specific challenges.}
\emph{FER2013}: strong class imbalance (Disgust has 436 training images, 16.5$\times$ fewer than
Happy); large variation in head pose, illumination (Fig.~\ref{fig:intensity}), age, occlusions
(hands, glasses, watermarks, text); low resolution, which makes landmark detection unreliable;
label noise --- identical images with different labels, non-faces and ambiguous expressions; and
duplicates across the official splits (8\,\% of PrivateTest also appears in Training), which
slightly inflates test scores. \emph{KDEF}: very few samples (784 training images, 56 people), so
models overfit easily and the 196-image test set gives wide confidence intervals; the identity of a
person is a strong confounder (same face shape across 7 expressions), which is why the split must be
subject-independent; posed, exaggerated expressions with no natural variability; colour and high
resolution that must be removed to compare with FER.

\paragraph{Shared challenges.} The same seven classes with intrinsically confusable pairs (fear vs
surprise, sad vs neutral, angry vs disgust); expressions are subtle local deformations that must be
separated from person-specific facial morphology; and both datasets must pass through the same
face crop at 48$\times$48 pixels, which removes fine texture.

\begin{figure}[h]\centering
\includegraphics[width=0.38\linewidth]{figures/data_mean_faces.png}\hfill
\includegraphics[width=0.58\linewidth]{figures/data_intensity.png}
\caption{Left: mean face of FER2013 and of KDEF for crop margins 0--0.3 (framing mismatch).
Right: pixel-intensity and per-image contrast distributions (acquisition mismatch).}
\label{fig:intensity}
\end{figure}
```
