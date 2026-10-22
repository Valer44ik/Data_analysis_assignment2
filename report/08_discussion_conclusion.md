# 8. Discussion and conclusion

Covers grading criteria 3–5: feature analysis, model analysis (strengths, limitations, suitable
applications), critical interpretation and limitations.

```latex
\section{Discussion}
\textbf{Features.} HOG is the most informative and robust single representation on FER2013
(local gradient structure, brightness invariance), landmarks add complementary geometry and are the
most \emph{transferable} information (normalised, illumination-independent) but are only as good as
the landmark detector at $48\times48$ px; PCA of the pixels mostly captures nuisance factors and
transfers worst. Binarisation removes the grey-level shading that carries expression and is not
recommended; contrast normalisation (CLAHE) is the only image-level change that \todo{helps}.

\textbf{Models.} \emph{RBF-SVM}: best accuracy, handles high-dimensional dense features, but
training scales quadratically with the number of samples and prediction with the number of support
vectors; no native probabilities --- suited to offline or moderate-rate applications.
\emph{Logistic regression}: linear, fast to train and to evaluate, calibrated probabilities and
inspectable weights; less accurate on FER but competitive on small clean data (KDEF) ---
suited to embedded/real-time use and as a baseline. \emph{Random forest}: robust to feature
scaling, little tuning, fast prediction and feature importances, but weak with many correlated
low-level features and biased to majority classes --- suited to compact, heterogeneous features
(e.g.\ geometric). \emph{Fuzzy TS model}: fully interpretable rules with plausible content at a
moderate cost in accuracy --- suited to settings that require explanations (e.g.\ clinical or
educational tools), not to maximise accuracy.

\textbf{Limitations.} FER2013's label noise bounds the attainable accuracy (human agreement on
FER2013 is reported around 65\,\%); KDEF results rest on 14 test subjects (wide confidence
intervals); ablations were run on a 12k subset and one random seed; landmark quality on FER was not
corrected; all methods are frame-based and ignore temporal dynamics.

\section{Conclusion}
\todo{3--4 sentences: best FER model and score; KDEF much easier but tiny; strong asymmetric
domain shift that preprocessing alone cannot remove; KDEF knowledge (geometry, prototypes, label
checking, merging) gives \todo{} on FER; demo works for clear expressions.}
```
