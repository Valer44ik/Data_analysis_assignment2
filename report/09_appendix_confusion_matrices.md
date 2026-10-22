# 9. Appendix — confusion matrices

The assignment asks for an appendix with the confusion matrices of every reported model, with no
long explanations. All figures are written by the notebooks into `figures/`.

```latex
\clearpage
\appendix
\section{Confusion matrices (row-normalised)}
\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{figures/appendix_task1_confusion_matrices.png}
\caption{Task 1, FER2013 PrivateTest: every classifier $\times$ feature combination of Table~\ref{tab:t1grid}.}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{figures/appendix_task2_confusion_matrices.png}
\caption{Task 2, KDEF test subjects: every combination of Table~\ref{tab:t2grid}.}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{figures/task3_transfer_cms.png}
\caption{Task 3: in-domain and cross-dataset (FER$\to$KDEF, KDEF$\to$FER).}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=0.32\linewidth]{figures/task4_fuzzy_cm.png}\hfill
\includegraphics[width=0.32\linewidth]{figures/task5_final_cm.png}\hfill
\includegraphics[width=0.32\linewidth]{figures/task1_best_cm.png}
\caption{Task 4 fuzzy classifier (KDEF test), Task 5 final model and Task 1 best model (FER PrivateTest).}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=0.6\linewidth]{figures/data_landmarks.png}
\caption{dlib landmarks on FER2013 (left) and KDEF (right) faces.}\label{fig:landmarks}
\end{figure}
```
