# 7. Task 6 — Real-time demo (10 %)

Answers: did the models perform well (yes/no, why), what problems did you run into, what fine
tuning was needed. Source: `group_2_emotion_recognition/demo_emotion.py`.
**To be completed by the team after testing the demo on a laptop webcam** — the bracketed items
are the observations to record.

```latex
\section{Task 6: Real-time demo}
\textbf{Implementation.} \texttt{demo\_emotion.py} loads a model bundle (fitted pipeline plus the
description of its features), detects faces with OpenCV's Haar cascade, crops each face with a
margin, converts it to a $48\times48$ gray image and computes exactly the training features with
the same code (\texttt{src/features.py}). Class scores (probabilities, or a soft-max of the SVM
decision values) are averaged over the last 10 frames to stabilise the prediction; the label,
confidence, per-class bars and frame rate are drawn on the image. Crop margin and smoothing can be
changed live (\texttt{+}/\texttt{-}, \texttt{s}).

\textbf{Did it work?} \todo{Yes/partly/no}. \todo{Observed: Happy and Surprise recognised
reliably and immediately; Neutral is the default; Sad/Fear/Disgust rarely or only with exaggerated
expressions.} This mirrors the per-class results of Tasks~1 and~5 and the fact that FER2013 was
collected from web photographs with different cameras, framing and lighting than a laptop webcam
--- another domain shift.

\textbf{Problems.} \todo{e.g.\ (1) frame rate: dlib landmarks + RBF-SVM with \textasciitilde N
support vectors run at \todo{} FPS; (2) flickering predictions between consecutive frames;
(3) crop framing: the Haar box differs from FER's crops, which changed predictions noticeably;
(4) backlight / dark rooms reduce contrast; (5) glasses and head rotation break the landmarks;
(6) bias towards Neutral/Happy.}

\textbf{Fine tuning.} \todo{e.g.\ temporal smoothing over 10 frames; crop margin tuned to
\todo{} (same value as selected in Task 3 for KDEF); CLAHE on the face crop; minimum face size
80\,px to avoid false detections; processing only the 3 largest faces.}
```
