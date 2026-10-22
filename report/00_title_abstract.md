# 0. Preamble, title and abstract

Overleaf: paste this block at the top of `main.tex`, then the other sections in order
(01 … 10). Upload the folders `figures/` and `results/` from `group_2_emotion_recognition/`
next to `main.tex` (the tables are `\input` from `results/*.tex`). Page limit: 8 pages + appendix.

```latex
\documentclass[10pt,a4paper]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{graphicx,booktabs,amsmath,hyperref,xcolor,float}
\usepackage[font=small]{caption}
\newcommand{\todo}[1]{\textcolor{red}{[#1]}}
\setlength{\parskip}{2pt}

\title{Emotion recognition with classical machine learning:\\FER2013 vs KDEF and the domain gap}
\author{Group 2 --- Gabriele Salvo, Val\=erijs Gon\v{c}arovs, Herman Chang\\5ARE0, Assignment 2}
\date{October 2026}

\begin{document}
\maketitle
\begin{abstract}
We compare logistic regression, RBF-SVM and random forests on HOG, facial-landmark and PCA features
for seven-class facial expression recognition on the in-the-wild FER2013 dataset and on the
studio-acquired KDEF dataset. The best FER2013 model (\todo{model+features}) reaches \todo{}
accuracy and \todo{} macro-F1 on PrivateTest, while the same pipeline reaches \todo{} macro-F1 on
unseen KDEF subjects. Cross-dataset transfer collapses in both directions (KDEF$\to$FER \todo{},
FER$\to$KDEF \todo{} macro-F1); we quantify the domain shift, show which features transfer best
and how far cropping, quality, intensity and feature-space alignment reduce it. An interpretable
Takagi--Sugeno fuzzy classifier on six geometric features reaches \todo{} on KDEF with rules that
agree with the Facial Action Coding System. Using KDEF-derived geometric features and emotion
prototypes, label verification, class balancing, augmentation and dataset merging, we improve
FER2013 to \todo{} macro-F1, and run the final model in real time on a webcam.
\end{abstract}
```
