# STGA‑PDQL: Spatio‑Temporal Aware Participant Selection with Local Differential Privacy in Mobile Crowdsensing

This repository contains the official implementation of the paper:  

**"Spatio-Temporal Aware Participant Selection with Local Differential Privacy in Mobile Crowdsensing"**  
*Mengzhe Tian, Yongjiao Sun, Yishu Wang, Hangxu Ji*  
Accepted by *IEEE Transactions on Dependable and Secure Computing*, 2026.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Datasets](#datasets)
- [Reproducibility Guide](#reproducibility-guide)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Data Preparation](#2-data-preparation)
  - [3. Train ST‑GAT Model](#3-train-stgat-model)
  - [4. Run Participant Selection (DQN)](#4-run-participant-selection-dqn)
  - [5. Reproduce Paper Tables & Figures](#5-reproduce-paper-tables--figures)
- [Code Structure](#code-structure)
- [Citation](#citation)
- [License](#license)

---

## Overview

We propose **STGA‑PDQL**, a novel framework for privacy‑preserving participant selection in mobile crowdsensing. It combines:
- **ST‑GAT** (Spatio‑Temporal Graph Attention Network) for trajectory prediction,
- **Local Differential Privacy (LDP)** mechanisms to protect participants’ locations,
- **Deep Q‑Learning (DQN)** to sequentially select participants and maximize long‑term profit under spatio‑temporal coverage constraints.

Experimental results on three real‑world datasets (MCM, GeoLife, T‑Drive) show that STGA‑PDQL outperforms state‑of‑the‑art baselines by 9.5%–18.7% in ST‑WSC and 10.8%–16.9% in net profit.
