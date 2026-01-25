# Baseline CNN - Error & Gap Analysis

Generated: 2026-01-24 18:23:53.609355

---

## Bonn EEG - Analysis

**Overall Accuracy**: 0.8833

**Class-wise Performance**:
  - Set A: P=1.000, R=0.600, F1=0.750
  - Set B: P=0.700, R=0.933, F1=0.800
  - Set D: P=0.938, R=1.000, F1=0.968
  - Set E (Seizure): P=1.000, R=1.000, F1=1.000

**Weakest Class**: Set A (F1=0.750)

**Common Misclassifications**:
  - Set A → Set B: 6 errors
  - Set B → Class 2: 1 errors

**Potential Failure Modes**:
  3. **Multi-class complexity** - Binary decomposition might help
  4. **Limited temporal modeling** - CNN lacks explicit sequence modeling
  5. **Raw signal variability** - May benefit from time-frequency features

---

## Epileptic Seizure - Analysis

**Overall Accuracy**: 0.6881

**Class-wise Performance**:
  - Seizure: P=0.983, R=0.846, F1=0.910
  - Tumor: P=0.658, R=0.417, F1=0.511
  - Healthy: P=0.554, R=0.794, F1=0.652
  - Eyes Closed: P=0.684, R=0.786, F1=0.731
  - Eyes Open: P=0.648, R=0.597, F1=0.621

**Weakest Class**: Tumor (F1=0.511)

**Common Misclassifications**:
  - Tumor → Healthy: 184 errors
  - Eyes Open → Eyes Closed: 105 errors
  - Eyes Closed → Eyes Open: 70 errors

**Potential Failure Modes**:
  2. **Poor Tumor detection** - May need class-specific features
  3. **Multi-class complexity** - Binary decomposition might help
  4. **Limited temporal modeling** - CNN lacks explicit sequence modeling
  5. **Raw signal variability** - May benefit from time-frequency features

---

## Epilepsy 122MB - Analysis

**Overall Accuracy**: 0.2952

**Class-wise Performance**:
  - Non-seizure: P=0.000, R=0.000, F1=0.000
  - Seizure: P=0.295, R=1.000, F1=0.456

**Weakest Class**: Non-seizure (F1=0.000)

**Common Misclassifications**:
  - Non-seizure → Seizure: 25956 errors

**Potential Failure Modes**:
  2. **Poor Non-seizure detection** - May need class-specific features
  4. **Limited temporal modeling** - CNN lacks explicit sequence modeling
  5. **Raw signal variability** - May benefit from time-frequency features

---

## Merged Epileptic + Bonn - Analysis

**Overall Accuracy**: 0.9524

 **Class Imbalance Detected**: Ratio 3.96:1
  - Majority class: 0 (1425 samples)
  - Minority class: 1 (360 samples)

**Class-wise Performance**:
  - Non-seizure: P=0.963, R=0.978, F1=0.970
  - Seizure: P=0.906, R=0.853, F1=0.878

**Weakest Class**: Seizure (F1=0.878)

**Common Misclassifications**:
  - Seizure → Non-seizure: 53 errors
  - Non-seizure → Seizure: 32 errors

**Potential Failure Modes**:
  1. **Class imbalance** - Model biased toward majority class
  4. **Limited temporal modeling** - CNN lacks explicit sequence modeling
  5. **Raw signal variability** - May benefit from time-frequency features

---

## Overall Conclusions

**Key Findings Across All Datasets:**

1. **Temporal Modeling Gap**: CNN lacks explicit temporal/sequential modeling
   - Recommendation: Add LSTM/GRU or Transformer layers

2. **Class Imbalance Issues**: Several datasets show significant imbalance
   - Recommendation: Implement focal loss or advanced sampling strategies

3. **Raw Signal Limitations**: Direct convolution may miss important patterns
   - Recommendation: Add attention mechanisms or multi-scale feature extraction

4. **Cross-Dataset Variability**: Performance varies significantly
   - Recommendation: Domain adaptation or transfer learning approaches

