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