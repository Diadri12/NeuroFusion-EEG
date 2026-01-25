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