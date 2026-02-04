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