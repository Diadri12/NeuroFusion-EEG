import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Modal,
  Animated,
  Dimensions,
  ScrollView,
} from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';

const { width, height } = Dimensions.get('window');

const FILE_TYPES = [
  { type: '.CSV',  color: '#5A189A', icon: 'file-delimited' },
  { type: '.EDF',  color: '#6A1B9A', icon: 'file-chart'     },
  { type: '.TXT',  color: '#7B2CBF', icon: 'file-document'  },
  { type: '.MAT',  color: '#9D4EDD', icon: 'file-code'      },
  { type: '.XLSX', color: '#B983FF', icon: 'file-excel'     },
  { type: '.XLS',  color: '#D0BFFF', icon: 'file-table'     },
];

const TIPS = [
  'Export from your EEG device as .CSV or .EDF for best results.',
  'MATLAB users: save as .MAT for direct compatibility.',
  'Excel exports (.XLSX / .XLS) are supported for pre-processed data.',
  'Text exports from EEG software should be saved as .TXT.',
];

const InvalidFormatDialog = ({ visible, onClose }) => {
  const scaleAnim  = useRef(new Animated.Value(0.7)).current;
  const fadeAnim   = useRef(new Animated.Value(0)).current;
  const shakeAnim  = useRef(new Animated.Value(0)).current;
  const iconBounce = useRef(new Animated.Value(0)).current;
  const tipFade    = useRef(new Animated.Value(0)).current;
  const badgeAnims = useRef(FILE_TYPES.map(() => new Animated.Value(0))).current;

  const [activeTip,    setActiveTip]    = useState(0);
  const [pressedType,  setPressedType]  = useState(null);

  // Cycle tips
  useEffect(() => {
    if (!visible) return;
    const interval = setInterval(() => {
      Animated.timing(tipFade, { toValue: 0, duration: 200, useNativeDriver: true }).start(() => {
        setActiveTip(prev => (prev + 1) % TIPS.length);
        Animated.timing(tipFade, { toValue: 1, duration: 300, useNativeDriver: true }).start();
      });
    }, 3000);
    return () => clearInterval(interval);
  }, [visible]);

  useEffect(() => {
    if (visible) {
      Animated.parallel([
        Animated.spring(scaleAnim, { toValue: 1, friction: 8, tension: 100, useNativeDriver: true }),
        Animated.timing(fadeAnim,  { toValue: 1, duration: 300, useNativeDriver: true }),
      ]).start();

      Animated.sequence([
        Animated.spring(iconBounce, { toValue: -8, friction: 4, tension: 80, useNativeDriver: true }),
        Animated.spring(iconBounce, { toValue: 0,  friction: 4, tension: 80, useNativeDriver: true }),
      ]).start();

      Animated.sequence([
        Animated.timing(shakeAnim, { toValue:  8, duration: 80, useNativeDriver: true }),
        Animated.timing(shakeAnim, { toValue: -8, duration: 80, useNativeDriver: true }),
        Animated.timing(shakeAnim, { toValue:  6, duration: 80, useNativeDriver: true }),
        Animated.timing(shakeAnim, { toValue:  0, duration: 80, useNativeDriver: true }),
      ]).start();

      Animated.stagger(60, badgeAnims.map(anim =>
        Animated.spring(anim, { toValue: 1, friction: 7, tension: 60, useNativeDriver: true })
      )).start();

      Animated.timing(tipFade, { toValue: 1, duration: 400, useNativeDriver: true }).start();

    } else {
      scaleAnim.setValue(0.7);
      fadeAnim.setValue(0);
      shakeAnim.setValue(0);
      iconBounce.setValue(0);
      tipFade.setValue(0);
      badgeAnims.forEach(a => a.setValue(0));
      setActiveTip(0);
      setPressedType(null);
    }
  }, [visible]);

  const handleClose = () => {
    Animated.parallel([
      Animated.timing(scaleAnim, { toValue: 0.7, duration: 200, useNativeDriver: true }),
      Animated.timing(fadeAnim,  { toValue: 0,   duration: 200, useNativeDriver: true }),
    ]).start(() => onClose());
  };

  return (
    <Modal transparent visible={visible} animationType="none" onRequestClose={handleClose}>
      <Animated.View style={[styles.overlay, { opacity: fadeAnim }]}>
        <TouchableOpacity style={styles.backdrop} activeOpacity={1} onPress={handleClose} />

        <Animated.View style={[styles.dialog, { opacity: fadeAnim, transform: [{ scale: scaleAnim }] }]}>

          {/* Close button */}
          <TouchableOpacity style={styles.closeBtn} onPress={handleClose} activeOpacity={0.7}>
            <MaterialCommunityIcons name="close" size={16} color="#666" />
          </TouchableOpacity>

          {/* Icon */}
          <Animated.View style={[
            styles.iconWrap,
            { transform: [{ translateX: shakeAnim }, { translateY: iconBounce }] },
          ]}>
            <View style={styles.iconOuter}>
              <View style={styles.iconInner}>
                <MaterialCommunityIcons name="alert-circle" size={34} color="#E63946" />
              </View>
            </View>
            <Animated.View style={[styles.pulseRing, styles.pulseRing1, { opacity: fadeAnim }]} />
            <Animated.View style={[styles.pulseRing, styles.pulseRing2, { opacity: fadeAnim }]} />
          </Animated.View>

          {/* Title */}
          <Text style={styles.title}>Invalid File Format</Text>
          <Text style={styles.subtitle}>Please upload a valid EEG file format</Text>

          <View style={styles.divider} />

          {/* Badges */}
          <Text style={styles.sectionLabel}>SUPPORTED FORMATS</Text>
          <View style={styles.badgesRow}>
            {FILE_TYPES.map((ft, index) => (
              <Animated.View
                key={ft.type}
                style={{
                  opacity: badgeAnims[index],
                  transform: [{
                    translateY: badgeAnims[index].interpolate({
                      inputRange: [0, 1], outputRange: [12, 0],
                    }),
                  }],
                }}
              >
                <TouchableOpacity
                  onPressIn={() => setPressedType(ft.type)}
                  onPressOut={() => setPressedType(null)}
                  activeOpacity={0.85}
                  style={[
                    styles.badge,
                    { backgroundColor: ft.color },
                    pressedType === ft.type && styles.badgePressed,
                  ]}
                >
                  <MaterialCommunityIcons name={ft.icon} size={11} color="#FFF" />
                  <Text style={styles.badgeText}>{ft.type}</Text>
                </TouchableOpacity>
              </Animated.View>
            ))}
          </View>

          {/* Tip box */}
          <View style={styles.tipBox}>
            <MaterialCommunityIcons name="lightbulb-outline" size={13} color="#B844FF" />
            <Animated.Text style={[styles.tipText, { opacity: tipFade }]}>
              {TIPS[activeTip]}
            </Animated.Text>
          </View>

          {/* Try again button */}
          <TouchableOpacity style={styles.tryAgainBtn} onPress={handleClose} activeOpacity={0.85}>
            <MaterialCommunityIcons name="upload" size={14} color="#FFF" />
            <Text style={styles.tryAgainText}>Try Again</Text>
          </TouchableOpacity>

        </Animated.View>
      </Animated.View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  overlay: {
    flex: 1, backgroundColor: 'rgba(0,0,0,0.55)',
    justifyContent: 'center', alignItems: 'center',
    paddingHorizontal: 20,
  },
  backdrop: { ...StyleSheet.absoluteFillObject },

  dialog: {
    width: '100%',
    maxHeight: height * 0.78,
    backgroundColor: '#FFFFFF',
    borderRadius: 24, padding: 20,
    alignItems: 'center',
    shadowColor: '#000', shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.2, shadowRadius: 20, elevation: 12,
  },

  closeBtn: {
    position: 'absolute', top: 12, right: 12,
    width: 26, height: 26, borderRadius: 13,
    backgroundColor: '#F5F5F5',
    justifyContent: 'center', alignItems: 'center', zIndex: 10,
  },

  // ── Icon ──
  iconWrap: { marginBottom: 12, alignItems: 'center', justifyContent: 'center' },
  iconOuter: {
    width: 68, height: 68, borderRadius: 34,
    backgroundColor: '#FEE2E2',
    alignItems: 'center', justifyContent: 'center',
  },
  iconInner: {
    width: 52, height: 52, borderRadius: 26,
    backgroundColor: '#FECACA',
    alignItems: 'center', justifyContent: 'center',
  },
  pulseRing: {
    position: 'absolute', borderRadius: 999,
    borderWidth: 1.5, borderColor: '#E63946',
  },
  pulseRing1: { width: 78,  height: 78,  opacity: 0.2  },
  pulseRing2: { width: 90,  height: 90,  opacity: 0.1  },

  // ── Text ──
  title:    { fontSize: 18, fontWeight: '800', color: '#E63946', marginBottom: 4, textAlign: 'center' },
  subtitle: { fontSize: 12, color: '#6B7280', textAlign: 'center', lineHeight: 18, marginBottom: 12 },
  divider:  { width: '100%', height: 1, backgroundColor: '#F0F0F0', marginBottom: 12 },

  // ── Badges ──
  sectionLabel: { fontSize: 9, fontWeight: '700', color: '#9CA3AF', letterSpacing: 1.2, marginBottom: 10 },
  badgesRow:    { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: 6, marginBottom: 12 },
  badge: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
    paddingHorizontal: 10, paddingVertical: 6,
    borderRadius: 10,
  },
  badgePressed: { opacity: 0.75 },
  badgeText:    { color: '#FFF', fontSize: 11, fontWeight: '700' },

  // ── Tip box ──
  tipBox: {
    flexDirection: 'row', alignItems: 'flex-start', gap: 6,
    backgroundColor: '#F5F0FF', borderRadius: 10,
    padding: 10, width: '100%', marginBottom: 14,
    borderWidth: 1, borderColor: '#E8D5FF',
  },
  tipText: { flex: 1, fontSize: 11, color: '#7C3AED', lineHeight: 16 },

  // ── Button ──
  tryAgainBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    backgroundColor: '#B844FF', borderRadius: 18,
    paddingVertical: 11, paddingHorizontal: 28,
    shadowColor: '#B844FF', shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.3, shadowRadius: 6, elevation: 4,
  },
  tryAgainText: { color: '#FFF', fontSize: 14, fontWeight: '700' },
});

export default InvalidFormatDialog;