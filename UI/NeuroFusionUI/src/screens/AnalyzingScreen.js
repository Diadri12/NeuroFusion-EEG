import React, { useEffect, useRef, useState } from 'react';
import {
  View, Text, StyleSheet, StatusBar,
  SafeAreaView, Platform, Alert, Animated,
} from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import axios from 'axios';
 
// Railway API
const API_URL = 'https://neurofusion-eeg-production.up.railway.app/predict/csv';
 
const SUPPORTED_TYPES = ['.csv', '.txt', '.mat', '.edf', '.xls', '.xlsx'];
 
const STEPS = [
  { label: 'Loading Data',        icon: 'database-import'   },
  { label: 'Extracting Features', icon: 'chart-scatter-plot' },
  { label: 'Running Model',       icon: 'brain'              },
  { label: 'Generating Results',  icon: 'check-circle'       },
];
 
// Component
const AnalyzingScreen = ({ fileUri, fileType }) => {
  const router = useRouter();
 
  const [currentStep, setCurrentStep] = useState(0);
  const [statusMsg,   setStatusMsg]   = useState('Preparing your file');
 
  // Animations
  const fadeAnim    = useRef(new Animated.Value(0)).current;
  const pulseAnim   = useRef(new Animated.Value(1)).current;
  const ring1       = useRef(new Animated.Value(1)).current;
  const ring2       = useRef(new Animated.Value(1)).current;
  const ring3       = useRef(new Animated.Value(1)).current;
  const progressAnim= useRef(new Animated.Value(0)).current;
  const stepAnims   = useRef(STEPS.map(() => new Animated.Value(0))).current;
 
  // ── Mount: start animations then analysis ──
  useEffect(() => {
    // Fade in
    Animated.timing(fadeAnim, { toValue: 1, duration: 500, useNativeDriver: true }).start();
 
    // Brain pulse
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.12, duration: 900, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1,    duration: 900, useNativeDriver: true }),
      ])
    ).start();
 
    // Ripple rings
    const ripple = (anim, delay) =>
      Animated.loop(
        Animated.sequence([
          Animated.delay(delay),
          Animated.timing(anim, { toValue: 1.6, duration: 1400, useNativeDriver: true }),
          Animated.timing(anim, { toValue: 1,   duration: 0,    useNativeDriver: true }),
        ])
      ).start();
    ripple(ring1, 0);
    ripple(ring2, 450);
    ripple(ring3, 900);
 
    // Stagger step rows in
    Animated.stagger(150,
      stepAnims.map(a =>
        Animated.spring(a, { toValue: 1, friction: 7, tension: 50, useNativeDriver: true })
      )
    ).start();
 
    // Start analysis after short delay so UI renders first
    const timer = setTimeout(() => {
      if (fileUri) startAnalysis();
    }, 600);
 
    return () => clearTimeout(timer);
  }, []);
 
  // Animate progress bar on step change
  useEffect(() => {
    Animated.timing(progressAnim, {
      toValue: (currentStep + 1) / STEPS.length,
      duration: 400,
      useNativeDriver: false,
    }).start();
  }, [currentStep]);
 
  const advanceStep = (step, msg) => {
    setCurrentStep(step);
    setStatusMsg(msg);
  };
 
  // Main analysis flow
  const startAnalysis = async () => {
    // Resolve filename and extension
    let name      = fileUri.split('/').pop() || 'uploaded_file';
    let extension;
 
    if (Platform.OS === 'web') {
      if (!fileType) {
        Alert.alert('Error', 'Cannot detect file type.');
        router.back();
        return;
      }
      extension = fileType.toLowerCase().replace('.', '');
      name      = `${name.replace(/\.[^/.]+$/, '')}.${extension}`;
    } else {
      extension = name.split('.').pop()?.toLowerCase();
    }
 
    if (!SUPPORTED_TYPES.includes(`.${extension}`)) {
      Alert.alert(
        'Unsupported Format',
        `The file type ".${extension}" is not supported.\nPlease upload a CSV file.`
      );
      router.back();
      return;
    }
 
    await analyzeFile(name, extension);
  };
 
  const analyzeFile = async (fileName, extension) => {
    try {
      // Load
      advanceStep(0, 'Loading your EEG file');
 
      const formData = new FormData();
 
      if (Platform.OS === 'web') {
        const fetchResp = await fetch(fileUri);
        const blob      = await fetchResp.blob();
        formData.append('file', new File([blob], fileName));
      } else {
        formData.append('file', {
          uri:  fileUri,
          name: fileName,
          type: 'application/octet-stream',
        });
      }
 
      await new Promise(r => setTimeout(r, 300));
 
      // Features
      advanceStep(1, 'Extracting EEG features');
      await new Promise(r => setTimeout(r, 300));
 
      // Model call
      advanceStep(2, 'Running NeuroFusion model');
 
      let data;
      if (Platform.OS === 'web') {
        const fetchResp = await fetch(API_URL, { method: 'POST', body: formData });
        if (!fetchResp.ok) {
          const e = await fetchResp.json().catch(() => ({}));
          Alert.alert('Analysis Failed', e?.detail || 'Server error. Please try again.');
          router.back();
          return;
        }
        data = await fetchResp.json();
      } else {
        const response = await axios.post(API_URL, formData, { timeout: 120000, validateStatus: () => true });
        if (response.status !== 200) {
          Alert.alert('Analysis Failed', response.data?.detail || 'Server error. Please try again.');
          router.back();
          return;
        }
        data = response.data;
      }
 
      // Results
      advanceStep(3, 'Generating results');
      await new Promise(r => setTimeout(r, 400));
      const urgency    = data.overall_urgency;   // 'critical' | 'high' | 'low'
      const isSeizure  = urgency === 'critical';
 
      // Build params to pass to result screens
      const params = {
        overall_urgency    : urgency,
        overall_status     : data.overall_status,
        overall_colour     : data.overall_colour,
        advice             : data.advice,
        total_windows      : String(data.total_windows ?? 0),
        time_taken_sec     : String(data.time_taken_sec ?? 0),
        file_name          : data.file_name ?? fileName,
        dominant_label     : data.dominant_prediction?.label ?? '',
        interictal_pct     : String(data.class_distribution?.Interictal?.pct  ?? 0),
        preictal_pct       : String(data.class_distribution?.Preictal?.pct    ?? 0),
        ictal_pct          : String(data.class_distribution?.Ictal?.pct       ?? 0),
        interictal_count   : String(data.class_distribution?.Interictal?.count ?? 0),
        preictal_count     : String(data.class_distribution?.Preictal?.count   ?? 0),
        ictal_count        : String(data.class_distribution?.Ictal?.count      ?? 0),
      };
 
      // Route to correct result screen
      if (isSeizure) {
        router.replace({ pathname: '/seizure-detected',    params });
      } else {
        router.replace({ pathname: '/no-seizure-detected', params });
      }
 
    } catch (err) {
      console.error('Analysis error:', err);
      Alert.alert(
        'Network Error',
        'Could not connect to the server.\nPlease check your connection and try again.'
      );
      router.back();
    }
  };
 
  // Progress bar interpolation
  const progressWidth = progressAnim.interpolate({
    inputRange:  [0, 1],
    outputRange: ['0%', '100%'],
  });
 
  // Render
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#B844FF" />
 
      <Animated.View style={[styles.content, { opacity: fadeAnim }]}>
 
        {/* Brain with ripple rings */}
        <View style={styles.brainWrap}>
          <Animated.View style={[styles.ring, styles.ring3, { transform: [{ scale: ring3 }] }]} />
          <Animated.View style={[styles.ring, styles.ring2, { transform: [{ scale: ring2 }] }]} />
          <Animated.View style={[styles.ring, styles.ring1, { transform: [{ scale: ring1 }] }]} />
          <Animated.View style={[styles.iconContainer, { transform: [{ scale: pulseAnim }] }]}>
            <MaterialCommunityIcons name="brain" size={64} color="#FFFFFF" />
          </Animated.View>
        </View>
 
        {/* Title */}
        <Text style={styles.title}>Analyzing EEG Signals</Text>
        <Text style={styles.subtitle}>{statusMsg}</Text>
 
        {/* Progress bar */}
        <View style={styles.progressBg}>
          <Animated.View style={[styles.progressFill, { width: progressWidth }]} />
        </View>
        <Text style={styles.progressLabel}>
          Step {currentStep + 1} of {STEPS.length}
        </Text>
 
        {/* Steps */}
        <View style={styles.stepsContainer}>
          {STEPS.map((step, index) => {
            const isDone   = index < currentStep;
            const isActive = index === currentStep;
            return (
              <Animated.View
                key={index}
                style={[
                  styles.stepRow,
                  {
                    opacity: stepAnims[index],
                    transform: [{
                      translateX: stepAnims[index].interpolate({
                        inputRange: [0, 1], outputRange: [-20, 0],
                      }),
                    }],
                  },
                ]}
              >
                {/* Circle */}
                <View style={[
                  styles.stepCircle,
                  isDone   && styles.stepCircleDone,
                  isActive && styles.stepCircleActive,
                ]}>
                  {isDone ? (
                    <MaterialCommunityIcons name="check" size={14} color="#B844FF" />
                  ) : (
                    <MaterialCommunityIcons
                      name={step.icon}
                      size={14}
                      color={isActive ? '#B844FF' : 'rgba(255,255,255,0.4)'}
                    />
                  )}
                </View>
 
                {/* Label */}
                <Text style={[
                  styles.stepText,
                  isDone   && styles.stepTextDone,
                  isActive && styles.stepTextActive,
                ]}>
                  {step.label}
                </Text>
 
                {/* Active dots */}
                {isActive && (
                  <View style={styles.dotsRow}>
                    {[0, 1, 2].map(i => <View key={i} style={styles.dot} />)}
                  </View>
                )}
 
                {/* Done tag */}
                {isDone && <Text style={styles.doneTag}>Done</Text>}
              </Animated.View>
            );
          })}
        </View>
 
        {/* Footer */}
        <View style={styles.footer}>
          <MaterialCommunityIcons name="shield-check" size={14} color="rgba(255,255,255,0.6)" />
          <Text style={styles.footerText}>Powered by NeuroFusion-EEG · BiLSTM + SupCon</Text>
        </View>
 
      </Animated.View>
    </SafeAreaView>
  );
};
 
// Styles
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#B844FF' },
  content:   { flex: 1, padding: 24, justifyContent: 'center' },
 
  // Brain
  brainWrap: {
    alignSelf: 'center', alignItems: 'center', justifyContent: 'center',
    marginBottom: 28, width: 160, height: 160,
  },
  ring: {
    position: 'absolute', borderRadius: 999,
    borderWidth: 1.5, borderColor: 'rgba(255,255,255,0.3)',
  },
  ring1: { width: 120, height: 120 },
  ring2: { width: 140, height: 140 },
  ring3: { width: 160, height: 160 },
  iconContainer: {
    width: 100, height: 100, borderRadius: 50,
    borderWidth: 2, borderColor: '#FFFFFF',
    backgroundColor: 'rgba(255,255,255,0.15)',
    justifyContent: 'center', alignItems: 'center',
  },
 
  // Titles
  title:    { fontSize: 24, fontWeight: '800', color: '#FFF', textAlign: 'center', marginBottom: 6 },
  subtitle: { fontSize: 14, color: 'rgba(255,255,255,0.8)', textAlign: 'center', marginBottom: 20, minHeight: 20 },
 
  // Progress
  progressBg: {
    height: 6, backgroundColor: 'rgba(255,255,255,0.25)',
    borderRadius: 6, overflow: 'hidden', marginBottom: 6,
  },
  progressFill:  { height: '100%', backgroundColor: '#FFF', borderRadius: 6 },
  progressLabel: { fontSize: 11, color: 'rgba(255,255,255,0.7)', textAlign: 'right', marginBottom: 24 },
 
  // Steps
  stepsContainer: { gap: 14, marginBottom: 28 },
  stepRow:        { flexDirection: 'row', alignItems: 'center', gap: 14 },
  stepCircle: {
    width: 32, height: 32, borderRadius: 16,
    borderWidth: 2, borderColor: 'rgba(255,255,255,0.3)',
    backgroundColor: 'transparent',
    justifyContent: 'center', alignItems: 'center',
  },
  stepCircleActive: { borderColor: '#FFF', backgroundColor: '#FFF' },
  stepCircleDone:   { borderColor: '#FFF', backgroundColor: '#FFF' },
  stepText:         { flex: 1, fontSize: 15, color: 'rgba(255,255,255,0.5)', fontWeight: '500' },
  stepTextActive:   { color: '#FFF', fontWeight: '700' },
  stepTextDone:     { color: 'rgba(255,255,255,0.75)', fontWeight: '600' },
 
  dotsRow: { flexDirection: 'row', gap: 4 },
  dot:     { width: 5, height: 5, borderRadius: 3, backgroundColor: 'rgba(255,255,255,0.7)' },
  doneTag: { fontSize: 11, color: 'rgba(255,255,255,0.6)', fontWeight: '600' },
 
  // Footer
  footer:     { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6 },
  footerText: { fontSize: 11, color: 'rgba(255,255,255,0.5)' },
});
 
export default AnalyzingScreen;