import React, { useEffect, useRef } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet,
  StatusBar, SafeAreaView, ScrollView, Animated,
} from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useLocalSearchParams, useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
 
// Animated bar for class distribution
const StatBar = ({ label, pct, count, color, delay }) => {
  const barAnim  = useRef(new Animated.Value(0)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;
 
  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, { toValue: 1, duration: 400, delay, useNativeDriver: true }),
      Animated.timing(barAnim,  { toValue: pct / 100, duration: 800, delay, useNativeDriver: false }),
    ]).start();
  }, []);
 
  const barWidth = barAnim.interpolate({ inputRange: [0, 1], outputRange: ['0%', '100%'] });
 
  return (
    <Animated.View style={[styles.statRow, { opacity: fadeAnim }]}>
      <View style={styles.statLabelRow}>
        <Text style={styles.statLabel}>{label}</Text>
        <Text style={styles.statValue}>{pct}% <Text style={styles.statCount}>({count})</Text></Text>
      </View>
      <View style={styles.barBg}>
        <Animated.View style={[styles.barFill, { width: barWidth, backgroundColor: color }]} />
      </View>
    </Animated.View>
  );
};
 
// Main Screen
const NoSeizureDetectedScreen = () => {
  const router = useRouter();
  const params = useLocalSearchParams();
 
  // Parse params
  const fileName       = params.file_name      || 'unknown';
  const timeTaken      = params.time_taken_sec || '0';
  const totalWindows   = params.total_windows  || '0';
  const advice         = params.advice         || 'No concerning EEG patterns detected at this time.';
  const interictalPct  = parseFloat(params.interictal_pct  || '0');
  const preictalPct    = parseFloat(params.preictal_pct    || '0');
  const ictalPct       = parseFloat(params.ictal_pct       || '0');
  const interictalCnt  = params.interictal_count || '0';
  const preictalCnt    = params.preictal_count   || '0';
  const ictalCnt       = params.ictal_count      || '0';
 
  // Animations
  const iconScale   = useRef(new Animated.Value(0)).current;
  const contentFade = useRef(new Animated.Value(0)).current;
  const checkBounce = useRef(new Animated.Value(0)).current;
 
  useEffect(() => {
    saveToHistory();
 
    // Icon pop-in
    Animated.spring(iconScale, {
      toValue: 1, friction: 4, tension: 60, useNativeDriver: true,
    }).start();
 
    // Check bounce after icon
    setTimeout(() => {
      Animated.sequence([
        Animated.timing(checkBounce, { toValue: -10, duration: 150, useNativeDriver: true }),
        Animated.spring(checkBounce, { toValue: 0, friction: 4, useNativeDriver: true }),
      ]).start();
    }, 400);
 
    // Content fade in
    Animated.timing(contentFade, {
      toValue: 1, duration: 600, delay: 200, useNativeDriver: true,
    }).start();
  }, []);
 
  const saveToHistory = async () => {
    try {
      const existing = await AsyncStorage.getItem('analysisHistory');
      const history  = existing ? JSON.parse(existing) : [];
      const entry    = {
        id: Date.now().toString(),
        fileName, timeTaken, totalWindows,
        interictalPct, preictalPct, ictalPct,
        interictalCnt, preictalCnt, ictalCnt,
        advice,
        date: new Date().toLocaleString(),
        result: 'No Seizure Detected',
        urgency: 'low',
        colour: 'green',
      };
      await AsyncStorage.setItem('analysisHistory', JSON.stringify([entry, ...history]));
    } catch (e) { console.log('History save error:', e); }
  };
 
  const goToDashboard = () => router.replace('/(tabs)/dashboard');
 
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#1A8A4A" />
 
      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
 
        {/* ── Check Icon ── */}
        <Animated.View style={[styles.iconWrap, { transform: [{ scale: iconScale }] }]}>
          <Animated.View style={[styles.iconRing, { transform: [{ translateY: checkBounce }] }]}>
            <MaterialCommunityIcons name="check-circle" size={72} color="#FFF" />
          </Animated.View>
          <Text style={styles.statusBadge}>ALL CLEAR</Text>
        </Animated.View>
 
        <Animated.View style={{ opacity: contentFade }}>
 
          {/* ── Title ── */}
          <Text style={styles.title}>No Seizure Detected</Text>
          <Text style={styles.subtitle}>EEG analysis shows normal interictal activity</Text>
 
          {/* ── Advice Card ── */}
          <View style={styles.adviceCard}>
            <MaterialCommunityIcons name="shield-check" size={22} color="#1A8A4A" />
            <Text style={styles.adviceText}>{advice}</Text>
          </View>
 
          {/* ── Summary Stats ── */}
          <View style={styles.statsRow}>
            <View style={styles.statCard}>
              <MaterialCommunityIcons name="file-document" size={20} color="rgba(255,255,255,0.7)" />
              <Text style={styles.statCardValue} numberOfLines={1}>{fileName}</Text>
              <Text style={styles.statCardLabel}>File</Text>
            </View>
            <View style={styles.statCard}>
              <MaterialCommunityIcons name="timer" size={20} color="rgba(255,255,255,0.7)" />
              <Text style={styles.statCardValue}>{timeTaken}s</Text>
              <Text style={styles.statCardLabel}>Time taken</Text>
            </View>
            <View style={styles.statCard}>
              <MaterialCommunityIcons name="waves" size={20} color="rgba(255,255,255,0.7)" />
              <Text style={styles.statCardValue}>{totalWindows}</Text>
              <Text style={styles.statCardLabel}>Windows</Text>
            </View>
          </View>
 
          {/* ── Class Distribution ── */}
          <View style={styles.distCard}>
            <Text style={styles.distTitle}>Class Distribution</Text>
            <StatBar label="Interictal"  pct={interictalPct}  count={interictalCnt}  color="#27AE60" delay={100} />
            <StatBar label="Preictal"    pct={preictalPct}    count={preictalCnt}    color="#F39C12" delay={250} />
            <StatBar label="Ictal"       pct={ictalPct}       count={ictalCnt}       color="#E74C3C" delay={400} />
          </View>
 
          {/* ── Disclaimer ── */}
          <Text style={styles.disclaimer}>
            This is not a medical diagnosis. Always consult your doctor
            if you have any health concerns.
          </Text>
 
          {/* ── Button ── */}
          <TouchableOpacity style={styles.button} onPress={goToDashboard} activeOpacity={0.85}>
            <MaterialCommunityIcons name="home" size={20} color="#FFF" />
            <Text style={styles.buttonText}>Back to Dashboard</Text>
          </TouchableOpacity>
 
        </Animated.View>
      </ScrollView>
    </SafeAreaView>
  );
};
 
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#1A8A4A' },
  scroll:    { padding: 24, paddingTop: 16, paddingBottom: 40 },
 
  // Icon
  iconWrap: { alignItems: 'center', marginTop: 8, marginBottom: 24 },
  iconRing: {
    width: 130, height: 130, borderRadius: 65,
    borderWidth: 3, borderColor: 'rgba(255,255,255,0.6)',
    backgroundColor: 'rgba(255,255,255,0.15)',
    justifyContent: 'center', alignItems: 'center',
  },
  statusBadge: {
    marginTop: 10, backgroundColor: 'rgba(0,0,0,0.2)',
    color: '#FFF', fontSize: 12, fontWeight: '800',
    paddingHorizontal: 14, paddingVertical: 4,
    borderRadius: 20, letterSpacing: 1.5,
  },
 
  // Title
  title:    { fontSize: 30, fontWeight: '800', color: '#FFF', textAlign: 'center' },
  subtitle: { fontSize: 14, color: 'rgba(255,255,255,0.75)', textAlign: 'center', marginTop: 6, marginBottom: 20 },
 
  // Advice
  adviceCard: {
    flexDirection: 'row', alignItems: 'flex-start', gap: 10,
    backgroundColor: '#FFF', borderRadius: 14, padding: 16, marginBottom: 20,
  },
  adviceText: { flex: 1, fontSize: 14, color: '#1A8A4A', fontWeight: '600', lineHeight: 20 },
 
  // Stats row
  statsRow: { flexDirection: 'row', gap: 10, marginBottom: 20 },
  statCard: {
    flex: 1, backgroundColor: 'rgba(0,0,0,0.15)', borderRadius: 14,
    padding: 12, alignItems: 'center', gap: 4,
  },
  statCardValue: { fontSize: 14, fontWeight: '700', color: '#FFF', textAlign: 'center' },
  statCardLabel: { fontSize: 11, color: 'rgba(255,255,255,0.6)', textAlign: 'center' },
 
  // Distribution
  distCard: {
    backgroundColor: 'rgba(0,0,0,0.15)', borderRadius: 16,
    padding: 18, marginBottom: 20,
  },
  distTitle: { fontSize: 16, fontWeight: '700', color: '#FFF', marginBottom: 16 },
 
  // Stat bar
  statRow:      { marginBottom: 14 },
  statLabelRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 6 },
  statLabel:    { fontSize: 13, color: 'rgba(255,255,255,0.85)', fontWeight: '600' },
  statValue:    { fontSize: 13, color: '#FFF', fontWeight: '700' },
  statCount:    { color: 'rgba(255,255,255,0.6)', fontWeight: '400' },
  barBg:   { height: 8, backgroundColor: 'rgba(255,255,255,0.2)', borderRadius: 4, overflow: 'hidden' },
  barFill: { height: '100%', borderRadius: 4 },
 
  // Disclaimer
  disclaimer: {
    fontSize: 12, color: 'rgba(255,255,255,0.65)',
    textAlign: 'center', lineHeight: 18,
    marginBottom: 24, paddingHorizontal: 8,
  },
 
  // Button
  button: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    backgroundColor: '#B844FF', borderRadius: 28,
    paddingVertical: 16, gap: 8,
    shadowColor: '#B844FF', shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4, shadowRadius: 8, elevation: 6,
  },
  buttonText: { color: '#FFF', fontSize: 17, fontWeight: '700' },
});
 
export default NoSeizureDetectedScreen;