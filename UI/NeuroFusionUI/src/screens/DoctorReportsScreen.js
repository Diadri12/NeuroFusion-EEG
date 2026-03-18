import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet,
  StatusBar, SafeAreaView, ScrollView,
  Animated, Dimensions,
} from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect } from '@react-navigation/native';
 
const { width } = Dimensions.get('window');
 
// Animated bar
const Bar = ({ pct, color, delay = 0 }) => {
  const anim = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(anim, {
      toValue: pct / 100, duration: 800, delay, useNativeDriver: false,
    }).start();
  }, [pct]);
  return (
    <View style={barStyles.bg}>
      <Animated.View style={[barStyles.fill, {
        width: anim.interpolate({ inputRange: [0, 1], outputRange: ['0%', '100%'] }),
        backgroundColor: color,
      }]} />
    </View>
  );
};
const barStyles = StyleSheet.create({
  bg:   { flex: 1, height: 8, backgroundColor: '#F0F0F0', borderRadius: 4, overflow: 'hidden' },
  fill: { height: '100%', borderRadius: 4 },
});
 
// Stat card
const StatCard = ({ icon, label, value, color, fadeAnim, slideAnim }) => (
  <Animated.View style={[
    styles.statCard,
    { opacity: fadeAnim, transform: [{ translateY: slideAnim }] },
  ]}>
    <View style={[styles.statIconWrap, { backgroundColor: color + '20' }]}>
      <MaterialCommunityIcons name={icon} size={22} color={color} />
    </View>
    <Text style={[styles.statValue, { color }]}>{value}</Text>
    <Text style={styles.statLabel}>{label}</Text>
  </Animated.View>
);
 
// Main Screen
const DoctorReportsScreen = () => {
  const [history,   setHistory]   = useState([]);
  const [activeTab, setActiveTab] = useState('overview'); // overview | timeline
 
  const headerFade = useRef(new Animated.Value(0)).current;
  const cardAnims  = useRef([
    { fade: new Animated.Value(0), slide: new Animated.Value(20) },
    { fade: new Animated.Value(0), slide: new Animated.Value(20) },
    { fade: new Animated.Value(0), slide: new Animated.Value(20) },
    { fade: new Animated.Value(0), slide: new Animated.Value(20) },
  ]).current;
 
  useFocusEffect(useCallback(() => {
    loadHistory();
    Animated.timing(headerFade, { toValue: 1, duration: 400, useNativeDriver: true }).start();
    cardAnims.forEach(({ fade, slide }, i) => {
      Animated.parallel([
        Animated.timing(fade,  { toValue: 1, duration: 400, delay: i * 80, useNativeDriver: true }),
        Animated.spring(slide, { toValue: 0, friction: 8, delay: i * 80, useNativeDriver: true }),
      ]).start();
    });
  }, []));
 
  const loadHistory = async () => {
    try {
      const stored = await AsyncStorage.getItem('analysisHistory');
      setHistory(stored ? JSON.parse(stored) : []);
    } catch (e) { console.log('Reports load error:', e); }
  };
 
  // ── Computed stats ──
  const total      = history.length;
  const seizures   = history.filter(h => h.urgency === 'critical' || h.result === 'Seizure Detected').length;
  const clear      = total - seizures;
  const seizurePct = total > 0 ? Math.round((seizures / total) * 100) : 0;
  const clearPct   = total > 0 ? Math.round((clear   / total) * 100) : 0;
 
  // Average class distribution across all records
  const avgInterictal = total > 0
    ? (history.reduce((s, h) => s + (parseFloat(h.interictalPct) || 0), 0) / total).toFixed(1)
    : '0';
  const avgPreictal = total > 0
    ? (history.reduce((s, h) => s + (parseFloat(h.preictalPct)   || 0), 0) / total).toFixed(1)
    : '0';
  const avgIctal = total > 0
    ? (history.reduce((s, h) => s + (parseFloat(h.ictalPct)      || 0), 0) / total).toFixed(1)
    : '0';
 
  const STATS = [
    { icon: 'clipboard-pulse',  label: 'Total Scans',    value: String(total),    color: '#B844FF' },
    { icon: 'alert-circle',     label: 'Seizures',        value: String(seizures), color: '#E74C3C' },
    { icon: 'check-circle',     label: 'Clear',           value: String(clear),    color: '#1A8A4A' },
    { icon: 'percent',          label: 'Seizure Rate',    value: `${seizurePct}%`, color: '#F39C12' },
  ];
 
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#B844FF" />
 
      {/*Header*/}
      <Animated.View style={[styles.header, { opacity: headerFade }]}>
        <View style={styles.headerTop}>
          <MaterialCommunityIcons name="chart-bar" size={26} color="#FFF" />
          <Text style={styles.headerTitle}>Reports</Text>
        </View>
        <Text style={styles.headerSub}>EEG Analysis Summary</Text>
      </Animated.View>
 
      {/* Tab toggle*/}
      <View style={styles.tabRow}>
        {['overview', 'timeline'].map(tab => (
          <TouchableOpacity
            key={tab}
            style={[styles.tabBtn, activeTab === tab && styles.tabBtnActive]}
            onPress={() => setActiveTab(tab)}
          >
            <Text style={[styles.tabBtnText, activeTab === tab && styles.tabBtnTextActive]}>
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
 
      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
 
        {activeTab === 'overview' ? (
          <>
            {/* Stat cards*/}
            <View style={styles.statsGrid}>
              {STATS.map((s, i) => (
                <StatCard
                  key={s.label}
                  icon={s.icon}
                  label={s.label}
                  value={s.value}
                  color={s.color}
                  fadeAnim={cardAnims[i].fade}
                  slideAnim={cardAnims[i].slide}
                />
              ))}
            </View>
 
            {/* Outcome breakdown*/}
            <View style={styles.card}>
              <Text style={styles.cardTitle}>Outcome Breakdown</Text>
              <View style={styles.distRow}>
                <Text style={styles.distLabel}>Seizure</Text>
                <Bar pct={seizurePct} color="#E74C3C" delay={100} />
                <Text style={styles.distPct}>{seizurePct}%</Text>
              </View>
              <View style={styles.distRow}>
                <Text style={styles.distLabel}>Clear</Text>
                <Bar pct={clearPct}   color="#1A8A4A" delay={250} />
                <Text style={styles.distPct}>{clearPct}%</Text>
              </View>
            </View>
 
            {/*Avg class distribution*/}
            <View style={styles.card}>
              <Text style={styles.cardTitle}>Avg Class Distribution</Text>
              <Text style={styles.cardSub}>Across all {total} scans</Text>
              <View style={styles.distRow}>
                <Text style={styles.distLabel}>Interictal</Text>
                <Bar pct={parseFloat(avgInterictal)} color="#27AE60" delay={100} />
                <Text style={styles.distPct}>{avgInterictal}%</Text>
              </View>
              <View style={styles.distRow}>
                <Text style={styles.distLabel}>Preictal</Text>
                <Bar pct={parseFloat(avgPreictal)}   color="#F39C12" delay={250} />
                <Text style={styles.distPct}>{avgPreictal}%</Text>
              </View>
              <View style={styles.distRow}>
                <Text style={styles.distLabel}>Ictal</Text>
                <Bar pct={parseFloat(avgIctal)}      color="#E74C3C" delay={400} />
                <Text style={styles.distPct}>{avgIctal}%</Text>
              </View>
            </View>
 
            {/* Model info*/}
            <View style={styles.card}>
              <Text style={styles.cardTitle}>Model Information</Text>
              {[
                { label: 'Architecture', value: 'Dual-Branch CNN + MLP' },
                { label: 'Training',     value: 'SupCon + Balanced'     },
                { label: 'Macro F1',     value: '0.2763'                },
                { label: 'AUC-ROC',      value: '0.5016'                },
                { label: 'MCC',          value: '+0.0064'               },
                { label: 'McNemar p',    value: '0.0020'                },
              ].map(row => (
                <View key={row.label} style={styles.infoRow}>
                  <Text style={styles.infoLabel}>{row.label}</Text>
                  <Text style={styles.infoValue}>{row.value}</Text>
                </View>
              ))}
            </View>
          </>
        ) : (
          /* Timeline tab*/
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Scan Timeline</Text>
            {history.length === 0 ? (
              <View style={styles.emptyWrap}>
                <MaterialCommunityIcons name="clipboard-text-off" size={48} color="#DDD" />
                <Text style={styles.emptyText}>No scans recorded yet</Text>
              </View>
            ) : (
              [...history]
                .sort((a, b) => new Date(b.date) - new Date(a.date))
                .map((item, index) => {
                  const isSeizure = item.urgency === 'critical' || item.result === 'Seizure Detected';
                  const color     = isSeizure ? '#E74C3C' : '#1A8A4A';
                  const icon      = isSeizure ? 'alert-circle' : 'check-circle';
                  return (
                    <View key={item.id} style={styles.timelineRow}>
                      {/* Line */}
                      <View style={styles.timelineLeft}>
                        <View style={[styles.timelineDot, { backgroundColor: color }]}>
                          <MaterialCommunityIcons name={icon} size={12} color="#FFF" />
                        </View>
                        {index < history.length - 1 && <View style={styles.timelineLine} />}
                      </View>
                      {/* Content */}
                      <View style={styles.timelineContent}>
                        <View style={styles.timelineTopRow}>
                          <Text style={[styles.timelineResult, { color }]}>{item.result}</Text>
                          <Text style={styles.timelineDate}>{item.date}</Text>
                        </View>
                        <Text style={styles.timelineFile} numberOfLines={1}>
                          {item.fileName || 'EEG Recording'}
                        </Text>
                        <Text style={styles.timelineMeta}>
                          {item.totalWindows ?? '—'} windows · {item.timeTaken ?? '—'}s
                        </Text>
                      </View>
                    </View>
                  );
                })
            )}
          </View>
        )}
 
      </ScrollView>
    </SafeAreaView>
  );
};
 
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F4F5F9' },
 
  // Header
  header: {
    backgroundColor: '#B844FF',
    paddingHorizontal: 20, paddingTop: 16, paddingBottom: 24,
  },
  headerTop:  { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 4 },
  headerTitle: { fontSize: 22, fontWeight: '800', color: '#FFF' },
  headerSub:   { fontSize: 13, color: 'rgba(255,255,255,0.75)' },
 
  // Tab toggle
  tabRow: {
    flexDirection: 'row',
    backgroundColor: '#FFF',
    borderBottomWidth: 1, borderBottomColor: '#EFEFEF',
  },
  tabBtn: {
    flex: 1, paddingVertical: 12, alignItems: 'center',
    borderBottomWidth: 2, borderBottomColor: 'transparent',
  },
  tabBtnActive:     { borderBottomColor: '#B844FF' },
  tabBtnText:       { fontSize: 14, fontWeight: '600', color: '#AAA' },
  tabBtnTextActive: { color: '#B844FF' },
 
  scroll: { padding: 16, paddingBottom: 100 },
 
  // Stat grid
  statsGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10, marginBottom: 14 },
  statCard: {
    width: (width - 42) / 2,
    backgroundColor: '#FFF', borderRadius: 16, padding: 16,
    alignItems: 'center', gap: 6,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06, shadowRadius: 6, elevation: 3,
  },
  statIconWrap: { width: 44, height: 44, borderRadius: 22, justifyContent: 'center', alignItems: 'center' },
  statValue:    { fontSize: 24, fontWeight: '800' },
  statLabel:    { fontSize: 11, color: '#999', textAlign: 'center' },
 
  // Card
  card: {
    backgroundColor: '#FFF', borderRadius: 16, padding: 18, marginBottom: 14,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06, shadowRadius: 6, elevation: 3,
  },
  cardTitle: { fontSize: 16, fontWeight: '700', color: '#1A1A2E', marginBottom: 4 },
  cardSub:   { fontSize: 12, color: '#AAA', marginBottom: 14 },
 
  // Distribution row
  distRow:   { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 12 },
  distLabel: { width: 68, fontSize: 12, color: '#555', fontWeight: '600' },
  distPct:   { width: 36, fontSize: 12, color: '#333', fontWeight: '700', textAlign: 'right' },
 
  // Model info
  infoRow:   { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 8, borderBottomWidth: 1, borderBottomColor: '#F5F5F5' },
  infoLabel: { fontSize: 13, color: '#888' },
  infoValue: { fontSize: 13, fontWeight: '700', color: '#333' },
 
  // Timeline
  timelineRow: { flexDirection: 'row', gap: 12, marginBottom: 4 },
  timelineLeft: { alignItems: 'center', width: 24 },
  timelineDot:  { width: 24, height: 24, borderRadius: 12, justifyContent: 'center', alignItems: 'center' },
  timelineLine: { flex: 1, width: 2, backgroundColor: '#EFEFEF', marginTop: 4, marginBottom: 4 },
  timelineContent: { flex: 1, paddingBottom: 16 },
  timelineTopRow:  { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 },
  timelineResult:  { fontSize: 13, fontWeight: '700' },
  timelineDate:    { fontSize: 11, color: '#AAA' },
  timelineFile:    { fontSize: 12, color: '#888', marginBottom: 2 },
  timelineMeta:    { fontSize: 11, color: '#BBB' },
 
  // Empty
  emptyWrap: { alignItems: 'center', paddingVertical: 40, gap: 10 },
  emptyText: { fontSize: 14, color: '#CCC' },
});
 
export default DoctorReportsScreen;