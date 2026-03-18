import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
  SafeAreaView,
  ScrollView,
  Animated,
} from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { auth } from '../../src/config/firebase';
import { onAuthStateChanged } from 'firebase/auth';
import { useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect } from '@react-navigation/native';
 
 
 
const STATS = [
  { label: 'Total Scans', value: '47', icon: 'lightning-bolt' },
  { label: 'This Week',   value: '3',  icon: 'chart-line'     },
  { label: 'Streak Days', value: '12', icon: 'fire'           },
];
 
 
const DashboardScreen = ({ onNavigateToAnalyze }) => {
  const router = useRouter();
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnims = useRef([
    new Animated.Value(30),
    new Animated.Value(30),
    new Animated.Value(30),
    new Animated.Value(30),
  ]).current;
 
  const [expandedHistory, setExpandedHistory] = useState(null);
  const [displayName, setDisplayName] = useState('');
  const [lastResult,    setLastResult]    = useState(null);
  const [recentHistory, setRecentHistory] = useState([]);
 
  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 600,
        useNativeDriver: true,
      }),
      ...slideAnims.map((anim, index) =>
        Animated.spring(anim, {
          toValue: 0,
          delay: index * 100,
          friction: 8,
          tension: 40,
          useNativeDriver: true,
        })
      ),
    ]).start();
  }, []);
 
  useFocusEffect(
    React.useCallback(() => {
      const loadLastResult = async () => {
        try {
          const stored  = await AsyncStorage.getItem('analysisHistory');
          const history = stored ? JSON.parse(stored) : [];
          if (history.length > 0) setLastResult(history[0]);
          else setLastResult(null);
 
          // Build recent list: last seizure record first, then up to 2 others
          // Sort newest-first — keep natural order, do NOT pin seizure to top
          const sorted = [...history].sort((a, b) => new Date(b.date) - new Date(a.date));
          setRecentHistory(sorted.slice(0, 3));
        } catch (e) { console.log('Load last result error:', e); }
      };
      loadLastResult();
    }, [])
  );
 
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user?.displayName) {
        setDisplayName(user.displayName);
      } else if (user?.email) {
        setDisplayName(user.email.split("@")[0]);
      }
    });
    return () => unsubscribe();
  }, []);
 
 
 
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#B844FF" />
 
      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
 
        {/* Header */}
        <Animated.View style={[styles.header, { opacity: fadeAnim }]}>
          <View style={styles.statusBar}>
            <Text style={styles.time}>10:00</Text>
            <View style={styles.statusIcons}>
              <MaterialCommunityIcons name="signal"  size={16} color="#FFF" />
              <MaterialCommunityIcons name="wifi"    size={16} color="#FFF" style={{ marginLeft: 4 }} />
              <MaterialCommunityIcons name="battery" size={16} color="#FFF" style={{ marginLeft: 4 }} />
            </View>
          </View>
          <View style={styles.userSection}>
            <View style={styles.avatarLarge}>
              <MaterialCommunityIcons name="account" size={40} color="#B844FF" />
            </View>
            <Text style={styles.welcomeTextLarge}>Welcome Back, {displayName}</Text>
          </View>
        </Animated.View>
 
        {/* Model Status Card */}
        <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY: slideAnims[0] }] }}>
          <View style={styles.statusCard}>
            <Text style={styles.statusLabel}>Model Status: </Text>
            <View style={styles.statusBadge}>
              <Text style={styles.statusBadgeText}>READY</Text>
            </View>
          </View>
        </Animated.View>
 
        {/* Stats Row */}
        <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY: slideAnims[0] }] }}>
          <View style={styles.statsRow}>
            {STATS.map(s => (
              <View key={s.label} style={styles.statCard}>
                <MaterialCommunityIcons name={s.icon} size={20} color="#B844FF" />
                <Text style={styles.statValue}>{s.value}</Text>
                <Text style={styles.statLabel}>{s.label}</Text>
              </View>
            ))}
          </View>
        </Animated.View>
 
        {/* Analyze EEG Data Card */}
        <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY: slideAnims[1] }] }}>
          <TouchableOpacity
            style={styles.analyzeCard}
            onPress={onNavigateToAnalyze}
            activeOpacity={0.8}
          >
            <View style={styles.analyzeHeader}>
              <MaterialCommunityIcons name="file-chart" size={32} color="#B844FF" />
              <Text style={styles.analyzeTitle}>Analyze EEG Data</Text>
            </View>
            <Text style={styles.analyzeDescription}>
              Upload EEG files for seizure detection analysis
            </Text>
            <View style={styles.analyzeButton}>
              <Text style={styles.analyzeButtonText}>Start Analysis</Text>
            </View>
          </TouchableOpacity>
        </Animated.View>
 
        {/* Last Result Card */}
        <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY: slideAnims[2] }] }}>
          <View style={styles.lastResultCard}>
            <View style={styles.lastResultHeader}>
              <MaterialCommunityIcons name="history" size={22} color="#B844FF" />
              <Text style={styles.lastResultTitle}>Last Result</Text>
            </View>
            {lastResult ? (
              <View style={styles.lastResultBody}>
                <View style={styles.lastResultLeft}>
                  <MaterialCommunityIcons
                    name={lastResult.urgency === 'critical' ? 'alert-circle' : 'check-circle'}
                    size={18}
                    color={lastResult.urgency === 'critical' ? '#E74C3C' : '#1A8A4A'}
                  />
                  <View>
                    <Text style={[
                      styles.lastResultValue,
                      { color: lastResult.urgency === 'critical' ? '#E74C3C' : '#1A8A4A' }
                    ]}>
                      {lastResult.result}
                    </Text>
                    <Text style={styles.lastResultMeta}>{lastResult.date}</Text>
                  </View>
                </View>
                <View style={[
                  styles.confBadge,
                  { backgroundColor: lastResult.urgency === 'critical' ? '#FEE2E2' : '#D1FAE5' }
                ]}>
                  <Text style={[
                    styles.confBadgeText,
                    { color: lastResult.urgency === 'critical' ? '#E74C3C' : '#1A8A4A' }
                  ]}>
                    {lastResult.totalWindows} windows
                  </Text>
                </View>
              </View>
            ) : (
              <View style={styles.lastResultEmpty}>
                <MaterialCommunityIcons name="chart-timeline-variant" size={20} color="#D1D5DB" />
                <Text style={styles.lastResultEmptyText}>No analysis yet</Text>
              </View>
            )}
          </View>
        </Animated.View>
 
        {/* History Card */}
        <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY: slideAnims[3] }] }}>
          <View style={styles.historyCard}>
            <View style={styles.historyHeader}>
              <MaterialCommunityIcons name="history" size={28} color="#B844FF" />
              <Text style={styles.historyTitle}>History</Text>
            </View>
 
            {recentHistory.length === 0 ? (
              <View style={styles.historyEmpty}>
                <MaterialCommunityIcons name="clipboard-text-off" size={28} color="#E0E0E0" />
                <Text style={styles.historyEmptyText}>No history yet</Text>
              </View>
            ) : recentHistory.map((item, index) => {
                const lastSeizureId = recentHistory.find(
                  h => h.urgency === 'critical' || h.result === 'Seizure Detected'
                )?.id;
                const isSeizure     = item.urgency === 'critical' || item.result === 'Seizure Detected';
                const color         = isSeizure ? '#E74C3C' : '#1A8A4A';
                const bg            = isSeizure ? '#FEE2E2' : '#D1FAE5';
                const icon          = isSeizure ? 'alert-circle' : 'check-circle';
                const isExpanded    = expandedHistory === item.id;
                const isLastSeizure = isSeizure && item.id === lastSeizureId;
 
                return (
                  <View key={item.id}>
                    {index > 0 && <View style={styles.historyDivider} />}
 
                    <TouchableOpacity
                      onPress={() => setExpandedHistory(prev => prev === item.id ? null : item.id)}
                      activeOpacity={0.75}
                      style={styles.historyRow}
                    >
                      <View style={[styles.historyIcon, { backgroundColor: bg }]}>
                        <MaterialCommunityIcons name={icon} size={18} color={color} />
                      </View>
                      <View style={styles.historyInfo}>
                        <View style={styles.historyTopLine}>
                          <Text style={[styles.historyResult, { color }]}>{item.result}</Text>
                        </View>
                        <Text style={styles.historyDate}>{item.date}</Text>
                      </View>
                      <MaterialCommunityIcons
                        name={isExpanded ? 'chevron-up' : 'chevron-down'}
                        size={18}
                        color="#B844FF"
                      />
                    </TouchableOpacity>
 
                    {isExpanded && (
                      <View style={styles.historyExpanded}>
                        <View style={styles.distMiniRow}>
                          <Text style={styles.distMiniLabel}>Interictal</Text>
                          <View style={styles.distMiniBg}>
                            <View style={[styles.distMiniFill, { width: `${item.interictalPct ?? 0}%`, backgroundColor: '#27AE60' }]} />
                          </View>
                          <Text style={styles.distMiniPct}>{item.interictalPct ?? 0}%</Text>
                        </View>
                        <View style={styles.distMiniRow}>
                          <Text style={styles.distMiniLabel}>Preictal</Text>
                          <View style={styles.distMiniBg}>
                            <View style={[styles.distMiniFill, { width: `${item.preictalPct ?? 0}%`, backgroundColor: '#F39C12' }]} />
                          </View>
                          <Text style={styles.distMiniPct}>{item.preictalPct ?? 0}%</Text>
                        </View>
                        <View style={styles.distMiniRow}>
                          <Text style={styles.distMiniLabel}>Ictal</Text>
                          <View style={styles.distMiniBg}>
                            <View style={[styles.distMiniFill, { width: `${item.ictalPct ?? 0}%`, backgroundColor: '#E74C3C' }]} />
                          </View>
                          <Text style={styles.distMiniPct}>{item.ictalPct ?? 0}%</Text>
                        </View>
                      </View>
                    )}
                  </View>
                );
              })}
 
            {/* View Full History button — navigates to history screen */}
            <TouchableOpacity
              style={styles.viewHistoryButton}
              onPress={() => router.push('/history')}
              activeOpacity={0.8}
            >
              <MaterialCommunityIcons name="history" size={18} color="#B844FF" />
              <Text style={styles.viewHistoryText}>View Full History</Text>
            </TouchableOpacity>
          </View>
        </Animated.View>
 
      </ScrollView>
    </SafeAreaView>
  );
};
 
const styles = StyleSheet.create({
  container:  { flex: 1, backgroundColor: '#B844FF' },
  scrollView: { flex: 1 },
  header: {
    backgroundColor: '#B844FF',
    paddingHorizontal: 20, paddingTop: 10, paddingBottom: 24,
  },
  statusBar: {
    flexDirection: 'row', justifyContent: 'space-between',
    alignItems: 'center', marginBottom: 20,
  },
  time:        { color: '#FFFFFF', fontSize: 14, fontWeight: '600' },
  statusIcons: { flexDirection: 'row', alignItems: 'center' },
  userSection: { alignItems: 'center' },
  avatarLarge: {
    width: 70, height: 70, borderRadius: 35,
    backgroundColor: '#FFFFFF',
    justifyContent: 'center', alignItems: 'center', marginBottom: 12,
  },
  welcomeTextLarge: { color: '#FFFFFF', fontSize: 18, fontWeight: '700' },
  statusCard: {
    backgroundColor: '#E8D5FF',
    marginHorizontal: 20, marginBottom: 16,
    padding: 20, borderRadius: 16,
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
  },
  statusLabel:     { fontSize: 16, fontWeight: '600', color: '#333' },
  statusBadge:     { backgroundColor: '#4CAF50', paddingHorizontal: 16, paddingVertical: 6, borderRadius: 12 },
  statusBadgeText: { color: '#FFFFFF', fontWeight: 'bold', fontSize: 14 },
  analyzeCard: {
    backgroundColor: '#FFFFFF',
    marginHorizontal: 20, marginBottom: 16, padding: 20, borderRadius: 16,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1, shadowRadius: 8, elevation: 3,
  },
  analyzeHeader:      { flexDirection: 'row', alignItems: 'center', marginBottom: 12 },
  analyzeTitle:       { fontSize: 18, fontWeight: 'bold', color: '#333', marginLeft: 12 },
  analyzeDescription: { fontSize: 14, color: '#666', marginBottom: 16, lineHeight: 20 },
  analyzeButton:      { backgroundColor: '#B844FF', paddingVertical: 12, borderRadius: 12, alignItems: 'center' },
  analyzeButtonText:  { color: '#FFFFFF', fontWeight: '600', fontSize: 16 },
  historyCard: {
    backgroundColor: '#FFFFFF',
    marginHorizontal: 20, marginBottom: 100, padding: 20, borderRadius: 16,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1, shadowRadius: 8, elevation: 3,
  },
  historyHeader:  { flexDirection: 'row', alignItems: 'center', marginBottom: 16 },
  historyTitle:   { fontSize: 18, fontWeight: 'bold', color: '#333', marginLeft: 12 },
  historyDivider: { height: 1, backgroundColor: '#F0F0F0', marginVertical: 4 },
  historyEmpty:     { alignItems: 'center', paddingVertical: 24, gap: 8 },
  historyEmptyText: { fontSize: 13, color: '#D1D5DB' },
  historyTopLine:   { flexDirection: 'row', alignItems: 'center', gap: 6, flexWrap: 'wrap' },
  latestBadge:      { backgroundColor: '#FEE2E2', borderRadius: 6, paddingHorizontal: 6, paddingVertical: 2 },
  latestBadgeText:  { fontSize: 9, fontWeight: '800', color: '#E74C3C', letterSpacing: 0.3 },
  distMiniRow:      { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 6 },
  distMiniLabel:    { width: 60, fontSize: 11, color: '#666', fontWeight: '600' },
  distMiniBg:       { flex: 1, height: 5, backgroundColor: '#E0E0E0', borderRadius: 3, overflow: 'hidden' },
  distMiniFill:     { height: '100%', borderRadius: 3 },
  distMiniPct:      { width: 30, fontSize: 11, color: '#555', fontWeight: '700', textAlign: 'right' },
  viewHistoryButton: {
    marginTop: 16, paddingVertical: 12, alignItems: 'center',
    borderWidth: 1, borderColor: '#B844FF', borderRadius: 12,
    flexDirection: 'row', justifyContent: 'center', gap: 8,
  },
  viewHistoryText: { color: '#B844FF', fontWeight: '600', fontSize: 16 },
 
  statsRow: {
    flexDirection: 'row',
    marginHorizontal: 20, marginBottom: 16, gap: 8,
  },
  statCard: {
    flex: 1, backgroundColor: '#FFFFFF', borderRadius: 14, paddingVertical: 14,
    alignItems: 'center', gap: 5,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08, shadowRadius: 6, elevation: 2,
  },
  statValue: { fontSize: 20, fontWeight: '800', color: '#3B0764' },
  statLabel: { fontSize: 9, color: '#9CA3AF', textAlign: 'center' },
 
  lastResultCard: {
    backgroundColor: '#FFFFFF',
    marginHorizontal: 20, marginBottom: 16, padding: 16, borderRadius: 16,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08, shadowRadius: 6, elevation: 2,
  },
  lastResultHeader: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 10 },
  lastResultTitle:  { fontSize: 14, fontWeight: '700', color: '#333' },
  lastResultBody:   { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  lastResultLeft:       { flexDirection: 'row', alignItems: 'center', gap: 8 },
  lastResultValue:      { fontSize: 15, fontWeight: '800', color: '#4CAF50' },
  lastResultEmpty:      { flexDirection: 'row', alignItems: 'center', gap: 8, paddingVertical: 4 },
  lastResultEmptyText:  { fontSize: 14, color: '#D1D5DB', fontWeight: '500' },
  lastResultMeta:   { fontSize: 11, color: '#9CA3AF', marginTop: 2 },
  confBadge:        { backgroundColor: '#D1FAE5', borderRadius: 8, paddingHorizontal: 10, paddingVertical: 4 },
  confBadgeText:    { fontSize: 11, fontWeight: '700', color: '#4CAF50' },
 
  historyRow: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingVertical: 8 },
  historyIcon: { width: 38, height: 38, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  historyInfo:   { flex: 1 },
  historyDate:   { fontSize: 14, color: '#333', fontWeight: '600', marginBottom: 2 },
  historyResult: { fontSize: 14, fontWeight: '600' },
 
  historyExpanded: { backgroundColor: '#F5F3FF', borderRadius: 10, padding: 12, marginBottom: 4 },
  confLabel:  { fontSize: 11, color: '#6B7280', marginBottom: 6 },
  confBarBg:  { height: 6, backgroundColor: '#E0E0E0', borderRadius: 6, overflow: 'hidden', marginBottom: 4 },
  confBarFill: { height: '100%', borderRadius: 6 },
  confPercent: { fontSize: 12, fontWeight: '700', color: '#333' },
});
 
export default DashboardScreen;