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

const RECENT_EVENTS = [
  { id: '1', patient: 'Diadri W.',  type: 'C2 Ictal',     confidence: 94, time: '08:23', date: 'Today',     ictal: true,  preictal: false },
  { id: '2', patient: 'Priya F.',   type: 'C1 Preictal',   confidence: 71, time: '07:44', date: 'Today',     ictal: false, preictal: true  },
  { id: '3', patient: 'Amal P.',    type: 'C0 Interictal', confidence: 98, time: '16:05', date: 'Yesterday', ictal: false, preictal: false },
  { id: '4', patient: 'Nimal S.',   type: 'C0 Interictal', confidence: 91, time: '11:22', date: 'Yesterday', ictal: false, preictal: false },
];

const METRICS = [
  { label: 'Accuracy',  value: '28.2%' },
  { label: 'AUC-ROC',   value: '0.502' },
  { label: 'McNemar p', value: '0.002' },
  { label: 'Kappa',     value: '0.005' },
];


const DoctorDashboardScreen = ({ navigation }) => {
  const fadeAnim   = useRef(new Animated.Value(0)).current;
  const slideAnims = useRef([
    new Animated.Value(30),
    new Animated.Value(30),
    new Animated.Value(30),
    new Animated.Value(30),
  ]).current;

  const [activeTab, setActiveTab]           = useState('all');
  const [expandedEvent, setExpandedEvent]   = useState(null);
  const [displayName, setDisplayName] = useState('');

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

  const filteredEvents = RECENT_EVENTS.filter(e => {
    if (activeTab === 'all')      return true;
    if (activeTab === 'ictal')    return e.ictal;
    if (activeTab === 'preictal') return e.preictal;
    if (activeTab === 'safe')     return !e.ictal && !e.preictal;
    return true;
  });

  const getEventStyle = (event) => {
    if (event.ictal)    return { color: '#EF4444', bg: '#FEE2E2', icon: 'alert-circle'  };
    if (event.preictal) return { color: '#F59E0B', bg: '#FEF3C7', icon: 'lightning-bolt' };
    return               { color: '#10B981', bg: '#D1FAE5', icon: 'check-circle'       };
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#B844FF" />

      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>

        {/* ── Header ── */}
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
              <MaterialCommunityIcons name="doctor" size={40} color="#B844FF" />
            </View>
            <Text style={styles.welcomeText}>Welcome, Dr. {displayName}</Text>
            <View style={styles.roleBadge}>
              <MaterialCommunityIcons name="stethoscope" size={14} color="#B844FF" />
              <Text style={styles.roleBadgeText}>Neurologist</Text>
            </View>
          </View>

          {/* Model chip row */}
          <View style={styles.chipRow}>
            {['BiLSTM + SupCon', 'F1: 0.2763', 'MCC: 0.0064'].map(chip => (
              <View key={chip} style={styles.chip}>
                <Text style={styles.chipText}>{chip}</Text>
              </View>
            ))}
          </View>
        </Animated.View>

        {/* ── Class Summary ── */}
        <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY: slideAnims[0] }] }}>
          <View style={styles.summaryRow}>
            {[
              { label: 'C0\nInterictal', count: 28, color: '#10B981', border: '#10B981' },
              { label: 'C1\nPreictal',   count: 11, color: '#F59E0B', border: '#F59E0B' },
              { label: 'C2\nIctal',      count: 4,  color: '#EF4444', border: '#EF4444' },
            ].map(c => (
              <View key={c.label} style={[styles.summaryCard, { borderTopColor: c.border }]}>
                <Text style={[styles.summaryNum, { color: c.color }]}>{c.count}</Text>
                <Text style={styles.summaryLabel}>{c.label}</Text>
              </View>
            ))}
          </View>
        </Animated.View>

        {/* ── Model Status ── */}
        <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY: slideAnims[1] }] }}>
          <View style={styles.statusCard}>
            <Text style={styles.statusLabel}>Model Status: </Text>
            <View style={styles.statusBadge}>
              <Text style={styles.statusBadgeText}>READY</Text>
            </View>
          </View>
        </Animated.View>

        {/* ── Recent Detections ── */}
        <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY: slideAnims[2] }] }}>
          <View style={styles.card}>
            <View style={styles.cardHeader}>
              <MaterialCommunityIcons name="brain" size={26} color="#B844FF" />
              <Text style={styles.cardTitle}>Recent Detections</Text>
            </View>

            {/* Filter tabs */}
            <View style={styles.tabRow}>
              {[
                { id: 'all',      label: 'All'      },
                { id: 'ictal',    label: 'Ictal'    },
                { id: 'preictal', label: 'Preictal' },
                { id: 'safe',     label: 'Safe'     },
              ].map(tab => (
                <TouchableOpacity
                  key={tab.id}
                  onPress={() => setActiveTab(tab.id)}
                  style={[styles.tab, activeTab === tab.id && styles.tabActive]}
                >
                  <Text style={[styles.tabText, activeTab === tab.id && styles.tabTextActive]}>
                    {tab.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            {/* Event list */}
            {filteredEvents.map((event, index) => {
              const es         = getEventStyle(event);
              const isExpanded = expandedEvent === event.id;

              return (
                <View key={event.id}>
                  {index > 0 && <View style={styles.divider} />}

                  <TouchableOpacity
                    onPress={() => setExpandedEvent(prev => prev === event.id ? null : event.id)}
                    activeOpacity={0.75}
                    style={styles.eventRow}
                  >
                    <View style={[styles.eventIcon, { backgroundColor: es.bg }]}>
                      <MaterialCommunityIcons name={es.icon} size={20} color={es.color} />
                    </View>
                    <View style={styles.eventInfo}>
                      <View style={styles.eventTopRow}>
                        <Text style={styles.eventPatient}>{event.patient}</Text>
                        <Text style={styles.eventTime}>{event.date !== 'Today' ? event.date : event.time}</Text>
                      </View>
                      <View style={styles.eventBottomRow}>
                        <View style={[styles.typeBadge, { backgroundColor: es.bg }]}>
                          <Text style={[styles.typeText, { color: es.color }]}>{event.type}</Text>
                        </View>
                        <Text style={styles.confText}>Conf: {event.confidence}%</Text>
                      </View>
                    </View>
                    <MaterialCommunityIcons
                      name={isExpanded ? 'chevron-up' : 'chevron-down'}
                      size={20}
                      color="#B844FF"
                    />
                  </TouchableOpacity>

                  {/* Expanded */}
                  {isExpanded && (
                    <View style={styles.expandedBox}>
                      <View style={styles.confBarBg}>
                        <View style={[
                          styles.confBarFill,
                          { width: `${event.confidence}%`, backgroundColor: es.color },
                        ]} />
                      </View>
                      <Text style={styles.confBarLabel}>Confidence: {event.confidence}%</Text>
                      <TouchableOpacity style={styles.viewReportBtn}>
                        <MaterialCommunityIcons name="file-chart" size={14} color="#B844FF" />
                        <Text style={styles.viewReportText}>View Full Report</Text>
                      </TouchableOpacity>
                    </View>
                  )}
                </View>
              );
            })}
          </View>
        </Animated.View>

        {/* ── Model Diagnostics ── */}
        <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY: slideAnims[3] }] }}>
          <View style={[styles.card, { marginBottom: 100 }]}>
            <View style={styles.cardHeader}>
              <MaterialCommunityIcons name="chart-line" size={26} color="#B844FF" />
              <Text style={styles.cardTitle}>Model Diagnostics</Text>
            </View>
            <View style={styles.metricsGrid}>
              {METRICS.map(m => (
                <View key={m.label} style={styles.metricCard}>
                  <Text style={styles.metricValue}>{m.value}</Text>
                  <Text style={styles.metricLabel}>{m.label}</Text>
                </View>
              ))}
            </View>
            <View style={styles.modelInfoRow}>
              <MaterialCommunityIcons name="information-outline" size={14} color="#9CA3AF" />
              <Text style={styles.modelInfoText}>
                SupCon + Balanced selected. McNemar p=0.002 (statistically significant).
              </Text>
            </View>
          </View>
        </Animated.View>

      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container:  { flex: 1, backgroundColor: '#B844FF' },
  scrollView: { flex: 1 },

  // ── Header ──
  header: {
    backgroundColor: '#B844FF',
    paddingHorizontal: 20,
    paddingTop: 10,
    paddingBottom: 24,
  },
  statusBar: {
    flexDirection: 'row', justifyContent: 'space-between',
    alignItems: 'center', marginBottom: 20,
  },
  time:        { color: '#FFF', fontSize: 14, fontWeight: '600' },
  statusIcons: { flexDirection: 'row', alignItems: 'center' },
  userSection: { alignItems: 'center' },
  avatarLarge: {
    width: 70, height: 70, borderRadius: 35,
    backgroundColor: '#FFF',
    justifyContent: 'center', alignItems: 'center',
    marginBottom: 10,
  },
  welcomeText: { color: '#FFF', fontSize: 18, fontWeight: '700', marginBottom: 6 },
  roleBadge: {
    flexDirection: 'row', alignItems: 'center', gap: 5,
    backgroundColor: '#FFF',
    paddingHorizontal: 14, paddingVertical: 4,
    borderRadius: 20, marginBottom: 14,
  },
  roleBadgeText: { color: '#B844FF', fontWeight: '700', fontSize: 12 },
  chipRow: { flexDirection: 'row', gap: 6, flexWrap: 'wrap', justifyContent: 'center' },
  chip: {
    backgroundColor: 'rgba(255,255,255,0.18)',
    borderRadius: 8, paddingHorizontal: 10, paddingVertical: 4,
    borderWidth: 1, borderColor: 'rgba(255,255,255,0.25)',
  },
  chipText: { fontSize: 10, fontWeight: '600', color: '#EDE9FE' },

  // ── Summary ──
  summaryRow: {
    flexDirection: 'row',
    marginHorizontal: 20, marginBottom: 14,
    gap: 8,
  },
  summaryCard: {
    flex: 1, backgroundColor: '#FFF',
    borderRadius: 14, paddingVertical: 14,
    alignItems: 'center', gap: 4,
    borderTopWidth: 3,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08, shadowRadius: 6, elevation: 3,
  },
  summaryNum:   { fontSize: 22, fontWeight: '800' },
  summaryLabel: { fontSize: 9, color: '#6B7280', fontWeight: '600', textAlign: 'center' },

  // ── Status card (matches DashboardScreen) ──
  statusCard: {
    backgroundColor: '#E8D5FF',
    marginHorizontal: 20, marginBottom: 14,
    padding: 20, borderRadius: 16,
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
  },
  statusLabel: { fontSize: 16, fontWeight: '600', color: '#333' },
  statusBadge: {
    backgroundColor: '#4CAF50',
    paddingHorizontal: 16, paddingVertical: 6,
    borderRadius: 12,
  },
  statusBadgeText: { color: '#FFF', fontWeight: 'bold', fontSize: 14 },

  // ── Generic card ──
  card: {
    backgroundColor: '#FFF',
    marginHorizontal: 20, marginBottom: 14,
    padding: 20, borderRadius: 16,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1, shadowRadius: 8, elevation: 3,
  },
  cardHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 14 },
  cardTitle:  { fontSize: 18, fontWeight: 'bold', color: '#333', marginLeft: 10 },
  divider:    { height: 1, backgroundColor: '#F3E8FF', marginVertical: 4 },

  // ── Filter tabs ──
  tabRow: { flexDirection: 'row', gap: 6, marginBottom: 12 },
  tab: {
    paddingHorizontal: 12, paddingVertical: 6,
    borderRadius: 20, borderWidth: 1, borderColor: '#E8D5FF',
    backgroundColor: '#FFF',
  },
  tabActive: { backgroundColor: '#B844FF', borderColor: '#B844FF' },
  tabText:       { fontSize: 11, fontWeight: '600', color: '#9CA3AF' },
  tabTextActive: { color: '#FFF' },

  // ── Event row ──
  eventRow: {
    flexDirection: 'row', alignItems: 'center',
    gap: 12, paddingVertical: 10,
  },
  eventIcon: {
    width: 42, height: 42, borderRadius: 13,
    alignItems: 'center', justifyContent: 'center',
  },
  eventInfo:      { flex: 1 },
  eventTopRow:    { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 5 },
  eventBottomRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  eventPatient:   { fontSize: 13, fontWeight: '700', color: '#333' },
  eventTime:      { fontSize: 10, color: '#9CA3AF' },
  typeBadge:      { borderRadius: 6, paddingHorizontal: 8, paddingVertical: 2 },
  typeText:       { fontSize: 10, fontWeight: '700' },
  confText:       { fontSize: 11, color: '#6B7280' },

  expandedBox: {
    backgroundColor: '#F5F3FF',
    borderRadius: 12, padding: 12, marginBottom: 4,
  },
  confBarBg: {
    height: 6, backgroundColor: '#EDE9FE',
    borderRadius: 6, overflow: 'hidden', marginBottom: 4,
  },
  confBarFill:   { height: '100%', borderRadius: 6 },
  confBarLabel:  { fontSize: 11, color: '#6B7280', marginBottom: 10 },
  viewReportBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    gap: 6, paddingVertical: 9,
    borderRadius: 10, borderWidth: 1.5, borderColor: '#B844FF',
  },
  viewReportText: { fontSize: 12, fontWeight: '600', color: '#B844FF' },

  metricsGrid: { flexDirection: 'row', gap: 8, marginBottom: 12 },
  metricCard: {
    flex: 1, backgroundColor: '#F5F3FF',
    borderRadius: 10, padding: 10,
  },
  metricValue: { fontSize: 15, fontWeight: '800', color: '#7C3AED' },
  metricLabel: { fontSize: 9, color: '#6B7280', marginTop: 2 },
  modelInfoRow: {
    flexDirection: 'row', alignItems: 'flex-start', gap: 6,
  },
  modelInfoText: { fontSize: 11, color: '#9CA3AF', flex: 1, lineHeight: 16 },
});

export default DoctorDashboardScreen;