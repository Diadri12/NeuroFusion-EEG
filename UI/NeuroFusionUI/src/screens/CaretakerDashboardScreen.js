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

const PATIENTS = [
  {
    id: '1',
    name: 'Diadri Weerasekera',
    age: 24,
    status: 'safe',
    lastScan: '2h ago',
    riskScore: 12,
    lastResult: 'No Seizure',
  },
  {
    id: '2',
    name: 'Priya Fernando',
    age: 31,
    status: 'warning',
    lastScan: '45m ago',
    riskScore: 67,
    lastResult: 'Preictal',
  },
  {
    id: '3',
    name: 'Amal Perera',
    age: 19,
    status: 'safe',
    lastScan: '4h ago',
    riskScore: 8,
    lastResult: 'No Seizure',
  },
];


const CaretakerDashboardScreen = ({ navigation }) => {
  const fadeAnim  = useRef(new Animated.Value(0)).current;
  const slideAnims = useRef([
    new Animated.Value(30),
    new Animated.Value(30),
    new Animated.Value(30),
    new Animated.Value(30),
  ]).current;

  const [expandedPatient, setExpandedPatient] = useState(null);
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

  const togglePatient = (id) => {
    setExpandedPatient(prev => (prev === id ? null : id));
  };

  const safePts    = PATIENTS.filter(p => p.status === 'safe').length;
  const warningPts = PATIENTS.filter(p => p.status === 'warning').length;

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#B844FF" />

      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>

        {/* ── Header ── */}
        <Animated.View style={[styles.header, { opacity: fadeAnim }]}>
          <View style={styles.statusBar}>
            <Text style={styles.time}>10:00</Text>
            <View style={styles.statusIcons}>
              <MaterialCommunityIcons name="signal"   size={16} color="#FFF" />
              <MaterialCommunityIcons name="wifi"     size={16} color="#FFF" style={{ marginLeft: 4 }} />
              <MaterialCommunityIcons name="battery"  size={16} color="#FFF" style={{ marginLeft: 4 }} />
            </View>
          </View>

          <View style={styles.userSection}>
            <View style={styles.avatarLarge}>
              <MaterialCommunityIcons name="account-heart" size={40} color="#B844FF" />
            </View>
            <Text style={styles.welcomeText}>Welcome Back, {displayName}</Text>
            <View style={styles.roleBadge}>
              <MaterialCommunityIcons name="shield-account" size={14} color="#B844FF" />
              <Text style={styles.roleBadgeText}>Caretaker</Text>
            </View>
          </View>
        </Animated.View>

        {/* ── Alert Banner ── */}
        <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY: slideAnims[0] }] }}>
          <View style={styles.alertBanner}>
            <MaterialCommunityIcons name="alert-circle" size={22} color="#F59E0B" />
            <View style={styles.alertText}>
              <Text style={styles.alertTitle}>Attention Required</Text>
              <Text style={styles.alertSub}>Priya Fernando — elevated risk detected</Text>
            </View>
          </View>
        </Animated.View>

        {/* ── Summary Row ── */}
        <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY: slideAnims[1] }] }}>
          <View style={styles.summaryRow}>
            <View style={styles.summaryCard}>
              <Text style={styles.summaryNum}>{PATIENTS.length}</Text>
              <Text style={styles.summaryLabel}>Patients</Text>
            </View>
            <View style={[styles.summaryCard, { borderTopColor: '#10B981' }]}>
              <Text style={[styles.summaryNum, { color: '#10B981' }]}>{safePts}</Text>
              <Text style={styles.summaryLabel}>Safe</Text>
            </View>
            <View style={[styles.summaryCard, { borderTopColor: '#F59E0B' }]}>
              <Text style={[styles.summaryNum, { color: '#F59E0B' }]}>{warningPts}</Text>
              <Text style={styles.summaryLabel}>At Risk</Text>
            </View>
            <View style={[styles.summaryCard, { borderTopColor: '#EF4444' }]}>
              <Text style={[styles.summaryNum, { color: '#EF4444' }]}>0</Text>
              <Text style={styles.summaryLabel}>Critical</Text>
            </View>
          </View>
        </Animated.View>

        {/* ── Patient List ── */}
        <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY: slideAnims[2] }] }}>
          <View style={styles.card}>
            <View style={styles.cardHeader}>
              <MaterialCommunityIcons name="account-group" size={26} color="#B844FF" />
              <Text style={styles.cardTitle}>Your Patients</Text>
            </View>

            {PATIENTS.map((patient, index) => {
              const isSafe     = patient.status === 'safe';
              const isExpanded = expandedPatient === patient.id;
              const borderColor = isSafe ? '#10B981' : '#F59E0B';
              const iconName    = isSafe ? 'check-circle' : 'alert-circle';
              const iconColor   = isSafe ? '#10B981' : '#F59E0B';

              return (
                <View key={patient.id}>
                  {index > 0 && <View style={styles.divider} />}

                  <TouchableOpacity
                    onPress={() => togglePatient(patient.id)}
                    activeOpacity={0.75}
                    style={[styles.patientRow, { borderLeftColor: borderColor }]}
                  >
                    {/* Icon */}
                    <View style={[styles.patientIcon, { backgroundColor: isSafe ? '#D1FAE5' : '#FEF3C7' }]}>
                      <MaterialCommunityIcons name={iconName} size={22} color={iconColor} />
                    </View>

                    {/* Info */}
                    <View style={styles.patientInfo}>
                      <Text style={styles.patientName}>{patient.name}</Text>
                      <Text style={styles.patientMeta}>Age {patient.age} · Last scan {patient.lastScan}</Text>

                      {/* Risk bar */}
                      <View style={styles.riskBarBg}>
                        <View style={[
                          styles.riskBarFill,
                          {
                            width: `${patient.riskScore}%`,
                            backgroundColor: patient.riskScore > 50 ? '#F59E0B' : '#10B981',
                          },
                        ]} />
                      </View>
                      <Text style={styles.riskLabel}>Risk: {patient.riskScore}%</Text>
                    </View>

                    {/* Chevron */}
                    <MaterialCommunityIcons
                      name={isExpanded ? 'chevron-up' : 'chevron-down'}
                      size={22}
                      color="#B844FF"
                    />
                  </TouchableOpacity>

                  {/* Expanded detail */}
                  {isExpanded && (
                    <View style={styles.expandedBox}>
                      <View style={styles.expandedRow}>
                        <Text style={styles.expandedKey}>Last Result</Text>
                        <Text style={[
                          styles.expandedVal,
                          { color: patient.status === 'safe' ? '#10B981' : '#F59E0B' },
                        ]}>
                          {patient.lastResult}
                        </Text>
                      </View>
                      <View style={styles.expandedBtnRow}>
                        <TouchableOpacity style={styles.expandedBtnOutline}>
                          <MaterialCommunityIcons name="history" size={14} color="#B844FF" />
                          <Text style={styles.expandedBtnOutlineText}>View History</Text>
                        </TouchableOpacity>
                        <TouchableOpacity style={styles.expandedBtnFill}>
                          <MaterialCommunityIcons name="phone" size={14} color="#FFF" />
                          <Text style={styles.expandedBtnFillText}>Contact</Text>
                        </TouchableOpacity>
                      </View>
                    </View>
                  )}
                </View>
              );
            })}
          </View>
        </Animated.View>

        {/* ── Emergency Contact ── */}
        <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY: slideAnims[3] }] }}>
          <View style={styles.emergencyWrap}>
            <TouchableOpacity style={styles.emergencyBtn} activeOpacity={0.8}>
              <MaterialCommunityIcons name="phone-alert" size={20} color="#EF4444" />
              <Text style={styles.emergencyText}>Emergency Contact</Text>
            </TouchableOpacity>
          </View>
        </Animated.View>

      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container:    { flex: 1, backgroundColor: '#B844FF' },
  scrollView:   { flex: 1 },

  // ── Header ──
  header: {
    backgroundColor: '#B844FF',
    paddingHorizontal: 20,
    paddingTop: 10,
    paddingBottom: 24,
  },
  statusBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
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
    borderRadius: 20,
  },
  roleBadgeText: { color: '#B844FF', fontWeight: '700', fontSize: 12 },

  // ── Alert ──
  alertBanner: {
    flexDirection: 'row', alignItems: 'center', gap: 12,
    backgroundColor: '#FEF3C7',
    marginHorizontal: 20, marginBottom: 14,
    padding: 14, borderRadius: 14,
    borderWidth: 1, borderColor: '#FDE68A',
  },
  alertText:  {},
  alertTitle: { fontSize: 13, fontWeight: '700', color: '#F59E0B' },
  alertSub:   { fontSize: 12, color: '#92400E', marginTop: 1 },

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
    borderTopWidth: 3, borderTopColor: '#B844FF',
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08, shadowRadius: 6, elevation: 3,
  },
  summaryNum:   { fontSize: 22, fontWeight: '800', color: '#B844FF' },
  summaryLabel: { fontSize: 10, color: '#6B7280' },

  // ── Generic card ──
  card: {
    backgroundColor: '#FFF',
    marginHorizontal: 20, marginBottom: 14,
    padding: 20, borderRadius: 16,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1, shadowRadius: 8, elevation: 3,
  },
  cardHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 16 },
  cardTitle:  { fontSize: 18, fontWeight: 'bold', color: '#333', marginLeft: 10 },
  divider:    { height: 1, backgroundColor: '#F3E8FF', marginVertical: 6 },

  // ── Patient row ──
  patientRow: {
    flexDirection: 'row', alignItems: 'center', gap: 12,
    paddingVertical: 10,
    borderLeftWidth: 4, borderLeftColor: '#10B981',
    paddingLeft: 10, borderRadius: 4,
  },
  patientIcon: {
    width: 44, height: 44, borderRadius: 13,
    alignItems: 'center', justifyContent: 'center',
  },
  patientInfo:  { flex: 1 },
  patientName:  { fontSize: 13, fontWeight: '700', color: '#333' },
  patientMeta:  { fontSize: 11, color: '#6B7280', marginTop: 1 },
  riskBarBg: {
    height: 4, backgroundColor: '#EDE9FE',
    borderRadius: 4, overflow: 'hidden', marginTop: 6,
  },
  riskBarFill: { height: '100%', borderRadius: 4 },
  riskLabel:   { fontSize: 10, color: '#9CA3AF', marginTop: 2 },

  // ── Expanded ──
  expandedBox: {
    backgroundColor: '#F5F3FF',
    borderRadius: 12, padding: 12, marginBottom: 6,
  },
  expandedRow: {
    flexDirection: 'row', justifyContent: 'space-between', marginBottom: 10,
  },
  expandedKey: { fontSize: 12, color: '#6B7280' },
  expandedVal: { fontSize: 12, fontWeight: '700' },
  expandedBtnRow: { flexDirection: 'row', gap: 10 },
  expandedBtnOutline: {
    flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    gap: 5, paddingVertical: 9,
    borderRadius: 10, borderWidth: 1.5, borderColor: '#B844FF',
  },
  expandedBtnOutlineText: { fontSize: 12, fontWeight: '600', color: '#B844FF' },
  expandedBtnFill: {
    flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    gap: 5, paddingVertical: 9,
    borderRadius: 10, backgroundColor: '#B844FF',
  },
  expandedBtnFillText: { fontSize: 12, fontWeight: '600', color: '#FFF' },

  // ── Emergency ──
  emergencyWrap: { paddingHorizontal: 20, marginBottom: 100 },
  emergencyBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    gap: 8, paddingVertical: 15,
    borderRadius: 16, backgroundColor: '#FFF',
    borderWidth: 2, borderColor: '#FEE2E2',
  },
  emergencyText: { fontSize: 14, fontWeight: '700', color: '#EF4444' },
});

export default CaretakerDashboardScreen;